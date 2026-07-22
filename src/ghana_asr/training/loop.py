"""Accelerate-based Whisper training loop (explicit, debuggable, production-style)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import yaml
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    get_cosine_schedule_with_warmup,
)

from ghana_asr.config import AppConfig
from ghana_asr.data.dataset import DataCollatorSpeechSeq2SeqWithPadding, GhanaSpeechASRDataset
from ghana_asr.utils.logging import get_logger
from ghana_asr.utils.seeding import seed_everything

logger = get_logger(__name__)


def _maybe_freeze_encoder(model: WhisperForConditionalGeneration, freeze: bool) -> None:
    if not freeze:
        return
    for param in model.model.encoder.parameters():
        param.requires_grad = False
    logger.info("Encoder frozen")


def run_training(cfg: AppConfig) -> Path:
    seed_everything(cfg.experiment.seed)
    out_dir = Path(cfg.experiment.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "run_config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.raw, f, sort_keys=False)

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        mixed_precision="bf16" if cfg.training.bf16 else ("fp16" if cfg.training.fp16 else "no"),
        log_with=cfg.training.report_to,
        project_dir=str(out_dir),
    )
    if accelerator.is_main_process:
        accelerator.init_trackers(cfg.experiment.name)

    logger.info("Loading processor/model: %s", cfg.model.pretrained)
    processor = WhisperProcessor.from_pretrained(cfg.model.pretrained)
    model = WhisperForConditionalGeneration.from_pretrained(cfg.model.pretrained)
    model.generation_config.language = cfg.data.whisper_language
    model.generation_config.task = cfg.data.task
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False
    if cfg.training.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    _maybe_freeze_encoder(model, cfg.model.freeze_encoder)

    manifest_dir = Path(cfg.data.manifest_dir)
    train_ds = GhanaSpeechASRDataset(
        manifest_dir / "train_cache.parquet",
        processor=processor,
        sampling_rate=cfg.data.sampling_rate,
        whisper_language=cfg.data.whisper_language,
        task=cfg.data.task,
    )
    eval_ds = GhanaSpeechASRDataset(
        manifest_dir / "validation_cache.parquet",
        processor=processor,
        sampling_rate=cfg.data.sampling_rate,
        whisper_language=cfg.data.whisper_language,
        task=cfg.data.task,
    )
    collator = DataCollatorSpeechSeq2SeqWithPadding(processor)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.per_device_train_batch_size,
        shuffle=True,
        num_workers=cfg.training.dataloader_num_workers,
        pin_memory=cfg.training.dataloader_pin_memory,
        persistent_workers=cfg.training.dataloader_num_workers > 0,
        collate_fn=collator,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=cfg.training.per_device_eval_batch_size,
        shuffle=False,
        num_workers=min(2, cfg.training.dataloader_num_workers),
        pin_memory=cfg.training.dataloader_pin_memory,
        collate_fn=collator,
    )

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    steps_per_epoch = math.ceil(len(train_loader) / cfg.training.gradient_accumulation_steps)
    if cfg.training.max_steps and cfg.training.max_steps > 0:
        total_steps = cfg.training.max_steps
    else:
        total_steps = steps_per_epoch * cfg.training.num_train_epochs
    warmup = cfg.training.warmup_steps
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup, num_training_steps=total_steps
    )

    model, optimizer, train_loader, eval_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, eval_loader, scheduler
    )

    # Sanity check one batch before the long run.
    model.train()
    first = next(iter(train_loader))
    with torch.no_grad():
        out = model(**first)
        loss0 = out.loss.detach().float().item()
    logger.info("Sanity batch loss=%.4f labels_shape=%s", loss0, tuple(first["labels"].shape))
    if not math.isfinite(loss0) or loss0 <= 0:
        raise RuntimeError(f"Sanity loss unhealthy: {loss0}")

    global_step = 0
    best_loss = float("inf")
    best_dir = out_dir / "best"
    progress = tqdm(total=total_steps, disable=not accelerator.is_main_process, desc="train")

    for epoch in range(cfg.training.num_train_epochs):
        model.train()
        for batch in train_loader:
            with accelerator.accumulate(model):
                # Feature hygiene — rare corrupt clips can poison bf16/fp16 runs.
                if not torch.isfinite(batch["input_features"]).all():
                    logger.warning("Skipping batch with non-finite input features at step=%s", global_step)
                    continue
                outputs = model(**batch)
                loss = outputs.loss
                if not torch.isfinite(loss):
                    logger.warning("Skipping non-finite loss at step=%s", global_step)
                    optimizer.zero_grad(set_to_none=True)
                    continue
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(
                        model.parameters(), cfg.training.max_grad_norm
                    )
                    if grad_norm is not None and not torch.isfinite(grad_norm):
                        logger.warning("Skipping non-finite grad_norm at step=%s", global_step)
                        optimizer.zero_grad(set_to_none=True)
                        continue
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    progress.update(1)

                    if global_step % cfg.training.logging_steps == 0:
                        lr = scheduler.get_last_lr()[0]
                        logger.info(
                            "step=%s epoch=%.3f loss=%.4f grad_norm=%s lr=%.3e",
                            global_step,
                            epoch + global_step / max(total_steps, 1),
                            loss.detach().float().item(),
                            float(grad_norm) if grad_norm is not None else None,
                            lr,
                        )
                        accelerator.log(
                            {
                                "train/loss": loss.detach().float().item(),
                                "train/lr": lr,
                                "train/grad_norm": float(grad_norm)
                                if grad_norm is not None
                                else 0.0,
                            },
                            step=global_step,
                        )

                    if global_step % cfg.training.eval_steps == 0:
                        val_loss = _evaluate(model, eval_loader, accelerator)
                        logger.info("step=%s val_loss=%.4f", global_step, val_loss)
                        accelerator.log({"eval/loss": val_loss}, step=global_step)
                        if val_loss < best_loss:
                            best_loss = val_loss
                            _save_best(accelerator, model, processor, best_dir)
                            logger.info("New best checkpoint -> %s", best_dir)
                        model.train()

                    if global_step % cfg.training.save_steps == 0:
                        ckpt = out_dir / f"checkpoint-{global_step}"
                        _save_best(accelerator, model, processor, ckpt)

                    if global_step >= total_steps:
                        break
        if global_step >= total_steps:
            break

    progress.close()
    # Final save
    if not best_dir.exists():
        _save_best(accelerator, model, processor, best_dir)
    summary = {
        "author": cfg.experiment.author,
        "experiment": cfg.experiment.name,
        "base_model": cfg.model.pretrained,
        "subsets": cfg.data.subsets,
        "best_val_loss": best_loss,
        "global_step": global_step,
    }
    (out_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    accelerator.end_training()
    logger.info("Training complete. Best model -> %s", best_dir)
    return best_dir


@torch.no_grad()
def _evaluate(model, eval_loader, accelerator: Accelerator) -> float:
    model.eval()
    losses = []
    for batch in eval_loader:
        outputs = model(**batch)
        loss = accelerator.gather(outputs.loss.detach()).mean()
        losses.append(float(loss.float().cpu()))
    return sum(losses) / max(len(losses), 1)


def _save_best(accelerator: Accelerator, model, processor, path: Path) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()
    unwrapped = accelerator.unwrap_model(model)
    if accelerator.is_main_process:
        unwrapped.save_pretrained(path)
        processor.save_pretrained(path)
