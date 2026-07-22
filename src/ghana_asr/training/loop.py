"""Training orchestration for Whisper fine-tuning."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from ghana_asr.config import AppConfig
from ghana_asr.data.dataset import DataCollatorSpeechSeq2SeqWithPadding, GhanaSpeechASRDataset
from ghana_asr.evaluation.metrics import MetricComputer
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

    logger.info("Loading processor/model: %s", cfg.model.pretrained)
    processor = WhisperProcessor.from_pretrained(cfg.model.pretrained)
    model = WhisperForConditionalGeneration.from_pretrained(cfg.model.pretrained)

    model.generation_config.language = cfg.data.whisper_language
    model.generation_config.task = cfg.data.task
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    if cfg.training.gradient_checkpointing:
        model.config.use_cache = False
        model.gradient_checkpointing_enable()

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
    metrics = MetricComputer(processor)

    tcfg = cfg.training
    args = Seq2SeqTrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=tcfg.num_train_epochs,
        per_device_train_batch_size=tcfg.per_device_train_batch_size,
        per_device_eval_batch_size=tcfg.per_device_eval_batch_size,
        gradient_accumulation_steps=tcfg.gradient_accumulation_steps,
        learning_rate=tcfg.learning_rate,
        warmup_ratio=tcfg.warmup_ratio,
        weight_decay=tcfg.weight_decay,
        max_grad_norm=tcfg.max_grad_norm,
        lr_scheduler_type=tcfg.lr_scheduler_type,
        bf16=tcfg.bf16 and torch.cuda.is_available(),
        fp16=tcfg.fp16 and torch.cuda.is_available() and not tcfg.bf16,
        gradient_checkpointing=tcfg.gradient_checkpointing,
        dataloader_num_workers=tcfg.dataloader_num_workers,
        dataloader_pin_memory=tcfg.dataloader_pin_memory,
        eval_strategy=tcfg.evaluation_strategy,
        eval_steps=tcfg.eval_steps,
        save_steps=tcfg.save_steps,
        save_total_limit=tcfg.save_total_limit,
        logging_steps=tcfg.logging_steps,
        predict_with_generate=tcfg.predict_with_generate,
        generation_max_length=tcfg.generation_max_length,
        load_best_model_at_end=tcfg.load_best_model_at_end,
        metric_for_best_model=tcfg.metric_for_best_model,
        greater_is_better=tcfg.greater_is_better,
        report_to=tcfg.report_to,
        remove_unused_columns=tcfg.remove_unused_columns,
        logging_dir=str(out_dir / "tb"),
        run_name=cfg.experiment.name,
    )

    trainer = Seq2SeqTrainer(
        args=args,
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        processing_class=processor,
        compute_metrics=metrics,
    )

    logger.info(
        "Starting training | train=%s eval=%s | device=%s",
        len(train_ds),
        len(eval_ds),
        "cuda" if torch.cuda.is_available() else "cpu",
    )
    train_result = trainer.train()
    metrics_out = train_result.metrics
    trainer.log_metrics("train", metrics_out)
    trainer.save_metrics("train", metrics_out)
    trainer.save_state()

    best_dir = out_dir / "best"
    trainer.save_model(str(best_dir))
    processor.save_pretrained(str(best_dir))

    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    summary = {
        "author": cfg.experiment.author,
        "experiment": cfg.experiment.name,
        "base_model": cfg.model.pretrained,
        "subsets": cfg.data.subsets,
        "train_metrics": metrics_out,
        "eval_metrics": eval_metrics,
    }
    (out_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Training complete. Best model -> %s", best_dir)
    return best_dir
