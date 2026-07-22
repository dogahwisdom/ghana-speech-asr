"""CLI: evaluate a checkpoint on the held-out test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from ghana_asr.config import load_config
from ghana_asr.data.dataset import DataCollatorSpeechSeq2SeqWithPadding, GhanaSpeechASRDataset
from ghana_asr.evaluation.metrics import MetricComputer
from ghana_asr.utils.logging import get_logger
from ghana_asr.utils.seeding import seed_everything

logger = get_logger(__name__)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate Ghana ASR checkpoint (Wisdom Dogah)")
    parser.add_argument("--config", type=str, default="configs/whisper_akan_ewe.yaml")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint dir (default: <output_dir>/best)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["validation", "test"],
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    seed_everything(cfg.experiment.seed)
    ckpt = Path(args.checkpoint) if args.checkpoint else Path(cfg.experiment.output_dir) / "best"

    processor = WhisperProcessor.from_pretrained(ckpt)
    model = WhisperForConditionalGeneration.from_pretrained(ckpt)
    model.generation_config.language = cfg.data.whisper_language
    model.generation_config.task = cfg.data.task
    model.generation_config.forced_decoder_ids = None

    ds = GhanaSpeechASRDataset(
        Path(cfg.data.manifest_dir) / f"{args.split}_cache.parquet",
        processor=processor,
        sampling_rate=cfg.data.sampling_rate,
        whisper_language=cfg.data.whisper_language,
        task=cfg.data.task,
    )
    collator = DataCollatorSpeechSeq2SeqWithPadding(processor)
    metrics = MetricComputer(processor)

    targs = Seq2SeqTrainingArguments(
        output_dir=str(Path(cfg.experiment.output_dir) / f"eval_{args.split}"),
        per_device_eval_batch_size=cfg.training.per_device_eval_batch_size,
        dataloader_num_workers=cfg.training.dataloader_num_workers,
        predict_with_generate=True,
        generation_max_length=cfg.training.generation_max_length,
        bf16=cfg.training.bf16 and torch.cuda.is_available(),
        report_to=[],
        remove_unused_columns=False,
    )
    trainer = Seq2SeqTrainer(
        args=targs,
        model=model,
        eval_dataset=ds,
        data_collator=collator,
        processing_class=processor,
        compute_metrics=metrics,
    )
    result = trainer.evaluate()
    out = Path(cfg.experiment.output_dir) / f"metrics_{args.split}.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("Eval (%s): %s", args.split, result)
    logger.info("Wrote %s", out)


if __name__ == "__main__":
    main()
