"""CLI: train Whisper ASR."""

from __future__ import annotations

import argparse

from ghana_asr.config import load_config
from ghana_asr.training.loop import run_training
from ghana_asr.utils.logging import get_logger

logger = get_logger(__name__)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train Ghana Speech Whisper ASR (Wisdom Dogah)")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/whisper_akan_ewe.yaml",
        help="Path to experiment YAML config",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    logger.info("Author: %s | Experiment: %s", cfg.experiment.author, cfg.experiment.name)
    best = run_training(cfg)
    logger.info("Best checkpoint: %s", best)


if __name__ == "__main__":
    main()
