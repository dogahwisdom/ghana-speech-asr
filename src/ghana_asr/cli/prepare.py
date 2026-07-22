"""CLI: prepare stratified manifests."""

from __future__ import annotations

import argparse

from ghana_asr.config import load_config
from ghana_asr.data.manifests import build_manifests
from ghana_asr.utils.logging import get_logger
from ghana_asr.utils.seeding import seed_everything

logger = get_logger(__name__)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare Ghana Speech ASR manifests (Wisdom Dogah)")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/whisper_akan_ewe.yaml",
        help="Path to experiment YAML config",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    seed_everything(cfg.experiment.seed)
    logger.info("Author: %s | Experiment: %s", cfg.experiment.author, cfg.experiment.name)
    paths = build_manifests(cfg.data, seed=cfg.experiment.seed)
    logger.info("Done. Manifests: %s", {k: str(v) for k, v in paths.items()})


if __name__ == "__main__":
    main()
