"""CLI: push best checkpoint + model card to Hugging Face Hub."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import HfApi, login

from ghana_asr.config import load_config
from ghana_asr.utils.logging import get_logger

logger = get_logger(__name__)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Push Ghana ASR model to HF Hub (Wisdom Dogah)")
    parser.add_argument("--config", type=str, default="configs/whisper_akan_ewe.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--model-card", type=str, default="model_card/MODEL_CARD.md")
    parser.add_argument("--token", type=str, default=None, help="HF token (or use HF_TOKEN env)")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    ckpt = Path(args.checkpoint) if args.checkpoint else Path(cfg.experiment.output_dir) / "best"
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    card_src = Path(args.model_card)
    if card_src.exists():
        shutil.copyfile(card_src, ckpt / "README.md")
        logger.info("Attached model card from %s", card_src)

    if args.token:
        login(token=args.token)

    api = HfApi()
    api.create_repo(cfg.push.repo_id, private=cfg.push.private, exist_ok=True)
    api.upload_folder(
        folder_path=str(ckpt),
        repo_id=cfg.push.repo_id,
        commit_message=f"Upload {cfg.experiment.name} by {cfg.experiment.author}",
    )
    logger.info("Pushed to https://huggingface.co/%s", cfg.push.repo_id)


if __name__ == "__main__":
    main()
