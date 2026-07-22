# Ghana Speech ASR — Wisdom Dogah

Production-grade Whisper fine-tuning pipeline for the [Ghana Speech](https://huggingface.co/datasets/ghananlpcommunity/ghana-speech) dataset on NVIDIA H200 (Ghana NLP / AISCA).

**Author:** Wisdom Dogah  
**License (models / data derivatives):** CC BY-NC 4.0  
**Target languages (v1):** Asante Twi · Akuapem Twi · Fante · Ewe

This repository is intentionally structured like a serious ML training codebase — config-driven experiments, indexed manifests, reproducible splits, WER/CER evaluation, and Hub publishing — not a one-off notebook.

## Why this design

| Practice | Implementation |
|---|---|
| Config-driven runs | `configs/*.yaml` |
| Leakage-aware splits | Hash-bucket by `language + source_file` |
| Indexed I/O | Manifest stores `parquet_path` + `row_index` |
| Robust audio decode | `soundfile` on embedded bytes (no `torchcodec` required) |
| Standard trainer | Hugging Face `Seq2SeqTrainer` + Whisper |
| Metrics that matter | WER + CER via `evaluate` / `jiwer` |
| Session safety | Checkpoints + TensorBoard under `outputs/`; push to HF before wipe |

## Ghana NLP session rules (followed)

From the H200 quickstart / model-card template:

1. Dataset lives at `/data/ghana-speech` (42 language subsets, 16 kHz mono).
2. License is **CC BY-NC 4.0** — non-commercial only.
3. Share trained models with **`ghananlpcommunity`** on Hugging Face and include a model card.
4. `/workspace` is **wiped** at window end — export / push before the deadline.
5. Use **`tmux`** for long-running jobs.

## Repository layout

```text
configs/                  Experiment YAML
src/ghana_asr/
  cli/                    prepare | train | evaluate | push
  data/                   manifests + indexed dataset + collator
  evaluation/             WER/CER
  training/               Seq2Seq training loop
  utils/                  seeding, audio, logging
model_card/MODEL_CARD.md  Hub README template (Ghana NLP format)
scripts/                  Thin wrappers for H200 / tmux
tests/                    Unit tests for split hashing & config
```

## Quickstart (H200 Jupyter session)

```bash
# inside /workspace after cloning this repo
python -m pip install -U pip
python -m pip install -e .

# 1) Build stratified manifests (capped hours for a competitive first run)
python -m ghana_asr.cli.prepare --config configs/whisper_akan_ewe.yaml

# 2) Train under tmux
tmux new -s ghana-asr
python -m ghana_asr.cli.train --config configs/whisper_akan_ewe.yaml
# detach: Ctrl-b d

# 3) Evaluate held-out test split
python -m ghana_asr.cli.evaluate --config configs/whisper_akan_ewe.yaml --split test

# 4) Push to Hugging Face (requires `huggingface-cli login`)
python -m ghana_asr.cli.push --config configs/whisper_akan_ewe.yaml
```

## Experiment defaults (v1)

- **Base model:** `openai/whisper-large-v3-turbo` (swap to `openai/whisper-large-v3` in YAML for a longer full-quality run)
- **Train budget:** up to **40 hours / language** (Asante, Akuapem, Fante, Ewe)
- **Eval budget:** up to **2 hours / language** for validation and test
- **Precision:** bf16 · gradient checkpointing · cosine LR · early-best by WER

Edit `configs/whisper_akan_ewe.yaml` to change subsets, hours, or hyperparameters.

## Local development

```bash
python -m pip install -e ".[dev]"
pytest -q
```

## Attribution

Designed, engineered, and maintained by **Wisdom Dogah**.  
Compute courtesy of **AI Skills and Compute Africa (AISCA)** / Ghana NLP H200.
