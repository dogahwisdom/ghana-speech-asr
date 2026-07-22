"""Typed config loading for reproducible experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SplitRatios:
    train: float = 0.90
    validation: float = 0.05
    test: float = 0.05

    def validate(self) -> None:
        total = self.train + self.validation + self.test
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")


@dataclass
class DataConfig:
    root: str
    subsets: list[str]
    whisper_language: str
    task: str
    sampling_rate: int
    min_duration_s: float
    max_duration_s: float
    split_ratios: SplitRatios
    max_train_hours_per_language: float
    max_eval_hours_per_language: float
    manifest_dir: str
    num_proc: int = 8


@dataclass
class ModelConfig:
    pretrained: str
    freeze_encoder: bool = False


@dataclass
class TrainingConfig:
    num_train_epochs: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    warmup_ratio: float
    weight_decay: float
    max_grad_norm: float
    lr_scheduler_type: str
    bf16: bool
    fp16: bool
    gradient_checkpointing: bool
    dataloader_num_workers: int
    dataloader_pin_memory: bool
    evaluation_strategy: str
    eval_steps: int
    save_steps: int
    save_total_limit: int
    logging_steps: int
    predict_with_generate: bool
    generation_max_length: int
    load_best_model_at_end: bool
    metric_for_best_model: str
    greater_is_better: bool
    report_to: list[str]
    remove_unused_columns: bool
    group_by_length: bool


@dataclass
class PushConfig:
    repo_id: str
    private: bool = False


@dataclass
class ExperimentConfig:
    name: str
    seed: int
    output_dir: str
    author: str


@dataclass
class AppConfig:
    experiment: ExperimentConfig
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    push: PushConfig
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


def _as_split(d: dict[str, float]) -> SplitRatios:
    ratios = SplitRatios(**d)
    ratios.validate()
    return ratios


def load_config(path: str | Path) -> AppConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    data_raw = dict(raw["data"])
    data_raw["split_ratios"] = _as_split(data_raw["split_ratios"])

    return AppConfig(
        experiment=ExperimentConfig(**raw["experiment"]),
        data=DataConfig(**data_raw),
        model=ModelConfig(**raw["model"]),
        training=TrainingConfig(**raw["training"]),
        push=PushConfig(**raw["push"]),
        raw=raw,
    )
