"""Unit tests for config + deterministic split hashing."""

from __future__ import annotations

from ghana_asr.config import load_config
from ghana_asr.data.manifests import _assign_split, _stable_bucket


def test_load_default_config():
    cfg = load_config("configs/whisper_akan_ewe.yaml")
    assert cfg.experiment.author == "Wisdom Dogah"
    assert "Asante_Twi_twi" in cfg.data.subsets
    assert abs(cfg.data.split_ratios.train + cfg.data.split_ratios.validation + cfg.data.split_ratios.test - 1.0) < 1e-9


def test_stable_bucket_deterministic():
    a = _stable_bucket("Asante_Twi|1CH.1.1461", 42)
    b = _stable_bucket("Asante_Twi|1CH.1.1461", 42)
    c = _stable_bucket("Asante_Twi|1CH.1.1462", 42)
    assert a == b
    assert 0.0 <= a < 1.0
    assert a != c


def test_assign_split_ratios_cover_all():
    ratios = {"train": 0.9, "validation": 0.05, "test": 0.05}
    labels = {_assign_split(f"src-{i}", "Ewe", 42, ratios) for i in range(200)}
    assert labels <= {"train", "validation", "test"}
    assert "train" in labels
