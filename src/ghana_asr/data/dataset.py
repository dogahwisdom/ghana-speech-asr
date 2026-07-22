"""Dataset loading from materialized split caches."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset
from transformers import WhisperProcessor

from ghana_asr.utils.audio import decode_audio_bytes
from ghana_asr.utils.logging import get_logger

logger = get_logger(__name__)


class GhanaSpeechASRDataset(Dataset):
    """Whisper fine-tuning dataset backed by a dense split cache parquet."""

    def __init__(
        self,
        cache_path: str | Path,
        processor: WhisperProcessor,
        sampling_rate: int = 16000,
        whisper_language: str = "yo",
        task: str = "transcribe",
    ) -> None:
        self.cache_path = Path(cache_path)
        if not self.cache_path.exists():
            raise FileNotFoundError(
                f"Missing cache {self.cache_path}. Run: python -m ghana_asr.cli.prepare"
            )
        self._pf = pq.ParquetFile(self.cache_path)
        self._num_rows = self._pf.metadata.num_rows
        self._rg_starts = self._build_row_group_starts()
        self._rg_cache: dict[int, Any] = {}
        self.processor = processor
        self.sampling_rate = sampling_rate
        self.whisper_language = whisper_language
        self.task = task

        tokenizer = processor.tokenizer
        tokenizer.set_prefix_tokens(language=whisper_language, task=task)
        self._prefix_ids: list[int] = list(tokenizer.prefix_tokens)
        self._eos_id = tokenizer.eos_token_id
        self._sot_id = tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
        logger.info(
            "Opened cache %s with %s examples | prefix_ids=%s",
            self.cache_path,
            self._num_rows,
            self._prefix_ids,
        )

    def _build_row_group_starts(self) -> list[int]:
        starts = [0]
        for rg in range(self._pf.num_row_groups):
            starts.append(starts[-1] + self._pf.metadata.row_group(rg).num_rows)
        return starts

    def __len__(self) -> int:
        return self._num_rows

    def _locate(self, idx: int) -> tuple[int, int]:
        if idx < 0 or idx >= self._num_rows:
            raise IndexError(idx)
        lo, hi = 0, len(self._rg_starts) - 2
        while lo <= hi:
            mid = (lo + hi) // 2
            if self._rg_starts[mid + 1] <= idx:
                lo = mid + 1
            elif self._rg_starts[mid] > idx:
                hi = mid - 1
            else:
                return mid, idx - self._rg_starts[mid]
        raise IndexError(idx)

    def _get_row(self, idx: int) -> dict[str, Any]:
        rg_id, offset = self._locate(idx)
        table = self._rg_cache.get(rg_id)
        if table is None:
            table = self._pf.read_row_group(
                rg_id,
                columns=["id", "language", "text", "duration", "subset", "audio"],
            )
            if len(self._rg_cache) >= 4:
                self._rg_cache.clear()
            self._rg_cache[rg_id] = table
        row = table.slice(offset, 1).to_pydict()
        return {k: v[0] for k, v in row.items()}

    def _tokenize_labels(self, text: str) -> list[int]:
        tokenizer = self.processor.tokenizer
        body = tokenizer(text, add_special_tokens=False)["input_ids"]
        labels = self._prefix_ids + body
        if self._eos_id is not None:
            labels = labels + [self._eos_id]
        return labels

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self._get_row(idx)
        audio = decode_audio_bytes(row["audio"], target_sr=self.sampling_rate)
        features = self.processor.feature_extractor(
            audio,
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
        )
        input_features = features.input_features.squeeze(0)
        labels = self._tokenize_labels(row["text"])

        return {
            "input_features": input_features,
            "labels": labels,
            "text": row["text"],
        }


class DataCollatorSpeechSeq2SeqWithPadding:
    def __init__(self, processor: WhisperProcessor, sot_id: int | None = None) -> None:
        self.processor = processor
        self.sot_id = (
            sot_id
            if sot_id is not None
            else processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
        )

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Always pass plain Python lists into tokenizer.pad for reliable masks.
        label_features = [{"input_ids": list(f["labels"])} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if self.sot_id is not None and (labels[:, 0] == self.sot_id).all().item():
            labels = labels[:, 1:]

        # Guard: empty supervision would silently yield loss=0 / NaN grads.
        valid = (labels != -100).any(dim=1)
        if not bool(valid.all().item()):
            raise RuntimeError("Batch contains examples with no supervised label tokens")

        batch["labels"] = labels
        return batch
