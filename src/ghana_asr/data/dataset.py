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
    """Whisper fine-tuning dataset backed by a dense split cache parquet.

    Builds a row-group index once, then reads only the needed row group per
    sample (standard pattern for large Parquet corpora).
    """

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
        logger.info("Opened cache %s with %s examples", self.cache_path, self._num_rows)

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
        # Binary search over cumulative starts.
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
            # Keep a small LRU-ish cache of row groups.
            if len(self._rg_cache) >= 4:
                self._rg_cache.clear()
            self._rg_cache[rg_id] = table
        row = table.slice(offset, 1).to_pydict()
        return {k: v[0] for k, v in row.items()}

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self._get_row(idx)
        audio = decode_audio_bytes(row["audio"], target_sr=self.sampling_rate)

        features = self.processor.feature_extractor(
            audio,
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
        )
        input_features = features.input_features.squeeze(0)

        self.processor.tokenizer.set_prefix_tokens(
            language=self.whisper_language,
            task=self.task,
        )
        labels = self.processor.tokenizer(row["text"], return_tensors="pt").input_ids.squeeze(0)

        return {
            "input_features": input_features,
            "labels": labels,
            "id": row["id"],
            "language": row["language"],
            "subset": row["subset"],
            "text": row["text"],
            "duration": float(row["duration"]),
        }


class DataCollatorSpeechSeq2SeqWithPadding:
    def __init__(self, processor: WhisperProcessor) -> None:
        self.processor = processor

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        bos = self.processor.tokenizer.bos_token_id
        if bos is not None and (labels[:, 0] == bos).all().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch
