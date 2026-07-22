"""ASR evaluation metrics (WER / CER)."""

from __future__ import annotations

from dataclasses import dataclass

import evaluate
import numpy as np
from transformers import WhisperProcessor


@dataclass
class MetricComputer:
    processor: WhisperProcessor

    def __post_init__(self) -> None:
        self.wer_metric = evaluate.load("wer")
        self.cer_metric = evaluate.load("cer")

    def __call__(self, pred) -> dict[str, float]:
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # Replace ignored label tokens.
        label_ids = np.where(label_ids != -100, label_ids, self.processor.tokenizer.pad_token_id)

        pred_str = self.processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = self.processor.batch_decode(label_ids, skip_special_tokens=True)

        # Normalize whitespace for fairer WER on orthographic variation.
        pred_str = [" ".join(s.split()) for s in pred_str]
        label_str = [" ".join(s.split()) for s in label_str]

        wer = 100.0 * self.wer_metric.compute(predictions=pred_str, references=label_str)
        cer = 100.0 * self.cer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": float(wer), "cer": float(cer)}
