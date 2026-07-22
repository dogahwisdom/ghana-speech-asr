"""Audio decoding without requiring torchcodec."""

from __future__ import annotations

import io
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torchaudio


def decode_audio_bytes(audio_obj: dict[str, Any], target_sr: int = 16000) -> np.ndarray:
    """Decode HuggingFace-style audio dict ``{"bytes": ..., "path": ...}`` to mono float32."""
    raw = audio_obj.get("bytes")
    if raw is None:
        raise ValueError("audio object missing 'bytes'")

    # Prefer torchaudio (fast resample); fall back to soundfile + linear interp.
    try:
        waveform, sr = torchaudio.load(io.BytesIO(raw))  # [channels, time]
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        audio = waveform.squeeze(0).numpy().astype(np.float32, copy=False)
        return audio
    except Exception:
        audio, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
        if getattr(audio, "ndim", 1) > 1:
            audio = audio.mean(axis=-1)
        if sr != target_sr:
            duration = audio.shape[0] / float(sr)
            target_len = int(round(duration * target_sr))
            if target_len <= 0:
                return np.zeros(0, dtype=np.float32)
            x_old = np.linspace(0.0, 1.0, num=audio.shape[0], endpoint=False)
            x_new = np.linspace(0.0, 1.0, num=target_len, endpoint=False)
            audio = np.interp(x_new, x_old, audio).astype(np.float32)
        else:
            audio = np.asarray(audio, dtype=np.float32)
        return audio
