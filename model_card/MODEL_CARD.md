---
license: cc-by-nc-4.0
language:
  - tw
  - ee
  - fat
tags:
  - ghana-nlp
  - speech
  - ghana-speech
  - whisper
  - automatic-speech-recognition
  - asante-twi
  - akuapem-twi
  - fante
  - ewe
library_name: transformers
pipeline_tag: automatic-speech-recognition
---

# Whisper Akan–Ewe ASR (Ghana Speech)

**Author:** Wisdom Dogah

## Overview
Fine-tuned Whisper model for automatic speech recognition on major Ghanaian languages:
**Asante Twi**, **Akuapem Twi**, **Fante**, and **Ewe**, trained on the Ghana Speech dataset.

## Training data
Trained on the [Ghana Speech](https://huggingface.co/datasets/ghananlpcommunity/ghana-speech)
dataset (audio + text, 42 Ghanaian language subsets), licensed CC BY-NC 4.0.

Subsets used in this release:
- `Asante_Twi_twi`
- `Akuapem_Twi_twi`
- `Fante_fat`
- `Ewe_ewe`

## Intended use & license
**Non-commercial use only** (CC BY-NC 4.0, inherited from the training data).

## How to use
```python
from transformers import pipeline

asr = pipeline(
    "automatic-speech-recognition",
    model="ghananlpcommunity/whisper-akan-ewe-wisdom-dogah",
    chunk_length_s=30,
)
result = asr("path/to/audio.wav")
print(result["text"])
```

## Training details
- Base model / architecture: `openai/whisper-large-v3-turbo`
- Language subset(s): Asante Twi, Akuapem Twi, Fante, Ewe
- Hardware: NVIDIA H200 (Ghana NLP)
- Notes: bf16, cosine LR, stratified splits by language + source chapter, WER/CER monitored on held-out validation

## Acknowledgements
Compute resources provided by **AI Skills and Compute Africa (AISCA)**.
Trained on the Ghana NLP H200 GPU. Please keep derivatives non-commercial and
share improvements back with the Ghana NLP community (`ghananlpcommunity`).

Designed and engineered by **Wisdom Dogah**.
