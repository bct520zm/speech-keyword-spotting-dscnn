# Lightweight Real-Time Keyword Spotting with DS-CNN

A lightweight real-time speech keyword spotting system implemented with
Python, PyTorch and TorchAudio.

The project covers the complete speech algorithm pipeline, including
speech preprocessing, STFT, Log-Mel/MFCC feature extraction, data augmentation,
DS-CNN training, robustness evaluation and real-time microphone inference.

---

## Overview

The system recognizes 10 speech commands:

- yes
- no
- up
- down
- left
- right
- on
- off
- stop
- go

Two additional classes are introduced:

- unknown
- silence

Therefore, the final classification task contains 12 classes.

The complete pipeline is:

```text
Microphone / WAV
       ↓
16 kHz Mono Audio
       ↓
Waveform Preprocessing
       ↓
STFT
       ↓
Mel Filter Bank
       ↓
Log-Mel Spectrogram
       ↓
SpecAugment
       ↓
DS-CNN
       ↓
Softmax
       ↓
Confidence Threshold
       ↓
Temporal Smoothing
       ↓
Keyword Detection
