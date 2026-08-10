# Robust Keyword Spotting System

A resume-ready speech algorithm project implemented with Python + PyTorch.

## What this project contains

- Google Speech Commands dataset through `torchaudio.datasets.SPEECHCOMMANDS`
- 12-class task:
  - yes, no, up, down, left, right, on, off, stop, go
  - unknown
  - silence
- Waveform preprocessing:
  - mono
  - 16 kHz
  - pad/trim to 1 second
- Data augmentation:
  - random time shift
  - random gain
  - speed perturbation
  - additive noise with random SNR
- Acoustic features:
  - Log-Mel Spectrogram
  - MFCC
  - SpecAugment (time masking + frequency masking)
- Models:
  - CNN
  - DS-CNN
- Evaluation:
  - Accuracy
  - Precision / Recall / F1
  - Confusion Matrix
  - parameter count
  - inference latency
  - optional noise robustness test
- Real-time microphone demo:
  - 1-second rolling buffer
  - 100 ms inference step
  - probability smoothing
  - confidence threshold
  - cooldown

---

# 1. Recommended environment

Use Python 3.10 or 3.11 for the least friction.

Create a virtual environment:

## Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

## macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install PyTorch and TorchAudio first.

CPU-only quick start:

```bash
pip install torch torchaudio
```

If you have an NVIDIA GPU, use the official PyTorch installation command that matches your CUDA environment.

Then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

---

# 2. First run: smoke test

This does NOT download the dataset.

```bash
python smoke_test.py
```

Expected ending:

```text
Smoke test passed.
```

---

# 3. Learn the speech features first

This downloads Speech Commands on first use.

```bash
python plot_features.py
```

Outputs:

```text
outputs/features/waveform.png
outputs/features/spectrogram.png
outputs/features/logmel.png
outputs/features/mfcc.png
```

Study in this order:

1. waveform
2. STFT spectrogram
3. Mel filter bank idea
4. Log-Mel
5. MFCC

Important default parameters:

```text
sample_rate = 16000 Hz
win_length  = 400 samples = 25 ms
hop_length  = 160 samples = 10 ms
n_fft       = 512
n_mels      = 40
n_mfcc      = 20
```

---

# 4. Fastest training path: Quick Mode

Start with DS-CNN + Log-Mel.

```bash
python train.py --quick --model dscnn --feature logmel --epochs 8
```

Quick Mode limits the number of examples per class so you can verify the entire training pipeline quickly.

After training:

```bash
python evaluate.py --quick --checkpoint outputs/best_model.pt
```

For noise robustness:

```bash
python evaluate.py --quick --checkpoint outputs/best_model.pt --noise-test
```

---

# 5. Full experiment for your resume

After Quick Mode works:

```bash
python train.py --model dscnn --feature logmel --epochs 15
```

Then:

```bash
python evaluate.py --checkpoint outputs/best_model.pt --noise-test
```

Record:

- best validation accuracy
- test accuracy
- Precision / Recall / F1
- parameter count
- inference latency
- 20 / 10 / 5 / 0 dB SNR accuracy

Never invent resume numbers. Use your real output.

---

# 6. CNN vs DS-CNN experiment

Quick comparison:

```bash
python compare_models.py --quick --epochs 8
```

Full comparison:

```bash
python compare_models.py --epochs 15
```

Outputs are stored in:

```text
experiments/cnn/
experiments/dscnn/
```

Compare:

| Model | Test Accuracy | Parameters | Latency |
|---|---:|---:|---:|
| CNN | fill in | fill in | fill in |
| DS-CNN | fill in | fill in | fill in |

---

# 7. MFCC vs Log-Mel experiment

Train MFCC:

```bash
python train.py --quick --model dscnn --feature mfcc --epochs 8 --output-dir experiments/dscnn_mfcc
```

Evaluate:

```bash
python evaluate.py --quick --checkpoint experiments/dscnn_mfcc/best_model.pt --output-dir experiments/dscnn_mfcc
```

Train Log-Mel:

```bash
python train.py --quick --model dscnn --feature logmel --epochs 8 --output-dir experiments/dscnn_logmel
```

Evaluate:

```bash
python evaluate.py --quick --checkpoint experiments/dscnn_logmel/best_model.pt --output-dir experiments/dscnn_logmel
```

---

# 8. Real-time microphone demo

After training:

```bash
python realtime_demo.py --checkpoint outputs/best_model.pt
```

You can adjust the trigger threshold:

```bash
python realtime_demo.py --checkpoint outputs/best_model.pt --threshold 0.85
```

Say:

```text
yes
no
up
down
left
right
on
off
stop
go
```

The program ignores `unknown` and `silence`.

---

# 9. Files you should study, in order

Fastest learning route:

1. `config.py`
2. `plot_features.py`
3. `audio_features.py`
4. `dataset.py`
5. `augmentation.py`
6. `models.py`
7. `train.py`
8. `evaluate.py`
9. `realtime_demo.py`
10. `compare_models.py`

Do NOT try to memorize all code.

For every file, answer:

1. What does this module receive?
2. What does it output?
3. Why is it necessary?
4. Which hyperparameters matter?
5. What would happen if it were removed?

---

# 10. Project architecture

```text
Microphone / WAV
      |
      v
16 kHz, mono, 1 second
      |
      v
Waveform augmentation
      |
      v
STFT
      |
      v
Mel Filter Bank
      |
      v
Log-Mel / MFCC
      |
      v
SpecAugment
      |
      v
CNN / DS-CNN
      |
      v
Softmax probabilities
      |
      v
Threshold + smoothing + cooldown
      |
      v
Keyword event
```

---

# 11. Interview points you must understand

## STFT

Why frame speech?

Speech is non-stationary globally but can be treated as approximately stationary over a short frame.

Why 25 ms window and 10 ms hop?

They are common speech-processing choices balancing frequency resolution, time resolution, and computational cost.

## Log-Mel

Understand:

```text
Waveform -> STFT -> Power Spectrum -> Mel Filter Bank -> Log
```

## MFCC

Understand:

```text
Log-Mel -> DCT -> MFCC
```

## Data augmentation

Explain why training with shifted, speed-perturbed, and noisy samples improves robustness to distribution changes.

## Unknown class

Without it, an arbitrary word may be forced into one of the command classes.

## Silence class

Prevents background quiet segments from being interpreted as a command.

## DS-CNN

Normal convolution parameter scale:

```text
K*K*Cin*Cout
```

Depthwise separable convolution parameter scale:

```text
K*K*Cin + Cin*Cout
```

That is why DS-CNN can be much smaller.

## Real-time system

Offline classification alone is not enough.

A usable demo also needs:

- rolling audio buffer
- confidence threshold
- temporal smoothing
- cooldown
- unknown / silence rejection

---

# 12. Resume template

Use REAL numbers from your experiments.

```text
Robust Keyword Spotting System | Python, PyTorch

- Built an end-to-end 12-class keyword spotting pipeline covering waveform
  preprocessing, Log-Mel/MFCC feature extraction, CNN/DS-CNN training and
  real-time microphone inference.
- Implemented time shift, gain, speed perturbation, additive-noise augmentation
  and SpecAugment, and evaluated robustness under multiple SNR conditions.
- Compared CNN and depthwise-separable CNN in test accuracy, trainable
  parameters and inference latency; analyzed errors with precision, recall,
  F1-score and confusion matrices.
- Implemented a real-time detection pipeline using a rolling audio window,
  confidence threshold, probability smoothing and cooldown logic to reduce
  repeated false triggers.
- Achieved XX.XX% test accuracy with X parameters and X ms model inference
  latency on [your hardware].
```

---

# 13. Suggested 2-day emergency schedule

## Day 1 morning
- install environment
- smoke test
- plot features
- understand STFT / Log-Mel / MFCC

## Day 1 afternoon
- read dataset.py
- read models.py
- run Quick training
- run evaluation

## Day 1 evening
- understand training loop
- understand confusion matrix
- run noise test

## Day 2 morning
- run CNN vs DS-CNN comparison
- run Log-Mel vs MFCC comparison

## Day 2 afternoon
- run real-time microphone demo
- adjust threshold
- collect screenshots / metrics

## Day 2 evening
- write README notes in your own words
- fill real numbers into resume
- prepare interview explanations

---

# 14. Troubleshooting

## `ModuleNotFoundError: torchaudio`

Install PyTorch and TorchAudio together:

```bash
pip install torch torchaudio
```

## Torch / TorchAudio version mismatch

They should be matching compatible releases. Reinstall both together using the official PyTorch installation command.

## Microphone does not work

Check available devices:

```python
import sounddevice as sd
print(sd.query_devices())
```

Then verify microphone permission in your operating system.

## DataLoader issues on Windows

Keep:

```text
--num-workers 0
```

The project already defaults to zero.

## Training is too slow

Use:

```bash
python train.py --quick --epochs 5
```

Then switch to full training only after the pipeline works.
