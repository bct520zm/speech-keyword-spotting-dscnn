import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import torchaudio
import soundfile as sf

from config import AudioConfig
from audio_features import FeatureExtractor, pad_or_trim


def load_waveform(args, cfg):
    if args.wav:
        audio, sr = sf.read(args.wav, dtype="float32")
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        waveform = torch.from_numpy(audio).unsqueeze(0)
        if sr != cfg.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, cfg.sample_rate)
        waveform = pad_or_trim(waveform, cfg.num_samples)
        return waveform, f"WAV: {Path(args.wav).name}"

    ds = torchaudio.datasets.SPEECHCOMMANDS(
        root=args.data_dir,
        download=True,
        subset="testing",
    )
    waveform, sr, label, speaker, utt = ds[args.index]
    if sr != cfg.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, cfg.sample_rate)
    waveform = pad_or_trim(waveform, cfg.num_samples)
    return waveform, f"SpeechCommands label={label}"


def save_waveform_plot(waveform, cfg, output_dir):
    t = np.arange(waveform.size(-1)) / cfg.sample_rate
    plt.figure(figsize=(9, 4))
    plt.plot(t, waveform.squeeze().numpy())
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Waveform")
    plt.tight_layout()
    plt.savefig(output_dir / "waveform.png", dpi=170)
    plt.close()


def save_spectrogram_plot(waveform, cfg, output_dir):
    spec = torch.stft(
        waveform.squeeze(0),
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length,
        window=torch.hann_window(cfg.win_length),
        return_complex=True,
    ).abs().pow(2)
    spec_db = 10 * torch.log10(spec.clamp_min(1e-10))

    plt.figure(figsize=(9, 5))
    plt.imshow(spec_db.numpy(), origin="lower", aspect="auto")
    plt.xlabel("Time Frame")
    plt.ylabel("Frequency Bin")
    plt.title("STFT Spectrogram")
    plt.colorbar(label="Power (dB)")
    plt.tight_layout()
    plt.savefig(output_dir / "spectrogram.png", dpi=170)
    plt.close()


def save_feature_plot(feature, title, filename, output_dir):
    x = feature.squeeze().detach().cpu().numpy()
    plt.figure(figsize=(9, 5))
    plt.imshow(x, origin="lower", aspect="auto")
    plt.xlabel("Time Frame")
    plt.ylabel("Feature Bin")
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=170)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", default=None)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output-dir", default="outputs/features")
    args = parser.parse_args()

    cfg = AudioConfig()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    waveform, description = load_waveform(args, cfg)
    print(description)

    logmel = FeatureExtractor("logmel", cfg, specaugment=False)
    mfcc = FeatureExtractor("mfcc", cfg, specaugment=False)

    save_waveform_plot(waveform, cfg, output_dir)
    save_spectrogram_plot(waveform, cfg, output_dir)
    save_feature_plot(logmel(waveform), "Log-Mel Spectrogram", "logmel.png", output_dir)
    save_feature_plot(mfcc(waveform), "MFCC", "mfcc.png", output_dir)

    print("Saved feature figures to:", output_dir)


if __name__ == "__main__":
    main()
