import torch
import torch.nn as nn
import torchaudio

from config import AudioConfig


def pad_or_trim(waveform: torch.Tensor, target_length: int) -> torch.Tensor:
    """waveform: [1, T]"""
    if waveform.size(-1) < target_length:
        pad = target_length - waveform.size(-1)
        waveform = torch.nn.functional.pad(waveform, (0, pad))
    elif waveform.size(-1) > target_length:
        waveform = waveform[..., :target_length]
    return waveform


class FeatureExtractor(nn.Module):
    """
    Converts waveform [1, T] or [B, 1, T] into:
      Log-Mel: [1, n_mels, time] / [B, 1, n_mels, time]
      MFCC:    [1, n_mfcc, time] / [B, 1, n_mfcc, time]
    """

    def __init__(self, feature_type="logmel", audio_cfg=None, specaugment=False):
        super().__init__()
        self.cfg = audio_cfg or AudioConfig()
        self.feature_type = feature_type.lower()
        self.specaugment = specaugment

        if self.feature_type == "logmel":
            self.transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.cfg.sample_rate,
                n_fft=self.cfg.n_fft,
                win_length=self.cfg.win_length,
                hop_length=self.cfg.hop_length,
                n_mels=self.cfg.n_mels,
                power=2.0,
            )
        elif self.feature_type == "mfcc":
            self.transform = torchaudio.transforms.MFCC(
                sample_rate=self.cfg.sample_rate,
                n_mfcc=self.cfg.n_mfcc,
                melkwargs={
                    "n_fft": self.cfg.n_fft,
                    "win_length": self.cfg.win_length,
                    "hop_length": self.cfg.hop_length,
                    "n_mels": self.cfg.n_mels,
                    "power": 2.0,
                },
            )
        else:
            raise ValueError("feature_type must be 'logmel' or 'mfcc'")

        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=6)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=12)

    @staticmethod
    def normalize(x):
        dims = (-2, -1)
        mean = x.mean(dim=dims, keepdim=True)
        std = x.std(dim=dims, keepdim=True).clamp_min(1e-5)
        return (x - mean) / std

    def forward(self, waveform):
        # Accept [T], [1,T], [B,1,T]
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        x = self.transform(waveform)

        if self.feature_type == "logmel":
            x = torch.log(x.clamp_min(1e-6))

        x = self.normalize(x)

        if self.specaugment and self.training:
            x = self.freq_mask(x)
            x = self.time_mask(x)

        # Single sample: [1,F,T] stays as [1,F,T]
        # Batched input [B,1,T] -> torchaudio usually returns [B,1,F,T].
        return x
