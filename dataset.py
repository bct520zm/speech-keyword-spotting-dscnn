import random
from collections import defaultdict

import torch
from torch.utils.data import Dataset
import torchaudio

from config import AudioConfig, COMMANDS, LABELS, LABEL_TO_INDEX
from audio_features import pad_or_trim
from augmentation import WaveformAugmentor


class BalancedSpeechCommands(Dataset):
    """
    Balanced wrapper around torchaudio.datasets.SPEECHCOMMANDS.

    Classes:
      10 target commands + unknown + silence

    - target command: real Speech Commands sample
    - unknown: sample from all non-target spoken words
    - silence: synthetic zero/very-low-noise waveform
    """

    def __init__(
        self,
        root="data",
        subset="training",
        download=True,
        audio_cfg=None,
        augment=False,
        max_per_class=None,
        seed=42,
    ):
        super().__init__()
        self.cfg = audio_cfg or AudioConfig()
        self.subset = subset
        self.base = torchaudio.datasets.SPEECHCOMMANDS(
            root=root,
            download=download,
            subset=subset,
        )
        self.augment = WaveformAugmentor(sample_rate=self.cfg.sample_rate) if augment else None

        rng = random.Random(seed)
        by_label = defaultdict(list)
        unknown_indices = []

        # get_metadata avoids decoding every waveform.
        for i in range(len(self.base)):
            _, _, label, _, _ = self.base.get_metadata(i)
            if label in COMMANDS:
                by_label[label].append(i)
            else:
                unknown_indices.append(i)

        refs = []

        # Choose target classes.
        selected_counts = []
        for label in COMMANDS:
            indices = by_label[label]
            rng.shuffle(indices)
            if max_per_class is not None:
                indices = indices[:max_per_class]
            selected_counts.append(len(indices))
            refs.extend([("real", idx, label) for idx in indices])

        # Balance unknown and silence to approximately one normal class.
        normal_count = int(sum(selected_counts) / max(len(selected_counts), 1))
        if max_per_class is not None:
            normal_count = min(normal_count, max_per_class)

        rng.shuffle(unknown_indices)
        unknown_indices = unknown_indices[:normal_count]
        refs.extend([("real", idx, "unknown") for idx in unknown_indices])
        refs.extend([("silence", -1, "silence") for _ in range(normal_count)])

        rng.shuffle(refs)
        self.refs = refs

        print(
            f"[{subset}] total={len(self.refs)}, "
            f"per target≈{normal_count}, classes={len(LABELS)}"
        )

    def __len__(self):
        return len(self.refs)

    def _resample_if_needed(self, waveform, sample_rate):
        if sample_rate != self.cfg.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, sample_rate, self.cfg.sample_rate
            )
        return waveform

    def __getitem__(self, index):
        kind, base_index, mapped_label = self.refs[index]

        if kind == "silence":
            # Low-amplitude noise is more realistic than perfect zeros.
            waveform = torch.randn(1, self.cfg.num_samples) * 0.002
        else:
            waveform, sample_rate, raw_label, _, _ = self.base[base_index]
            waveform = self._resample_if_needed(waveform, sample_rate)

            # Convert to mono.
            if waveform.size(0) > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            waveform = pad_or_trim(waveform, self.cfg.num_samples)

        if self.augment is not None:
            waveform = self.augment(waveform)

        target = LABEL_TO_INDEX[mapped_label]
        return waveform.float(), target


def build_datasets(root="data", quick=False, audio_cfg=None, seed=42):
    cfg = audio_cfg or AudioConfig()

    if quick:
        limits = {
            "training": 700,
            "validation": 120,
            "testing": 120,
        }
    else:
        limits = {
            "training": None,
            "validation": None,
            "testing": None,
        }

    train_ds = BalancedSpeechCommands(
        root=root,
        subset="training",
        download=True,
        audio_cfg=cfg,
        augment=True,
        max_per_class=limits["training"],
        seed=seed,
    )
    val_ds = BalancedSpeechCommands(
        root=root,
        subset="validation",
        download=True,
        audio_cfg=cfg,
        augment=False,
        max_per_class=limits["validation"],
        seed=seed,
    )
    test_ds = BalancedSpeechCommands(
        root=root,
        subset="testing",
        download=True,
        audio_cfg=cfg,
        augment=False,
        max_per_class=limits["testing"],
        seed=seed,
    )
    return train_ds, val_ds, test_ds
