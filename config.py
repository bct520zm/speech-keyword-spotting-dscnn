from dataclasses import dataclass, asdict
import torch


COMMANDS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
LABELS = COMMANDS + ["unknown", "silence"]
LABEL_TO_INDEX = {label: i for i, label in enumerate(LABELS)}


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    clip_seconds: float = 1.0
    n_fft: int = 512
    win_length: int = 400       # 25 ms at 16 kHz
    hop_length: int = 160       # 10 ms at 16 kHz
    n_mels: int = 40
    n_mfcc: int = 20

    @property
    def num_samples(self):
        return int(self.sample_rate * self.clip_seconds)


@dataclass
class TrainConfig:
    seed: int = 42
    epochs: int = 12
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 0
    model_name: str = "dscnn"
    feature_type: str = "logmel"


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def config_to_dict(audio_cfg, train_cfg):
    return {
        "audio": asdict(audio_cfg),
        "train": asdict(train_cfg),
        "labels": LABELS,
    }
