"""
No dataset download is required.
Checks:
- feature extractor output
- CNN forward
- DS-CNN forward
"""
import torch

from config import AudioConfig, LABELS
from audio_features import FeatureExtractor
from models import build_model


def main():
    cfg = AudioConfig()
    waveform = torch.randn(4, 1, cfg.num_samples) * 0.01

    for feature_type in ["logmel", "mfcc"]:
        extractor = FeatureExtractor(feature_type, cfg, specaugment=False)
        feature = extractor(waveform)
        if feature.dim() == 3:
            feature = feature.unsqueeze(1)
        print(feature_type, "feature shape:", tuple(feature.shape))

        for model_name in ["cnn", "dscnn"]:
            model = build_model(model_name, num_classes=len(LABELS))
            logits = model(feature)
            print(model_name, "logits shape:", tuple(logits.shape))
            assert logits.shape == (4, len(LABELS))

    print("Smoke test passed.")


if __name__ == "__main__":
    main()
