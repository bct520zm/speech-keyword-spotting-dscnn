import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

from config import AudioConfig, LABELS, get_device
from dataset import BalancedSpeechCommands
from audio_features import FeatureExtractor
from augmentation import add_noise_at_snr
from models import build_model, count_parameters
from utils import measure_inference_latency


def load_system(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    labels = checkpoint.get("labels", LABELS)
    audio_cfg = AudioConfig(**checkpoint.get("audio_config", {}))
    model_name = checkpoint["model_name"]
    feature_type = checkpoint["feature_type"]

    feature_extractor = FeatureExtractor(
        feature_type=feature_type,
        audio_cfg=audio_cfg,
        specaugment=False,
    ).to(device)
    feature_extractor.eval()

    model = build_model(model_name, num_classes=len(labels)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    return model, feature_extractor, labels, audio_cfg, checkpoint


@torch.no_grad()
def predict_loader(model, feature_extractor, loader, device):
    y_true, y_pred = [], []

    for waveforms, targets in loader:
        waveforms = waveforms.to(device)
        features = feature_extractor(waveforms)
        if features.dim() == 3:
            features = features.unsqueeze(1)
        logits = model(features)
        preds = logits.argmax(dim=1)

        y_true.extend(targets.numpy().tolist())
        y_pred.extend(preds.cpu().numpy().tolist())

    return y_true, y_pred


def plot_confusion_matrix(cm, labels, output_path):
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True Label",
        xlabel="Predicted Label",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    threshold = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > threshold else "black",
                fontsize=7,
            )
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


@torch.no_grad()
def noise_stress_test(model, feature_extractor, dataset, device, snr_values):
    results = {}
    for snr_db in snr_values:
        correct, total = 0, 0
        for waveform, target in dataset:
            noisy = add_noise_at_snr(waveform, snr_db).unsqueeze(0).to(device)
            feat = feature_extractor(noisy)
            if feat.dim() == 3:
                feat = feat.unsqueeze(1)
            pred = model(feat).argmax(dim=1).item()
            correct += int(pred == target)
            total += 1
        results[snr_db] = correct / max(total, 1)
        print(f"SNR {snr_db:>4} dB -> accuracy={results[snr_db]:.4f}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/best_model.pt")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--noise-test", action="store_true")
    args = parser.parse_args()

    device = get_device()
    print("Device:", device)

    model, feature_extractor, labels, audio_cfg, checkpoint = load_system(
        args.checkpoint, device
    )

    test_ds = BalancedSpeechCommands(
        root=args.data_dir,
        subset="testing",
        download=True,
        audio_cfg=audio_cfg,
        augment=False,
        max_per_class=120 if args.quick else None,
        seed=42,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    y_true, y_pred = predict_loader(
        model, feature_extractor, test_loader, device
    )
    acc = accuracy_score(y_true, y_pred)
    print(f"\nTest accuracy: {acc:.4f}")
    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=list(range(len(labels))),
            target_names=labels,
            digits=4,
            zero_division=0,
        )
    )

    cm = confusion_matrix(
        y_true, y_pred, labels=list(range(len(labels)))
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(cm, labels, output_dir / "confusion_matrix.png")
    print("Saved:", output_dir / "confusion_matrix.png")

    # Model size and latency.
    example_waveform = torch.zeros(1, 1, audio_cfg.num_samples, device=device)
    example_feature = feature_extractor(example_waveform)
    if example_feature.dim() == 3:
        example_feature = example_feature.unsqueeze(1)

    latency_ms = measure_inference_latency(
        model, example_feature, device, warmup=10, runs=50
    )
    print(f"Trainable parameters: {count_parameters(model):,}")
    print(f"Model-only inference latency: {latency_ms:.3f} ms")

    if args.noise_test:
        print("\nNoise robustness test:")
        noise_stress_test(
            model,
            feature_extractor,
            test_ds,
            device,
            snr_values=[20, 10, 5, 0],
        )


if __name__ == "__main__":
    main()
