import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import AudioConfig, TrainConfig, LABELS, get_device, config_to_dict
from dataset import build_datasets
from audio_features import FeatureExtractor
from models import build_model, count_parameters
from utils import set_seed, AverageMeter, save_history


def run_epoch(model, feature_extractor, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    feature_extractor.train(is_train)

    loss_meter = AverageMeter()
    correct = 0
    total = 0

    pbar = tqdm(loader, leave=False)
    for waveforms, targets in pbar:
        waveforms = waveforms.to(device)
        targets = targets.to(device)

        with torch.set_grad_enabled(is_train):
            features = feature_extractor(waveforms)
            # Input should be [B,1,F,T].
            if features.dim() == 3:
                features = features.unsqueeze(1)

            logits = model(features)
            loss = criterion(logits, targets)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        batch_size = targets.size(0)
        loss_meter.update(loss.item(), batch_size)
        preds = logits.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += batch_size

        pbar.set_postfix(loss=f"{loss_meter.avg:.4f}", acc=f"{correct/max(total,1):.3f}")

    return loss_meter.avg, correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--model", choices=["cnn", "dscnn"], default="dscnn")
    parser.add_argument("--feature", choices=["logmel", "mfcc"], default="logmel")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    audio_cfg = AudioConfig()
    train_cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_workers=args.num_workers,
        model_name=args.model,
        feature_type=args.feature,
    )

    set_seed(train_cfg.seed)
    device = get_device()
    print("Device:", device)

    train_ds, val_ds, _ = build_datasets(
        root=args.data_dir,
        quick=args.quick,
        audio_cfg=audio_cfg,
        seed=train_cfg.seed,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    feature_extractor = FeatureExtractor(
        feature_type=args.feature,
        audio_cfg=audio_cfg,
        specaugment=True,
    ).to(device)

    model = build_model(args.model, num_classes=len(LABELS)).to(device)
    print(f"Model: {args.model}")
    print(f"Trainable parameters: {count_parameters(model):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best_model.pt"

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }
    best_val_acc = -1.0

    for epoch in range(1, train_cfg.epochs + 1):
        print(f"\nEpoch {epoch}/{train_cfg.epochs}")

        train_loss, train_acc = run_epoch(
            model, feature_extractor, train_loader, criterion, device, optimizer
        )
        val_loss, val_acc = run_epoch(
            model, feature_extractor, val_loader, criterion, device, optimizer=None
        )

        scheduler.step(val_acc)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                "model_state": model.state_dict(),
                "model_name": args.model,
                "feature_type": args.feature,
                "labels": LABELS,
                "audio_config": audio_cfg.__dict__,
                "train_config": train_cfg.__dict__,
                "best_val_acc": best_val_acc,
            }
            torch.save(checkpoint, best_path)
            print(f"Saved best checkpoint -> {best_path}")

        save_history(history, output_dir)

    print(f"\nDone. Best validation accuracy: {best_val_acc:.4f}")
    print("Next: python evaluate.py --checkpoint outputs/best_model.pt")


if __name__ == "__main__":
    main()
