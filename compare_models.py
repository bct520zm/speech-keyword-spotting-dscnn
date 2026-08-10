"""
Run two experiments automatically:
1) CNN + Log-Mel
2) DS-CNN + Log-Mel

This script launches train.py and evaluate.py as subprocesses.
For a fast demonstration, keep --quick.
"""
import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd):
    print("\n$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()

    for model_name in ["cnn", "dscnn"]:
        out = Path("experiments") / model_name
        train_cmd = [
            sys.executable, "train.py",
            "--model", model_name,
            "--feature", "logmel",
            "--epochs", str(args.epochs),
            "--output-dir", str(out),
        ]
        eval_cmd = [
            sys.executable, "evaluate.py",
            "--checkpoint", str(out / "best_model.pt"),
            "--output-dir", str(out),
        ]
        if args.quick:
            train_cmd.append("--quick")
            eval_cmd.append("--quick")

        run(train_cmd)
        run(eval_cmd)


if __name__ == "__main__":
    main()
