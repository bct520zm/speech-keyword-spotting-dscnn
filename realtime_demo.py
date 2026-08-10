import argparse
import queue
import time
from collections import deque

import numpy as np
import sounddevice as sd
import torch

from config import get_device
from evaluate import load_system


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/best_model.pt")
    parser.add_argument("--threshold", type=float, default=0.80)
    parser.add_argument("--smoothing", type=int, default=3)
    parser.add_argument("--cooldown", type=float, default=1.0)
    parser.add_argument("--block-ms", type=int, default=100)
    args = parser.parse_args()

    device = get_device()
    model, feature_extractor, labels, audio_cfg, _ = load_system(
        args.checkpoint, device
    )

    audio_queue = queue.Queue()
    block_size = int(audio_cfg.sample_rate * args.block_ms / 1000)
    buffer = deque(maxlen=audio_cfg.num_samples)
    prob_history = deque(maxlen=args.smoothing)
    last_trigger_time = 0.0

    def callback(indata, frames, time_info, status):
        if status:
            print(status)
        audio_queue.put(indata[:, 0].copy())

    print("Real-time Keyword Spotting")
    print("Labels:", labels)
    print("Press Ctrl+C to stop.")
    print()

    with sd.InputStream(
        channels=1,
        samplerate=audio_cfg.sample_rate,
        blocksize=block_size,
        dtype="float32",
        callback=callback,
    ):
        try:
            while True:
                block = audio_queue.get()
                buffer.extend(block.tolist())

                if len(buffer) < audio_cfg.num_samples:
                    continue

                waveform_np = np.asarray(buffer, dtype=np.float32)
                waveform = torch.from_numpy(waveform_np).view(1, 1, -1).to(device)

                with torch.no_grad():
                    feature = feature_extractor(waveform)
                    if feature.dim() == 3:
                        feature = feature.unsqueeze(1)
                    logits = model(feature)
                    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

                prob_history.append(probs)
                avg_probs = np.mean(np.stack(prob_history), axis=0)
                pred_idx = int(np.argmax(avg_probs))
                confidence = float(avg_probs[pred_idx])
                label = labels[pred_idx]

                now = time.time()
                if (
                    confidence >= args.threshold
                    and label not in {"unknown", "silence"}
                    and now - last_trigger_time >= args.cooldown
                    and len(prob_history) == args.smoothing
                ):
                    print(
                        f"Detected: {label.upper():<8} "
                        f"confidence={confidence*100:5.1f}%"
                    )
                    last_trigger_time = now

        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
