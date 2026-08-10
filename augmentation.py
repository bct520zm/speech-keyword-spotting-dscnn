import random
import torch
import torch.nn.functional as F


class WaveformAugmentor:
    def __init__(
        self,
        time_shift_prob=0.5,
        max_shift_ms=100,
        noise_prob=0.5,
        snr_db_range=(5.0, 20.0),
        gain_prob=0.3,
        speed_prob=0.25,
        speed_range=(0.9, 1.1),
        sample_rate=16000,
    ):
        self.time_shift_prob = time_shift_prob
        self.max_shift_samples = int(sample_rate * max_shift_ms / 1000)
        self.noise_prob = noise_prob
        self.snr_db_range = snr_db_range
        self.gain_prob = gain_prob
        self.speed_prob = speed_prob
        self.speed_range = speed_range

    def __call__(self, waveform):
        """waveform: [1, T]"""
        x = waveform.clone()

        if random.random() < self.time_shift_prob:
            shift = random.randint(-self.max_shift_samples, self.max_shift_samples)
            x = torch.roll(x, shifts=shift, dims=-1)
            if shift > 0:
                x[..., :shift] = 0
            elif shift < 0:
                x[..., shift:] = 0

        if random.random() < self.speed_prob:
            speed = random.uniform(*self.speed_range)
            old_len = x.size(-1)
            new_len = max(1, int(old_len / speed))
            x = F.interpolate(
                x.unsqueeze(0),
                size=new_len,
                mode="linear",
                align_corners=False,
            ).squeeze(0)
            if new_len < old_len:
                x = F.pad(x, (0, old_len - new_len))
            else:
                x = x[..., :old_len]

        if random.random() < self.gain_prob:
            gain = random.uniform(0.7, 1.3)
            x = x * gain

        if random.random() < self.noise_prob:
            snr_db = random.uniform(*self.snr_db_range)
            x = add_noise_at_snr(x, snr_db)

        return x.clamp(-1.0, 1.0)


def add_noise_at_snr(waveform, snr_db):
    """Adds Gaussian noise at the requested SNR. waveform: [..., T]"""
    signal_power = waveform.pow(2).mean().clamp_min(1e-8)
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise = torch.randn_like(waveform) * torch.sqrt(noise_power)
    return (waveform + noise).clamp(-1.0, 1.0)
