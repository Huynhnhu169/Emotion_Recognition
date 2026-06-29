"""Audio augmentation helpers used only for the training split."""

from __future__ import annotations

from collections.abc import Iterable

import librosa
import numpy as np


def add_noise(y: np.ndarray, noise_rate: float = 0.05, rng: np.random.Generator | None = None) -> np.ndarray:
    """Add amplitude-scaled Gaussian noise."""
    rng = rng or np.random.default_rng()
    amplitude = noise_rate * rng.uniform() * float(np.max(np.abs(y)) or 1.0)
    return y + amplitude * rng.normal(size=y.shape[0])


def time_stretch(y: np.ndarray, rate: float = 0.9) -> np.ndarray:
    """Apply librosa time stretching."""
    return librosa.effects.time_stretch(y, rate=rate)


def pitch_shift(y: np.ndarray, sr: int, n_steps: float = 0.7) -> np.ndarray:
    """Apply pitch shifting in fractional semitone steps."""
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)


def augment_audio(
    y: np.ndarray,
    sr: int,
    methods: Iterable[str],
    noise_rate: float = 0.05,
    stretch_rate: float = 0.9,
    pitch_steps: float = 0.7,
    rng: np.random.Generator | None = None,
) -> list[tuple[str, np.ndarray]]:
    """Return named augmentations for one audio signal.

    The caller decides whether to include the original signal. Validation and
    test callers should not use this function.
    """
    rng = rng or np.random.default_rng()
    augmented: list[tuple[str, np.ndarray]] = []

    for method in methods:
        if method == "noise":
            augmented.append((method, add_noise(y, noise_rate=noise_rate, rng=rng)))
        elif method == "stretch":
            augmented.append((method, time_stretch(y, rate=stretch_rate)))
        elif method == "pitch_shift":
            augmented.append((method, pitch_shift(y, sr=sr, n_steps=pitch_steps)))
        elif method == "pitch_shift_noise":
            shifted = pitch_shift(y, sr=sr, n_steps=pitch_steps)
            augmented.append((method, add_noise(shifted, noise_rate=noise_rate, rng=rng)))
        else:
            raise ValueError(f"Unknown augmentation method: {method}")

    return augmented
