"""Audio loading and feature extraction for speech emotion recognition."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np


def load_audio(
    audio_path: str | Path,
    sample_rate: int = 22050,
    duration: float | None = 2.5,
    offset: float = 0.6,
) -> tuple[np.ndarray, int]:
    """Load one audio file with the project preprocessing defaults."""
    y, sr = librosa.load(audio_path, sr=sample_rate, duration=duration, offset=offset)
    return y, sr


def zero_crossing_rate(y: np.ndarray, frame_length: int = 2048, hop_length: int = 512) -> np.ndarray:
    return np.squeeze(
        librosa.feature.zero_crossing_rate(y=y, frame_length=frame_length, hop_length=hop_length)
    )


def rms_energy(y: np.ndarray, frame_length: int = 2048, hop_length: int = 512) -> np.ndarray:
    return np.squeeze(librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length))


def mfcc(
    y: np.ndarray,
    sr: int,
    n_mfcc: int = 20,
    hop_length: int = 512,
    flatten: bool = True,
) -> np.ndarray:
    values = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    values = np.squeeze(values.T)
    return np.ravel(values) if flatten else values


def extract_feature_vector(
    y: np.ndarray,
    sr: int,
    fixed_length: int = 5120,
    frame_length: int = 2048,
    hop_length: int = 512,
    n_mfcc: int = 20,
) -> np.ndarray:
    """Extract the same ZCR + RMS + MFCC feature vector used by the notebook."""
    y = librosa.util.fix_length(y, size=fixed_length)
    zcr_feature = zero_crossing_rate(y, frame_length=frame_length, hop_length=hop_length)
    rms_feature = rms_energy(y, frame_length=frame_length, hop_length=hop_length)
    mfcc_feature = mfcc(y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    return np.hstack((zcr_feature, rms_feature, mfcc_feature)).astype(np.float32)


def extract_feature_vector_from_file(audio_path: str | Path, audio_config: dict, feature_config: dict) -> np.ndarray:
    y, sr = load_audio(
        audio_path,
        sample_rate=audio_config.get("sample_rate", 22050),
        duration=audio_config.get("duration", 2.5),
        offset=audio_config.get("offset", 0.6),
    )
    return extract_feature_vector(y, sr, **feature_config)
