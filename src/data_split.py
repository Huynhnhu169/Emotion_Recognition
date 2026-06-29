"""Leakage-safe dataset manifest and split utilities.

This module splits original audio files before any augmentation is applied.
Training code may augment only the training rows returned here.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


EMODB_EMOTION_MAP = {
    "W": "anger",
    "L": "boredom",
    "E": "disgust",
    "A": "fear",
    "F": "happiness",
    "T": "sadness",
    "N": "neutral",
}


@dataclass(frozen=True)
class SplitConfig:
    test_size: float
    val_size: float
    random_seed: int
    stratify: bool = True


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def infer_emodb_label(audio_path: str | Path) -> str:
    """Infer an EmoDB label from filenames such as 03a01Nc.wav."""
    stem = Path(audio_path).stem
    if len(stem) > 5 and stem[5] in EMODB_EMOTION_MAP:
        return EMODB_EMOTION_MAP[stem[5]]
    for char in stem:
        if char in EMODB_EMOTION_MAP:
            return EMODB_EMOTION_MAP[char]
    raise ValueError(f"Could not infer EmoDB emotion label from filename: {audio_path}")


def build_manifest_from_directory(
    data_dir: str | Path,
    audio_extensions: list[str] | tuple[str, ...],
    label_source: str = "emodb_filename",
) -> pd.DataFrame:
    """Scan a data directory and return one row per original audio file."""
    data_dir = Path(data_dir)
    rows: list[dict[str, str]] = []
    extensions = {ext.lower() for ext in audio_extensions}

    for path in sorted(data_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        if label_source != "emodb_filename":
            raise ValueError("Directory scanning currently supports label_source='emodb_filename'.")
        rows.append({"path": str(path), "label": infer_emodb_label(path)})

    if not rows:
        raise FileNotFoundError(
            f"No audio files with extensions {sorted(extensions)} found under {data_dir}."
        )

    return pd.DataFrame(rows)


def load_manifest(config: dict) -> pd.DataFrame:
    """Load a CSV manifest or build one from the configured data directory."""
    paths = config["paths"]
    data_cfg = config["data"]
    manifest_csv = paths.get("manifest_csv")

    if manifest_csv:
        manifest = pd.read_csv(manifest_csv)
        path_col = data_cfg.get("path_column", "path")
        label_col = data_cfg.get("label_column", "label")
        manifest = manifest.rename(columns={path_col: "path", label_col: "label"})
        missing = {"path", "label"} - set(manifest.columns)
        if missing:
            raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")
        return manifest[["path", "label"]].copy()

    return build_manifest_from_directory(
        data_dir=paths["data_dir"],
        audio_extensions=data_cfg.get("audio_extensions", [".wav"]),
        label_source=data_cfg.get("label_source", "emodb_filename"),
    )


def _stratify_or_none(labels: pd.Series, enabled: bool, split_size: float) -> pd.Series | None:
    if not enabled:
        return None
    counts = labels.value_counts()
    if counts.empty or counts.min() < 2:
        return None
    split_count = math.ceil(len(labels) * split_size)
    if split_count < labels.nunique():
        return None
    return labels


def split_manifest(manifest: pd.DataFrame, split_config: SplitConfig) -> dict[str, pd.DataFrame]:
    """Split original file rows into train, validation, and test sets."""
    required = {"path", "label"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")

    test_stratify = _stratify_or_none(manifest["label"], split_config.stratify, split_config.test_size)
    train_val, test = train_test_split(
        manifest,
        test_size=split_config.test_size,
        random_state=split_config.random_seed,
        shuffle=True,
        stratify=test_stratify,
    )

    val_relative = split_config.val_size / (1.0 - split_config.test_size)
    val_stratify = _stratify_or_none(train_val["label"], split_config.stratify, val_relative)
    train, val = train_test_split(
        train_val,
        test_size=val_relative,
        random_state=split_config.random_seed,
        shuffle=True,
        stratify=val_stratify,
    )

    return {
        "train": train.reset_index(drop=True),
        "val": val.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }


def split_config_from_project_config(config: dict) -> SplitConfig:
    data_cfg = config["data"]
    training_cfg = config["training"]
    return SplitConfig(
        test_size=float(data_cfg.get("test_size", 0.1)),
        val_size=float(data_cfg.get("val_size", 0.1)),
        random_seed=int(training_cfg.get("random_seed", 42)),
        stratify=bool(data_cfg.get("stratify", True)),
    )


def save_splits(splits: dict[str, pd.DataFrame], splits_dir: str | Path) -> None:
    splits_dir = Path(splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in splits.items():
        frame.to_csv(splits_dir / f"{name}.csv", index=False)


def load_or_create_splits(config: dict, save: bool = True) -> dict[str, pd.DataFrame]:
    splits_dir = Path(config["paths"]["splits_dir"])
    expected = {name: splits_dir / f"{name}.csv" for name in ("train", "val", "test")}
    if all(path.exists() for path in expected.values()):
        return {name: pd.read_csv(path) for name, path in expected.items()}

    manifest = load_manifest(config)
    splits = split_manifest(manifest, split_config_from_project_config(config))
    if save:
        save_splits(splits, splits_dir)
    return splits


def main() -> None:
    parser = argparse.ArgumentParser(description="Create leakage-safe train/val/test splits.")
    parser.add_argument("--config", default="configs/emotion_model.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    splits = load_or_create_splits(config, save=True)
    for name, frame in splits.items():
        counts = frame["label"].value_counts().to_dict()
        print(f"{name}: {len(frame)} files, labels={counts}")


if __name__ == "__main__":
    main()
