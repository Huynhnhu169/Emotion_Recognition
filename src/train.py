"""Training entry point for the leakage-safe emotion model pipeline."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
import yaml
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight

try:
    from .augment import augment_audio
    from .data_split import load_or_create_splits
    from .features import extract_feature_vector, load_audio
except ImportError:  # pragma: no cover - supports direct script execution
    from augment import augment_audio
    from data_split import load_or_create_splits
    from features import extract_feature_vector, load_audio


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_feature_matrix(
    rows,
    config: dict,
    split_name: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, str]]]:
    """Build features for a split.

    Augmentation is intentionally limited to the training split.
    """
    audio_cfg = config["audio"]
    feature_cfg = config["features"]
    aug_cfg = config["augmentation"]
    X: list[np.ndarray] = []
    y: list[str] = []
    audit_rows: list[dict[str, str]] = []

    for row in rows.itertuples(index=False):
        audio, sr = load_audio(
            row.path,
            sample_rate=audio_cfg.get("sample_rate", 22050),
            duration=audio_cfg.get("duration", 2.5),
            offset=audio_cfg.get("offset", 0.6),
        )

        variants: list[tuple[str, np.ndarray]] = []
        if split_name == "train" and aug_cfg.get("include_original", True):
            variants.append(("original", audio))
        elif split_name != "train":
            variants.append(("original", audio))

        if split_name == "train" and aug_cfg.get("enabled", True):
            variants.extend(
                augment_audio(
                    audio,
                    sr,
                    methods=aug_cfg.get("methods", []),
                    noise_rate=aug_cfg.get("noise_rate", 0.05),
                    stretch_rate=aug_cfg.get("stretch_rate", 0.9),
                    pitch_steps=aug_cfg.get("pitch_steps", 0.7),
                    rng=rng,
                )
            )

        for variant_name, variant_audio in variants:
            X.append(extract_feature_vector(variant_audio, sr, **feature_cfg))
            y.append(row.label)
            audit_rows.append({"path": row.path, "label": row.label, "variant": variant_name})

    return np.vstack(X), np.array(y), audit_rows


def build_model(input_shape: tuple[int, int], num_classes: int, model_cfg: dict):
    """Build the Conv1D + LSTM architecture from the cleaned notebook."""
    import tensorflow as tf
    from tensorflow.keras.layers import BatchNormalization, Conv1D, Dense, Dropout, Flatten, LSTM, MaxPooling1D
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam

    model = Sequential()
    model.add(tf.keras.Input(shape=input_shape))
    filters = model_cfg.get("conv_filters", [512, 256, 128])
    kernel_size = model_cfg.get("kernel_size", 7)
    dropout = model_cfg.get("dropout", 0.3)

    for filter_count in filters:
        model.add(Conv1D(filter_count, kernel_size=kernel_size, strides=1, padding="same", activation="relu"))
        model.add(MaxPooling1D(pool_size=2, strides=2, padding="same"))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))

    model.add(LSTM(model_cfg.get("lstm_units", 128), return_sequences=True))
    model.add(Dropout(dropout))
    model.add(Flatten())

    for units in model_cfg.get("dense_units", [128, 64, 32]):
        model.add(Dense(units, activation="relu"))

    model.add(Dense(num_classes, activation="softmax"))
    model.compile(
        optimizer=Adam(learning_rate=model_cfg.get("learning_rate", 0.001)),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train(config: dict) -> dict:
    import tensorflow as tf

    seed = int(config["training"].get("random_seed", 42))
    rng = np.random.default_rng(seed)
    tf.keras.utils.set_random_seed(seed)

    output_dir = Path(config["paths"]["output_dir"])
    model_dir = Path(config["paths"]["model_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    splits = load_or_create_splits(config, save=True)
    train_X, train_labels, train_audit = build_feature_matrix(splits["train"], config, "train", rng)
    val_X, val_labels, _ = build_feature_matrix(splits["val"], config, "val", rng)
    test_X, test_labels, _ = build_feature_matrix(splits["test"], config, "test", rng)

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_labels)
    y_val = label_encoder.transform(val_labels)
    y_test = label_encoder.transform(test_labels)

    scaler = MinMaxScaler()
    train_X = scaler.fit_transform(train_X)
    val_X = scaler.transform(val_X)
    test_X = scaler.transform(test_X)

    train_X = train_X.reshape(train_X.shape[0], train_X.shape[1], 1)
    val_X = val_X.reshape(val_X.shape[0], val_X.shape[1], 1)
    test_X = test_X.reshape(test_X.shape[0], test_X.shape[1], 1)

    model = build_model(train_X.shape[1:], len(label_encoder.classes_), config["model"])

    class_weight = None
    if config["training"].get("use_class_weights", True):
        classes = np.unique(y_train)
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
        class_weight = dict(zip(classes, weights))

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=int(config["training"].get("early_stopping_patience", 10)),
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=float(config["training"].get("reduce_lr_factor", 0.2)),
            patience=int(config["training"].get("reduce_lr_patience", 5)),
            min_lr=float(config["training"].get("min_learning_rate", 0.00001)),
        ),
    ]

    history = model.fit(
        train_X,
        y_train,
        validation_data=(val_X, y_val),
        epochs=int(config["training"].get("epochs", 100)),
        batch_size=int(config["training"].get("batch_size", 32)),
        class_weight=class_weight,
        callbacks=callbacks,
    )

    test_loss, test_accuracy = model.evaluate(test_X, y_test, verbose=0)

    model_path = model_dir / config["paths"].get("model_file", "emotion_model.keras")
    scaler_path = model_dir / config["paths"].get("scaler_file", "scaler.pkl")
    encoder_path = model_dir / config["paths"].get("label_encoder_file", "label_encoder.pkl")
    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(label_encoder, encoder_path)

    audit_path = output_dir / "train_feature_audit.csv"
    with audit_path.open("w", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "variant"])
        writer.writeheader()
        for item in train_audit:
            writer.writerow(item)

    summary = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "classes": label_encoder.classes_.tolist(),
        "train_original_files": int(len(splits["train"])),
        "val_original_files": int(len(splits["val"])),
        "test_original_files": int(len(splits["test"])),
        "train_feature_rows": int(train_X.shape[0]),
        "val_feature_rows": int(val_X.shape[0]),
        "test_feature_rows": int(test_X.shape[0]),
        "history": {key: [float(v) for v in values] for key, values in history.history.items()},
    }
    with (output_dir / "training_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the speech emotion recognition model.")
    parser.add_argument("--config", default="configs/emotion_model.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and planned split without training.")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.dry_run:
        splits = load_or_create_splits(config, save=False)
        for name, frame in splits.items():
            print(f"{name}: {len(frame)} original files")
        return

    summary = train(config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
