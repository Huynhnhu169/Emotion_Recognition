"""Evaluate a trained emotion model on the clean test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import yaml
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

try:
    from .data_split import load_or_create_splits
    from .features import extract_feature_vector_from_file
except ImportError:  # pragma: no cover
    from data_split import load_or_create_splits
    from features import extract_feature_vector_from_file


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def evaluate(config: dict) -> dict:
    import tensorflow as tf

    model_dir = Path(config["paths"]["model_dir"])
    model = tf.keras.models.load_model(model_dir / config["paths"].get("model_file", "emotion_model.keras"))
    scaler = joblib.load(model_dir / config["paths"].get("scaler_file", "scaler.pkl"))
    label_encoder = joblib.load(model_dir / config["paths"].get("label_encoder_file", "label_encoder.pkl"))

    test_split = load_or_create_splits(config, save=True)["test"]
    X = np.vstack(
        [
            extract_feature_vector_from_file(path, config["audio"], config["features"])
            for path in test_split["path"].tolist()
        ]
    )
    y_true = label_encoder.transform(test_split["label"].tolist())
    X = scaler.transform(X).reshape(X.shape[0], X.shape[1], 1)

    probabilities = model.predict(X, verbose=0)
    y_pred = np.argmax(probabilities, axis=1)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=label_encoder.classes_,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classes": label_encoder.classes_.tolist(),
        "test_original_files": int(len(test_split)),
    }

    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "evaluation.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the trained emotion model.")
    parser.add_argument("--config", default="configs/emotion_model.yaml")
    args = parser.parse_args()

    metrics = evaluate(load_config(args.config))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
