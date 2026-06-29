"""Single-file inference for a trained speech emotion model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import yaml

try:
    from .features import extract_feature_vector_from_file
except ImportError:  # pragma: no cover
    from features import extract_feature_vector_from_file


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def predict(audio_path: str | Path, config: dict) -> dict:
    import tensorflow as tf

    model_dir = Path(config["paths"]["model_dir"])
    model = tf.keras.models.load_model(model_dir / config["paths"].get("model_file", "emotion_model.keras"))
    scaler = joblib.load(model_dir / config["paths"].get("scaler_file", "scaler.pkl"))
    label_encoder = joblib.load(model_dir / config["paths"].get("label_encoder_file", "label_encoder.pkl"))

    features = extract_feature_vector_from_file(audio_path, config["audio"], config["features"])
    features = scaler.transform(features.reshape(1, -1)).reshape(1, features.shape[0], 1)
    probabilities = model.predict(features, verbose=0)[0]
    best_idx = int(np.argmax(probabilities))

    return {
        "audio_path": str(audio_path),
        "predicted_label": str(label_encoder.inverse_transform([best_idx])[0]),
        "confidence": float(probabilities[best_idx]),
        "probabilities": {
            str(label): float(probabilities[idx])
            for idx, label in enumerate(label_encoder.classes_)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict emotion for one audio file.")
    parser.add_argument("--config", default="configs/emotion_model.yaml")
    parser.add_argument("--audio", required=True, help="Path to a local audio file.")
    args = parser.parse_args()

    result = predict(args.audio, load_config(args.config))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
