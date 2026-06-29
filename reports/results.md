# Results

## Current Status

Corrected evaluation metrics are not available yet.

The previous notebook reported very high test performance, but that result may be inflated because augmented feature rows were created before the train/test split. That workflow can place near-duplicate versions of the same original audio file into both training and test sets.

This repository now uses a leakage-safe protocol:

1. Build a manifest with one row per original audio file.
2. Split original files into train, validation, and test sets.
3. Apply augmentation only to the training set.
4. Keep validation and test sets clean.
5. Fit scalers only on training features.

## Corrected Metrics

TODO after retraining with the corrected pipeline:

| Protocol | Dataset | Accuracy | Weighted F1 | Notes |
| --- | --- | ---: | ---: | --- |
| File-level split, train-only augmentation | EmoDB | TODO | TODO | Regenerate with `python -m src.train` and `python -m src.evaluate`. |

## Reporting Checklist

- Add the exact dataset version/source.
- Add train/validation/test file counts.
- Add confusion matrix generated from the clean test split.
- Add classification report from `runs/emotion_model/evaluation.json`.
- Do not compare against the old notebook result as a valid benchmark.
