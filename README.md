# Speech Emotion Recognition

This repository is a reproducible Speech Emotion Recognition project focused on classifying emotion from audio recordings. It refactors the original notebook workflow into a small Python pipeline with a leakage-safe evaluation protocol.

## Problem Statement

Given a short speech audio clip, predict its emotion class from acoustic features such as zero-crossing rate, RMS energy, and MFCCs.

The current implementation is configured for the EmoDB filename convention, where emotion labels can be inferred from original `.wav` filenames.

## Current Status

The evaluation protocol has been rebuilt to avoid augmentation leakage. Corrected metrics are still TODO and should be regenerated before this project is presented as a finished benchmark.

The old notebook result should not be treated as final because augmentation was applied before splitting the data.

## Dataset Notes

Datasets are not committed to this repository.

Expected local layout for the default config:

```text
wav/
  03a01Nc.wav
  03a01Wa.wav
  ...
```

You can also provide a CSV manifest by setting `paths.manifest_csv` in `configs/emotion_model.yaml`. The manifest must contain audio paths and labels. By default, the expected columns are `path` and `label`.

To download EmoDB into the default `wav/` directory, run:

```bash
python scripts/download_emodb.py --output-dir wav
```

If Kaggle blocks direct download in your environment, download the EmoDB ZIP manually from Kaggle and extract the `.wav` files into `wav/`.

## Leakage-Safe Evaluation Protocol

This project uses file-level splitting:

1. Build a manifest with one row per original audio file.
2. Split original files into train, validation, and test sets.
3. Apply augmentation only to the training set.
4. Keep validation and test audio clean.
5. Fit preprocessing scalers only on training features.
6. Evaluate once on the clean test split.

This avoids train/test leakage from augmented duplicates.

## Method

Feature extraction follows the original notebook:

- load audio at 22,050 Hz
- use a 2.5 second window with 0.6 second offset
- fix audio length to 5,120 samples
- extract zero-crossing rate, RMS energy, and MFCC features

The model architecture is preserved from the cleaned training notebook:

- three Conv1D blocks
- LSTM layer
- dense classifier head
- softmax output over emotion classes

## Installation

For Colab, see [COLAB.md](COLAB.md). Colab usually already includes TensorFlow with GPU support, so it uses `requirements-colab.txt`.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS/Linux, activate with:

```bash
source .venv/bin/activate
```

## Prepare Data

Place the EmoDB `.wav` files under `wav/`, run the download script above, or update `configs/emotion_model.yaml` to point to your dataset location or manifest CSV.

Create leakage-safe splits:

```bash
python -m src.data_split --config configs/emotion_model.yaml
```

Generated split files are written under `runs/` and are ignored by Git.

## Train

```bash
python -m src.train --config configs/emotion_model.yaml
```

Training saves local artifacts under `models/`:

- `emotion_model.keras`
- `scaler.pkl`
- `label_encoder.pkl`

These files are ignored by Git. See `models/README.md`.

## Evaluate

```bash
python -m src.evaluate --config configs/emotion_model.yaml
```

Evaluation writes `runs/emotion_model/evaluation.json`. Copy only validated, lightweight summaries into `reports/results.md`.

## Inference

```bash
python -m src.inference --config configs/emotion_model.yaml --audio path/to/audio.wav
```

## Project Structure

```text
Emotion_Recognition/
  README.md
  LICENSE
  requirements.txt
  requirements-colab.txt
  COLAB.md
  .gitignore
  configs/
    emotion_model.yaml
    emotion_model_colab.yaml
  src/
    features.py
    augment.py
    data_split.py
    train.py
    evaluate.py
    inference.py
  notebooks/
    01_exploration.ipynb
    02_training_clean.ipynb
    03_demo.ipynb
  models/
    README.md
  scripts/
    download_emodb.py
  reports/
    results.md
  assets/
    README.md
```

## Limitations

- Corrected metrics have not been regenerated yet.
- The default label parser is specific to EmoDB filenames.
- The current split is file-level, not speaker-independent. A speaker-independent split would be a stronger evaluation if speaker IDs are available.
- No model artifact is shipped with the repository.

## Future Work

- Retrain and publish corrected metrics.
- Add a confusion matrix from the clean test split.
- Add a speaker-independent evaluation option.
- Add a lightweight demo screenshot or short inference example without committing raw audio.
- Add tests for label parsing, splitting, and feature shapes.
