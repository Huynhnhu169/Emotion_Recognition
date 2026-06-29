# Run on Google Colab

Use this workflow when the local GPU is too limited. The code still uses the same leakage-safe protocol: split original audio files first, then augment only the training split.

## 1. Select GPU Runtime

In Colab:

```text
Runtime > Change runtime type > Hardware accelerator > GPU
```

Check GPU:

```python
!nvidia-smi
```

## 2. Get the Repository

If the repository is already on GitHub:

```python
!git clone https://github.com/Huynhnhu169/Emotion_Recognition.git
%cd Emotion_Recognition
```

If you upload the project folder manually, change directory to that folder instead:

```python
%cd /content/Emotion_Recognition
```

## 3. Install Dependencies

Colab usually includes TensorFlow with GPU support. Install the lighter Colab requirements first:

```python
!python -m pip install -q -r requirements-colab.txt
```

Check TensorFlow:

```python
import tensorflow as tf
print(tf.__version__)
print(tf.config.list_physical_devices("GPU"))
```

If TensorFlow is missing, install the full requirements:

```python
!python -m pip install -q -r requirements.txt
```

## 4. Prepare EmoDB Data

The config expects EmoDB `.wav` files under:

```text
wav/
```

Option A: download from Kaggle URL:

```python
!python scripts/download_emodb.py --output-dir wav
```

If Kaggle blocks direct download, upload the EmoDB ZIP to Colab and extract only `.wav` files:

```python
!python scripts/download_emodb.py --archive /content/emodb.zip --output-dir wav --skip-existing
```

Option B: copy your existing `wav/` folder from Google Drive:

```python
from google.colab import drive
drive.mount("/content/drive")
!mkdir -p wav
!cp "/content/drive/MyDrive/Emotion_detect/wav/"*.wav ./wav/
```

Do not commit `wav/` to GitHub.

## 5. Create Leakage-Safe Splits

```python
!python -m src.data_split --config configs/emotion_model_colab.yaml
```

Expected local EmoDB count is 535 `.wav` files. A typical 80/10/10 split is:

```text
train: 427 files
val: 54 files
test: 54 files
```

## 6. Train

```python
!python -m src.train --config configs/emotion_model_colab.yaml
```

Training writes local artifacts to:

```text
models/
runs/emotion_model_colab/
```

These outputs are ignored by Git.

## 7. Evaluate

```python
!python -m src.evaluate --config configs/emotion_model_colab.yaml
```

After evaluation, inspect:

```python
import json
with open("runs/emotion_model_colab/evaluation.json", "r", encoding="utf-8") as f:
    metrics = json.load(f)
metrics
```

Only copy corrected metrics into `reports/results.md` after this run finishes.

## 8. Inference

```python
!python -m src.inference --config configs/emotion_model_colab.yaml --audio wav/03a01Nc.wav
```

## 9. Save Artifacts to Drive

Optional:

```python
from google.colab import drive
drive.mount("/content/drive")
!mkdir -p "/content/drive/MyDrive/Emotion_Recognition_Artifacts"
!cp -r models runs "/content/drive/MyDrive/Emotion_Recognition_Artifacts/"
```

Keep these artifacts private unless you document exactly how they were produced.
