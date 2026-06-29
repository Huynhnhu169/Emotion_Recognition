# Model Artifacts

Trained model artifacts are intentionally excluded from GitHub.

Expected files after training:

- `models/emotion_model.keras`
- `models/scaler.pkl`
- `models/label_encoder.pkl`

Generate them with:

```bash
python -m src.train --config configs/emotion_model.yaml
```

Older artifacts from `DPL_project_files/` are not assumed to be present. If you keep a private copy of those files, document where they came from, which dataset/protocol produced them, and whether they were trained with the corrected leakage-safe split.
