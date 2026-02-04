# Swar Boundary RLHF (PPO) — Minimal, CSV-only

This is an **end-to-end** PPO RLHF loop that works **only from your model CSV outputs**.
No audio feature extraction. No model re-training. It learns a **calibration layer** on top
of your model's probabilities:

    new_logit = a * old_logit + b
    new_prob  = sigmoid(new_logit)
    new_pred  = (new_prob >= 0.5)

It uses **PPO-style clipped objective** on the logged actions (your `pred`) with rewards from:
- `label` column (supervised proxy), OR
- a human feedback CSV (recommended).

## Files
- `ppo_calibrate.py`  -> trains (a,b), prints accuracy before/after, saves `swar_boundary_calib.pt/json`
- `apply_calibration.py` -> applies saved calibration to any frames CSV
- `metrics.py` -> metrics helpers
- `feedback_template.csv` -> example feedback file

## Required input columns (frames CSV)
Must have: `wav, frame_idx, prob, pred`
For evaluation / reward (choose one):
- `label` column in frames CSV **OR**
- `feedback.csv` with: `wav, frame_idx, correct_label` (0/1) OR `reward` (+1/-1)

## Typical usage (run in the same directory as your CSVs)
### 1) Train calibration with PPO (using labels in CSV)
```bash
python ppo_calibrate.py --frames_csv trainval_all_frames.csv
```

### 2) OR train with human feedback
```bash
python ppo_calibrate.py --frames_csv inference_all_frames.csv --feedback_csv feedback.csv
```

### 3) Apply calibration to inference CSV
```bash
python apply_calibration.py --in_csv inference_all_frames.csv --out_csv inference_all_frames_calib.csv
```

Outputs:
- `swar_boundary_calib.pt` and `swar_boundary_calib.json` (a,b)
- Accuracy + Precision/Recall/F1 printed to terminal
