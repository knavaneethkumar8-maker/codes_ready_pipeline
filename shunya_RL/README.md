# Akshar Boundary RLHF (PPO) — FULL (generates prob CSV)

Problem you had: `jal_dataset.csv` / `jal_Seq.csv` doesn't have `prob` column.
This package **generates** an inference CSV with `prob,pred` from `akshar_boundary_v1.pt`,
then you can do RLHF PPO using **feedback only**.

## 1) Generate inference CSV (adds prob,pred)
```bash
python make_inference_csv_akshar.py --in_csv jal_dataset.csv --out_csv akshar_infer.csv
```

## 2) Generate feedback template (you fill correct_label)
```bash
python make_feedback_template.py --in_csv akshar_infer.csv --out_csv feedback_akshar.csv --n 200
```

Open `feedback_akshar.csv`, fill `correct_label` with 0/1.

## 3) PPO train using feedback only (no label needed)
```bash
python ppo_calibrate_akshar.py --csv akshar_infer.csv --feedback_csv feedback_akshar.csv --key_col start
```

Outputs:
- akshar_boundary_calib.pt
- akshar_boundary_calib.json

## 4) Apply calibration to inference CSV
```bash
python apply_calibration_akshar.py --in_csv akshar_infer.csv --out_csv akshar_infer_calib.csv
```

### Notes
- WAV paths: we keep whatever is in `jal_dataset.csv`. If the wav is in the same directory, it works.
- If wav path includes folders, it still works as long as that relative path exists.
- Threshold for `pred` default: 0.5 (you can change with --thresh)
