
# SL (Supervised Learning) – training + inference only (WAV + TextGrid -> model, WAV -> JSON)

This package contains **only**:
- `train_sl.py` : trains the SL character classifier from a **WAV + TextGrid**
- `infer_sl.py` : runs inference on a **WAV** and produces your **grid JSON** (akash/agni/vayu/jal/prithvi)

It does **NOT** include any boundary / RL calibration (you said you'll integrate that later in `app.py`).

## 1) Unzip

```bash
unzip -o sl_only.zip
cd sl_only
```

## 2) (Optional) Create a venv

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3) Training (input = wav + TextGrid)

Example using your sample:

```bash
python train_sl.py --wav DataTrain/chunk_10.wav --textgrid DataTrain/chunk_10.TextGrid --epochs 6
```

It saves:
- `models/sl_char.pt` (weights)

## 4) Inference (input = wav)

```bash
python infer_sl.py --wav DataInfer/chunk_10.wav --out Out/chunk_10.json
```

Output JSON matches your structure with:
- metadata
- grids[]
  - tiers.akash (1 cell)
  - tiers.agni (2 cells)
  - tiers.vayu (4 cells)
  - tiers.jal (8 cells)
  - tiers.prithvi (24 cells)

### Folder Convention

- Put training pairs in `DataTrain/` (each folder can contain wav+TextGrid).
- Put inference wavs in `DataInfer/`.
- Output goes to `Out/`.

## Notes

- We train using the **'जल'** tier from your TextGrid because in your provided sample the **'पृथ्वी'** tier has empty marks.
- Inference predicts at **27ms** (jal cells) and expands that prediction down to 9ms (prithvi).
