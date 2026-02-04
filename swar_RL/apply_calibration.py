import argparse, json
import numpy as np
import pandas as pd
import torch

def logit(p):
    p = np.clip(p, 1e-6, 1-1e-6)
    return np.log(p / (1 - p))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", default="inference_all_frames.csv")
    ap.add_argument("--calib_pt", default="swar_boundary_calib.pt")
    ap.add_argument("--out_csv", default="inference_all_frames_calib.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    if "prob" not in df.columns:
        raise SystemExit("Input CSV must have 'prob' column")

    calib = torch.load(args.calib_pt, map_location="cpu")
    a = float(calib["a"]); b = float(calib["b"])

    old_logits = logit(df["prob"].astype(float).to_numpy())
    new_logits = a * old_logits + b
    new_prob = sigmoid(new_logits)
    new_pred = (new_prob >= 0.5).astype(int)

    df["prob_calib"] = new_prob
    df["pred_calib"] = new_pred

    df.to_csv(args.out_csv, index=False)
    print(f"[OK] wrote {args.out_csv}  (prob_calib, pred_calib)")

if __name__ == "__main__":
    main()
