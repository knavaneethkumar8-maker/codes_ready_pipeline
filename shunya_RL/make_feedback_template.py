import argparse
import pandas as pd
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", default="akshar_infer.csv")
    ap.add_argument("--out_csv", default="feedback_akshar.csv")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--strategy", default="uncertain", choices=["uncertain","random","positives"])
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    need = {"wav","start","prob","pred"}
    if not need.issubset(set(df.columns)):
        raise SystemExit(f"in_csv must contain {sorted(list(need))}")

    d = df.copy()
    d = d.dropna(subset=["prob"])
    if args.strategy == "uncertain":
        d["u"] = (d["prob"] - 0.5).abs()
        d = d.sort_values("u", ascending=True)
    elif args.strategy == "positives":
        d = d[d["pred"] == 1].copy()
    else:
        d = d.sample(frac=1.0, random_state=0)

    out = d[["wav","start"]].head(args.n).copy()
    out["correct_label"] = ""  # you fill 0/1
    out.to_csv(args.out_csv, index=False)
    print(f"[OK] wrote {args.out_csv} (fill correct_label 0/1)")

if __name__ == "__main__":
    main()
