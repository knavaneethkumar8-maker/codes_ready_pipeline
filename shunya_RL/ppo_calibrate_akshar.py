import argparse, json, os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from metrics import print_metrics

CAND_KEYS = ["start", "t", "time", "boundary_time", "frame_idx", "idx", "i"]

def pick_key(df, forced=None):
    if forced:
        if forced not in df.columns:
            raise SystemExit(f"--key_col '{forced}' not in CSV columns")
        return forced
    for k in CAND_KEYS:
        if k in df.columns:
            return k
    raise SystemExit(f"No key column found. Add one of {CAND_KEYS} or pass --key_col <colname>")

def pick_prob_pred(df):
    prob_col = None
    for c in ["prob", "p", "score", "probability"]:
        if c in df.columns:
            prob_col = c
            break
    if prob_col is None:
        raise SystemExit("CSV must have a probability column: prob (or p/score/probability)")
    pred_col = "pred" if "pred" in df.columns else None
    return prob_col, pred_col

def logit(p):
    p = np.clip(p, 1e-6, 1-1e-6)
    return np.log(p / (1 - p))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def bern_logprob_from_logits(logits, action01):
    return -F.binary_cross_entropy_with_logits(logits, action01, reduction="none")

def bern_entropy_from_logits(logits):
    p = torch.sigmoid(logits)
    eps = 1e-8
    return -(p * torch.log(p + eps) + (1 - p) * torch.log(1 - p + eps))

def wav_base(s):
    if pd.isna(s): return s
    return os.path.basename(str(s))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="akshar_infer.csv")
    ap.add_argument("--feedback_csv", default=None)
    ap.add_argument("--key_col", default=None)
    ap.add_argument("--save_pt", default="akshar_boundary_calib.pt")
    ap.add_argument("--save_json", default="akshar_boundary_calib.json")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent_coef", type=float, default=0.01)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    key = pick_key(df, args.key_col)
    prob_col, pred_col = pick_prob_pred(df)
    if pred_col is None:
        df["pred"] = (df[prob_col].astype(float) >= 0.5).astype(int)
        pred_col = "pred"

    if "wav" in df.columns:
        df["_wav_base"] = df["wav"].apply(wav_base)

    # feedback required for this flow (as you asked)
    if not args.feedback_csv:
        raise SystemExit("Provide --feedback_csv (feedback-only mode).")
    fb = pd.read_csv(args.feedback_csv)

    if "correct_label" in fb.columns and fb["correct_label"].astype(str).str.strip().eq("").any():
        raise SystemExit("feedback_csv has empty correct_label. Fill 0/1 then run PPO.")

    if "wav" in fb.columns:
        fb["_wav_base"] = fb["wav"].apply(wav_base)

    merge_keys = [key]
    if "_wav_base" in df.columns and "_wav_base" in fb.columns:
        merge_keys = ["_wav_base", key]

    if key not in fb.columns:
        raise SystemExit(f"feedback_csv must contain key column '{key}'")

    df = df.merge(fb, on=merge_keys, how="inner")
    if len(df) == 0:
        raise SystemExit(f"No overlap after merge. Merge keys used: {merge_keys}")

    if "reward" in df.columns:
        reward_np = df["reward"].astype(float).to_numpy()
    elif "correct_label" in df.columns:
        correct = df["correct_label"].astype(int).to_numpy()
        reward_np = ((df[pred_col].astype(int).to_numpy() == correct).astype(np.float32) * 2 - 1)
    else:
        raise SystemExit("feedback_csv needs reward OR correct_label")

    old_logits_np = logit(df[prob_col].astype(float).to_numpy())
    action_np = df[pred_col].astype(int).to_numpy().astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    old_logits = torch.tensor(old_logits_np, dtype=torch.float32, device=device)
    action = torch.tensor(action_np, dtype=torch.float32, device=device)
    reward = torch.tensor(reward_np, dtype=torch.float32, device=device)

    a = torch.nn.Parameter(torch.tensor(1.0, device=device))
    b = torch.nn.Parameter(torch.tensor(0.0, device=device))
    opt = torch.optim.AdamW([a,b], lr=args.lr)

    with torch.no_grad():
        old_logp = bern_logprob_from_logits(old_logits, action)

    baseline = reward.mean().detach()
    adv = reward - baseline
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    n = len(df)
    idx = torch.arange(n, device=device)

    for ep in range(args.epochs):
        perm = idx[torch.randperm(n)]
        ep_loss = 0.0
        for s in range(0, n, args.batch):
            batch = perm[s:s+args.batch]
            ol = old_logits[batch]
            ab = action[batch]
            advb = adv[batch]
            oldlp = old_logp[batch]

            new_logits = a * ol + b
            logp = bern_logprob_from_logits(new_logits, ab)
            ratio = torch.exp(logp - oldlp)

            surr1 = ratio * advb
            surr2 = torch.clamp(ratio, 1.0 - args.clip, 1.0 + args.clip) * advb
            pi_loss = -(torch.min(surr1, surr2)).mean()

            ent = bern_entropy_from_logits(new_logits).mean()
            loss = pi_loss - args.ent_coef * ent

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([a,b], 1.0)
            opt.step()
            ep_loss += float(loss.detach().cpu())

        print(f"[PPO] epoch {ep+1}/{args.epochs} loss={ep_loss:.4f}  a={float(a.detach().cpu()):.4f}  b={float(b.detach().cpu()):.4f}")

    torch.save({"a": float(a.detach().cpu()), "b": float(b.detach().cpu()), "key_col": key, "prob_col": prob_col}, args.save_pt)
    with open(args.save_json, "w") as f:
        json.dump({"a": float(a.detach().cpu()), "b": float(b.detach().cpu()), "key_col": key, "prob_col": prob_col}, f, indent=2)

    print(f"\n[OK] Saved: {args.save_pt} and {args.save_json}")

if __name__ == "__main__":
    main()
