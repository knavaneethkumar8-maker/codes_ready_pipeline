import argparse, json, math
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from metrics import print_metrics

def logit(p):
    p = np.clip(p, 1e-6, 1-1e-6)
    return np.log(p / (1 - p))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def bern_logprob_from_logits(logits, action01):
    # returns log prob for Bernoulli with given logits and actions (0/1 float)
    # logp = -BCEWithLogits
    return -F.binary_cross_entropy_with_logits(logits, action01, reduction="none")

def bern_entropy_from_logits(logits):
    p = torch.sigmoid(logits)
    eps = 1e-8
    return -(p * torch.log(p + eps) + (1 - p) * torch.log(1 - p + eps))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_csv", default="inference_all_frames.csv")
    ap.add_argument("--feedback_csv", default=None, help="Optional: wav,frame_idx + correct_label OR reward")
    ap.add_argument("--save_pt", default="swar_boundary_calib.pt")
    ap.add_argument("--save_json", default="swar_boundary_calib.json")

    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent_coef", type=float, default=0.01)
    args = ap.parse_args()

    df = pd.read_csv(args.frames_csv)
    need = {"wav","frame_idx","prob","pred"}
    if not need.issubset(set(df.columns)):
        raise SystemExit(f"frames_csv must have columns: {sorted(list(need))}. Found: {list(df.columns)}")

    # Merge feedback if provided
    if args.feedback_csv:
        fb = pd.read_csv(args.feedback_csv)
        if not {"wav","frame_idx"}.issubset(set(fb.columns)):
            raise SystemExit("feedback_csv must have wav,frame_idx and (correct_label or reward)")
        df = df.merge(fb, on=["wav","frame_idx"], how="inner")
        if len(df) == 0:
            raise SystemExit("No overlap between frames_csv and feedback_csv on (wav,frame_idx).")
        if "reward" in df.columns:
            reward_np = df["reward"].astype(float).to_numpy()
        elif "correct_label" in df.columns:
            correct = df["correct_label"].astype(int).to_numpy()
            reward_np = ((df["pred"].astype(int).to_numpy() == correct).astype(np.float32) * 2 - 1)
        else:
            raise SystemExit("feedback_csv needs reward OR correct_label")
        y_true = df["correct_label"].astype(int).to_numpy() if "correct_label" in df.columns else None
    else:
        # Use label column as supervised proxy reward
        if "label" not in df.columns:
            raise SystemExit("No feedback_csv provided and frames_csv has no 'label' column. Provide feedback_csv.")
        y = df["label"].astype(int).to_numpy()
        reward_np = ((df["pred"].astype(int).to_numpy() == y).astype(np.float32) * 2 - 1)
        y_true = y

    # Old policy logits from prob
    old_logits_np = logit(df["prob"].astype(float).to_numpy())
    action_np = df["pred"].astype(int).to_numpy().astype(np.float32)

    # Print BEFORE metrics (if label is available)
    if "label" in df.columns:
        print_metrics("BEFORE (raw model)", df["label"].astype(int).to_numpy(), df["pred"].astype(int).to_numpy())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    old_logits = torch.tensor(old_logits_np, dtype=torch.float32, device=device)
    action = torch.tensor(action_np, dtype=torch.float32, device=device)
    reward = torch.tensor(reward_np, dtype=torch.float32, device=device)

    # Calibration params: new_logit = a*old_logit + b
    a = torch.nn.Parameter(torch.tensor(1.0, device=device))
    b = torch.nn.Parameter(torch.tensor(0.0, device=device))
    opt = torch.optim.AdamW([a,b], lr=args.lr)

    # old logp (frozen behavior policy) for PPO ratio
    with torch.no_grad():
        old_logp = bern_logprob_from_logits(old_logits, action)

    # simple baseline: moving average
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

    # Evaluate AFTER (calibrated)
    new_logits_np = (float(a.detach().cpu()) * old_logits_np + float(b.detach().cpu()))
    new_prob = sigmoid(new_logits_np)
    new_pred = (new_prob >= 0.5).astype(int)

    if "label" in df.columns:
        print_metrics("AFTER (PPO calibrated)", df["label"].astype(int).to_numpy(), new_pred)

    # Save calibration
    torch.save({"a": float(a.detach().cpu()), "b": float(b.detach().cpu())}, args.save_pt)
    with open(args.save_json, "w") as f:
        json.dump({"a": float(a.detach().cpu()), "b": float(b.detach().cpu())}, f, indent=2)

    print(f"\n[OK] Saved: {args.save_pt} and {args.save_json}")
    print("To apply on any CSV: python apply_calibration.py --in_csv <frames.csv> --out_csv <out.csv>")

if __name__ == "__main__":
    main()
