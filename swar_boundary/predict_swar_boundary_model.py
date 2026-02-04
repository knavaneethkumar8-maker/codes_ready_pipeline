import os, glob, csv, json, argparse
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn

# =========================
# YOUR UPDATED AKSHAR SET
# =========================
AKSHAR_SET = {
    "अ", "आ", "इ", "उ", "ए", "ओ",
    "क", "ख", "ग", "घ", "च", "छ", "ज", "झ",
    "ट", "ठ", "ड", "ढ", "त", "थ", "द", "ध",
    "प", "फ", "ब", "भ",
    "न", "म",
    "य", "र", "ल", "व",
    "स", "ह", "०"
}

VYANJAN_SET = {
    "क", "ख", "ग", "घ", "च", "छ", "ज", "झ",
    "ट", "ठ", "ड", "ढ", "त", "थ", "द", "ध",
    "प", "फ", "ब", "भ",
    "न", "म",
    "य", "र", "ल", "व",
    "स", "ह"
}

def is_vyanjan(tok: str) -> bool:
    return tok in VYANJAN_SET

# =========================
# TextGrid parsing
# =========================
def read_textgrid_lines(path: str):
    for enc in ("utf-16", "utf-8"):
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read().splitlines()
        except UnicodeError:
            continue
    raise UnicodeError(f"Unsupported encoding: {path}")

def parse_interval_tier(lines, tier_name: str):
    intervals = []
    in_tier = False
    is_it = False
    xmin = xmax = None
    for ln in lines:
        s = ln.strip()
        if s.startswith("class ="):
            is_it = ('"IntervalTier"' in s)
        if s.startswith("name ="):
            name = s.split("=", 1)[1].strip().strip('"')
            in_tier = is_it and (name == tier_name)
        if in_tier and s.startswith("xmin ="):
            xmin = float(s.split("=", 1)[1].strip())
        if in_tier and s.startswith("xmax ="):
            xmax = float(s.split("=", 1)[1].strip())
        if in_tier and s.startswith("text ="):
            txt = s.split("=", 1)[1].strip().strip('"')
            if xmin is not None and xmax is not None:
                intervals.append((xmin, xmax, txt))
            xmin = xmax = None
    return intervals

def clean_tok(t: str) -> str:
    t = t.strip().replace(" ", "")
    if t == "":
        return "०"
    if t not in AKSHAR_SET:
        return "०"
    return t

# same label logic as training
def build_unit_start_labels(tokens):
    T = len(tokens)
    y = np.zeros(T, dtype=np.int64)
    prev_ns = None
    for i in range(T):
        cur = tokens[i]
        if cur == "०":
            continue
        if prev_ns is None:
            y[i] = 1
        else:
            if is_vyanjan(cur) and (not is_vyanjan(prev_ns)):
                y[i] = 1
        prev_ns = cur
    return y

# =========================
# Frame features
# =========================
def frame_signal(sig: np.ndarray, win: int, hop: int) -> np.ndarray:
    n = 1 + max(0, (len(sig) - win) // hop)
    return np.lib.stride_tricks.as_strided(
        sig,
        shape=(n, win),
        strides=(sig.strides[0] * hop, sig.strides[0])
    ).copy()

def zcr(x: np.ndarray) -> float:
    return float(np.mean(np.abs(np.diff(np.sign(x))) > 0))

def extract_frame_feat(frame: np.ndarray, n_fft=512, n_bins=32) -> np.ndarray:
    frame = frame.astype(np.float32)
    frame = frame - frame.mean()
    energy = float(np.sqrt(np.mean(frame**2) + 1e-12))
    z = zcr(frame)
    w = np.hanning(len(frame)).astype(np.float32)
    X = np.fft.rfft(frame * w, n=n_fft)
    mag = np.abs(X) + 1e-6
    logmag = np.log(mag)
    spec = logmag[1:1 + n_bins].astype(np.float32)
    return np.concatenate([spec, np.array([energy, z], np.float32)], axis=0)

class TinyMLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

def confusion_counts(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    acc = float((tp + tn) / max(len(y_true), 1))
    return {"acc": acc, "tp": tp, "tn": tn, "fp": fp, "fn": fn, "n": int(len(y_true)), "pos": int(y_true.sum())}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=".")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tier", default=None)  # if None, use ckpt config tier
    ap.add_argument("--th", type=float, default=None)  # if None, use ckpt threshold

    ap.add_argument("--out_csv", default="inference_all_frames.csv")
    ap.add_argument("--out_segments_csv", default="inference_all_segments.csv")
    ap.add_argument("--out_metrics", default="inference_metrics.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt.get("config", {})

    sr = int(cfg.get("sr", 16000))
    win_ms = float(cfg.get("win_ms", 27.0))
    hop_ms = float(cfg.get("hop_ms", 27.0))
    ctx = int(cfg.get("ctx", 5))

    tier = args.tier if args.tier is not None else str(cfg.get("tier", "जल"))
    th = float(args.th) if args.th is not None else float(cfg.get("threshold", 0.7))

    win = int(round(win_ms / 1000.0 * sr))
    hop = int(round(hop_ms / 1000.0 * sr))
    h = ctx // 2

    # init model
    # infer in_dim from config akshar_set not possible; compute from feat size:
    d = 32 + 2  # bins + energy + zcr
    in_dim = ctx * d
    model = TinyMLP(in_dim).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    wavs = glob.glob(os.path.join(args.data_dir, "**/*.wav"), recursive=True)

    frame_rows = []
    seg_rows = []

    all_y = []
    all_p = []

    for wav in wavs:
        tg = os.path.splitext(wav)[0] + ".TextGrid"
        if not os.path.exists(tg):
            continue

        # tokens + gt
        lines = read_textgrid_lines(tg)
        intervals = parse_interval_tier(lines, tier)
        tokens = [clean_tok(txt) for _, _, txt in intervals]
        y_gt = build_unit_start_labels(tokens)

        # audio frames
        y, sr0 = sf.read(wav)
        if y.ndim > 1:
            y = y.mean(axis=1)
        y = y.astype(np.float32)

        if sr0 != sr:
            import librosa
            y = librosa.resample(y.astype(np.float32), orig_sr=sr0, target_sr=sr)


        frames = frame_signal(y, win, hop)

        T = min(len(frames), len(tokens))
        if T <= h:
            print("SKIP (too short after align):", wav)
            continue
        
        frames = frames[:T]
        tokens = tokens[:T]
        y_gt = y_gt[:T]
        
        feats = np.stack([extract_frame_feat(frames[i]) for i in range(T)], axis=0)


        # predict per non-silence frame (with ctx)
        pred_starts = []  # for segment building (times)
        for i in range(h, T - h):
            if tokens[i] == "०":
                continue

            x = feats[i - h:i + h + 1].reshape(-1).astype(np.float32)
            xb = torch.from_numpy(x).unsqueeze(0).to(device)

            with torch.no_grad():
                prob = torch.sigmoid(model(xb)).item()

            pred = 1 if prob >= th else 0
            label = int(y_gt[i])
            time_s = i * (hop / sr)

            frame_rows.append({
                "wav": wav,
                "frame_idx": i,
                "time_s": float(time_s),
                "token": tokens[i],
                "label": label,
                "prob": float(prob),
                "pred": pred
            })

            all_y.append(label)
            all_p.append(pred)

            if pred == 1:
                pred_starts.append(float(time_s))

        # build segments for this wav using [start_i, start_{i+1})
        pred_starts = sorted(pred_starts)
        if len(pred_starts) >= 1:
            for k, s in enumerate(pred_starts):
                e = pred_starts[k + 1] if (k + 1) < len(pred_starts) else float((T - 1) * (hop / sr))
                if e > s:
                    seg_rows.append({
                        "wav": wav,
                        "seg_idx": k + 1,
                        "start": float(s),
                        "end": float(e),
                        "duration": float(e - s)
                    })

        print("done:", os.path.relpath(wav, args.data_dir),
              "frames_used:", sum(1 for t in tokens[h:T-h] if t != "０"),
              "pred_starts:", len(pred_starts))

    # write combined frame csv
    cols = ["wav", "frame_idx", "time_s", "token", "label", "prob", "pred"]
    with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in frame_rows:
            w.writerow({k: r.get(k) for k in cols})
    print("WROTE:", args.out_csv)

    # write combined segments csv
    seg_cols = ["wav", "seg_idx", "start", "end", "duration"]
    with open(args.out_segments_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=seg_cols)
        w.writeheader()
        for r in seg_rows:
            w.writerow({k: r.get(k) for k in seg_cols})
    print("WROTE:", args.out_segments_csv)

    metrics = confusion_counts(all_y, all_p)
    print("\n[GLOBAL CONFUSION]", metrics)

    with open(args.out_metrics, "w", encoding="utf-8") as f:
        json.dump({"threshold": th, "tier": tier, "metrics": metrics}, f, ensure_ascii=False, indent=2)
    print("WROTE:", args.out_metrics)

if __name__ == "__main__":
    main()
