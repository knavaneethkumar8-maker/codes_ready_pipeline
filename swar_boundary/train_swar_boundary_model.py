import os, glob, csv, json, argparse, random
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

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

# Consonants for boundary logic
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
# TextGrid parsing (IntervalTier)
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
    # anything outside AKSHAR_SET -> silence token
    if t not in AKSHAR_SET:
        return "०"
    return t

# =========================
# Frame features (LIGHT, no torchaudio/librosa)
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

# =========================
# YOUR boundary label logic
# =========================
# For non-silence frames only:
# label[i]=1 if:
#   - first non-silence frame in stream
#   - OR token[i] is VYANJAN and previous non-silence token is NOT VYANJAN
#
# This produces for: भ इ इ व व अ अ अ
# labels:           1 0 0 1 0 0 0 0
def build_unit_start_labels(tokens):
    T = len(tokens)
    y = np.zeros(T, dtype=np.int64)

    prev_ns = None  # previous non-silence token
    for i in range(T):
        cur = tokens[i]
        if cur == "०":
            continue

        if prev_ns is None:
            y[i] = 1  # first token of stream
        else:
            if is_vyanjan(cur) and (not is_vyanjan(prev_ns)):
                y[i] = 1

        prev_ns = cur

    return y

# =========================
# Dataset
# =========================
class BoundaryDataset(Dataset):
    def __init__(self, samples, sr, win, hop, ctx):
        self.samples = samples
        self.sr = sr
        self.win = win
        self.hop = hop
        self.ctx = ctx
        self.h = ctx // 2

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        # item: dict with wav, frame_idx, time_s, token, label, feat_stack
        # feat_stack is context-stacked vector already
        x = torch.from_numpy(item["x"]).float()
        y = torch.tensor(float(item["label"])).float()
        return x, y, item["wav"], int(item["frame_idx"]), float(item["time_s"]), item["token"], int(item["label"])

def confusion_counts(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    acc = float((tp + tn) / max(len(y_true), 1))
    return {"acc": acc, "tp": tp, "tn": tn, "fp": fp, "fn": fn, "n": int(len(y_true)), "pos": int(y_true.sum())}

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

def build_samples_for_pair(wav_path, tg_path, tier, sr, win, hop, ctx):
    # load textgrid tokens
    lines = read_textgrid_lines(tg_path)
    intervals = parse_interval_tier(lines, tier)
    tokens = [clean_tok(txt) for _, _, txt in intervals]
    y_gt = build_unit_start_labels(tokens)

    # load audio and frame it
    y, sr0 = sf.read(wav_path)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float32)

    if sr0 != sr:
        # resample on the fly
        import librosa
        y = librosa.resample(y.astype(np.float32), orig_sr=sr0, target_sr=sr)

    frames = frame_signal(y, win, hop)
    T = min(len(frames), len(tokens))
    frames = frames[:T]
    tokens = tokens[:T]
    y_gt = y_gt[:T]

    feats = np.stack([extract_frame_feat(frames[i]) for i in range(T)], axis=0)  # (T,D)

    assert ctx % 2 == 1
    h = ctx // 2

    samples = []
    for i in range(h, T - h):
        if tokens[i] == "०":
            continue  # silence filtered out
        x = feats[i - h:i + h + 1].reshape(-1).astype(np.float32)
        time_s = i * (hop / sr)  # frame start time
        samples.append({
            "wav": wav_path,
            "frame_idx": i,
            "time_s": float(time_s),
            "token": tokens[i],
            "label": int(y_gt[i]),
            "x": x
        })
    return samples

def write_combined_csv(path, records):
    cols = ["split", "wav", "frame_idx", "time_s", "token", "label", "prob", "pred"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in records:
            w.writerow({k: r.get(k) for k in cols})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=".")
    ap.add_argument("--tier", default="जल")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--win_ms", type=float, default=27.0)
    ap.add_argument("--hop_ms", type=float, default=27.0)
    ap.add_argument("--ctx", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--th", type=float, default=0.7)

    ap.add_argument("--out_pt", default="swar_boundary_v1.pt")
    ap.add_argument("--out_csv", default="trainval_all_frames.csv")
    ap.add_argument("--out_metrics", default="trainval_metrics.json")

    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    sr = args.sr
    win = int(round(args.win_ms / 1000.0 * sr))
    hop = int(round(args.hop_ms / 1000.0 * sr))

    # gather pairs
    wavs = glob.glob(os.path.join(args.data_dir, "**/*.wav"), recursive=True)
    pairs = []
    for w in wavs:
        tg = os.path.splitext(w)[0] + ".TextGrid"
        if os.path.exists(tg):
            pairs.append((w, tg))

    if not pairs:
        raise SystemExit("No wav/TextGrid pairs found")

    # build samples from ALL files
    all_samples = []
    for w, tg in pairs:
        try:
            s = build_samples_for_pair(w, tg, args.tier, sr, win, hop, args.ctx)
            if len(s) > 0:
                all_samples.extend(s)
                print("loaded:", os.path.relpath(w, args.data_dir), "samples:", len(s))
        except Exception as e:
            print("skip:", w, "err:", e)

    if len(all_samples) == 0:
        raise SystemExit("No usable samples built. Check tier name and sample rate.")

    # split train/val
    idx = np.arange(len(all_samples))
    np.random.shuffle(idx)
    split = int(0.8 * len(idx))
    tr_idx = idx[:split]
    va_idx = idx[split:]

    train_samples = [all_samples[i] for i in tr_idx]
    val_samples   = [all_samples[i] for i in va_idx]

    train_ds = BoundaryDataset(train_samples, sr, win, hop, args.ctx)
    val_ds   = BoundaryDataset(val_samples, sr, win, hop, args.ctx)

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    val_dl   = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    # model
    in_dim = train_samples[0]["x"].shape[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyMLP(in_dim).to(device)

    # class weight
    pos = sum(s["label"] for s in train_samples)
    neg = len(train_samples) - pos
    pos_weight = torch.tensor([neg / max(pos, 1)], dtype=torch.float32).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    print("\nTOTAL train:", len(train_samples), "val:", len(val_samples))
    print("pos_weight:", float(pos_weight.item()), "threshold:", args.th)

    # train loop
    for ep in range(args.epochs):
        model.train()
        losses = []
        for xb, yb, *_ in train_dl:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        print(f"ep={ep} loss={np.mean(losses):.4f}")

    # eval + write combined csv (train+val)
    def collect(dl, split_name):
        model.eval()
        recs = []
        y_true = []
        y_pred = []
        with torch.no_grad():
            for xb, yb_float, wav, frame_idx, time_s, token, y_int in dl:
                prob = torch.sigmoid(model(xb.to(device))).cpu().numpy()
                pred = (prob >= args.th).astype(int)

                y_true.extend(y_int.numpy().astype(int).tolist())
                y_pred.extend(pred.astype(int).tolist())

                for i in range(len(prob)):
                    recs.append({
                        "split": split_name,
                        "wav": wav[i],
                        "frame_idx": int(frame_idx[i]),
                        "time_s": float(time_s[i]),
                        "token": token[i],
                        "label": int(y_int[i]),
                        "prob": float(prob[i]),
                        "pred": int(pred[i]),
                    })
        return recs, y_true, y_pred

    train_eval_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=False, num_workers=0)
    val_eval_dl   = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    rec_tr, ytr, ptr = collect(train_eval_dl, "train")
    rec_va, yva, pva = collect(val_eval_dl, "val")

    m_tr = confusion_counts(ytr, ptr)
    m_va = confusion_counts(yva, pva)

    print("\n[CONFUSION TRAIN]", m_tr)
    print("[CONFUSION VAL]  ", m_va)

    write_combined_csv(args.out_csv, rec_tr + rec_va)
    print("WROTE:", args.out_csv)

    # save pt
    ckpt = {
        "model_state_dict": model.state_dict(),
        "config": {
            "sr": sr,
            "win_ms": args.win_ms,
            "hop_ms": args.hop_ms,
            "ctx": args.ctx,
            "tier": args.tier,
            "threshold": args.th,
            "akshar_set": sorted(list(AKSHAR_SET))
        }
    }
    torch.save(ckpt, args.out_pt)
    print("SAVED:", args.out_pt)

    # save metrics json
    outm = {
        "train": m_tr,
        "val": m_va,
        "threshold": args.th,
        "config": ckpt["config"]
    }
    with open(args.out_metrics, "w", encoding="utf-8") as f:
        json.dump(outm, f, ensure_ascii=False, indent=2)
    print("SAVED:", args.out_metrics)

if __name__ == "__main__":
    main()
