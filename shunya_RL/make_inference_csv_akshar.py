import argparse, os, math
import pandas as pd
import numpy as np
import torch
import torchaudio.transforms as T
import librosa

# ---- Model + FeatureEngine (copied from your gemini_jal_model.py, inference-side) ----
SR = 16000
WIN_MS = 27
WIN_SAMPLES = int(SR * (WIN_MS / 1000))
NUM_WINDOWS = 5
CONTEXT_SAMPLES = WIN_SAMPLES * NUM_WINDOWS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BoundaryNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 4))
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(32 * 4 * NUM_WINDOWS, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, 1)
        )

    def forward(self, x):
        # x: (B,5,3,M,T)
        b, n, c, m, t = x.shape
        x = x.view(b * n, c, m, t)
        x = self.feature_extractor(x)
        x = x.view(b, -1)
        return self.classifier(x).squeeze(-1)

class FeatureEngine:
    def __init__(self):
        self.mel_spec = T.MelSpectrogram(SR, n_mels=40, n_fft=256, hop_length=64)

    def extract(self, audio_tensor):
        # audio_tensor: 1D float tensor (samples)
        mel = self.mel_spec(audio_tensor)
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)

        # manual deltas (short-window safe)
        delta = mel[:, 1:] - mel[:, :-1]
        delta = torch.nn.functional.pad(delta, (1, 0))
        delta2 = delta[:, 1:] - delta[:, :-1]
        delta2 = torch.nn.functional.pad(delta2, (1, 0))

        feats = torch.stack([mel, delta, delta2], dim=0)  # (3, M, T)
        return feats.float()

def load_wav(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    return y.astype(np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", default="jal_dataset.csv", help="Must have columns: wav,start,end,(label optional)")
    ap.add_argument("--out_csv", default="akshar_infer.csv")
    ap.add_argument("--model_pt", default="akshar_boundary_v1.pt")
    ap.add_argument("--thresh", type=float, default=0.5, help="threshold for pred column")
    ap.add_argument("--max_rows", type=int, default=0, help="0 = all, else limit rows for quick test")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    if "wav" not in df.columns:
        raise SystemExit("Input CSV must have 'wav' column")
    if "start" not in df.columns:
        raise SystemExit("Input CSV must have 'start' column (seconds)")

    if args.max_rows and args.max_rows > 0:
        df = df.head(args.max_rows).copy()

    ckpt = torch.load(args.model_pt, map_location=DEVICE)
    model = BoundaryNet().to(DEVICE)
    # supports both raw state_dict or dict with model_state_dict
    sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(sd, strict=True)
    model.eval()
    fe = FeatureEngine()

    # cache audio by wav path
    cache = {}
    probs = []
    ok = 0
    for i, r in df.iterrows():
        wav = str(r["wav"])
        if wav not in cache:
            if not os.path.exists(wav):
                # if wav is just filename, try in current dir
                alt = os.path.basename(wav)
                if os.path.exists(alt):
                    wav = alt
                else:
                    probs.append(np.nan)
                    continue
            cache[wav] = load_wav(wav)

        y = cache[wav]
        t = float(r["start"])
        center = int(round(t * SR))

        # build 5 windows centered so that middle window corresponds to [t, t+win)
        # same logic as generate_tg: middle window index 2 in context
        ctx_start = center - 2 * WIN_SAMPLES
        ctx_end = ctx_start + CONTEXT_SAMPLES

        # pad edges
        if ctx_start < 0:
            pad = -ctx_start
            chunk = np.pad(y[:ctx_end], (pad, 0))
        elif ctx_end > len(y):
            pad = ctx_end - len(y)
            chunk = np.pad(y[ctx_start:], (0, pad))
        else:
            chunk = y[ctx_start:ctx_end]

        if len(chunk) != CONTEXT_SAMPLES:
            chunk = np.pad(chunk, (0, max(0, CONTEXT_SAMPLES - len(chunk))))[:CONTEXT_SAMPLES]

        windows = np.split(chunk, NUM_WINDOWS)
        feats = torch.stack([fe.extract(torch.from_numpy(w)) for w in windows], dim=0)  # (5,3,M,T)
        feats = feats.unsqueeze(0).to(DEVICE)  # (1,5,3,M,T)

        with torch.no_grad():
            p = torch.sigmoid(model(feats)).item()
        probs.append(p)
        ok += 1
        if ok % 500 == 0:
            print(f"[infer] processed {ok} rows")

    df = df.copy()
    df["prob"] = probs
    df["pred"] = (df["prob"].astype(float) >= args.thresh).astype(int)

    # drop NaNs (missing wav) but keep for debugging if you want
    print(f"[DONE] wrote prob/pred. missing_prob={int(pd.isna(df['prob']).sum())} / {len(df)}")
    df.to_csv(args.out_csv, index=False)
    print(f"[OK] wrote {args.out_csv}")

if __name__ == "__main__":
    main()
