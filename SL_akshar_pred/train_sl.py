
import os
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import config as cfg
from textgrid_parser import parse_textgrid
from features import load_wav_mono, mfcc_energy_zcr
from model import SLNet

class JalDataset(Dataset):
    def __init__(self, wav_path: str, tg_path: str, classes):
        self.items = []
        self.classes = classes
        self.cls2i = {c:i for i,c in enumerate(classes)}

        x, sr = load_wav_mono(wav_path, cfg.SAMPLE_RATE)
        tiers = parse_textgrid(tg_path)
        jal = tiers.get("जल", [])
        if not jal:
            raise RuntimeError("Tier 'जल' not found in TextGrid.")

        for xmin, xmax, mark in jal:
            lab = mark.strip()
            if lab == "":
                lab = "None"
            else:
                lab = lab[0]
                if lab not in self.cls2i:
                    lab = "None"

            start = int(xmin * sr)
            end = int(xmax * sr)
            seg = x[start:end]
            feats = mfcc_energy_zcr(seg, sr=sr, win_ms=20, hop_ms=5)  # (T,15) T≈2
            self.items.append((torch.tensor(feats), self.cls2i[lab]))

    def __len__(self): 
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]

def collate(batch):
    xs, ys = zip(*batch)
    xs_pad = pad_sequence(xs, batch_first=True)  # (B,T,15)
    ys = torch.tensor(ys, dtype=torch.long)
    return xs_pad, ys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True, help="Path to training WAV")
    ap.add_argument("--textgrid", required=True, help="Path to training TextGrid")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out_model", default=os.path.join(cfg.MODEL_DIR, "sl_char.pt"))
    args = ap.parse_args()

    os.makedirs(cfg.MODEL_DIR, exist_ok=True)

    # Load classes (fixed)
    with open(cfg.CLASSES_PATH, "r", encoding="utf-8") as f:
        classes = json.load(f)

    ds = JalDataset(args.wav, args.textgrid, classes)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, collate_fn=collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"TRAIN device: {device} | samples: {len(ds)}")

    model = SLNet(len(classes)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    for ep in range(args.epochs):
        model.train()
        total = 0.0
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            logits, _ = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            total += float(loss.item())
        print(f"epoch {ep+1}/{args.epochs} loss={total/len(dl):.4f}")

    torch.save(model.state_dict(), args.out_model)
    print(f"SAVED: {args.out_model}")

if __name__ == "__main__":
    main()
