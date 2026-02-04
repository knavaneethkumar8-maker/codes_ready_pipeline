
import os
import json
import math
import argparse
from datetime import datetime

import numpy as np
import torch

import config as cfg
from features import load_wav_mono, mfcc_energy_zcr
from model import SLNet

DEFAULT_CELL = {
    "status": "NEW",
    "is_locked": False,
    "metadata": {},
}

def make_id(audio_id, grid_idx, cell_global_idx):
    return f"{audio_id}_{grid_idx}_{cell_global_idx}"

def infer_one(wav_path: str, out_json: str):
    audio_id = os.path.basename(wav_path)
    x, sr = load_wav_mono(wav_path, cfg.SAMPLE_RATE)
    duration_ms = int(round(len(x) * 1000 / sr))

    # load classes + model
    with open(cfg.CLASSES_PATH, "r", encoding="utf-8") as f:
        classes = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SLNet(len(classes)).to(device)
    model_path = os.path.join(cfg.MODEL_DIR, "sl_char.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing model: {model_path} (run train_sl.py first)")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    grid_ms = cfg.GRID_MS
    total_grids = int(math.ceil(duration_ms / grid_ms))

    # We predict at JAL resolution (27ms), then expand down to prithvi 9ms
    jal_ms = 27
    jal_per_grid = 8
    prithvi_per_jal = 3

    grids = []

    for g in range(total_grids):
        g_start = g * grid_ms
        g_end = min((g + 1) * grid_ms, duration_ms)

        # 1) jal predictions for this grid
        jal_cells = []
        for j in range(jal_per_grid):
            c_start = g_start + j * jal_ms
            c_end = min(c_start + jal_ms, duration_ms)
            # extract waveform
            s = int(c_start * sr / 1000)
            e = int(c_end * sr / 1000)
            seg = x[s:e]
            feats = mfcc_energy_zcr(seg, sr=sr, win_ms=20, hop_ms=5)  # (T,15)
            xb = torch.tensor(feats, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                _, probs = model(xb)
                p, idx = torch.max(probs[0], dim=0)
            text = classes[int(idx)]
            conf = float(p.item())
            jal_cells.append((text, conf, c_start, c_end))

        # 2) build tiers by aggregation/expansion
        tiers_out = {}

        # akash: 1 cell (pick most confident jal)
        ak_text, ak_conf, _, _ = max(jal_cells, key=lambda t: t[1])
        tiers_out["akash"] = {
            "name": "आकाश",
            "index": 0,
            "start_ms": g_start,
            "end_ms": g_end,
            "cells": [{
                "id": make_id(audio_id, g, 1),
                "index": 1,
                "start_ms": g_start,
                "end_ms": g_end,
                "text": ak_text + " ",
                "conf": round(ak_conf, 4),
                **DEFAULT_CELL
            }]
        }

        # agni: 2 cells (each aggregates 4 jal)
        agni_cells=[]
        for a in range(2):
            seg = jal_cells[a*4:(a+1)*4]
            t, c, _, _ = max(seg, key=lambda t: t[1])
            st = g_start + a*108
            en = min(st+108, duration_ms)
            agni_cells.append((t,c,st,en))
        tiers_out["agni"] = {"name":"अग्नि","index":1,"start_ms":g_start,"end_ms":g_end,"cells":[]}
        for idx,(t,c,st,en) in enumerate(agni_cells, start=1):
            tiers_out["agni"]["cells"].append({
                "id": make_id(audio_id, g, 1+idx),
                "index": idx,
                "start_ms": st,
                "end_ms": en,
                "text": t + " ",
                "conf": round(float(c),4),
                **DEFAULT_CELL
            })

        # vayu: 4 cells (each aggregates 2 jal)
        tiers_out["vayu"] = {"name":"वायु","index":2,"start_ms":g_start,"end_ms":g_end,"cells":[]}
        for v in range(4):
            seg = jal_cells[v*2:(v+1)*2]
            t,c,_,_ = max(seg, key=lambda t: t[1])
            st=g_start+v*54
            en=min(st+54, duration_ms)
            tiers_out["vayu"]["cells"].append({
                "id": make_id(audio_id, g, 10+v),
                "index": v+1,
                "start_ms": st,
                "end_ms": en,
                "text": t + " ",
                "conf": round(float(c),4),
                **DEFAULT_CELL
            })

        # jal: 8 cells (direct)
        tiers_out["jal"]={"name":"जल","index":3,"start_ms":g_start,"end_ms":g_end,"cells":[]}
        for j,(t,c,st,en) in enumerate(jal_cells, start=1):
            tiers_out["jal"]["cells"].append({
                "id": make_id(audio_id, g, 20+j),
                "index": j,
                "start_ms": st,
                "end_ms": en,
                "text": t + " ",
                "conf": round(float(c),4),
                **DEFAULT_CELL
            })

        # prithvi: 24 cells (expand each jal into 3 x 9ms)
        tiers_out["prithvi"]={"name":"पृथ्वी","index":4,"start_ms":g_start,"end_ms":g_end,"cells":[]}
        p_idx=1
        for (t,c,st,en) in jal_cells:
            for k in range(prithvi_per_jal):
                pst = st + k*9
                pen = min(pst+9, duration_ms)
                tiers_out["prithvi"]["cells"].append({
                    "id": make_id(audio_id, g, 40+p_idx),
                    "index": p_idx,
                    "start_ms": pst,
                    "end_ms": pen,
                    "text": t + " ",
                    "conf": round(float(c),4),
                    **DEFAULT_CELL
                })
                p_idx += 1

        grids.append({
            "id": f"{audio_id}_{g}",
            "index": g,
            "start_ms": g_start,
            "end_ms": g_end,
            "status": "NEW",
            "is_locked": False,
            "metadata": {},
            "tiers": tiers_out
        })

    out = {
        "metadata": {
            "audio_id": audio_id,
            "duration_ms": duration_ms,
            "grid_size_ms": grid_ms,
            "total_grids": total_grids,
            "uploaded_at": datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
            "status": "NEW",
        },
        "grids": grids
    }

    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"WROTE: {out_json}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    infer_one(args.wav, args.out)

if __name__ == "__main__":
    main()
