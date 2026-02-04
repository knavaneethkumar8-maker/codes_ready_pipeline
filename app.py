import os
import io
import json
import math
import tempfile
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from flask import Flask, jsonify, request

import numpy as np
import torch
import soundfile as sf
import librosa


# ============================================================
# Flask app
# ============================================================
app = Flask(__name__)
REPO = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _ok(data: Any, **meta):
    payload = {"ok": True, "data": data}
    if meta:
        payload["meta"] = meta
    return jsonify(payload)


def _err(msg: str, status: int = 400, **extra):
    payload: Dict[str, Any] = {"ok": False, "error": {"message": msg}}
    if extra:
        payload["error"].update(extra)
    return jsonify(payload), status


def _require_file(path: str, hint: str = ""):
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}" + (f" | {hint}" if hint else ""))


def _save_upload(field: str, suffix: str) -> Optional[str]:
    """Save an uploaded file (multipart) to a temp file. Returns path or None."""
    f = request.files.get(field)
    if not f:
        return None
    fd, p = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    f.save(p)
    return p


# ============================================================
# SL (Supervised Learning) – grid JSON
# Folder: SL_akshar_pred/
# ============================================================
SL_DIR = os.path.join(REPO, "SL_akshar_pred")
SL_MODELS_DIR = os.path.join(SL_DIR, "models")
SL_MODEL_PATH = os.path.join(SL_MODELS_DIR, "sl_char.pt")
SL_CLASSES_PATH = os.path.join(SL_MODELS_DIR, "classes.json")


def _sl_import():
    """Import SL modules from SL_akshar_pred without requiring pip packaging."""
    import importlib.util

    def _load(name: str, rel: str):
        path = os.path.join(SL_DIR, rel)
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(mod)
        return mod

    cfg = _load("sl_cfg", "config.py")
    features = _load("sl_features", "features.py")
    model_mod = _load("sl_model", "model.py")
    tg_parser = _load("sl_tg", "textgrid_parser.py")
    return cfg, features, model_mod, tg_parser


_SL = {}


def sl_get() -> Tuple[Any, Any, Any, Any, torch.nn.Module, list]:
    """Load SL modules + model once, keep globally."""
    global _SL
    if _SL:
        return _SL["cfg"], _SL["features"], _SL["model_mod"], _SL["tg"], _SL["model"], _SL["classes"]

    cfg, features, model_mod, tg = _sl_import()

    _require_file(SL_CLASSES_PATH, "Expected SL_akshar_pred/classes.json")
    with open(SL_CLASSES_PATH, "r", encoding="utf-8") as f:
        classes = json.load(f)

    model = model_mod.SLNet(len(classes)).to(DEVICE)
    if os.path.exists(SL_MODEL_PATH):
        model.load_state_dict(torch.load(SL_MODEL_PATH, map_location=DEVICE))
        model.eval()
    else:
        # model not trained yet – keep uninitialized but present
        model.eval()

    _SL = {"cfg": cfg, "features": features, "model_mod": model_mod, "tg": tg, "model": model, "classes": classes}
    return cfg, features, model_mod, tg, model, classes


DEFAULT_CELL = {"status": "NEW", "is_locked": False, "metadata": {}}


def _make_cell_id(audio_id: str, grid_idx: int, cell_global_idx: int) -> str:
    return f"{audio_id}_{grid_idx}_{cell_global_idx}"


@torch.no_grad()
def sl_infer_grid_json(wav_path: str, audio_id: Optional[str] = None) -> Dict[str, Any]:
    cfg, features, _, _, model, classes = sl_get()

    audio_id = audio_id or os.path.basename(wav_path)
    x, sr = features.load_wav_mono(wav_path, cfg.SAMPLE_RATE)
    duration_ms = int(round(len(x) * 1000 / sr))

    if not os.path.exists(SL_MODEL_PATH):
        raise FileNotFoundError(f"Missing SL model: {SL_MODEL_PATH} (call /sl/train first)")

    grid_ms = int(getattr(cfg, "GRID_MS", 216))
    total_grids = int(math.ceil(duration_ms / grid_ms))

    jal_ms = 27
    jal_per_grid = 8
    prithvi_per_jal = 3

    grids = []
    for g in range(total_grids):
        g_start = g * grid_ms
        g_end = min((g + 1) * grid_ms, duration_ms)

        # 1) jal predictions
        jal_cells = []  # (text, conf, start_ms, end_ms)
        for j in range(jal_per_grid):
            c_start = g_start + j * jal_ms
            c_end = min(c_start + jal_ms, duration_ms)
            s = int(c_start * sr / 1000)
            e = int(c_end * sr / 1000)
            seg = x[s:e]
            feats = features.mfcc_energy_zcr(seg, sr=sr, win_ms=20, hop_ms=5)  # (T,15)
            xb = torch.tensor(feats, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            _, probs = model(xb)
            p, idx = torch.max(probs[0], dim=0)
            text = classes[int(idx)]
            conf = float(p.item())
            jal_cells.append((text, conf, int(c_start), int(c_end)))

        tiers_out: Dict[str, Any] = {}

        # akash (1 cell): highest confidence
        ak_text, ak_conf, _, _ = max(jal_cells, key=lambda t: t[1])
        tiers_out["akash"] = {
            "name": "आकाश",
            "index": 0,
            "start_ms": g_start,
            "end_ms": g_end,
            "cells": [
                {
                    "id": _make_cell_id(audio_id, g, 1),
                    "index": 1,
                    "start_ms": g_start,
                    "end_ms": g_end,
                    "text": ak_text + " ",
                    "conf": round(float(ak_conf), 4),
                    **DEFAULT_CELL,
                }
            ],
        }

        # agni (2 cells): each aggregates 4 jal
        tiers_out["agni"] = {"name": "अग्नि", "index": 1, "start_ms": g_start, "end_ms": g_end, "cells": []}
        for a in range(2):
            seg = jal_cells[a * 4 : (a + 1) * 4]
            t, c, _, _ = max(seg, key=lambda t: t[1])
            st = g_start + a * 108
            en = min(st + 108, duration_ms)
            tiers_out["agni"]["cells"].append(
                {
                    "id": _make_cell_id(audio_id, g, 1 + (a + 1)),
                    "index": a + 1,
                    "start_ms": st,
                    "end_ms": en,
                    "text": t + " ",
                    "conf": round(float(c), 4),
                    **DEFAULT_CELL,
                }
            )

        # vayu (4 cells): each aggregates 2 jal
        tiers_out["vayu"] = {"name": "वायु", "index": 2, "start_ms": g_start, "end_ms": g_end, "cells": []}
        for v in range(4):
            seg = jal_cells[v * 2 : (v + 1) * 2]
            t, c, _, _ = max(seg, key=lambda t: t[1])
            st = g_start + v * 54
            en = min(st + 54, duration_ms)
            tiers_out["vayu"]["cells"].append(
                {
                    "id": _make_cell_id(audio_id, g, 10 + v),
                    "index": v + 1,
                    "start_ms": st,
                    "end_ms": en,
                    "text": t + " ",
                    "conf": round(float(c), 4),
                    **DEFAULT_CELL,
                }
            )

        # jal (8 cells)
        tiers_out["jal"] = {"name": "जल", "index": 3, "start_ms": g_start, "end_ms": g_end, "cells": []}
        for j, (t, c, st, en) in enumerate(jal_cells, start=1):
            tiers_out["jal"]["cells"].append(
                {
                    "id": _make_cell_id(audio_id, g, 20 + j),
                    "index": j,
                    "start_ms": st,
                    "end_ms": en,
                    "text": t + " ",
                    "conf": round(float(c), 4),
                    **DEFAULT_CELL,
                }
            )

        # prithvi (24 cells): expand each jal into 3 x 9ms
        tiers_out["prithvi"] = {"name": "पृथ्वी", "index": 4, "start_ms": g_start, "end_ms": g_end, "cells": []}
        p_idx = 1
        for (t, c, st, en) in jal_cells:
            _ = en
            for k in range(prithvi_per_jal):
                pst = st + k * 9
                pen = min(pst + 9, duration_ms)
                tiers_out["prithvi"]["cells"].append(
                    {
                        "id": _make_cell_id(audio_id, g, 40 + p_idx),
                        "index": p_idx,
                        "start_ms": pst,
                        "end_ms": pen,
                        "text": t + " ",
                        "conf": round(float(c), 4),
                        **DEFAULT_CELL,
                    }
                )
                p_idx += 1

        grids.append(
            {
                "id": f"{audio_id}_{g}",
                "index": g,
                "start_ms": g_start,
                "end_ms": g_end,
                "status": "NEW",
                "is_locked": False,
                "metadata": {},
                "tiers": tiers_out,
            }
        )

    return {
        "metadata": {
            "audio_id": audio_id,
            "duration_ms": duration_ms,
            "grid_size_ms": grid_ms,
            "total_grids": total_grids,
            "uploaded_at": datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
            "status": "NEW",
        },
        "grids": grids,
    }


def sl_train_from_files(wav_path: str, tg_path: str, epochs: int = 6, batch: int = 64, lr: float = 1e-3) -> Dict[str, Any]:
    cfg, features, model_mod, tg, _, classes = sl_get()

    # Build dataset like SL_akshar_pred/train_sl.py (tier 'जल' → class by first char)
    x, sr = features.load_wav_mono(wav_path, cfg.SAMPLE_RATE)
    tiers = tg.parse_textgrid(tg_path)
    jal = tiers.get("जल", [])
    if not jal:
        raise RuntimeError("Tier 'जल' not found in TextGrid.")

    cls2i = {c: i for i, c in enumerate(classes)}
    X = []
    Y = []
    for xmin, xmax, mark in jal:
        lab = (mark or "").strip()
        if lab == "":
            lab = "None"
        else:
            lab = lab[0]
            if lab not in cls2i:
                lab = "None"

        s = int(float(xmin) * sr)
        e = int(float(xmax) * sr)
        seg = x[s:e]
        feats = features.mfcc_energy_zcr(seg, sr=sr, win_ms=20, hop_ms=5)
        X.append(torch.tensor(feats, dtype=torch.float32))
        Y.append(int(cls2i[lab]))

    if not X:
        raise RuntimeError("No samples built from TextGrid tier 'जल'.")

    # pad to (B,T,15)
    from torch.nn.utils.rnn import pad_sequence
    Xp = pad_sequence(X, batch_first=True)
    Yt = torch.tensor(Y, dtype=torch.long)

    ds = torch.utils.data.TensorDataset(Xp, Yt)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True)

    model = model_mod.SLNet(len(classes)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = torch.nn.CrossEntropyLoss()

    model.train()
    losses = []
    for ep in range(epochs):
        total = 0.0
        for xb, yb in dl:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            opt.zero_grad()
            logits, _ = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            total += float(loss.item())
        losses.append(total / max(1, len(dl)))

    os.makedirs(SL_MODELS_DIR, exist_ok=True)
    torch.save(model.state_dict(), SL_MODEL_PATH)

    # refresh global model
    _SL.clear()
    sl_get()

    return {"samples": int(len(Y)), "epochs": int(epochs), "losses": [round(float(x), 6) for x in losses], "model_path": SL_MODEL_PATH}


# ============================================================
# Swar Boundary + Swar RL calibration (27ms)
# Folder: swar_boundary/ and swar_RL/
# ============================================================
SWAR_BOUNDARY_DIR = os.path.join(REPO, "swar_boundary")
SWAR_BOUNDARY_CKPT = os.path.join(SWAR_BOUNDARY_DIR, "swar_boundary_v1.pt")

SWAR_RL_DIR = os.path.join(REPO, "swar_RL")
SWAR_CALIB_PT = os.path.join(SWAR_RL_DIR, "swar_boundary_calib.pt")


class _TinyMLP(torch.nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _extract_swar_frame_feat(frame: np.ndarray, n_fft: int = 512, n_bins: int = 32) -> np.ndarray:
    frame = frame.astype(np.float32)
    frame = frame - frame.mean()
    energy = float(np.sqrt(np.mean(frame**2) + 1e-12))
    z = float(np.mean(np.abs(np.diff(np.sign(frame))) > 0))
    w = np.hanning(len(frame)).astype(np.float32)
    X = np.fft.rfft(frame * w, n=n_fft)
    mag = np.abs(X) + 1e-6
    logmag = np.log(mag)
    spec = logmag[1 : 1 + n_bins].astype(np.float32)
    return np.concatenate([spec, np.array([energy, z], np.float32)], axis=0)


_SWAR = {}


def swar_get():
    """Load swar boundary model + (optional) calibration once."""
    global _SWAR
    if _SWAR:
        return _SWAR
    _require_file(SWAR_BOUNDARY_CKPT, "Expected swar_boundary/swar_boundary_v1.pt")
    ckpt = torch.load(SWAR_BOUNDARY_CKPT, map_location=DEVICE)
    cfg = ckpt.get("config", {})
    sr = int(cfg.get("sr", 16000))
    win_ms = float(cfg.get("win_ms", 27.0))
    hop_ms = float(cfg.get("hop_ms", 27.0))
    ctx = int(cfg.get("ctx", 5))
    d = 32 + 2
    in_dim = ctx * d
    model = _TinyMLP(in_dim).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    calib = None
    if os.path.exists(SWAR_CALIB_PT):
        c = torch.load(SWAR_CALIB_PT, map_location="cpu")
        a = float(c.get("a", float("nan")))
        b = float(c.get("b", float("nan")))
        if not (math.isnan(a) or math.isnan(b)):
            calib = {"a": a, "b": b}

    _SWAR = {"model": model, "sr": sr, "win_ms": win_ms, "hop_ms": hop_ms, "ctx": ctx, "calib": calib}
    return _SWAR


@torch.no_grad()
def swar_boundary_probs(wav_path: str) -> Dict[str, Any]:
    """Return per-frame swar boundary probs at hop_ms (default 27ms)."""
    s = swar_get()
    sr = s["sr"]
    win = int(round(s["win_ms"] / 1000.0 * sr))
    hop = int(round(s["hop_ms"] / 1000.0 * sr))
    ctx = int(s["ctx"])
    h = ctx // 2

    y, sr0 = sf.read(wav_path)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float32)
    if sr0 != sr:
        y = librosa.resample(y, orig_sr=sr0, target_sr=sr)

    # frame
    if len(y) < win:
        y = np.pad(y, (0, win - len(y)))
    T = 1 + max(0, (len(y) - win) // hop)
    frames = np.lib.stride_tricks.as_strided(
        y,
        shape=(T, win),
        strides=(y.strides[0] * hop, y.strides[0]),
    ).copy()

    feats = np.stack([_extract_swar_frame_feat(frames[i]) for i in range(T)], axis=0)  # (T,d)
    d = feats.shape[1]

    # context concat
    X = np.zeros((T, ctx * d), dtype=np.float32)
    for i in range(T):
        parts = []
        for j in range(i - h, i + h + 1):
            if j < 0 or j >= T:
                parts.append(np.zeros((d,), dtype=np.float32))
            else:
                parts.append(feats[j])
        X[i] = np.concatenate(parts, axis=0)

    xb = torch.from_numpy(X).to(DEVICE)
    logits = s["model"](xb)
    prob = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float64)

    out = {"hop_ms": int(round(s["hop_ms"])), "prob": prob.tolist()}

    # apply calibration if present
    if s["calib"] is not None:
        a = float(s["calib"]["a"])
        b = float(s["calib"]["b"])
        newp = _sigmoid_np(a * _logit(prob) + b)
        out["prob_calib"] = newp.tolist()
    return out


# ============================================================
# Shunya Akshar Boundary + Shunya RL calibration (27ms)
# Folder: shunya_boundary/ and shunya_RL/
# ============================================================
SHUNYA_BOUNDARY_DIR = os.path.join(REPO, "shunya_boundary")
SHUNYA_BOUNDARY_CKPT = os.path.join(SHUNYA_BOUNDARY_DIR, "akshar_boundary_v1.pt")

SHUNYA_RL_DIR = os.path.join(REPO, "shunya_RL")
SHUNYA_CALIB_PT = os.path.join(SHUNYA_RL_DIR, "akshar_boundary_calib.pt")

_SHUNYA = {}


def shunya_get():
    global _SHUNYA
    if _SHUNYA:
        return _SHUNYA

    # dynamic import to avoid packaging
    import importlib.util

    mod_path = os.path.join(SHUNYA_BOUNDARY_DIR, "gemini_jal_model.py")
    _require_file(mod_path, "Expected shunya_boundary/gemini_jal_model.py")
    spec = importlib.util.spec_from_file_location("shunya_gemini", mod_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)

    _require_file(SHUNYA_BOUNDARY_CKPT, "Expected shunya_boundary/akshar_boundary_v1.pt")
    ckpt = torch.load(SHUNYA_BOUNDARY_CKPT, map_location=DEVICE)
    cfg = ckpt.get("config", {})
    sr = int(cfg.get("sr", getattr(mod, "SR", 16000)))
    win_ms = float(cfg.get("win_ms", getattr(mod, "WIN_MS", 27.0)))
    num_windows = int(cfg.get("num_windows", getattr(mod, "NUM_WINDOWS", 5)))

    model = mod.BoundaryNet().to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    fe = mod.FeatureEngine()

    calib = None
    if os.path.exists(SHUNYA_CALIB_PT):
        c = torch.load(SHUNYA_CALIB_PT, map_location="cpu")
        a = float(c.get("a", float("nan")))
        b = float(c.get("b", float("nan")))
        # your current calib file has NaN -> we skip in that case
        if not (math.isnan(a) or math.isnan(b)):
            calib = {"a": a, "b": b}

    _SHUNYA = {"mod": mod, "model": model, "fe": fe, "sr": sr, "win_ms": win_ms, "num_windows": num_windows, "calib": calib}
    return _SHUNYA


@torch.no_grad()
def shunya_boundary_probs(wav_path: str) -> Dict[str, Any]:
    """Return per-frame akshar boundary probs at 27ms hop."""
    sh = shunya_get()
    sr = int(sh["sr"])
    win = int(round(float(sh["win_ms"]) / 1000.0 * sr))
    hop = win
    h = int(sh["num_windows"]) // 2

    y, sr0 = sf.read(wav_path)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float32)
    if sr0 != sr:
        y = librosa.resample(y, orig_sr=sr0, target_sr=sr)

    if len(y) < win:
        y = np.pad(y, (0, win - len(y)))

    T = 1 + max(0, (len(y) - win) // hop)
    frames = np.lib.stride_tricks.as_strided(
        y,
        shape=(T, win),
        strides=(y.strides[0] * hop, y.strides[0]),
    ).copy()

    # build context features for each frame
    probs = np.zeros((T,), dtype=np.float64)
    for i in range(T):
        windows = []
        for j in range(i - h, i + h + 1):
            if j < 0 or j >= T:
                w = np.zeros((win,), dtype=np.float32)
            else:
                w = frames[j]
            windows.append(torch.from_numpy(w).float())
        feats = torch.stack([sh["fe"].extract(w) for w in windows]).unsqueeze(0)  # (1,5,3,M,T)
        logit = sh["model"](feats.to(DEVICE))
        p = torch.sigmoid(logit).item()
        probs[i] = float(p)

    out = {"hop_ms": int(round(float(sh["win_ms"]))), "prob": probs.tolist()}
    if sh["calib"] is not None:
        a = float(sh["calib"]["a"])
        b = float(sh["calib"]["b"])
        newp = _sigmoid_np(a * _logit(probs) + b)
        out["prob_calib"] = newp.tolist()
    return out


def _interval_max(arr: Optional[list], start_ms: int, end_ms: int, hop_ms: int) -> Optional[float]:
    if not arr:
        return None
    i0 = max(0, int(start_ms // hop_ms))
    i1 = min(len(arr) - 1, int((max(end_ms - 1, start_ms)) // hop_ms))
    if i1 < i0:
        return None
    return float(max(arr[i0 : i1 + 1]))


def runall_json(wav_path: str, audio_id: Optional[str] = None) -> Dict[str, Any]:
    """RUNALL = SWAR boundary + SWAR RL + SHUNYA boundary + SHUNYA RL + SL.

    Output is your standard grid JSON, with extra model outputs stored in each cell's metadata:
      - swar_boundary_prob, swar_boundary_prob_calib
      - shunya_boundary_prob, shunya_boundary_prob_calib
    """
    grid = sl_infer_grid_json(wav_path, audio_id=audio_id)

    sw = None
    sh = None
    try:
        sw = swar_boundary_probs(wav_path)
    except Exception:
        sw = None
    try:
        sh = shunya_boundary_probs(wav_path)
    except Exception:
        sh = None

    sw_hop = int(sw.get("hop_ms", 27)) if sw else 27
    sh_hop = int(sh.get("hop_ms", 27)) if sh else 27
    sw_p = sw.get("prob") if sw else None
    sw_pc = sw.get("prob_calib") if sw else None
    sh_p = sh.get("prob") if sh else None
    sh_pc = sh.get("prob_calib") if sh else None

    for g in grid.get("grids", []):
        tiers = g.get("tiers", {})
        for _, tier in tiers.items():
            for cell in tier.get("cells", []):
                st = int(cell.get("start_ms", 0))
                en = int(cell.get("end_ms", st))
                md = cell.setdefault("metadata", {})
                md.update(
                    {
                        "swar_boundary_prob": _interval_max(sw_p, st, en, sw_hop),
                        "swar_boundary_prob_calib": _interval_max(sw_pc, st, en, sw_hop),
                        "shunya_boundary_prob": _interval_max(sh_p, st, en, sh_hop),
                        "shunya_boundary_prob_calib": _interval_max(sh_pc, st, en, sh_hop),
                    }
                )

    # Add summary into top-level metadata
    grid.setdefault("metadata", {})["runall"] = {
        "swar_boundary_loaded": bool(sw),
        "swar_rl_loaded": bool(sw and sw.get("prob_calib") is not None),
        "shunya_boundary_loaded": bool(sh),
        "shunya_rl_loaded": bool(sh and sh.get("prob_calib") is not None),
        "note": "shunya_RL calib is skipped automatically if calib a/b are NaN",
    }
    return grid


# ============================================================
# Routes
# ============================================================
@app.get("/health")
def health():
    return _ok({"status": "up", "device": str(DEVICE)})


@app.get("/debug/paths")
def debug_paths():
    info = {
        "REPO": REPO,
        "cwd": os.getcwd(),
        "SL_DIR": SL_DIR,
        "SL_CLASSES_PATH": SL_CLASSES_PATH,
        "SL_MODEL_PATH": SL_MODEL_PATH,
        "exists": {
            "SL_DIR": os.path.isdir(SL_DIR),
            "SL_CLASSES": os.path.exists(SL_CLASSES_PATH),
            "SL_MODEL": os.path.exists(SL_MODEL_PATH),
        },
        "folders_here": sorted([p for p in os.listdir(REPO) if os.path.isdir(os.path.join(REPO, p))])[:50],
    }
    return _ok(info)


@app.post("/sl/train")
def api_sl_train():
    """Train SL model from (wav + TextGrid).

    Accepts either:
    - multipart: wav=@file, textgrid=@file
    - json: {"wav_path":"...","textgrid_path":"..."}
    """
    tmp_wav = None
    tmp_tg = None
    try:
        body = request.get_json(silent=True) or {}
        wav_path = body.get("wav_path")
        tg_path = body.get("textgrid_path") or body.get("tg_path")

        if not wav_path:
            tmp_wav = _save_upload("wav", ".wav")
            wav_path = tmp_wav
        if not tg_path:
            tmp_tg = _save_upload("textgrid", ".TextGrid") or _save_upload("tg", ".TextGrid")
            tg_path = tmp_tg

        if not wav_path or not tg_path:
            return _err("Provide wav+textgrid either as multipart upload or wav_path/textgrid_path JSON")

        _require_file(wav_path)
        _require_file(tg_path)

        epochs = int(body.get("epochs", request.form.get("epochs", 6)))
        batch = int(body.get("batch", request.form.get("batch", 64)))
        lr = float(body.get("lr", request.form.get("lr", 1e-3)))

        out = sl_train_from_files(wav_path, tg_path, epochs=epochs, batch=batch, lr=lr)
        return _ok(out)
    except Exception as e:
        return _err("sl train failed", 500, detail=str(e))
    finally:
        for p in (tmp_wav, tmp_tg):
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass


@app.post("/sl/infer")
def api_sl_infer():
    """Infer SL grid JSON from wav.

    Accepts either:
    - multipart: wav=@file
    - json: {"wav_path":"..."}
    """
    tmp_wav = None
    try:
        body = request.get_json(silent=True) or {}
        wav_path = body.get("wav_path")
        audio_id = body.get("audio_id")

        if not wav_path:
            tmp_wav = _save_upload("wav", ".wav")
            wav_path = tmp_wav

        if not wav_path:
            return _err("Provide wav either as multipart upload or wav_path JSON")

        _require_file(wav_path)
        out = sl_infer_grid_json(wav_path, audio_id=audio_id)
        return jsonify(out)
    except Exception as e:
        return _err("sl infer failed", 500, detail=str(e))
    finally:
        if tmp_wav and os.path.exists(tmp_wav):
            try:
                os.remove(tmp_wav)
            except Exception:
                pass


@app.post("/runall")
def api_runall():
    """RUNALL: wav -> (swar boundary + swar_RL + shunya boundary + shunya_RL + SL) -> final grid JSON.

    Accepts either:
    - multipart: wav=@file
    - json: {"wav_path":"...", "audio_id":"..."}
    """
    tmp_wav = None
    try:
        body = request.get_json(silent=True) or {}
        wav_path = body.get("wav_path")
        audio_id = body.get("audio_id")

        if not wav_path:
            tmp_wav = _save_upload("wav", ".wav")
            wav_path = tmp_wav

        if not wav_path:
            return _err("Provide wav either as multipart upload or wav_path JSON")

        _require_file(wav_path)
        out = runall_json(wav_path, audio_id=audio_id)
        return jsonify(out)
    except Exception as e:
        return _err("runall failed", 500, detail=str(e))
    finally:
        if tmp_wav and os.path.exists(tmp_wav):
            try:
                os.remove(tmp_wav)
            except Exception:
                pass


if __name__ == "__main__":
    # default port 8000 to match your repo README
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8000")), debug=False)
