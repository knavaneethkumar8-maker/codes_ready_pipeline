# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchaudio.transforms as T
# from torch.utils.data import Dataset, DataLoader
# import numpy as np
# import librosa
# import csv, random
# from sklearn.metrics import confusion_matrix, classification_report

# # ===================== Config =====================
# SR = 16000
# WIN_MS = 27  # Your specific akshar duration
# WIN_SAMPLES = int(SR * (WIN_MS / 1000)) # ~432 samples
# CONTEXT_SAMPLES = WIN_SAMPLES * 5       # 5-window span (135ms total)
# BATCH_SIZE = 32
# LR = 5e-4
# EPOCHS = 40
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # ===================== Architecture =====================
# class BoundaryNet(nn.Module):
#     def __init__(self, n_mels=40):
#         super().__init__()
#         # Shared Feature Extractor for each 27ms window
#         # Focuses on Delta (Δ) and Delta-Delta (ΔΔ) via 2D Conv
#         self.feature_extractor = nn.Sequential(
#             nn.Conv2d(1, 16, kernel_size=(3, 3), padding=1),
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
#             nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.AdaptiveAvgPool2d((1, 4)) 
#         )
        
#         # Classifier sees 5 windows worth of features concatenated
#         self.classifier = nn.Sequential(
#             nn.Linear(32 * 4 * 5, 128),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(128, 1)
#         )

#     def forward(self, x):
#         # Input x: (Batch, 5, Mels, Time)
#         b, n, m, t = x.shape
#         x = x.view(b * n, 1, m, t) # Treat each window as a separate "image"
        
#         x = self.feature_extractor(x) # (B*5, 32, 1, 4)
#         x = x.view(b, -1)             # Concatenate all 5 windows: (B, 32*4*5)
        
#         return self.classifier(x).squeeze()

# # ===================== Processing & Dataset =====================
# class AksharBoundaryDataset(Dataset):
#     def __init__(self, rows):
#         self.rows = rows
#         self.mel_spec = T.MelSpectrogram(SR, n_mels=40, n_fft=256, hop_length=64)

#     def __len__(self): return len(self.rows)

#     def _get_features(self, audio):
#         # Extract Mel + Delta + Delta-Delta equivalent via CNN structure
#         # Normalizing each segment independently to preserve boundary energy
#         m = self.mel_spec(audio)
#         m = (m - m.mean()) / (m.std() + 1e-6)
#         return m

#     def __getitem__(self, idx):
#         wav, start, end, label = self.rows[idx]
        
#         # Exact Boundary Logic: 
#         # Load 2 windows before, current window, 2 windows after
#         total_dur = (WIN_MS * 5) / 1000
#         load_start = start - (2 * (WIN_MS / 1000))
        
#         try:
#             y, _ = librosa.load(wav, sr=SR, offset=load_start, duration=total_dur)
#         except:
#             y = np.zeros(CONTEXT_SAMPLES)

#         # Padding to ensure exact context length
#         if len(y) < CONTEXT_SAMPLES:
#             y = np.pad(y, (0, CONTEXT_SAMPLES - len(y)))
#         y = y[:CONTEXT_SAMPLES]
        
#         # Split into 5 windows and compute features per window
#         windows = np.split(y, 5)
#         feats = torch.stack([self._get_features(torch.from_numpy(w)) for w in windows])
        
#         return feats, torch.tensor(label, dtype=torch.float32)

# # ===================== Training & Evaluation =====================
# def train_and_eval():
#     # 1. Load Data
#     rows = []
#     with open("jal_dataset.csv") as f:
#         for r in csv.DictReader(f):
#             rows.append((r["wav"], float(r["start"]), float(r["end"]), int(r["label"])))
    
#     random.shuffle(rows)
#     split = int(0.8 * len(rows))
#     train_dl = DataLoader(AksharBoundaryDataset(rows[:split]), batch_size=BATCH_SIZE, shuffle=True)
#     test_dl = DataLoader(AksharBoundaryDataset(rows[split:]), batch_size=BATCH_SIZE)

#     # 2. Setup (Class weighting for imbalance)
#     pos_count = sum(r[3] for r in rows[:split])
#     neg_count = len(rows[:split]) - pos_count
#     pos_weight = torch.tensor([neg_count / max(pos_count, 1)]).to(DEVICE)
    
#     model = BoundaryNet().to(DEVICE)
#     optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
#     loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

#     # 3. Loop
#     for ep in range(EPOCHS):
#         model.train()
#         for x, y in train_dl:
#             x, y = x.to(DEVICE), y.to(DEVICE)
#             optimizer.zero_grad()
#             loss = loss_fn(model(x), y)
#             loss.backward()
#             optimizer.step()

#     # 4. Rigorous Evaluation
#     model.eval()
#     all_y, all_probs = [], []
#     with torch.no_grad():
#         for x, y in test_dl:
#             probs = torch.sigmoid(model(x.to(DEVICE)))
#             all_y.extend(y.numpy())
#             all_probs.extend(probs.cpu().numpy())

#     # Threshold Tuning for False Positive reduction
#     # Since FP is more harmful, we can raise the threshold from 0.5
#     best_thresh = 0.7 
#     preds = (np.array(all_probs) > best_thresh).astype(int)

#     print("\n--- BOUNDARY DETECTION REPORT (Threshold: 0.7) ---")
#     print(confusion_matrix(all_y, preds))
#     print(classification_report(all_y, preds))

# if __name__ == "__main__":
#     train_and_eval()
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
import torchaudio.functional as AF
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
import csv, random, os
from sklearn.metrics import confusion_matrix, classification_report

# ===================== Config =====================
SR = 16000
WIN_MS = 27 
WIN_SAMPLES = int(SR * (WIN_MS / 1000))
NUM_WINDOWS = 5
CONTEXT_SAMPLES = WIN_SAMPLES * NUM_WINDOWS
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "akshar_boundary_v1.pt"

# ===================== Model =====================
class BoundaryNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 3 Input Channels: [Mel, Delta, Delta-Delta]
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 4)) 
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 4 * NUM_WINDOWS, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        b, n, c, m, t = x.shape
        x = x.view(b * n, c, m, t) 
        x = self.feature_extractor(x)
        x = x.view(b, -1)
        return self.classifier(x).squeeze()

# ===================== Feature Engine =====================
class FeatureEngine:
    def __init__(self):
        self.mel_spec = T.MelSpectrogram(SR, n_mels=40, n_fft=256, hop_length=64)

    def extract(self, audio_tensor):
        # Create 3-channel feature: Static, Delta, Delta-Delta
        mel = self.mel_spec(audio_tensor)
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)
        
        # Manually compute deltas if window is too short for torchaudio.functional.compute_deltas
        delta = torch.zeros_like(mel)
        delta[:, 1:] = mel[:, 1:] - mel[:, :-1]
        
        delta2 = torch.zeros_like(delta)
        delta2[:, 1:] = delta[:, 1:] - delta[:, :-1]
        
        return torch.stack([mel, delta, delta2]) # (3, Mels, Time)

# ===================== Dataset =====================
class AksharBoundaryDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows
        self.fe = FeatureEngine()

    def __len__(self): return len(self.rows)

    def __getitem__(self, idx):
        wav, start, end, label = self.rows[idx]
        load_start = start - (2 * (WIN_MS / 1000))
        total_dur = (WIN_MS * 5) / 1000
        
        try:
            # Using librosa with explicit error handling
            y, _ = librosa.load(wav, sr=SR, offset=load_start, duration=total_dur)
        except:
            y = np.zeros(CONTEXT_SAMPLES)

        y = np.pad(y, (0, max(0, CONTEXT_SAMPLES - len(y))))[:CONTEXT_SAMPLES]
        windows = np.split(y, NUM_WINDOWS)
        
        feats = torch.stack([self.fe.extract(torch.from_numpy(w).float()) for w in windows])
        return feats, torch.tensor(label, dtype=torch.float32)

# ===================== Training & Saving =====================
def run_full_pipeline():
    rows = []
    with open("jal_dataset.csv") as f:
        for r in csv.DictReader(f):
            rows.append((r["wav"], float(r["start"]), float(r["end"]), int(r["label"])))
    
    random.shuffle(rows)
    split = int(0.8 * len(rows))
    train_dl = DataLoader(AksharBoundaryDataset(rows[:split]), batch_size=32, shuffle=True)
    test_dl = DataLoader(AksharBoundaryDataset(rows[split:]), batch_size=32)

    model = BoundaryNet().to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0]).to(DEVICE)) # Adjusted for precision
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

    print("🚀 Training...")
    for ep in range(30):
        model.train()
        for x, y in train_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad(); loss_fn(model(x), y).backward(); optimizer.step()

    # SAVE EVERYTHING
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'sr': SR,
            'win_ms': WIN_MS,
            'num_windows': NUM_WINDOWS,
            'context_samples': CONTEXT_SAMPLES
        }
    }, MODEL_NAME)
    print(f"✅ Model saved to {MODEL_NAME}")

    # Evaluation
    model.eval()
    y_true, y_prob = [], []
    with torch.no_grad():
        for x, y in test_dl:
            y_prob.extend(torch.sigmoid(model(x.to(DEVICE))).cpu().numpy())
            y_true.extend(y.numpy())
    
    y_pred = (np.array(y_prob) > 0.7).astype(int)
    print("\n" + classification_report(y_true, y_pred))

# ===================== Prediction Function =====================
def predict_akshar(wav_path, start_time, end_time):
    """Call this to predict on a single interval"""
    checkpoint = torch.load(MODEL_NAME, map_location=DEVICE)
    model = BoundaryNet().to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    fe = FeatureEngine()
    load_start = start_time - (2 * (WIN_MS / 1000))
    y, _ = librosa.load(wav_path, sr=SR, offset=load_start, duration=(WIN_MS*5)/1000)
    y = np.pad(y, (0, max(0, CONTEXT_SAMPLES - len(y))))[:CONTEXT_SAMPLES]
    
    windows = np.split(y, NUM_WINDOWS)
    feats = torch.stack([fe.extract(torch.from_numpy(w).float()) for w in windows]).unsqueeze(0)
    
    with torch.no_grad():
        prob = torch.sigmoid(model(feats.to(DEVICE))).item()
    return prob > 0.7, prob

if __name__ == "__main__":
    run_full_pipeline()