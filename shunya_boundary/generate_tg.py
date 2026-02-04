import torch
import numpy as np
import librosa
import os
from gemini_jal_model import BoundaryNet, FeatureEngine # Imports your existing model class

# ===================== Config =====================
SR = 16000
WIN_MS = 27
WIN_SAMPLES = int(SR * (WIN_MS / 1000))
NUM_WINDOWS = 5
CONTEXT_SAMPLES = WIN_SAMPLES * NUM_WINDOWS
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "akshar_boundary_v1.pt"

def write_textgrid(intervals, duration, output_path):
    """Writes a Praat Long TextGrid file."""
    with open(output_path, "w") as f:
        f.write('File type = "ooTextFile"\nObject class = "TextGrid"\n\n')
        f.write(f'xmin = 0\nxmax = {duration}\ntiers? <exists>\nsize = 1\nitem []:\n')
        f.write(f'    item [1]:\n        class = "IntervalTier"\n        name = "akshar_pred"\n')
        f.write(f'        xmin = 0\n        xmax = {duration}\n')
        f.write(f'        intervals: size = {len(intervals)}\n')
        
        for i, (start, end, label) in enumerate(intervals, 1):
            f.write(f'        intervals [{i}]:\n')
            f.write(f'            xmin = {start}\n            xmax = {end}\n')
            f.write(f'            text = "{label}"\n')

def run_inference(wav_path):
    # 1. Load Model & Setup
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model = BoundaryNet().to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    fe = FeatureEngine()

    # 2. Load Audio
    y, _ = librosa.load(wav_path, sr=SR)
    duration = len(y) / SR
    
    # 3. Predict in 27ms Hops
    raw_preds = []
    for i in range(0, len(y) - CONTEXT_SAMPLES, WIN_SAMPLES):
        chunk = y[i : i + CONTEXT_SAMPLES]
        windows = np.split(chunk, NUM_WINDOWS)
        
        # Prepare feature stack (3 channels: Mel, Delta, Delta-Delta)
        feats = torch.stack([fe.extract(torch.from_numpy(w).float()) for w in windows]).unsqueeze(0)
        
        with torch.no_grad():
            prob = torch.sigmoid(model(feats.to(DEVICE))).item()
        
        # We label the middle window (Window 3) of the 5-window context
        t_start = (i + (2 * WIN_SAMPLES)) / SR
        t_end = (i + (3 * WIN_SAMPLES)) / SR
        raw_preds.append((t_start, t_end, prob > 0.7))

    # 4. Merge Logic: Join consecutive 'akshar' predictions
    merged_intervals = []
    if not raw_preds: return
    
    curr_start, curr_end, curr_is_akshar = raw_preds[0]
    
    for next_start, next_end, next_is_akshar in raw_preds[1:]:
        if next_is_akshar == curr_is_akshar:
            curr_end = next_end # Extend the current interval
        else:
            merged_intervals.append((curr_start, curr_end, "akshar" if curr_is_akshar else ""))
            curr_start, curr_end, curr_is_akshar = next_start, next_end, next_is_akshar
            
    merged_intervals.append((curr_start, duration, "akshar" if curr_is_akshar else ""))

    # 5. Output
    tg_path = wav_path.replace(".wav", ".TextGrid")
    write_textgrid(merged_intervals, duration, tg_path)
    print(f"🏁 Created TextGrid: {tg_path}")

if __name__ == "__main__":
    # Example: Run on a specific file
    target_wav = "anjalichunk6.wav"
    if os.path.exists(target_wav):
        run_inference(target_wav)
