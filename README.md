# Codes Ready Pipeline – Flask Service

This repository exposes models as a single Flask service.

Currently enabled endpoints in `app.py`:

- **SL training** (wav + TextGrid) → writes `SL_akshar_pred/models/sl_char.pt`
- **SL inference** (wav) → returns the **grid JSON** format you shared (akash/agni/vayu/jal/prithvi)

Folder structure expected by the service:

codes_ready_pipeline/
│
├── app.py
├── shunya_boundary/
├── shunya_RL/
├── swar_boundary/
├── swar_RL/
├── SL_akshar_pred/
└── venv/   (optional)

--------------------------------------------------
1. Environment setup
--------------------------------------------------

Use Python 3.8 or higher.

Create and activate virtual environment if not already done.

python -m venv venv
source venv/bin/activate

Install dependencies.

pip install flask torch librosa soundfile numpy

--------------------------------------------------
2. Start the server
--------------------------------------------------

From repository root (same folder as app.py):

python app.py

Server runs on:
http://localhost:8000

--------------------------------------------------
3. Health check
--------------------------------------------------

curl http://localhost:8000/health

--------------------------------------------------
4. Debug paths (important)
--------------------------------------------------

This endpoint confirms all model files are detected correctly.

curl http://localhost:8000/debug/paths

All required paths should show exists = true.

--------------------------------------------------
5. SL APIs
--------------------------------------------------

### A) Train SL (wav + TextGrid)

Multipart upload (recommended):

curl -X POST http://localhost:8000/sl/train \
  -F "wav=@SL_akshar_pred/DataTrain/chunk_10.wav" \
  -F "textgrid=@SL_akshar_pred/DataTrain/chunk_10.TextGrid" \
  -F "epochs=6"

OR JSON with paths:

curl -X POST http://localhost:8000/sl/train \
  -H "Content-Type: application/json" \
  -d '{"wav_path":"SL_akshar_pred/DataTrain/chunk_10.wav","textgrid_path":"SL_akshar_pred/DataTrain/chunk_10.TextGrid","epochs":6}'

### B) Infer SL (wav → grid JSON)

Multipart upload:

curl -X POST http://localhost:8000/sl/infer \
  -F "wav=@SL_akshar_pred/DataInfer/chunk_10.wav" \
  -o out.json

OR JSON with path:

curl -X POST http://localhost:8000/sl/infer \
  -H "Content-Type: application/json" \
  -d '{"wav_path":"SL_akshar_pred/DataInfer/chunk_10.wav"}' \
  -o out.json

--------------------------------------------------
6. Notes
--------------------------------------------------

All paths are relative to repository root.

`/sl/train` trains using the **TextGrid tier 'जल'** (27ms). The model predicts at 27ms and expands down to 9ms cells in the JSON (for 'पृथ्वी').

--------------------------------------------------
