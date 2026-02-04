
import os
from pathlib import Path

# Project root = folder containing this file
PROJECT_ROOT = str(Path(__file__).resolve().parent)

DATA_DIR = os.path.join(PROJECT_ROOT, "DataTrain")
TEST_DIR = os.path.join(PROJECT_ROOT, "DataInfer")
OUT_DIR = os.path.join(PROJECT_ROOT, "Out")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

SAMPLE_RATE = 16000

GRID_MS = 216
TIERS = [
    ("akash", "आकाश", 216, 1),
    ("agni",  "अग्नि", 108, 2),
    ("vayu",  "वायु",  54,  4),
    ("jal",   "जल",    27,  8),
    ("prithvi","पृथ्वी", 9, 24),
]

CLASSES_PATH = os.path.join(MODEL_DIR, "classes.json")
