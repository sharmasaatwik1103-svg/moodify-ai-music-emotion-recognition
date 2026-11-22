import hashlib
import requests
import pandas as pd
from config import DATA_DIR, DATA_PATH, DATA_URL

EXPECTED_SHA256 = "f09044497e415fe22b999051cdee59d479db2486f73f896603ae4f7fe73c0ff6"

def download_dataset():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    response = requests.get(DATA_URL, timeout=120)
    response.raise_for_status()
    DATA_PATH.write_bytes(response.content)
    digest = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256:
        DATA_PATH.unlink(missing_ok=True)
        raise ValueError("Dataset checksum did not match. Download was removed.")
    return DATA_PATH

def load_dataset():
    if not DATA_PATH.exists():
        download_dataset()
    df = pd.read_csv(DATA_PATH)
    required = {"song_id", "valence_mean", "arousal_mean"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    return df
