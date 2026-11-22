from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
DB_PATH = BASE_DIR / "moodify_history.db"
DATA_PATH = DATA_DIR / "deam_song_level_features.csv"
MODEL_PATH = MODEL_DIR / "moodify_models.joblib"

DATA_URL = "https://github.com/asher0913/deam-song-level-features/releases/download/1/deam_song_level_features.csv"
TARGETS = ["valence_mean", "arousal_mean"]
