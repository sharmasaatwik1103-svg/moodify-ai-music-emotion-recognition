import sqlite3
from config import DB_PATH

def init_db():
    with sqlite3.connect(DB_PATH) as con:
        con.execute('''CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            filename TEXT,
            valence REAL,
            arousal REAL,
            mood TEXT,
            model_name TEXT
        )''')

def save_analysis(filename, valence, arousal, mood, model_name):
    init_db()
    with sqlite3.connect(DB_PATH) as con:
        con.execute(
            "INSERT INTO analyses(filename,valence,arousal,mood,model_name) VALUES(?,?,?,?,?)",
            (filename, float(valence), float(arousal), mood, model_name)
        )

def get_history():
    init_db()
    with sqlite3.connect(DB_PATH) as con:
        return con.execute(
            "SELECT created_at, filename, valence, arousal, mood, model_name FROM analyses ORDER BY id DESC"
        ).fetchall()
