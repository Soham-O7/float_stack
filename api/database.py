import sqlite3
from pathlib import Path

DB_PATH = Path("data/jobs.db")

def get_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                status TEXT,
                health_report TEXT,
                chosen_method TEXT,
                proof_table TEXT,
                error TEXT
            )
        """)
        conn.commit()

init_db()