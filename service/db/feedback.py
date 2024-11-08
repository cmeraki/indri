import sqlite3
from pathlib import Path
from threading import Lock
from datetime import datetime

class RealFakeFeedbackDB:
    _instance = None
    _lock = Lock()

    DB_FILE = Path("audio_feedback.db")

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(RealFakeFeedbackDB, cls).__new__(cls)
                cls._instance.initialize_db()
        return cls._instance

    def initialize_db(self):
        self.DB_FILE.touch(exist_ok=True)
        self.conn = sqlite3.connect(str(self.DB_FILE))
        self.c = self.conn.cursor()
        self.c.execute("""
            CREATE TABLE IF NOT EXISTS real_fake_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                audio_id TEXT,
                feedback INTEGER,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

    def insert_feedback(self, audio_id: str, feedback: int):
        with self.conn:
            timestamp = datetime.now().isoformat()
            self.c.execute("INSERT INTO real_fake_feedback (audio_id, feedback, timestamp) VALUES (?, ?, ?)", (audio_id, feedback, timestamp))
