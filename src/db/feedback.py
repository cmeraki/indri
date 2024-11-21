import time
import sqlite3
from pathlib import Path
from threading import Lock
from datetime import datetime
from queue import Queue
from contextlib import contextmanager

from ..logger import get_logger

logger = get_logger(__name__)

class RealFakeFeedbackDB:
    _instance = None
    _db_lock = Lock()

    DB_FILE = Path("audio_feedback.db")
    MAX_CONNECTIONS = 5

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(RealFakeFeedbackDB, cls).__new__(cls)
            cls._instance.initialize_db()
        return cls._instance

    def initialize_db(self):
        """Initialize the database and connection pool"""
        self.DB_FILE.touch(exist_ok=True)
        self._connection_pool = Queue(maxsize=self.MAX_CONNECTIONS)

        for _ in range(self.MAX_CONNECTIONS):
            conn = sqlite3.connect(str(self.DB_FILE))
            self._connection_pool.put(conn)

        with self.get_connection() as conn:
            c = conn.cursor()
            c.execute("""
                CREATE TABLE IF NOT EXISTS real_fake_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    audio_id TEXT,
                    feedback INTEGER,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    @contextmanager
    def get_connection(self):
        """Get a connection from the pool"""
        connection = self._connection_pool.get(timeout=5)
        try:
            yield connection
        finally:
            self._connection_pool.put(connection)

    def insert_feedback(self, audio_id: str, feedback: int):
        start_time = time.time()
        with self._db_lock:
            with self.get_connection() as conn:
                timestamp = datetime.now().isoformat()
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO real_fake_feedback (audio_id, feedback, timestamp) VALUES (?, ?, ?)",
                    (audio_id, feedback, timestamp)
                )
                conn.commit()

        end_time = time.time()
        logger.info(f'Inserted feedback for {audio_id} in {end_time - start_time:.4f} seconds')

    def __del__(self):
        """Cleanup connections when the instance is destroyed"""
        while not self._connection_pool.empty():
            conn = self._connection_pool.get_nowait()
            conn.close()
