import sqlite3
import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DB_PATH = os.path.join(BASE_DIR, "cybershield.db")


def get_connection():
    return sqlite3.connect(DB_PATH)


def initialize_database():

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions (

        id INTEGER PRIMARY KEY AUTOINCREMENT,

        text TEXT,

        result INTEGER,

        message TEXT,

        confidence REAL,

        severity TEXT,

        category TEXT,

        created_at TEXT

    )
    """)

    conn.commit()
    conn.close()


def save_prediction(text, prediction):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT id
        FROM predictions
        WHERE text = ?
        LIMIT 1
        """,
        (text,)
    )

    if cursor.fetchone():
        conn.close()
        return

    cursor.execute(
        """
        INSERT INTO predictions
        (
            text,
            result,
            message,
            confidence,
            severity,
            category,
            created_at
        )
        VALUES
        (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            text,
            prediction["result"],
            prediction["message"],
            prediction["confidence"],
            prediction["severity"],
            prediction["category"],
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
    )

    conn.commit()
    conn.close()


def get_statistics():

    conn = get_connection()
    cursor = conn.cursor()

    stats = {}

    cursor.execute("SELECT COUNT(*) FROM predictions")
    stats["total"] = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM predictions WHERE result=1")
    stats["cyberbullying"] = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM predictions WHERE result=0")
    stats["safe"] = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM predictions WHERE severity='high'")
    stats["high"] = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM predictions WHERE severity='medium'")
    stats["medium"] = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM predictions WHERE severity='low'")
    stats["low"] = cursor.fetchone()[0]

    cursor.execute("""
    SELECT
        text,
        message,
        confidence,
        severity,
        created_at
    FROM predictions
    ORDER BY id DESC
    LIMIT 20
    """)

    stats["recent"] = cursor.fetchall()

    conn.close()

    return stats


initialize_database()