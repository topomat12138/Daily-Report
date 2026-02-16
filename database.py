import sqlite3
from datetime import datetime, timezone

DB_NAME = "arxiv.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS papers (
        arxiv_id TEXT PRIMARY KEY,
        title TEXT,
        summary TEXT,
        published_utc TEXT,
        created_at TEXT
    )
    """)

    # Backward-compatible migration for existing DB files created without summary.
    cur.execute("PRAGMA table_info(papers)")
    columns = [row[1] for row in cur.fetchall()]
    if "summary" not in columns:
        cur.execute("ALTER TABLE papers ADD COLUMN summary TEXT")

    conn.commit()
    conn.close()

def insert_if_new(arxiv_id, title, summary, published):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()

    cur.execute("SELECT arxiv_id FROM papers WHERE arxiv_id = ?", (arxiv_id,))
    exists = cur.fetchone()

    if exists:
        conn.close()
        return False

    cur.execute("""
        INSERT INTO papers (arxiv_id, title, summary, published_utc, created_at)
        VALUES (?, ?, ?, ?, ?)
    """, (
        arxiv_id,
        title,
        summary,
        published,
        datetime.now(timezone.utc).isoformat()
    ))

    conn.commit()
    conn.close()
    return True

def prune_old_papers(cutoff_iso):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()

    cur.execute("DELETE FROM papers WHERE published_utc < ?", (cutoff_iso,))
    deleted_rows = cur.rowcount

    conn.commit()
    conn.close()
    return deleted_rows
