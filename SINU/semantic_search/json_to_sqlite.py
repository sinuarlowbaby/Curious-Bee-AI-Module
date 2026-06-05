import json
import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).parent
JSON_FILE = BASE_DIR / "research_dataset_1000_unique.json"
DB_FILE = BASE_DIR / "curious_bees.db"

with open(JSON_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

conn = sqlite3.connect(DB_FILE)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS posts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    author TEXT NOT NULL,
    tag TEXT NOT NULL,
    abstract TEXT NOT NULL,
    date TEXT NOT NULL,
    ts INTEGER NOT NULL
)
""")

cursor.execute("DELETE FROM posts")

for row in data:
    cursor.execute("""
    INSERT INTO posts (
        title,
        author,
        tag,
        abstract,
        date,
        ts
    )
    VALUES (?, ?, ?, ?, ?, ?)
    """, (
        row["title"],
        row["author"],
        row["tag"],
        row["abstract"],
        row["date"],
        row["ts"]
    ))

conn.commit()
conn.close()

print("Database created successfully")