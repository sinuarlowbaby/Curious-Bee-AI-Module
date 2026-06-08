import os
import psycopg2
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    print("DATABASE_URL not found.")
    print("Make sure .env is in the same folder as db_test.py")
    exit()

print("DATABASE_URL loaded successfully.")

connection = psycopg2.connect(DATABASE_URL)
cursor = connection.cursor()

cursor.execute("SELECT COUNT(*) FROM research_items;")
count = cursor.fetchone()[0]

print("Database connected successfully.")
print("Total research items:", count)

cursor.close()
connection.close()