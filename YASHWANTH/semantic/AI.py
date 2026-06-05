# pyrefly: ignore [missing-import]
from flask import Flask, render_template, request, jsonify

# pyrefly: ignore [missing-import]
from sentence_transformers import SentenceTransformer

# pyrefly: ignore [missing-import]
import chromadb

import os
import psycopg2
from pathlib import Path
from dotenv import load_dotenv


app = Flask(__name__)

# Load .env from same folder as AI.py
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

DATABASE_URL = os.getenv("DATABASE_URL")


# -----------------------------
# LOAD EMBEDDING MODEL
# -----------------------------

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# -----------------------------
# CHROMADB SETUP
# -----------------------------

client = chromadb.PersistentClient(path="./curiousbees_supabase_chroma_db")

collection = client.get_or_create_collection(
    name="research_scholar_search"
)


# -----------------------------
# FETCH DATA FROM SUPABASE
# -----------------------------

def fetch_all_items():
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL is missing. Check your .env file.")

    connection = psycopg2.connect(DATABASE_URL)
    cursor = connection.cursor()

    cursor.execute("""
        SELECT
            id,
            item_type,
            block,
            department,
            title,
            author,
            description,
            tags,
            status,
            likes,
            comments
        FROM research_items;
    """)

    rows = cursor.fetchall()

    cursor.close()
    connection.close()

    items = []

    for row in rows:
        items.append({
            "id": row[0],
            "type": row[1],
            "block": row[2],
            "department": row[3],
            "title": row[4],
            "author": row[5],
            "description": row[6],
            "tags": row[7].split(","),
            "status": row[8],
            "likes": row[9],
            "comments": row[10]
        })

    return items


# -----------------------------
# SYNC SUPABASE DATA TO CHROMADB
# -----------------------------

def sync_supabase_to_chromadb():
    items = fetch_all_items()

    for item in items:
        searchable_text = (
            item["type"] + " " +
            item["block"] + " " +
            item["department"] + " " +
            item["title"] + " " +
            item["author"] + " " +
            item["description"] + " " +
            " ".join(item["tags"]) + " " +
            item["status"]
        )

        embedding = model.encode(searchable_text).tolist()

        collection.upsert(
            ids=[item["id"]],
            documents=[searchable_text],
            embeddings=[embedding],
            metadatas=[{
                "type": item["type"],
                "block": item["block"],
                "department": item["department"],
                "title": item["title"],
                "author": item["author"],
                "description": item["description"],
                "tags": ", ".join(item["tags"]),
                "status": item["status"],
                "likes": item["likes"],
                "comments": item["comments"]
            }]
        )

    print("Supabase data synced to ChromaDB successfully.")


# -----------------------------
# EXACT SEARCH USING SUPABASE
# -----------------------------

def exact_match(query):
    query = query.lower().strip()
    like_query = "%" + query + "%"

    connection = psycopg2.connect(DATABASE_URL)
    cursor = connection.cursor()

    cursor.execute("""
        SELECT
            id,
            item_type,
            block,
            department,
            title,
            author,
            description,
            tags,
            status,
            likes,
            comments
        FROM research_items
        WHERE
            LOWER(item_type) = %s
            OR LOWER(block) = %s
            OR LOWER(department) = %s
            OR LOWER(tags) LIKE %s
            OR LOWER(title) LIKE %s;
    """, (query, query, query, like_query, like_query))

    rows = cursor.fetchall()

    cursor.close()
    connection.close()

    results = []

    for row in rows:
        results.append({
            "id": row[0],
            "type": row[1],
            "block": row[2],
            "department": row[3],
            "title": row[4],
            "author": row[5],
            "description": row[6],
            "tags": row[7].split(","),
            "status": row[8],
            "likes": row[9],
            "comments": row[10]
        })

    return results


# -----------------------------
# SEMANTIC SEARCH USING CHROMADB
# -----------------------------

def semantic_search(query, top_k=8):
    query_embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    similar_items = []

    for i in range(len(results["ids"][0])):
        metadata = results["metadatas"][0][i]

        similar_items.append({
            "id": results["ids"][0][i],
            "type": metadata["type"],
            "block": metadata["block"],
            "department": metadata["department"],
            "title": metadata["title"],
            "author": metadata["author"],
            "description": metadata["description"],
            "tags": metadata["tags"].split(", "),
            "status": metadata["status"],
            "likes": metadata["likes"],
            "comments": metadata["comments"]
        })

    return similar_items


# -----------------------------
# MAIN SEARCH FUNCTION
# -----------------------------

def search_research_items(query):
    exact_results = exact_match(query)

    if exact_results:
        return {
            "search_type": "supabase_exact_search",
            "results": exact_results
        }

    semantic_results = semantic_search(query)

    return {
        "search_type": "chromadb_semantic_search",
        "results": semantic_results
    }


# -----------------------------
# HOME PAGE
# -----------------------------

@app.route("/")
def home():
    return render_template("index.html")


# -----------------------------
# SEARCH API
# -----------------------------

@app.route("/search", methods=["POST"])
def search():
    data = request.json
    query = data["query"]

    output = search_research_items(query)

    return jsonify(output)


# -----------------------------
# RUN APP
# -----------------------------

if __name__ == "__main__":
    sync_supabase_to_chromadb()
    app.run(debug=True)