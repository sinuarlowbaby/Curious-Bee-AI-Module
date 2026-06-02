from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from langchain_core.documents import Document
from qdrant_client.models import Filter, FieldCondition, MatchValue
import logging
import os
import sqlite3

# Define logger for api.py
logger = logging.getLogger(__name__)

QDRANT_COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION_NAME", "RECOLAB")

app_router = APIRouter()


class VectorInput(BaseModel):
    title: str
    author: str
    tag: str
    abstract: str
    date: str
    ts: int   # epoch milliseconds sent from the browser


# ── Helper ───────────────────────────────────────────────────────────
def get_db(request: Request) -> str:
    """Return the SQLite db path stored on app.state."""
    return request.app.state.db_path


# ── POST /update_db ─────────────────────────────────────────
@app_router.post("/update_db")
async def update_db(payload: VectorInput, request: Request):
    try:
        # 1. Insert metadata into SQLite ─────────────────────────────
        db_path = get_db(request)
        with sqlite3.connect(db_path) as con:
            cur = con.execute(
                """
                INSERT INTO posts (title, author, tag, abstract, date, ts)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    payload.title,
                    payload.author,
                    payload.tag,
                    payload.abstract,
                    payload.date,
                    payload.ts,
                ),
            )
            sqlite_id = cur.lastrowid
            con.commit()
        logger.info(f"SQLite inserted post id={sqlite_id} title='{payload.title}'")

        # 2. Index abstract + metadata in Qdrant ──────────────────────
        vector_store = request.app.state.vector_store
        doc = Document(
            page_content=payload.abstract,
            metadata={
                "sqlite_id": sqlite_id,   # link back to SQLite row
                "title":     payload.title,
                "author":    payload.author,
                "tag":       payload.tag,
                "date":      payload.date,
            },
        )
        vector_store.add_documents([doc])
        logger.info(f"Qdrant indexed sqlite_id={sqlite_id}")

        return {
            "message": "Post saved to SQLite and indexed in Qdrant",
            "id": sqlite_id,
        }

    except Exception as e:
        logger.error(f"Error in /update_db: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── GET /search ────────────────────────────────────────────────
@app_router.get("/search")
async def search_vectors(query: str, request: Request):
    try:
        vector_store = request.app.state.vector_store
        db_path      = get_db(request)

        # Qdrant semantic search ──────────────────────────────────
        docs_with_scores = vector_store.similarity_search_with_score(query, k=5)

        results = []
        with sqlite3.connect(db_path) as con:
            con.row_factory = sqlite3.Row
            for doc, score in docs_with_scores:
                sqlite_id = doc.metadata.get("sqlite_id")

                # Fetch the authoritative row from SQLite if we have a link
                row = None
                if sqlite_id:
                    row = con.execute(
                        "SELECT * FROM posts WHERE id = ?", (sqlite_id,)
                    ).fetchone()

                if row:
                    results.append({
                        "id":       row["id"],
                        "title":    row["title"],
                        "author":   row["author"],
                        "tag":      row["tag"],
                        "abstract": row["abstract"],
                        "date":     row["date"],
                        "ts":       row["ts"],
                        "score":    float(score),
                    })
                else:
                    # Fallback: use whatever Qdrant metadata has
                    results.append({
                        "id":       None,
                        "title":    doc.metadata.get("title", "Untitled"),
                        "author":   doc.metadata.get("author", "Anonymous"),
                        "tag":      doc.metadata.get("tag", "General"),
                        "abstract": doc.page_content,
                        "date":     doc.metadata.get("date", ""),
                        "ts":       0,
                        "score":    float(score),
                    })

        return results

    except Exception as e:
        logger.error(f"Error in /search: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── GET /posts ────────────────────────────────────────────────
@app_router.get("/posts")
async def get_posts(request: Request):
    try:
        db_path = get_db(request)

        # Read directly from SQLite — fast, no Qdrant dependency
        with sqlite3.connect(db_path) as con:
            con.row_factory = sqlite3.Row
            rows = con.execute(
                "SELECT * FROM posts ORDER BY ts DESC LIMIT 100"
            ).fetchall()

        return [
            {
                "id":       row["id"],
                "title":    row["title"],
                "author":   row["author"],
                "tag":      row["tag"],
                "abstract": row["abstract"],
                "date":     row["date"],
                "ts":       row["ts"],
            }
            for row in rows
        ]

    except Exception as e:
        logger.error(f"Error in /posts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── DELETE /posts/{post_id} ───────────────────────────────────────────────────
@app_router.delete("/posts/{post_id}")
async def delete_post(post_id: int, request: Request):
    """
    Delete a post from both SQLite and Qdrant atomically.
    - SQLite: DELETE WHERE id = post_id
    - Qdrant: delete all points WHERE metadata.sqlite_id = post_id
    """
    try:
        db_path = get_db(request)

        # 1. Verify the post exists in SQLite and delete it ───────────────────
        with sqlite3.connect(db_path) as con:
            row = con.execute(
                "SELECT id, title FROM posts WHERE id = ?", (post_id,)
            ).fetchone()

            if row is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Post with id={post_id} not found in SQLite"
                )

            con.execute("DELETE FROM posts WHERE id = ?", (post_id,))
            con.commit()

        logger.info(f"SQLite deleted post id={post_id} title='{row[1]}'")

        # 2. Delete matching vectors from Qdrant by payload filter ─────────────
        # We stored sqlite_id inside metadata, so Qdrant key is "metadata.sqlite_id"
        client = request.app.state.client
        client.delete(
            collection_name=QDRANT_COLLECTION_NAME,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="metadata.sqlite_id",
                        match=MatchValue(value=post_id),
                    )
                ]
            ),
        )
        logger.info(f"Qdrant deleted vectors for sqlite_id={post_id}")

        return {
            "message": f"Post id={post_id} deleted from SQLite and Qdrant",
            "id": post_id,
        }

    except HTTPException:
        raise  # re-raise 404 as-is
    except Exception as e:
        logger.error(f"Error in DELETE /posts/{post_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
