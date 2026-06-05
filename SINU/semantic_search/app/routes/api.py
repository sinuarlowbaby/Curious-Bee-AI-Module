from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from langchain_core.documents import Document
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchText, OptimizersConfigDiff
import re
import time
import logging
import os
import sqlite3
from langsmith import traceable

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


# ── Helpers ───────────────────────────────────────────────────────────
def get_db(request: Request) -> str:
    """Return the SQLite db path stored on app.state."""
    return request.app.state.db_path


def _fts_query(query: str) -> str:
    """
    Sanitise a user query into a safe FTS5 MATCH expression.
    Each word becomes a separate OR term so partial matches still score.
    e.g. 'ml in health care' → 'ml OR in OR health OR care'
    """
    tokens = re.sub(r'[^\w\s]', ' ', query).split()
    return " OR ".join(tokens) if tokens else ""


def _rrf_merge(
    semantic_hits: list[tuple[int, float]],
    fts_hits:      list[int],
    k: int = 60,
    top_n: int = 20,
) -> list[tuple[int, float]]:
    """
    Reciprocal Rank Fusion.
    semantic_hits : [(sqlite_id, score), ...] ordered best-first
    fts_hits      : [sqlite_id, ...]          ordered best-first (SQLite rank ASC)
    Returns top_n (sqlite_id, rrf_score) pairs, best-first.
    """
    scores: dict[int, float] = {}
    for rank, (sid, _) in enumerate(semantic_hits, 1):
        scores[sid] = scores.get(sid, 0.0) + 1.0 / (k + rank)
    for rank, sid in enumerate(fts_hits, 1):
        scores[sid] = scores.get(sid, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]


def _wait_for_optimizer(client, collection_name: str, timeout: float = 5.0, interval: float = 0.1):
    """
    Temporarily lower indexing_threshold to 0 so Qdrant kicks off the optimizer
    immediately, then poll until it finishes (status == 'ok').
    Restores the original threshold afterwards.
    Raises TimeoutError if the optimizer doesn't finish within `timeout` seconds.
    """
    # Trigger optimizer by setting threshold to 0
    client.update_collection(
        collection_name=collection_name,
        optimizer_config=OptimizersConfigDiff(indexing_threshold=0),
    )

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        info = client.get_collection(collection_name)
        if info.optimizer_status.status == "ok":
            break
        time.sleep(interval)
    else:
        logger.warning(
            f"Qdrant optimizer did not finish within {timeout}s — "
            "new vector may not be immediately searchable."
        )

    # Restore default threshold (20000) so bulk inserts stay fast
    client.update_collection(
        collection_name=collection_name,
        optimizer_config=OptimizersConfigDiff(indexing_threshold=20000),
    )


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

            # Also insert into FTS5 keyword index ──────────────────────────
            con.execute(
                "INSERT INTO posts_fts(rowid, title, tag, abstract) VALUES (?, ?, ?, ?)",
                (sqlite_id, payload.title, payload.tag, payload.abstract),
            )
            con.commit()
        logger.info(f"SQLite inserted post id={sqlite_id} title='{payload.title}'")

        # 2. Index abstract + metadata in Qdrant ──────────────────────
        # Option 2: Weighted repetition — title repeated twice gives it ~2x
        # semantic weight over the abstract; tags surfaced once for keyword proximity.
        vector_store = request.app.state.vector_store
        enriched_content = (
            f"{payload.title}. {payload.title}.\n"
            f"Tags: {payload.tag}.\n"
            f"Abstract: {payload.abstract}"
        )
        doc = Document(
            page_content=enriched_content,
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

        # Force optimizer so the new vector is immediately searchable ────────
        client = request.app.state.client
        _wait_for_optimizer(client, QDRANT_COLLECTION_NAME)

        return {
            "message": "Post saved to SQLite and indexed in Qdrant",
            "id": sqlite_id,
        }

    except Exception as e:
        logger.error(f"Error in /update_db: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── GET /search ────────────────────────────────────────────────
@traceable(name="search_vectors", run_type="tool")
@app_router.get("/search")
async def search_vectors(query: str, request: Request, tag: str = None):
    """
    Hybrid search (semantic + keyword) over posts, merged with RRF.
    - query : natural-language search string (required)
    - tag   : optional tag filter applied to both semantic and keyword results.
    """
    try:
        vector_store = request.app.state.vector_store
        db_path      = get_db(request)

        # ── 1. Semantic search (Qdrant) ───────────────────────────────────────
        qdrant_filter = None
        if tag:
            qdrant_filter = Filter(
                must=[
                    FieldCondition(
                        key="metadata.tag",
                        match=MatchText(text=tag),
                    )
                ]
            )
            logger.info(f"Applying tag filter (MatchText): '{tag}'")

        # Fetch 50 candidates so RRF has enough diversity before trimming to 20
        semantic_docs = vector_store.similarity_search_with_score(
            query, k=50, filter=qdrant_filter
        )
        semantic_hits: list[tuple[int, float]] = [
            (int(doc.metadata["sqlite_id"]), float(score))
            for doc, score in semantic_docs
            if doc.metadata.get("sqlite_id") is not None
        ]
        logger.info(f"Semantic hits: {len(semantic_hits)}")

        # ── 2. Keyword search (SQLite FTS5) ──────────────────────────────────
        fts_hits: list[int] = []
        fts_q = _fts_query(query)
        if fts_q:
            with sqlite3.connect(db_path) as fts_con:
                fts_con.row_factory = sqlite3.Row
                try:
                    if tag:
                        fts_rows = fts_con.execute(
                            """
                            SELECT f.rowid
                            FROM   posts_fts f
                            JOIN   posts     p ON p.id = f.rowid
                            WHERE  posts_fts MATCH ?
                            AND    p.tag LIKE '%' || ? || '%'
                            ORDER  BY rank
                            LIMIT  50
                            """,
                            (fts_q, tag),
                        ).fetchall()
                    else:
                        fts_rows = fts_con.execute(
                            """
                            SELECT rowid
                            FROM   posts_fts
                            WHERE  posts_fts MATCH ?
                            ORDER  BY rank
                            LIMIT  50
                            """,
                            (fts_q,),
                        ).fetchall()
                    fts_hits = [row["rowid"] for row in fts_rows]
                    logger.info(f"FTS5 hits: {len(fts_hits)} (query='{fts_q}')")
                except Exception as fts_err:
                    logger.warning(f"FTS5 search failed, semantic only: {fts_err}")

        # ── 3. RRF merge ──────────────────────────────────────────────────────
        merged = _rrf_merge(semantic_hits, fts_hits, k=60, top_n=20)
        logger.info(f"RRF merged result count: {len(merged)}")

        # ── 4. Fetch full rows from SQLite in RRF order ───────────────────────
        semantic_score_map = {sid: sc for sid, sc in semantic_hits}
        results = []
        with sqlite3.connect(db_path) as con:
            con.row_factory = sqlite3.Row
            for sqlite_id, rrf_score in merged:
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
                        "score":    round(semantic_score_map.get(sqlite_id, rrf_score), 6),
                    })

        # Sort by score descending so highest-scoring results appear first ────
        results.sort(key=lambda x: x["score"], reverse=True)

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

            # Remove from FTS5 keyword index as well ────────────────────────
            con.execute("DELETE FROM posts_fts WHERE rowid = ?", (post_id,))
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

