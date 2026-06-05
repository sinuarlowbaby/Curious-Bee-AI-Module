"""
sync_sqlite_to_qdrant.py
--------------------------------------------------------------------------------
One-shot script: reads every row from curious_bees.db and indexes it
into the Qdrant vector store, using the same embedding model, metadata
structure, and enriched page_content format as the FastAPI app (/update_db).

page_content format (Option 2 — weighted repetition):
    "<title>. <title>.\nTags: <tag>.\nAbstract: <abstract>"

Title is repeated twice to give it ~2x semantic weight during embedding.
Tags are surfaced once for keyword proximity.

Run from inside  SINU/semantic search/app/  so that the .env is loaded
and the relative DB path resolves correctly:

    cd "SINU/semantic search/app"
    python sync_sqlite_to_qdrant.py

Options:
    --dry-run   Print what would be uploaded without touching Qdrant.
    --batch N   Upload N documents at a time (default: 32).
--------------------------------------------------------------------------------

To re-index existing posts (so old embeddings benefit from the new format), run:

    python sync_sqlite_to_qdrant.py --force-recreate

This drops the old collection and re-uploads everything with the enriched content.

to run 

python sync_sqlite_to_qdrant.py --dry-run        # Preview all rows — no upload
python sync_sqlite_to_qdrant.py                   # Full sync
python sync_sqlite_to_qdrant.py --batch 10       # Upload 10 at a time  
"""

import argparse
import os
import sqlite3
import sys
from langsmith import traceable

from dotenv import load_dotenv

load_dotenv()  # picks up QDRANT_URL, QDRANT_COLLECTION_NAME from .env

# ── Config ─────────────────────
@traceable(run_type="chain", name="sync_sqlite_to_qdrant")
def main():
    # ── Config ───────────────────────────────────────────────────────────────────

    BASE_DIR               = os.path.dirname(os.path.abspath(__file__))
    DB_PATH                = os.path.join(BASE_DIR, "curious_bees.db")
    QDRANT_URL             = os.environ.get("QDRANT_URL", "http://localhost:6333")
    QDRANT_COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION_NAME", "RECOLAB")
    EMBEDDING_MODEL_NAME   = "nomic-ai/nomic-embed-text-v1.5"   # same model as the FastAPI app
    VECTOR_SIZE            = 768                   # output dimension of the model

    # ── CLI args ──────────────────────────────────────────────────────────────────

    parser = argparse.ArgumentParser(description="Sync SQLite posts -> Qdrant")
    parser.add_argument("--dry-run",       action="store_true", help="Print rows only, no upload")
    parser.add_argument("--batch",         type=int, default=32, metavar="N", help="Batch size (default 32)")
    parser.add_argument("--force-recreate", action="store_true",
                        help="Delete the existing Qdrant collection and re-upload everything (use when changing embedding model)")
    args = parser.parse_args()

    # ── Validate DB exists ────────────────────────────────────────────────────────

    if not os.path.exists(DB_PATH):
        sys.exit(f"[!] Database not found: {DB_PATH}")

    # ── Load rows from SQLite ─────────────────────────────────────────────────────

    print(f"[DB]  Reading from  : {DB_PATH}")
    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute("SELECT * FROM posts ORDER BY id ASC").fetchall()

    if not rows:
        sys.exit("[i]  No rows found in the posts table. Nothing to sync.")

    print(f"[i]  Found {len(rows)} post(s) in SQLite\n")

    if args.dry_run:
        print("--- DRY RUN - no data will be written to Qdrant ---")
        for row in rows:
            print(f"  id={row['id']:>4}  title={row['title']!r}  author={row['author']!r}  tag={row['tag']!r}")
        print("\nDry run complete. Re-run without --dry-run to upload.")
        sys.exit(0)

    # ── Load embedding model ──────────────────────────────────────────────────────

    print("[*]  Loading embedding model ...")
    from langchain_huggingface import HuggingFaceEmbeddings

    # nomic-embed models require trust_remote_code=True
    # all-MiniLM models work fine without it
    _needs_trust = "nomic" in EMBEDDING_MODEL_NAME
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"trust_remote_code": True} if _needs_trust else {"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    print(f"[OK] Model ready: {EMBEDDING_MODEL_NAME} ({VECTOR_SIZE}-dim)\n")

    # ── Connect to Qdrant ─────────────────────────────────────────────────────────

    print(f"[*]  Connecting to Qdrant at {QDRANT_URL} ...")
    from qdrant_client import QdrantClient
    from qdrant_client.models import VectorParams, Distance
    from langchain_qdrant import QdrantVectorStore
    from langchain_core.documents import Document

    client = QdrantClient(url=QDRANT_URL)

    # Handle collection creation / recreation
    existing = [c.name for c in client.get_collections().collections]

    if args.force_recreate and QDRANT_COLLECTION_NAME in existing:
        print(f"  [!] --force-recreate: deleting collection '{QDRANT_COLLECTION_NAME}' ...")
        client.delete_collection(QDRANT_COLLECTION_NAME)
        existing = []  # treat as not existing
        print(f"  [!] Collection deleted.")

    if QDRANT_COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        print(f"  [+] Created Qdrant collection '{QDRANT_COLLECTION_NAME}' ({VECTOR_SIZE}-dim)")
    else:
        print(f"  [=] Using existing Qdrant collection '{QDRANT_COLLECTION_NAME}'")

    vector_store = QdrantVectorStore(
        client=client,
        embedding=embedding_model,
        collection_name=QDRANT_COLLECTION_NAME,
    )
    print("[OK] Qdrant ready\n")

    # ── Check which IDs are already indexed ──────────────────────────────────────
    # Scroll through Qdrant payloads and collect sqlite_ids already indexed,
    # so we can skip them on re-runs (idempotent sync).

    print("[*]  Checking for already-indexed posts ...")
    already_indexed: set = set()
    offset = None
    while True:
        response, offset = client.scroll(
            collection_name=QDRANT_COLLECTION_NAME,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in response:
            sid = (point.payload or {}).get("metadata", {}).get("sqlite_id")
            if sid is not None:
                already_indexed.add(int(sid))
        if offset is None:
            break

    new_rows = [r for r in rows if r["id"] not in already_indexed]
    print(f"  Already indexed : {len(already_indexed)} post(s)")
    print(f"  To upload       : {len(new_rows)} post(s)\n")

    if not new_rows:
        print("[OK] Everything is already in sync. Nothing to do.")
        sys.exit(0)

    # ── Upload in batches ─────────────────────────────────────────────────────────

    def chunked(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    total    = len(new_rows)
    uploaded = 0

    print(f"[^]  Uploading {total} post(s) in batches of {args.batch} ...\n")

    for batch in chunked(new_rows, args.batch):
        docs = []
        for row in batch:
            title    = row["title"]    or ""
            tag      = row["tag"]      or ""
            abstract = row["abstract"] or ""

            # Option 2: Weighted repetition — mirrors /update_db in api.py.
            # Title repeated twice → ~2x semantic weight; tags surfaced once.
            enriched_content = (
                f"{title}. {title}.\n"
                f"Tags: {tag}.\n"
                f"Abstract: {abstract}"
            )

            docs.append(
                Document(
                    page_content=enriched_content,
                    metadata={
                        "sqlite_id": row["id"],       # same field /update_db uses
                        "title":     title,
                        "author":    row["author"],
                        "tag":       tag,
                        "date":      row["date"] or "",
                    },
                )
            )

        vector_store.add_documents(docs)
        uploaded += len(batch)
        print(f"  [OK] {uploaded}/{total} uploaded")

    print(f"\n[DONE] {uploaded} post(s) synced to Qdrant collection '{QDRANT_COLLECTION_NAME}'.")


if __name__ == "__main__":
    main()
