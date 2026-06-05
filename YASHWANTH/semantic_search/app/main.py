from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
import logging
import os
import sqlite3
from pathlib import Path
from fastapi.templating import Jinja2Templates
from starlette.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore


load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / ".env")

QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
QDRANT_COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION_NAME")

ALLOWED_ORIGINS = ["http://localhost:8000", "http://localhost:3000"]

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app:FastAPI):
    try:
        app.state.client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        
        existing_collections = [c.name for c in app.state.client.get_collections().collections]
        if QDRANT_COLLECTION_NAME not in existing_collections:
            app.state.client.create_collection(
                collection_name=QDRANT_COLLECTION_NAME,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE),
            )
            logger.info(f"Created Qdrant collection '{QDRANT_COLLECTION_NAME}'")
    #     app.state.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

        app.state.embedding_model = HuggingFaceEmbeddings(
            model_name="nomic-ai/nomic-embed-text-v1.5",  # 768-dim, 547MB
            model_kwargs={"trust_remote_code": True},
            encode_kwargs={"normalize_embeddings": True}   # required for cosine similarity
        )
        # app.state.embedding_model = HuggingFaceEmbeddings(
        #     model_name="all-MiniLM-L6-v2",    # 384-dim, 50 MB
        #     model_kwargs={"device": "cpu"},
        #     encode_kwargs={"normalize_embeddings": True}
        # )



        # Initialize once at startup — not per-request
        app.state.vector_store = QdrantVectorStore(
            client=app.state.client,
            embedding=app.state.embedding_model,
            collection_name=QDRANT_COLLECTION_NAME,
        )

    #     # Parse REDIS_URL into host/port for redis-py
    #     _redis_host = REDIS_URL.replace("redis://", "").split(":")[0]
    #     _redis_port = int(REDIS_URL.replace("redis://", "").split(":")[1]) if ":" in REDIS_URL.replace("redis://", "") else 6379
    #     app.state.redis = redis.Redis(
    #         host=_redis_host,
    #         port=_redis_port,
    #         db=0,
    #         decode_responses=True,
    #     )
        app.state.sessions = {}

        # ── SQLite Setup ──────────────────────────────────────────
        # Store db path on app.state so routes can open connections
        _base = os.path.dirname(os.path.abspath(__file__))
        app.state.db_path = os.path.join(_base, "curious_bees.db")

        # Create the posts table if it doesn't exist yet
        with sqlite3.connect(app.state.db_path) as con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS posts (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    title     TEXT    NOT NULL,
                    author    TEXT    NOT NULL DEFAULT 'Anonymous',
                    tag       TEXT    NOT NULL DEFAULT 'Research',
                    abstract  TEXT,
                    date      TEXT,
                    ts        INTEGER NOT NULL  -- epoch milliseconds (from client)
                )
            """)

            # ── FTS5 virtual table for keyword search (hybrid search) ──────────
            # Porter stemmer: "searching" matches "search", "searches" etc.
            # Standalone table (not content=posts) for simple insert/delete sync.
            con.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS posts_fts
                USING fts5(title, tag, abstract, tokenize='porter ascii')
            """)

            # Sync any existing posts not yet in the FTS index (idempotent) ────
            already_fts = {
                r[0] for r in con.execute("SELECT rowid FROM posts_fts").fetchall()
            }
            to_sync = con.execute(
                "SELECT id, title, tag, abstract FROM posts"
            ).fetchall()
            new_fts_rows = [
                (r[0], r[1] or "", r[2] or "", r[3] or "")
                for r in to_sync
                if r[0] not in already_fts
            ]
            if new_fts_rows:
                con.executemany(
                    "INSERT INTO posts_fts(rowid, title, tag, abstract) VALUES (?, ?, ?, ?)",
                    new_fts_rows,
                )
                logger.info(f"FTS5: synced {len(new_fts_rows)} existing post(s) into keyword index")

            con.commit()
        logger.info(f"SQLite + FTS5 ready → {app.state.db_path}")

        logger.info("Server is ready!")
    except Exception as e:
        logger.error(f"Error initializing Server: {e}")
        raise e

    logger.info("FastAPI server is ready!")
    logger.info("Swagger UI  ->  http://localhost:8000/docs")
    logger.info("Home Page   ->  http://localhost:8000")

    yield  # App handles requests here

    logger.info("Shutting down server...")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
jinja2_env = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

from routes.api import app_router
app.include_router(app_router)

@app.get("/")
async def root(request: Request):
    return jinja2_env.TemplateResponse(request=request, name="research_platform.html", )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_config=None, log_level="info")
