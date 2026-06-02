from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from langchain_core.documents import Document
import logging
import os

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


@app_router.post("/add_vector")
async def add_vector(payload: VectorInput, request: Request):
    try:
        # 1. Access the vector store and embedding model from FastAPI app state
        vector_store = request.app.state.vector_store
        embedding_model = request.app.state.embedding_model
        
        # 2. Embed the text directly on the server to get the vector representation
        vector = embedding_model.embed_query(payload.abstract)
        
        # 3. Wrap abstract as page_content and add rich metadata for Qdrant
        doc = Document(
            page_content=payload.abstract,
            metadata={
                "title": payload.title,
                "author": payload.author,
                "tag": payload.tag,
                "date": payload.date
            }
        )
        
        # 4. Add the document to Qdrant (which internally indexes it)
        vector_store.add_documents([doc])
        
        # 5. Return success message along with the calculated embedding vector
        return {
            "message": "Vector added successfully",
            "embedding": vector  # Returns the 384-dimension float array
        }
    except Exception as e:
        logger.error(f"Error adding vector: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app_router.get("/search")
async def search_vectors(query: str, request: Request):
    try:
        vector_store = request.app.state.vector_store
        
        # Perform similarity search
        docs_with_scores = vector_store.similarity_search_with_score(query, k=5)
        
        # Format results for the HTML frontend
        results = []
        for doc, score in docs_with_scores:
            results.append({
                "title": doc.metadata.get("title", "Untitled"),
                "author": doc.metadata.get("author", "Anonymous"),
                "tag": doc.metadata.get("tag", "General"),
                "abstract": doc.page_content,
                "date": doc.metadata.get("date", ""),
                "score": float(score)  # Cosine similarity score
            })
        return results
    except Exception as e:
        logger.error(f"Error searching vectors: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app_router.get("/posts")
async def get_posts(request: Request):
    try:
        client = request.app.state.client
        
        # Scroll up to 100 points from Qdrant
        response, _ = client.scroll(
            collection_name=QDRANT_COLLECTION_NAME,
            limit=100,
            with_payload=True,
            with_vectors=False
        )
        
        results = []
        for point in response:
            payload = point.payload or {}
            page_content = payload.get("page_content", "")
            metadata = payload.get("metadata", {})
            
            # Use timestamp or point ID as numerical id and ts representation
            results.append({
                "id": point.id,
                "title": metadata.get("title", "Untitled"),
                "author": metadata.get("author", "Anonymous"),
                "tag": metadata.get("tag", "General"),
                "abstract": page_content,
                "date": metadata.get("date", ""),
                "ts": int(point.id) if isinstance(point.id, int) else 0
            })
        return results
    except Exception as e:
        logger.error(f"Error fetching posts: {e}")
        raise HTTPException(status_code=500, detail=str(e))