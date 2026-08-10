"""
main.py — FastAPI entry point
Run with: uvicorn main:app --reload
Docs at:  http://localhost:8000/docs
"""

from fastapi import FastAPI
from routes.recommendations import router

app = FastAPI(
    title="Academic User Recommendation API",
    description="Recommends researchers, professors, and students based on department, interest tags, research overlap, and activity.",
    version="2.0.0",
)

app.include_router(router, prefix="/api/v1")


@app.get("/health")
def health():
    return {"status": "ok"}