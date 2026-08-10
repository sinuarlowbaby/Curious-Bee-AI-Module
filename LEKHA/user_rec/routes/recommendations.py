"""
API Routes
───────────
GET  /users/{user_id}/recommendations     → get recommended users
GET  /tags                                → list all interest tags
POST /users/{user_id}/tags                → set user's interest tags (1–5)
POST /users/{user_id}/papers              → add a published paper
GET  /users/{user_id}/papers              → list a user's papers
PUT  /users/{user_id}/activity            → touch activity timestamp
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, field_validator
from db.connection import get_connection
from core.recommender import get_recommendations
from jobs.update_activity_scores import touch_user_activity

router = APIRouter()


# ─── Request / Response Models ────────────────────────────────────────────────

class SetTagsRequest(BaseModel):
    tag_ids: list[int]

    @field_validator("tag_ids")
    @classmethod
    def validate_tags(cls, v):
        if not (1 <= len(v) <= 5):
            raise ValueError("Select between 1 and 5 tags.")
        if len(v) != len(set(v)):
            raise ValueError("Duplicate tag IDs are not allowed.")
        return v


class AddPaperRequest(BaseModel):
    title:     str
    abstract:  str | None = None
    keywords:  list[str]           # extracted keywords/tags from the paper
    published_at: str | None = None  # YYYY-MM-DD


class RecommendationResponse(BaseModel):
    user_id:         str
    mode:            str
    recommendations: list[dict]


# ─── Recommendations ──────────────────────────────────────────────────────────

@router.get(
    "/users/{user_id}/recommendations",
    response_model=RecommendationResponse,
    summary="Get recommended users",
)
def recommend_users(
    user_id: str,
    limit:   int = Query(default=10, ge=1,   le=50),
    mode:    str = Query(default="both",     description="similar | collaborative | both"),
    conn       = Depends(get_connection),
):
    if mode not in ("similar", "collaborative", "both"):
        raise HTTPException(status_code=400, detail="mode must be 'similar', 'collaborative', or 'both'")

    cur = conn.cursor()
    cur.execute("SELECT id FROM users WHERE id = %s", (user_id,))
    if not cur.fetchone():
        cur.close()
        raise HTTPException(status_code=404, detail="User not found")
    cur.close()

    results = get_recommendations(user_id, conn, limit=limit, mode=mode)
    return RecommendationResponse(user_id=user_id, mode=mode, recommendations=results)


# ─── Tags ─────────────────────────────────────────────────────────────────────

@router.get("/tags", summary="List all interest tags grouped by category")
def list_tags(conn=Depends(get_connection)):
    cur = conn.cursor()
    cur.execute("SELECT id, name, category FROM tags ORDER BY category, name")
    rows = cur.fetchall()
    cur.close()

    grouped: dict[str, list] = {}
    for tag_id, name, category in rows:
        grouped.setdefault(category, []).append({"id": tag_id, "name": name})

    return {"tags": grouped}


@router.post("/users/{user_id}/tags", summary="Set user interest tags (1–5)")
def set_user_tags(
    user_id: str,
    body:    SetTagsRequest,
    conn   = Depends(get_connection),
):
    cur = conn.cursor()

    cur.execute("SELECT id FROM users WHERE id = %s", (user_id,))
    if not cur.fetchone():
        cur.close()
        raise HTTPException(status_code=404, detail="User not found")

    cur.execute("SELECT id FROM tags WHERE id = ANY(%s)", (body.tag_ids,))
    found   = {r[0] for r in cur.fetchall()}
    missing = set(body.tag_ids) - found
    if missing:
        cur.close()
        raise HTTPException(status_code=400, detail=f"Unknown tag IDs: {missing}")

    cur.execute("DELETE FROM user_tags WHERE user_id = %s", (user_id,))
    cur.executemany(
        "INSERT INTO user_tags (user_id, tag_id) VALUES (%s, %s)",
        [(user_id, tid) for tid in body.tag_ids],
    )
    conn.commit()
    cur.close()

    return {"user_id": user_id, "tag_ids": body.tag_ids, "message": "Tags updated."}


# ─── Papers ───────────────────────────────────────────────────────────────────

@router.post("/users/{user_id}/papers", summary="Add a published paper")
def add_paper(
    user_id: str,
    body:    AddPaperRequest,
    conn   = Depends(get_connection),
):
    cur = conn.cursor()

    cur.execute("SELECT id FROM users WHERE id = %s", (user_id,))
    if not cur.fetchone():
        cur.close()
        raise HTTPException(status_code=404, detail="User not found")

    if not body.keywords:
        cur.close()
        raise HTTPException(status_code=400, detail="At least one keyword is required.")

    # Insert paper
    cur.execute("""
        INSERT INTO papers (user_id, title, abstract, published_at)
        VALUES (%s, %s, %s, %s)
        RETURNING id
    """, (user_id, body.title, body.abstract, body.published_at))
    paper_id = cur.fetchone()[0]

    # Insert keywords (lowercase + strip for consistent matching)
    cur.executemany(
        "INSERT INTO paper_keywords (paper_id, keyword) VALUES (%s, %s) ON CONFLICT DO NOTHING",
        [(paper_id, kw.lower().strip()) for kw in body.keywords if kw.strip()]
    )

    conn.commit()
    cur.close()

    return {
        "paper_id": paper_id,
        "user_id":  user_id,
        "title":    body.title,
        "keywords": body.keywords,
        "message":  "Paper added successfully."
    }


@router.get("/users/{user_id}/papers", summary="List a user's published papers")
def list_papers(user_id: str, conn=Depends(get_connection)):
    cur = conn.cursor()

    cur.execute("SELECT id FROM users WHERE id = %s", (user_id,))
    if not cur.fetchone():
        cur.close()
        raise HTTPException(status_code=404, detail="User not found")

    cur.execute("""
        SELECT p.id, p.title, p.abstract, p.published_at,
               ARRAY_AGG(pk.keyword) AS keywords
        FROM papers p
        LEFT JOIN paper_keywords pk ON pk.paper_id = p.id
        WHERE p.user_id = %s
        GROUP BY p.id
        ORDER BY p.published_at DESC NULLS LAST
    """, (user_id,))

    rows = cur.fetchall()
    cur.close()

    papers = [
        {
            "id":           r[0],
            "title":        r[1],
            "abstract":     r[2],
            "published_at": str(r[3]) if r[3] else None,
            "keywords":     r[4] if r[4] != [None] else [],
        }
        for r in rows
    ]

    return {"user_id": user_id, "papers": papers}


# ─── Activity ─────────────────────────────────────────────────────────────────

@router.put("/users/{user_id}/activity", summary="Touch user activity")
def update_activity(user_id: str, conn=Depends(get_connection)):
    cur = conn.cursor()
    cur.execute("SELECT id FROM users WHERE id = %s", (user_id,))
    if not cur.fetchone():
        cur.close()
        raise HTTPException(status_code=404, detail="User not found")
    cur.close()

    touch_user_activity(user_id, conn)
    conn.commit()
    return {"user_id": user_id, "message": "Activity updated."}