"""
Academic Recommendation Engine
────────────────────────────────
Pure algorithmic scoring — no ML models.

Signals:
  1. Department Match     weight = 0.20  → same institutional department
  2. Interest Tag Overlap weight = 0.35  → Jaccard on user-selected interest tags
  3. Research Overlap     weight = 0.30  → Jaccard on keywords across all published papers
  4. Activity Score       weight = 0.15  → exponential decay based on last active timestamp

Modes:
  "similar"      → people with the same interests and research area
  "collaborative" → people in the same department (cross-interest discovery)
  "both"         → full weighted composite (default)
"""

from __future__ import annotations
from dataclasses import dataclass, field

# ─── Weights (must sum to 1.0) ────────────────────────────────────────────────
W_DEPARTMENT = 0.20
W_TAGS       = 0.35
W_RESEARCH   = 0.30
W_ACTIVITY   = 0.15


# ─── Data Model ───────────────────────────────────────────────────────────────

@dataclass
class UserProfile:
    user_id:           str
    username:          str        # ← add this
    department:        str | None
    role:              str        # ← add this
    tag_ids:           set[int]
    tag_categories:    set[str]
    research_keywords: set[str]
    activity_score:    float = 0.5


# ─── DB Fetch ─────────────────────────────────────────────────────────────────

def fetch_user_profile(cur, user_id: str) -> UserProfile:
    """Fetch everything needed to score a user."""

    # Department
    cur.execute("SELECT username, department, role FROM users WHERE id = %s", (user_id,))
    row = cur.fetchone()
    username   = row[0] if row else ""
    department = row[1] if row and row[1] else None
    role       = row[2] if row and row[2] else ""

    # Interest tags
    cur.execute("""
        SELECT t.id, t.category
        FROM user_tags ut
        JOIN tags t ON t.id = ut.tag_id
        WHERE ut.user_id = %s
    """, (user_id,))
    rows = cur.fetchall()
    tag_ids        = {r[0] for r in rows}
    tag_categories = {r[1] for r in rows}

    # Research keywords — union across ALL papers the user has published
    cur.execute("""
        SELECT DISTINCT pk.keyword
        FROM papers p
        JOIN paper_keywords pk ON pk.paper_id = p.id
        WHERE p.user_id = %s
    """, (user_id,))
    research_keywords = {r[0].lower().strip() for r in cur.fetchall()}

    # Activity score
    cur.execute("SELECT activity_score FROM user_activity WHERE user_id = %s", (user_id,))
    row = cur.fetchone()
    activity_score = float(row[0]) if row else 0.5

    return UserProfile(
        user_id=user_id,
        username=username,        # ← add
        department=department,
        role=role,                # ← add
        tag_ids=tag_ids,
        tag_categories=tag_categories,
        research_keywords=research_keywords,
        activity_score=activity_score,
)


def fetch_candidate_ids(cur, source_user_id: str) -> list[str]:
    """All users except the source."""
    cur.execute("SELECT id FROM users WHERE id != %s", (source_user_id,))
    return [r[0] for r in cur.fetchall()]


# ─── Individual Scoring Functions ─────────────────────────────────────────────

def department_score(a: UserProfile, b: UserProfile) -> float:
    """
    1.0 if both users are in the same department.
    0.0 if either has no department or they differ.
    Binary signal — you're either in the same dept or you're not.
    """
    if not a.department or not b.department:
        return 0.0
    return 1.0 if a.department.lower() == b.department.lower() else 0.0


def tag_similarity_score(a: UserProfile, b: UserProfile) -> float:
    """
    Jaccard similarity on interest tag IDs.
    Measures how much their self-reported interests overlap.

    Formula: |A ∩ B| / |A ∪ B|
    """
    if not a.tag_ids or not b.tag_ids:
        return 0.0
    intersection = len(a.tag_ids & b.tag_ids)
    union        = len(a.tag_ids | b.tag_ids)
    return intersection / union


def research_overlap_score(a: UserProfile, b: UserProfile) -> float:
    """
    Jaccard similarity on research keywords across all published papers.
    Measures how much their actual research output overlaps.

    Keywords are lowercased and stripped for consistent matching.
    Users with no papers score 0.0 on this signal.

    Formula: |A ∩ B| / |A ∪ B|
    """
    if not a.research_keywords or not b.research_keywords:
        return 0.0
    intersection = len(a.research_keywords & b.research_keywords)
    union        = len(a.research_keywords | b.research_keywords)
    return intersection / union


def activity_score_normalized(b: UserProfile) -> float:
    """Clamp activity score to [0.0, 1.0]."""
    return min(max(b.activity_score, 0.0), 1.0)


# ─── Composite Score ──────────────────────────────────────────────────────────

def compute_score(source: UserProfile, candidate: UserProfile) -> dict:
    """
    Compute full weighted score and return all signal breakdowns.
    """
    if source.user_id == candidate.user_id:
        return {"total": 0.0, "department": 0.0, "tags": 0.0, "research": 0.0, "activity": 0.0}

    dept     = department_score(source, candidate)
    tags     = tag_similarity_score(source, candidate)
    research = research_overlap_score(source, candidate)
    activity = activity_score_normalized(candidate)

    total = (
        W_DEPARTMENT * dept     +
        W_TAGS       * tags     +
        W_RESEARCH   * research +
        W_ACTIVITY   * activity
    )

    return {
        "total":      round(total,    4),
        "department": round(dept,     4),
        "tags":       round(tags,     4),
        "research":   round(research, 4),
        "activity":   round(activity, 4),
    }


# ─── Main Recommendation Function ─────────────────────────────────────────────

def get_recommendations(
    user_id:   str,
    conn,
    limit:     int   = 10,
    mode:      str   = "both",   # "similar" | "collaborative" | "both"
    min_score: float = 0.05,
) -> list[dict]:
    """
    Returns a ranked list of recommended users with scores and breakdowns.

    Modes:
      "similar"       → ranked purely by tag + research overlap (who shares your interests)
      "collaborative" → ranked purely by department match (who's in your institution)
      "both"          → full weighted composite (default, recommended)

    Returns:
        [
          {
            "user_id": "...",
            "score": 0.72,
            "breakdown": {
              "department": 1.0,
              "tags": 0.6,
              "research": 0.4,
              "activity": 0.9
            }
          },
          ...
        ]
    """
    cur = conn.cursor()

    source        = fetch_user_profile(cur, user_id)
    candidate_ids = fetch_candidate_ids(cur, user_id)

    scored = []

    for cid in candidate_ids:
        candidate = fetch_user_profile(cur, cid)
        scores    = compute_score(source, candidate)

        if mode == "similar":
            final = (W_TAGS * scores["tags"]) + (W_RESEARCH * scores["research"])
        elif mode == "collaborative":
            final = scores["department"]
        else:
            final = scores["total"]

        if final < min_score:
            continue

        scored.append({
            "user_id":    cid,
            "username":   candidate.username,
            "department": candidate.department,
            "role":       candidate.role,
            "score":      round(final, 4),
            "breakdown":  {
                "department": scores["department"],
                "tags":       scores["tags"],
                "research":   scores["research"],
                "activity":   scores["activity"],
    }
})

    cur.close()
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:limit]