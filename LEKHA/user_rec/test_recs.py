"""
test_recommendations.py
────────────────────────
Self-contained test — no PostgreSQL, no setup needed.
Uses SQLite in memory so you can run this immediately.

Run with:
    python test_recommendations.py
"""

import sqlite3
import uuid
import random
from dataclasses import dataclass

# ─── Weights ──────────────────────────────────────────────────────────────────
W_DEPARTMENT = 0.20
W_TAGS       = 0.35
W_RESEARCH   = 0.30
W_ACTIVITY   = 0.15


# ─── Data Model ───────────────────────────────────────────────────────────────
@dataclass
class UserProfile:
    user_id:           str
    username:          str
    department:        str
    role:              str
    tag_ids:           set
    research_keywords: set
    activity_score:    float


# ─── Scoring Functions ────────────────────────────────────────────────────────
def department_score(a: UserProfile, b: UserProfile) -> float:
    if not a.department or not b.department:
        return 0.0
    return 1.0 if a.department.lower() == b.department.lower() else 0.0


def tag_similarity_score(a: UserProfile, b: UserProfile) -> float:
    if not a.tag_ids or not b.tag_ids:
        return 0.0
    return len(a.tag_ids & b.tag_ids) / len(a.tag_ids | b.tag_ids)


def research_overlap_score(a: UserProfile, b: UserProfile) -> float:
    if not a.research_keywords or not b.research_keywords:
        return 0.0
    return len(a.research_keywords & b.research_keywords) / len(a.research_keywords | b.research_keywords)


def compute_score(source: UserProfile, candidate: UserProfile) -> dict:
    dept     = department_score(source, candidate)
    tags     = tag_similarity_score(source, candidate)
    research = research_overlap_score(source, candidate)
    activity = min(max(candidate.activity_score, 0.0), 1.0)
    total    = (W_DEPARTMENT * dept) + (W_TAGS * tags) + (W_RESEARCH * research) + (W_ACTIVITY * activity)
    return {
        "total":      round(total,    4),
        "department": round(dept,     4),
        "tags":       round(tags,     4),
        "research":   round(research, 4),
        "activity":   round(activity, 4),
    }


# ─── Setup SQLite ─────────────────────────────────────────────────────────────
def setup_db():
    conn = sqlite3.connect(":memory:")
    cur  = conn.cursor()

    cur.executescript("""
        CREATE TABLE users (
            id         TEXT PRIMARY KEY,
            username   TEXT UNIQUE NOT NULL,
            email      TEXT UNIQUE NOT NULL,
            full_name  TEXT,
            department TEXT,
            role       TEXT
        );

        CREATE TABLE tags (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            name     TEXT UNIQUE NOT NULL,
            category TEXT NOT NULL
        );

        CREATE TABLE user_tags (
            user_id TEXT,
            tag_id  INTEGER,
            PRIMARY KEY (user_id, tag_id)
        );

        CREATE TABLE papers (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id      TEXT,
            title        TEXT NOT NULL,
            published_at TEXT
        );

        CREATE TABLE paper_keywords (
            paper_id INTEGER,
            keyword  TEXT NOT NULL,
            PRIMARY KEY (paper_id, keyword)
        );

        CREATE TABLE user_activity (
            user_id        TEXT PRIMARY KEY,
            activity_score REAL DEFAULT 0.5
        );
    """)
    conn.commit()
    return conn


# ─── Dummy Data ───────────────────────────────────────────────────────────────
TAGS = [
    # AI
    (1,  "Machine Learning",     "AI"),
    (2,  "Deep Learning",        "AI"),
    (3,  "NLP",                  "AI"),
    (4,  "Computer Vision",      "AI"),
    (5,  "Reinforcement Learning","AI"),
    # Data Science
    (6,  "Statistical Modeling", "Data Science"),
    (7,  "Data Mining",          "Data Science"),
    (8,  "Bayesian Methods",     "Data Science"),
    # Biology
    (9,  "Genomics",             "Biology"),
    (10, "CRISPR",               "Biology"),
    (11, "Bioinformatics",       "Biology"),
    (12, "Neuroscience",         "Biology"),
    # Medicine
    (13, "Clinical Trials",      "Medicine"),
    (14, "Drug Discovery",       "Medicine"),
    (15, "Medical Imaging",      "Medicine"),
    # Physics
    (16, "Quantum Mechanics",    "Physics"),
    (17, "Astrophysics",         "Physics"),
    (18, "Particle Physics",     "Physics"),
    # Mathematics
    (19, "Graph Theory",         "Mathematics"),
    (20, "Optimization",         "Mathematics"),
    (21, "Cryptography",         "Mathematics"),
    # Engineering
    (22, "Robotics",             "Engineering"),
    (23, "Signal Processing",    "Engineering"),
    (24, "Nanotechnology",       "Engineering"),
    # Environment
    (25, "Climate Change",       "Environment"),
    (26, "Renewable Energy",     "Environment"),
    (27, "Ecology",              "Environment"),
]

# Each user: (username, full_name, department, role, tag_ids, paper_keywords, activity_score)
USERS = [
    (
        "alice.chen",       "Alice Chen",
        "Computer Science", "professor",
        {1, 2, 3, 4, 5},   # heavy AI focus
        {"neural networks", "transformer", "bert", "nlp", "deep learning", "attention mechanism"},
        0.95,
    ),
    (
        "bob.kumar",        "Bob Kumar",
        "Computer Science", "researcher",
        {1, 2, 4, 6, 7},   # AI + data science
        {"convolutional networks", "image classification", "deep learning", "data mining", "feature extraction"},
        0.80,
    ),
    (
        "carol.smith",      "Carol Smith",
        "Computer Science", "student",
        {3, 5, 8, 19, 20},  # NLP + math
        {"nlp", "reinforcement learning", "optimization", "graph neural networks", "text classification"},
        0.60,
    ),
    (
        "david.lee",        "David Lee",
        "Biology",          "professor",
        {9, 10, 11, 12, 6}, # genomics + bioinformatics
        {"crispr", "gene editing", "genomics", "dna sequencing", "rna splicing", "protein folding"},
        0.88,
    ),
    (
        "emma.jones",       "Emma Jones",
        "Biology",          "researcher",
        {9, 11, 12, 13, 14},# biology + medicine
        {"bioinformatics", "genomics", "drug discovery", "biomarkers", "clinical data"},
        0.72,
    ),
    (
        "frank.patel",      "Frank Patel",
        "Medicine",         "professor",
        {13, 14, 15, 6, 8}, # medicine + data
        {"clinical trials", "drug discovery", "medical imaging", "mri", "patient outcomes", "biomarkers"},
        0.65,
    ),
    (
        "grace.wu",         "Grace Wu",
        "Medicine",         "student",
        {13, 15, 1, 3, 6},  # medicine + AI (medical AI)
        {"medical imaging", "deep learning", "diagnosis", "neural networks", "clinical data"},
        0.45,
    ),
    (
        "henry.brown",      "Henry Brown",
        "Physics",          "professor",
        {16, 17, 18, 20, 21},# physics + math
        {"quantum entanglement", "particle physics", "dark matter", "astrophysics", "gravitational waves"},
        0.90,
    ),
    (
        "iris.tanaka",      "Iris Tanaka",
        "Physics",          "researcher",
        {16, 17, 19, 20, 6}, # physics + data
        {"quantum mechanics", "condensed matter", "superconductivity", "statistical modeling", "optimization"},
        0.55,
    ),
    (
        "james.garcia",     "James Garcia",
        "Mathematics",      "professor",
        {19, 20, 21, 8, 6},  # pure math
        {"graph theory", "cryptography", "optimization", "number theory", "topology"},
        0.70,
    ),
    (
        "kate.nguyen",      "Kate Nguyen",
        "Mathematics",      "student",
        {19, 20, 1, 6, 7},   # math + AI
        {"optimization", "machine learning", "graph theory", "bayesian methods", "statistical learning"},
        0.40,
    ),
    (
        "liam.roberts",     "Liam Roberts",
        "Engineering",      "researcher",
        {22, 23, 24, 1, 4},  # engineering + AI
        {"robotics", "signal processing", "autonomous systems", "computer vision", "sensor fusion"},
        0.78,
    ),
    (
        "maya.wilson",      "Maya Wilson",
        "Engineering",      "professor",
        {22, 23, 5, 20, 6},  # robotics + RL
        {"robotics", "control systems", "reinforcement learning", "optimization", "embedded systems"},
        0.83,
    ),
    (
        "noah.martinez",    "Noah Martinez",
        "Environment",      "researcher",
        {25, 26, 27, 6, 8},  # environment + data
        {"climate change", "carbon emissions", "renewable energy", "ecology", "sustainability"},
        0.50,
    ),
    (
        "olivia.taylor",    "Olivia Taylor",
        "Environment",      "professor",
        {25, 27, 6, 7, 8},   # environment + data science
        {"climate modeling", "biodiversity", "ecology", "statistical modeling", "environmental policy"},
        0.67,
    ),
    (
        "peter.anderson",   "Peter Anderson",
        "Computer Science", "researcher",
        {1, 3, 5, 20, 19},   # CS + math
        {"machine learning", "nlp", "optimization", "graph neural networks", "transformer"},
        0.91,
    ),
    (
        "quinn.thomas",     "Quinn Thomas",
        "Biology",          "student",
        {9, 10, 12, 1, 6},   # bio + AI (bioinformatics AI)
        {"genomics", "neural networks", "protein structure", "machine learning", "bioinformatics"},
        0.35,
    ),
    (
        "rachel.white",     "Rachel White",
        "Medicine",         "researcher",
        {14, 15, 1, 2, 6},   # medicine + deep learning
        {"drug discovery", "medical imaging", "deep learning", "cnn", "diagnosis", "clinical trials"},
        0.76,
    ),
    (
        "sam.harris",       "Sam Harris",
        "Physics",          "student",
        {16, 18, 19, 20, 21},# physics + math
        {"quantum computing", "particle physics", "optimization", "cryptography", "linear algebra"},
        0.30,
    ),
    (
        "tara.jackson",     "Tara Jackson",
        "Computer Science", "professor",
        {1, 2, 22, 23, 4},   # CS + robotics/CV
        {"deep learning", "computer vision", "robotics", "object detection", "autonomous driving"},
        0.88,
    ),
]


# ─── Insert Dummy Data ────────────────────────────────────────────────────────
def insert_dummy_data(conn):
    cur = conn.cursor()

    # Tags
    cur.executemany("INSERT OR IGNORE INTO tags (id, name, category) VALUES (?,?,?)", TAGS)

    # Users
    user_id_map = {}   # username → uuid
    for (username, full_name, department, role, tag_ids, keywords, activity) in USERS:
        uid = str(uuid.uuid4())
        user_id_map[username] = uid

        cur.execute("""
            INSERT INTO users (id, username, email, full_name, department, role)
            VALUES (?,?,?,?,?,?)
        """, (uid, username, f"{username}@university.edu", full_name, department, role))

        # Tags
        for tid in tag_ids:
            cur.execute("INSERT OR IGNORE INTO user_tags (user_id, tag_id) VALUES (?,?)", (uid, tid))

        # Paper + keywords (one paper per user for simplicity)
        cur.execute("""
            INSERT INTO papers (user_id, title, published_at)
            VALUES (?, ?, '2024-01-01')
        """, (uid, f"Research in {department} by {full_name}"))
        paper_id = cur.lastrowid

        for kw in keywords:
            cur.execute("INSERT OR IGNORE INTO paper_keywords (paper_id, keyword) VALUES (?,?)",
                        (paper_id, kw.lower().strip()))

        # Activity
        cur.execute("INSERT INTO user_activity (user_id, activity_score) VALUES (?,?)", (uid, activity))

    conn.commit()
    return user_id_map


# ─── Fetch Profile from SQLite ────────────────────────────────────────────────
def fetch_profile(conn, user_id: str) -> UserProfile:
    cur = conn.cursor()

    cur.execute("SELECT username, department, role FROM users WHERE id=?", (user_id,))
    row = cur.fetchone()
    username, department, role = row

    cur.execute("""
        SELECT t.id FROM user_tags ut JOIN tags t ON t.id = ut.tag_id WHERE ut.user_id=?
    """, (user_id,))
    tag_ids = {r[0] for r in cur.fetchall()}

    cur.execute("""
        SELECT DISTINCT pk.keyword
        FROM papers p JOIN paper_keywords pk ON pk.paper_id = p.id
        WHERE p.user_id=?
    """, (user_id,))
    research_keywords = {r[0] for r in cur.fetchall()}

    cur.execute("SELECT activity_score FROM user_activity WHERE user_id=?", (user_id,))
    row = cur.fetchone()
    activity_score = row[0] if row else 0.5

    return UserProfile(
        user_id=user_id,
        username=username,
        department=department,
        role=role,
        tag_ids=tag_ids,
        research_keywords=research_keywords,
        activity_score=activity_score,
    )


# ─── Recommend ────────────────────────────────────────────────────────────────
def recommend(conn, source_id: str, mode: str = "both", limit: int = 5) -> list[dict]:
    cur = conn.cursor()
    cur.execute("SELECT id FROM users WHERE id != ?", (source_id,))
    candidate_ids = [r[0] for r in cur.fetchall()]

    source = fetch_profile(conn, source_id)
    scored = []

    for cid in candidate_ids:
        candidate = fetch_profile(conn, cid)
        scores    = compute_score(source, candidate)

        if mode == "similar":
            final = (W_TAGS * scores["tags"]) + (W_RESEARCH * scores["research"])
        elif mode == "collaborative":
            final = scores["department"]
        else:
            final = scores["total"]

        if final < 0.05:
            continue

        scored.append({
            "username":   candidate.username,
            "department": candidate.department,
            "role":       candidate.role,
            "score":      round(final, 4),
            "breakdown":  scores,
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:limit]


# ─── Pretty Print ─────────────────────────────────────────────────────────────
def print_recommendations(source: UserProfile, results: list[dict], mode: str):
    print(f"\n{'═'*65}")
    print(f"  Recommendations for: {source.username}")
    print(f"  Department : {source.department}")
    print(f"  Role       : {source.role}")
    print(f"  Mode       : {mode}")
    print(f"{'═'*65}")

    if not results:
        print("  No recommendations found.")
        return

    for i, r in enumerate(results, 1):
        print(f"\n  #{i}  {r['username']:<20} [{r['role']:<12} | {r['department']}]")
        print(f"       Score      : {r['score']}")
        b = r["breakdown"]
        print(f"       Department : {b['department']}   Tags: {b['tags']}   Research: {b['research']}   Activity: {b['activity']}")

    print()


# ─── Run Tests ────────────────────────────────────────────────────────────────
def run_tests(conn, user_id_map: dict):
    test_cases = [
        # (username,        mode,            what we expect to see)
        ("alice.chen",     "both",          "Other CS professors and researchers with AI/NLP overlap"),
        ("alice.chen",     "similar",       "People with matching AI research keywords"),
        ("alice.chen",     "collaborative", "Everyone in Computer Science dept"),
        ("david.lee",      "both",          "Biology/Medicine researchers with genomics overlap"),
        ("frank.patel",    "both",          "Medicine + anyone doing medical AI (grace, rachel)"),
        ("henry.brown",    "both",          "Physics people, maybe math overlap"),
    ]

    for username, mode, note in test_cases:
        uid     = user_id_map[username]
        source  = fetch_profile(conn, uid)
        results = recommend(conn, uid, mode=mode, limit=5)
        print(f"\n  ── Expected: {note}")
        print_recommendations(source, results, mode)
        input("  Press Enter for next test...")


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n Setting up in-memory SQLite database...")
    conn = setup_db()

    print(" Inserting 20 dummy academic users...")
    user_id_map = insert_dummy_data(conn)
    print(f" Done. {len(user_id_map)} users created.\n")

    print("─" * 65)
    print("  Available test users:")
    print("─" * 65)
    for uname, uid in user_id_map.items():
        cur = conn.cursor()
        cur.execute("SELECT department, role FROM users WHERE id=?", (uid,))
        dept, role = cur.fetchone()
        print(f"  {uname:<22} [{role:<12} | {dept}]")

    print("\n Starting recommendation tests...\n")
    run_tests(conn, user_id_map)

    conn.close()
    print("Done.")