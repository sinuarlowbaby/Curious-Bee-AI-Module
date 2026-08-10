"""
seed_users.py — Generate academic dummy users with tags, papers, and keywords.
Run with: python seed_users.py
          python seed_users.py 30    ← custom count
"""

from __future__ import annotations
import os
import uuid
import random
import psycopg2
from dotenv import load_dotenv

load_dotenv()

conn = psycopg2.connect(
    host=os.getenv("DB_HOST", "localhost"),
    port=os.getenv("DB_PORT", 5432),
    dbname=os.getenv("DB_NAME", "yourdb"),
    user=os.getenv("DB_USER", "postgres"),
    password=os.getenv("DB_PASSWORD", ""),
)
cur = conn.cursor()

# ─── Fetch real tag IDs ───────────────────────────────────────────────────────
cur.execute("SELECT id, name, category FROM tags ORDER BY category")
all_tags = cur.fetchall()

tags_by_category: dict[str, list[int]] = {}
tag_name_by_id:   dict[int, str]       = {}

for tag_id, name, category in all_tags:
    tags_by_category.setdefault(category, []).append(tag_id)
    tag_name_by_id[tag_id] = name

all_tag_ids = list(tag_name_by_id.keys())
print(f"Loaded {len(all_tag_ids)} tags across {len(tags_by_category)} categories.\n")

# ─── Academic Personas ────────────────────────────────────────────────────────
personas = [
    {
        "role":       "professor",
        "department": "Computer Science",
        "tag_categories": ["AI", "Computer Science", "Data Science"],
        "paper_keywords_pool": [
            "neural networks", "deep learning", "transformer", "bert", "gpt",
            "image classification", "object detection", "nlp", "text generation",
            "machine learning", "reinforcement learning", "graph neural networks",
        ],
    },
    {
        "role":       "researcher",
        "department": "Biology",
        "tag_categories": ["Biology", "Medicine", "Data Science"],
        "paper_keywords_pool": [
            "genomics", "crispr", "gene editing", "dna sequencing", "rna",
            "protein folding", "bioinformatics", "cell biology", "mutations",
            "phylogenetics", "evolutionary biology", "synthetic biology",
        ],
    },
    {
        "role":       "student",
        "department": "Mathematics",
        "tag_categories": ["Mathematics", "Computer Science", "AI"],
        "paper_keywords_pool": [
            "optimization", "convex analysis", "graph theory", "number theory",
            "cryptography", "topology", "linear algebra", "probabilistic models",
            "differential equations", "stochastic processes",
        ],
    },
    {
        "role":       "professor",
        "department": "Medicine",
        "tag_categories": ["Medicine", "Biology", "Data Science"],
        "paper_keywords_pool": [
            "clinical trials", "drug discovery", "epidemiology", "medical imaging",
            "precision medicine", "patient outcomes", "biomarkers", "mri",
            "neural correlates", "mental health", "public health",
        ],
    },
    {
        "role":       "researcher",
        "department": "Physics",
        "tag_categories": ["Physics", "Mathematics", "Engineering"],
        "paper_keywords_pool": [
            "quantum mechanics", "particle physics", "condensed matter",
            "astrophysics", "dark matter", "gravitational waves",
            "thermodynamics", "quantum entanglement", "superconductivity",
        ],
    },
    {
        "role":       "researcher",
        "department": "Social Sciences",
        "tag_categories": ["Social Sciences", "Data Science", "Medicine"],
        "paper_keywords_pool": [
            "behavioral economics", "cognitive bias", "decision making",
            "social networks", "political polarization", "survey methods",
            "linguistics", "sentiment analysis", "cultural dynamics",
        ],
    },
    {
        "role":       "student",
        "department": "Engineering",
        "tag_categories": ["Engineering", "AI", "Computer Science"],
        "paper_keywords_pool": [
            "robotics", "signal processing", "control systems", "nanotechnology",
            "materials science", "biomedical devices", "sensor fusion",
            "autonomous systems", "embedded systems",
        ],
    },
    {
        "role":       "professor",
        "department": "Environment",
        "tag_categories": ["Environment", "Data Science", "Social Sciences"],
        "paper_keywords_pool": [
            "climate change", "carbon emissions", "renewable energy",
            "ecology", "biodiversity", "environmental policy",
            "sustainability", "ocean acidification", "deforestation",
        ],
    },
]

# ─── Paper title templates ────────────────────────────────────────────────────
PAPER_TEMPLATES = [
    "A Novel Approach to {kw1} Using {kw2}",
    "Investigating {kw1} in the Context of {kw2}",
    "{kw1} and {kw2}: A Comparative Study",
    "Advances in {kw1}: Implications for {kw2}",
    "Toward Better {kw1}: A {kw2}-Based Framework",
    "The Role of {kw1} in Modern {kw2} Research",
    "Exploring {kw1} with {kw2} Methods",
]

def generate_paper(persona: dict) -> dict:
    pool = persona["paper_keywords_pool"]
    keywords = random.sample(pool, min(random.randint(3, 6), len(pool)))
    k1, k2 = keywords[0].title(), keywords[1].title()
    title = random.choice(PAPER_TEMPLATES).format(kw1=k1, kw2=k2)
    return {
        "title":    title,
        "abstract": f"This paper explores {k1} and its relationship to {k2} in the domain of {persona['department']}.",
        "keywords": keywords,
        "published_at": f"{random.randint(2018, 2024)}-{random.randint(1,12):02d}-01",
    }


# ─── Tag picking ──────────────────────────────────────────────────────────────
def pick_tags(persona: dict, total: int = 5) -> list[int]:
    chosen = []
    for cat in persona["tag_categories"]:
        pool = tags_by_category.get(cat, [])
        if pool:
            chosen.append(random.choice(pool))
    chosen = list(set(chosen))[:4]
    remaining = total - len(chosen)
    if remaining > 0:
        extras = random.sample(
            [t for t in all_tag_ids if t not in chosen],
            min(remaining, len(all_tag_ids) - len(chosen))
        )
        chosen.extend(extras)
    return chosen[:total]


# ─── Names ────────────────────────────────────────────────────────────────────
FIRST_NAMES = [
    "Alex", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Avery", "Quinn",
    "Skyler", "Drew", "Blake", "Cameron", "Dakota", "Emerson", "Finley",
    "Harley", "Jamie", "Kendall", "Logan", "Robin", "Sasha", "Dana",
    "Reese", "Peyton", "Hayden", "Rowan", "Sage", "Phoenix", "River", "Elliot",
]
LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Wilson", "Moore", "Taylor", "Anderson", "Thomas", "Jackson",
    "White", "Harris", "Martin", "Thompson", "Young", "Lee", "Walker",
    "Hall", "Allen", "King", "Wright", "Scott", "Green", "Adams", "Baker",
]


# ─── Seed ─────────────────────────────────────────────────────────────────────
def seed(count: int = 20):
    print(f"Generating {count} academic dummy users...\n")

    used_names      = set()
    inserted_users  = 0
    inserted_papers = 0

    for i in range(count):
        persona = personas[i % len(personas)]

        while True:
            first = random.choice(FIRST_NAMES)
            last  = random.choice(LAST_NAMES)
            if (first, last) not in used_names:
                used_names.add((first, last))
                break

        user_id  = str(uuid.uuid4())
        username = f"{first.lower()}.{last.lower()}{random.randint(1, 99)}"
        email    = f"{username}@university.edu"
        tag_ids  = pick_tags(persona)
        activity = round(random.uniform(0.1, 1.0), 2)
        papers   = [generate_paper(persona) for _ in range(random.randint(1, 4))]

        try:
            # Insert user
            cur.execute("""
                INSERT INTO users (id, username, email, full_name, department, role)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (email) DO NOTHING
            """, (user_id, username, email, f"{first} {last}", persona["department"], persona["role"]))

            if cur.rowcount == 0:
                print(f"  SKIP  {username} (already exists)")
                continue

            # Tags
            for tid in tag_ids:
                cur.execute(
                    "INSERT INTO user_tags (user_id, tag_id) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                    (user_id, tid)
                )

            # Activity
            cur.execute("""
                UPDATE user_activity SET activity_score = %s WHERE user_id = %s
            """, (activity, user_id))

            # Papers + keywords
            for paper in papers:
                cur.execute("""
                    INSERT INTO papers (user_id, title, abstract, published_at)
                    VALUES (%s, %s, %s, %s) RETURNING id
                """, (user_id, paper["title"], paper["abstract"], paper["published_at"]))
                paper_id = cur.fetchone()[0]

                for kw in paper["keywords"]:
                    cur.execute(
                        "INSERT INTO paper_keywords (paper_id, keyword) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                        (paper_id, kw.lower().strip())
                    )
                inserted_papers += 1

            conn.commit()
            inserted_users += 1

            tag_names = [tag_name_by_id.get(t, str(t)) for t in tag_ids]
            print(f"  ✓  {username:<35} [{persona['role']:<12} | {persona['department']:<20}]  activity={activity}  papers={len(papers)}")
            print(f"      tags: {tag_names}")

        except Exception as e:
            conn.rollback()
            print(f"  ERROR inserting {username}: {e}")

    print(f"\n✓ Done. {inserted_users} users, {inserted_papers} papers inserted.")
    print("\nCopy a user ID to test recommendations:\n")

    cur.execute("SELECT id, username, department, role FROM users LIMIT 5")
    for row in cur.fetchall():
        print(f"  {row[0]}  →  {row[1]}  ({row[3]}, {row[2]})")


if __name__ == "__main__":
    import sys
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    seed(count)
    cur.close()
    conn.close()