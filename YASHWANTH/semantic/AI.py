from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer
import chromadb

app = Flask(__name__)

# -----------------------------
# SAMPLE RESEARCH POSTS
# -----------------------------

research_posts = [
    {
        "id": "post_1",
        "title": "Smart Farming AI",
        "author": "Dr. Kumar",
        "description": "AI based crop monitoring system using IoT sensors and machine learning.",
        "tags": ["AI", "Agriculture", "IoT"],
        "likes": 245,
        "comments": 35
    },
    {
        "id": "post_2",
        "title": "Crop Disease Detection",
        "author": "Dr. Priya",
        "description": "Deep learning model for identifying crop diseases from plant leaf images.",
        "tags": ["AI", "Agriculture", "DeepLearning", "ComputerVision"],
        "likes": 190,
        "comments": 25
    },
    {
        "id": "post_3",
        "title": "Healthcare Chatbot",
        "author": "Dr. Raj",
        "description": "An AI chatbot that helps users understand symptoms and basic healthcare information.",
        "tags": ["AI", "Healthcare", "Chatbot"],
        "likes": 175,
        "comments": 20
    },
    {
        "id": "post_4",
        "title": "Blockchain Voting System",
        "author": "Dr. Meena",
        "description": "A secure voting system using blockchain technology to prevent vote tampering.",
        "tags": ["Blockchain", "Security", "Voting"],
        "likes": 140,
        "comments": 15
    },
    {
        "id": "post_5",
        "title": "Medical Image Analysis",
        "author": "Dr. Arjun",
        "description": "Computer vision model for analyzing medical images and detecting diseases.",
        "tags": ["AI", "Healthcare", "ComputerVision"],
        "likes": 220,
        "comments": 30
    },
    {
        "id": "post_6",
        "title": "AI Traffic Management",
        "author": "Dr. Naveen",
        "description": "Smart traffic control system using artificial intelligence and real-time monitoring.",
        "tags": ["AI", "SmartCity", "IoT"],
        "likes": 180,
        "comments": 22
    },
    {
        "id": "post_7",
        "title": "Face Recognition Attendance",
        "author": "Dr. Karthik",
        "description": "Automated attendance system using facial recognition and deep learning.",
        "tags": ["AI", "DeepLearning", "ComputerVision"],
        "likes": 260,
        "comments": 40
    },
]

# -----------------------------
# LOAD EMBEDDING MODEL
# -----------------------------

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# -----------------------------
# CHROMADB SETUP
# -----------------------------

client = chromadb.PersistentClient(path="./researchhub_chroma_db")

collection = client.get_or_create_collection(
    name="research_posts"
)

# -----------------------------
# STORE POSTS IN CHROMADB
# -----------------------------

for post in research_posts:
    searchable_text = (
        post["title"] + " " +
        post["description"] + " " +
        " ".join(post["tags"])
    )

    embedding = model.encode(searchable_text).tolist()

    collection.upsert(
        ids=[post["id"]],
        documents=[searchable_text],
        embeddings=[embedding],
        metadatas=[{
            "title": post["title"],
            "author": post["author"],
            "description": post["description"],
            "tags": ", ".join(post["tags"]),
            "likes": post["likes"],
            "comments": post["comments"]
        }]
    )

print("Research posts stored successfully!")


# -----------------------------
# EXACT TAG MATCH
# -----------------------------

def exact_tag_match(query):
    query = query.lower().strip()
    matched_posts = []

    for post in research_posts:
        tags = [tag.lower() for tag in post["tags"]]

        if query in tags:
            matched_posts.append(post)

    return matched_posts


# -----------------------------
# SEMANTIC SEARCH
# -----------------------------

def semantic_search(query, top_k=5):
    query_embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    similar_posts = []

    for i in range(len(results["ids"][0])):
        metadata = results["metadatas"][0][i]

        similar_posts.append({
            "id": results["ids"][0][i],
            "title": metadata["title"],
            "author": metadata["author"],
            "description": metadata["description"],
            "tags": metadata["tags"].split(", "),
            "likes": metadata["likes"],
            "comments": metadata["comments"]
        })

    return similar_posts


# -----------------------------
# HOME PAGE
# -----------------------------

@app.route("/")
def home():
    return render_template("index.html")


# -----------------------------
# SEARCH API
# -----------------------------

@app.route("/search", methods=["POST"])
def search():
    data = request.json
    query = data["query"]

    exact_results = exact_tag_match(query)

    if exact_results:
        return jsonify({
            "search_type": "exact_tag_match",
            "results": exact_results
        })

    semantic_results = semantic_search(query)

    return jsonify({
        "search_type": "semantic_search",
        "results": semantic_results
    })


# -----------------------------
# RUN FLASK APP
# -----------------------------

if __name__ == "__main__":
    app.run(debug=True)