from sentence_transformers import SentenceTransformer
import chromadb
import json

# --------------------------------------------------
# SAMPLE RESEARCH POSTS
# --------------------------------------------------

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
        "title": "Fake News Detection",
        "author": "Dr. Suresh",
        "description": "Natural language processing model to detect fake news articles online.",
        "tags": ["AI", "NLP", "MachineLearning"],
        "likes": 210,
        "comments": 28
    },
    {
        "id": "post_7",
        "title": "Cyber Threat Detection",
        "author": "Dr. Harish",
        "description": "AI based cybersecurity system for identifying network attacks.",
        "tags": ["CyberSecurity", "AI", "Security"],
        "likes": 295,
        "comments": 50
    },
    {
        "id": "post_8",
        "title": "Student Performance Prediction",
        "author": "Dr. Vinod",
        "description": "Machine learning model for predicting academic performance of students.",
        "tags": ["AI", "Education", "MachineLearning"],
        "likes": 160,
        "comments": 17
    },
    {
        "id": "post_9",
        "title": "Smart Waste Management",
        "author": "Dr. Gayathri",
        "description": "IoT enabled smart waste collection and monitoring system.",
        "tags": ["IoT", "SmartCity", "Environment"],
        "likes": 145,
        "comments": 14
    },
    {
        "id": "post_10",
        "title": "Lung Cancer Prediction",
        "author": "Dr. Rekha",
        "description": "Deep learning model for early detection of lung cancer from CT scans.",
        "tags": ["Healthcare", "DeepLearning", "MedicalImaging"],
        "likes": 320,
        "comments": 52
    }
]

# --------------------------------------------------
# LOAD EMBEDDING MODEL
# --------------------------------------------------

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# --------------------------------------------------
# CREATE CHROMADB VECTOR DATABASE
# --------------------------------------------------

client = chromadb.PersistentClient(path="./researchhub_chroma_db")

collection = client.get_or_create_collection(
    name="research_posts"
)

# --------------------------------------------------
# STORE POSTS INTO CHROMADB
# --------------------------------------------------

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

print("Research posts stored successfully in ChromaDB.")


# --------------------------------------------------
# EXACT TAG MATCH FUNCTION
# --------------------------------------------------

def exact_tag_match(query):
    query = query.lower().strip()
    matched_posts = []

    for post in research_posts:
        tags = [tag.lower() for tag in post["tags"]]

        if query in tags:
            matched_posts.append(post)

    return matched_posts


# --------------------------------------------------
# SEMANTIC SEARCH FUNCTION
# --------------------------------------------------

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


# --------------------------------------------------
# MAIN SEARCH FUNCTION
# --------------------------------------------------

def search_research_posts(query):
    print("\nSearching:", query)

    exact_results = exact_tag_match(query)

    if exact_results:
        print("\nExact Tag Match Found")

        return {
            "search_type": "exact_tag_match",
            "results": exact_results
        }

    else:
        print("\nNo exact tag match found.")
        print("Running semantic search...")

        semantic_results = semantic_search(query)

        return {
            "search_type": "semantic_search",
            "results": semantic_results
        }


# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

def display_results(search_output):
    print("\nSearch Type:", search_output["search_type"])
    print("\nResearch Feed Results:\n")

    for post in search_output["results"]:
        print("📄", post["title"])
        print("👤", post["author"])
        print(post["description"])
        print("Tags:", ", ".join(post["tags"]))
        print("❤️", post["likes"], " 💬", post["comments"], " 🔖 Save")
        print("-" * 60)


# --------------------------------------------------
# RUN PROGRAM
# --------------------------------------------------

if __name__ == "__main__":
    user_query = input("Enter your search query: ")

    output = search_research_posts(user_query)

    display_results(output)

    print("\nJSON Output:\n")
    print(json.dumps(output, indent=4))