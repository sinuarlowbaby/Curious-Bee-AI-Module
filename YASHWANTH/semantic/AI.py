# pyrefly: ignore [missing-import]
from flask import Flask, render_template, request, jsonify

# pyrefly: ignore [missing-import]
from sentence_transformers import SentenceTransformer

# pyrefly: ignore [missing-import]
import chromadb


app = Flask(__name__)


# --------------------------------------------------
# RESEARCH SCHOLAR SEMANTIC SEARCH TEST DATA
# --------------------------------------------------
# type:
# supervisor  -> faculty guide
# opportunity -> research opening
# thread      -> discussion thread

research_items = [

    # ==================================================
    # ENGINEERING BLOCK - SUPERVISORS
    # ==================================================

    {
        "id": "eng_sup_1",
        "type": "supervisor",
        "block": "Engineering Block",
        "department": "Computer Science and Engineering",
        "title": "Dr. Priya Raman - AI and Computer Vision Supervisor",
        "author": "Dr. Priya Raman",
        "description": "Faculty supervisor working on artificial intelligence, crop disease detection, medical image analysis, deep learning, and computer vision.",
        "tags": ["AI", "DeepLearning", "ComputerVision", "MedicalImaging", "Agriculture"],
        "status": "Available for 3 scholars",
        "likes": 42,
        "comments": 8
    },
    {
        "id": "eng_sup_2",
        "type": "supervisor",
        "block": "Engineering Block",
        "department": "Information Technology",
        "title": "Dr. Arjun Menon - Cybersecurity and Blockchain Supervisor",
        "author": "Dr. Arjun Menon",
        "description": "Faculty guide for blockchain security, secure voting systems, cyber threat detection, network security, and privacy-focused systems.",
        "tags": ["Blockchain", "CyberSecurity", "Security", "Voting", "NetworkSecurity"],
        "status": "Available for 2 scholars",
        "likes": 37,
        "comments": 6
    },
    {
        "id": "eng_sup_3",
        "type": "supervisor",
        "block": "Engineering Block",
        "department": "Mechanical Engineering",
        "title": "Dr. Naveen Kumar - Robotics and Smart Systems Supervisor",
        "author": "Dr. Naveen Kumar",
        "description": "Research supervisor for robotics, automation, IoT systems, smart traffic management, autonomous robots, and sensor-based engineering solutions.",
        "tags": ["Robotics", "Automation", "IoT", "SmartSystems", "Sensors"],
        "status": "Available for 4 scholars",
        "likes": 31,
        "comments": 5
    },

    # ==================================================
    # ENGINEERING BLOCK - OPPORTUNITIES
    # ==================================================

    {
        "id": "eng_opp_1",
        "type": "opportunity",
        "block": "Engineering Block",
        "department": "Computer Science and Engineering",
        "title": "Research Opportunity: AI Based Crop Disease Detection",
        "author": "Dr. Priya Raman",
        "description": "Open research opportunity for scholars interested in plant leaf disease detection using CNN, transfer learning, image processing, and computer vision.",
        "tags": ["AI", "Agriculture", "CNN", "DeepLearning", "ComputerVision"],
        "status": "Applications Open",
        "likes": 58,
        "comments": 13
    },
    {
        "id": "eng_opp_2",
        "type": "opportunity",
        "block": "Engineering Block",
        "department": "Information Technology",
        "title": "Research Opportunity: Blockchain Based Secure Voting System",
        "author": "Dr. Arjun Menon",
        "description": "Opportunity for scholars to work on blockchain voting, smart contracts, digital identity, vote tampering prevention, and secure election systems.",
        "tags": ["Blockchain", "Security", "Voting", "SmartContracts", "DigitalIdentity"],
        "status": "Applications Open",
        "likes": 49,
        "comments": 10
    },
    {
        "id": "eng_opp_3",
        "type": "opportunity",
        "block": "Engineering Block",
        "department": "Electronics and Communication Engineering",
        "title": "Research Opportunity: IoT Based Smart Campus Monitoring",
        "author": "Dr. Kavitha S",
        "description": "Research opening for IoT-based smart campus monitoring using sensors, edge devices, wireless communication, and real-time analytics.",
        "tags": ["IoT", "Sensors", "SmartCampus", "EmbeddedSystems", "Analytics"],
        "status": "Applications Open",
        "likes": 44,
        "comments": 7
    },

    # ==================================================
    # ENGINEERING BLOCK - THREADS
    # ==================================================

    {
        "id": "eng_thread_1",
        "type": "thread",
        "block": "Engineering Block",
        "department": "Computer Science and Engineering",
        "title": "How can we improve accuracy in medical image classification?",
        "author": "Research Scholar - CSE",
        "description": "Discussion about CNN architectures, medical imaging datasets, data augmentation, feature extraction, model evaluation, and explainable AI.",
        "tags": ["MedicalImaging", "CNN", "ComputerVision", "ExplainableAI", "Healthcare"],
        "status": "Active Discussion",
        "likes": 75,
        "comments": 22
    },
    {
        "id": "eng_thread_2",
        "type": "thread",
        "block": "Engineering Block",
        "department": "Information Technology",
        "title": "Best architecture for blockchain voting in universities",
        "author": "Research Scholar - IT",
        "description": "Thread discussing blockchain voting design, authentication, smart contracts, privacy, scalability, and secure result verification.",
        "tags": ["Blockchain", "Voting", "Security", "Authentication", "Privacy"],
        "status": "Active Discussion",
        "likes": 66,
        "comments": 18
    },
    {
        "id": "eng_thread_3",
        "type": "thread",
        "block": "Engineering Block",
        "department": "Mechanical Engineering",
        "title": "Robotics projects for smart university automation",
        "author": "Research Scholar - Mechanical",
        "description": "Discussion on autonomous robots, campus delivery bots, robotic inspection, path planning, sensors, and automation in university environments.",
        "tags": ["Robotics", "Automation", "PathPlanning", "Sensors", "SmartCampus"],
        "status": "Active Discussion",
        "likes": 51,
        "comments": 15
    },

    # ==================================================
    # LAW BLOCK - SUPERVISORS
    # ==================================================

    {
        "id": "law_sup_1",
        "type": "supervisor",
        "block": "Law Block",
        "department": "Cyber Law",
        "title": "Dr. Meera Iyer - Cyber Law and Data Privacy Supervisor",
        "author": "Dr. Meera Iyer",
        "description": "Faculty supervisor for cyber law, data protection, privacy regulations, digital evidence, online fraud, and IT Act related research.",
        "tags": ["CyberLaw", "DataPrivacy", "ITAct", "DigitalEvidence", "OnlineFraud"],
        "status": "Available for 3 scholars",
        "likes": 39,
        "comments": 9
    },
    {
        "id": "law_sup_2",
        "type": "supervisor",
        "block": "Law Block",
        "department": "Intellectual Property Rights",
        "title": "Prof. Raghav Sharma - IPR and Technology Law Supervisor",
        "author": "Prof. Raghav Sharma",
        "description": "Research guide for intellectual property rights, patents, copyrights, AI-generated content, software licensing, and innovation law.",
        "tags": ["IPR", "PatentLaw", "Copyright", "TechnologyLaw", "AIRegulation"],
        "status": "Available for 2 scholars",
        "likes": 34,
        "comments": 6
    },
    {
        "id": "law_sup_3",
        "type": "supervisor",
        "block": "Law Block",
        "department": "Criminal Law",
        "title": "Dr. Farah Khan - Criminal Justice and Forensic Law Supervisor",
        "author": "Dr. Farah Khan",
        "description": "Supervisor for criminal justice, forensic evidence, cybercrime investigation, victim rights, and legal reform research.",
        "tags": ["CriminalLaw", "ForensicLaw", "CyberCrime", "JusticeSystem", "LegalReform"],
        "status": "Available for 1 scholar",
        "likes": 29,
        "comments": 5
    },

    # ==================================================
    # LAW BLOCK - OPPORTUNITIES
    # ==================================================

    {
        "id": "law_opp_1",
        "type": "opportunity",
        "block": "Law Block",
        "department": "Cyber Law",
        "title": "Research Opportunity: Data Privacy in Indian Universities",
        "author": "Dr. Meera Iyer",
        "description": "Opportunity to study student data privacy, consent, cybersecurity policies, legal compliance, and institutional data governance.",
        "tags": ["DataPrivacy", "CyberLaw", "UniversityPolicy", "Compliance", "StudentData"],
        "status": "Applications Open",
        "likes": 46,
        "comments": 11
    },
    {
        "id": "law_opp_2",
        "type": "opportunity",
        "block": "Law Block",
        "department": "Intellectual Property Rights",
        "title": "Research Opportunity: AI Generated Content and Copyright Law",
        "author": "Prof. Raghav Sharma",
        "description": "Open research topic on ownership, copyright protection, plagiarism, and legal challenges in AI-generated academic and creative content.",
        "tags": ["AIRegulation", "Copyright", "IPR", "TechnologyLaw", "Plagiarism"],
        "status": "Applications Open",
        "likes": 52,
        "comments": 14
    },
    {
        "id": "law_opp_3",
        "type": "opportunity",
        "block": "Law Block",
        "department": "Criminal Law",
        "title": "Research Opportunity: Cybercrime Investigation and Digital Evidence",
        "author": "Dr. Farah Khan",
        "description": "Research opportunity on cybercrime reporting, digital evidence admissibility, forensic procedure, and investigation challenges.",
        "tags": ["CyberCrime", "DigitalEvidence", "ForensicLaw", "CriminalLaw", "Investigation"],
        "status": "Applications Open",
        "likes": 41,
        "comments": 9
    },

    # ==================================================
    # LAW BLOCK - THREADS
    # ==================================================

    {
        "id": "law_thread_1",
        "type": "thread",
        "block": "Law Block",
        "department": "Cyber Law",
        "title": "How should universities handle student data privacy?",
        "author": "Research Scholar - Law",
        "description": "Discussion thread on privacy policies, consent forms, cloud storage, student records, and legal accountability in universities.",
        "tags": ["DataPrivacy", "CyberLaw", "UniversityPolicy", "Consent", "CloudStorage"],
        "status": "Active Discussion",
        "likes": 61,
        "comments": 19
    },
    {
        "id": "law_thread_2",
        "type": "thread",
        "block": "Law Block",
        "department": "Intellectual Property Rights",
        "title": "Can AI generated research content be copyrighted?",
        "author": "Research Scholar - IPR",
        "description": "Thread discussing copyright ownership, AI tools, originality, academic writing, plagiarism, and legal protection of AI-generated content.",
        "tags": ["AIRegulation", "Copyright", "IPR", "ResearchEthics", "Plagiarism"],
        "status": "Active Discussion",
        "likes": 70,
        "comments": 24
    },
    {
        "id": "law_thread_3",
        "type": "thread",
        "block": "Law Block",
        "department": "Criminal Law",
        "title": "Digital evidence challenges in cybercrime cases",
        "author": "Research Scholar - Criminal Law",
        "description": "Discussion on digital evidence collection, chain of custody, admissibility, forensic tools, and legal standards in cybercrime cases.",
        "tags": ["DigitalEvidence", "CyberCrime", "ForensicLaw", "CriminalLaw", "EvidenceLaw"],
        "status": "Active Discussion",
        "likes": 55,
        "comments": 16
    },

    # ==================================================
    # FACULTY OF SCIENCE AND HUMANITIES BLOCK - SUPERVISORS
    # ==================================================

    {
        "id": "snh_sup_1",
        "type": "supervisor",
        "block": "Faculty of Science and Humanities Block",
        "department": "Psychology",
        "title": "Dr. Ananya Bose - Psychology and Student Wellbeing Supervisor",
        "author": "Dr. Ananya Bose",
        "description": "Research supervisor for student mental health, academic stress, emotional intelligence, counselling, and wellbeing studies.",
        "tags": ["Psychology", "MentalHealth", "StudentWellbeing", "Counselling", "AcademicStress"],
        "status": "Available for 4 scholars",
        "likes": 45,
        "comments": 12
    },
    {
        "id": "snh_sup_2",
        "type": "supervisor",
        "block": "Faculty of Science and Humanities Block",
        "department": "English and Communication",
        "title": "Dr. Latha Menon - Language, Communication and Digital Humanities Supervisor",
        "author": "Dr. Latha Menon",
        "description": "Faculty guide for communication studies, academic writing, digital humanities, language learning, and media discourse analysis.",
        "tags": ["Communication", "English", "DigitalHumanities", "MediaStudies", "AcademicWriting"],
        "status": "Available for 3 scholars",
        "likes": 36,
        "comments": 8
    },
    {
        "id": "snh_sup_3",
        "type": "supervisor",
        "block": "Faculty of Science and Humanities Block",
        "department": "Mathematics",
        "title": "Dr. Suresh Babu - Applied Mathematics and Data Modelling Supervisor",
        "author": "Dr. Suresh Babu",
        "description": "Supervisor for applied mathematics, statistical modelling, optimization, data analysis, and mathematical methods for AI systems.",
        "tags": ["Mathematics", "Statistics", "DataModelling", "Optimization", "AI"],
        "status": "Available for 2 scholars",
        "likes": 33,
        "comments": 7
    },

    # ==================================================
    # FACULTY OF SCIENCE AND HUMANITIES BLOCK - OPPORTUNITIES
    # ==================================================

    {
        "id": "snh_opp_1",
        "type": "opportunity",
        "block": "Faculty of Science and Humanities Block",
        "department": "Psychology",
        "title": "Research Opportunity: Academic Stress among University Students",
        "author": "Dr. Ananya Bose",
        "description": "Open research opportunity to study academic pressure, exam stress, sleep patterns, counselling needs, and student wellbeing.",
        "tags": ["Psychology", "AcademicStress", "StudentWellbeing", "MentalHealth", "Counselling"],
        "status": "Applications Open",
        "likes": 47,
        "comments": 13
    },
    {
        "id": "snh_opp_2",
        "type": "opportunity",
        "block": "Faculty of Science and Humanities Block",
        "department": "English and Communication",
        "title": "Research Opportunity: Digital Communication among University Students",
        "author": "Dr. Latha Menon",
        "description": "Research opening on social media communication, academic writing, digital identity, online learning communication, and student engagement.",
        "tags": ["Communication", "DigitalMedia", "AcademicWriting", "StudentEngagement", "OnlineLearning"],
        "status": "Applications Open",
        "likes": 38,
        "comments": 9
    },
    {
        "id": "snh_opp_3",
        "type": "opportunity",
        "block": "Faculty of Science and Humanities Block",
        "department": "Mathematics",
        "title": "Research Opportunity: Statistical Modelling for Student Performance",
        "author": "Dr. Suresh Babu",
        "description": "Opportunity to build statistical and machine learning models for academic performance prediction and student learning analytics.",
        "tags": ["Statistics", "MachineLearning", "Education", "DataModelling", "Prediction"],
        "status": "Applications Open",
        "likes": 43,
        "comments": 10
    },

    # ==================================================
    # FACULTY OF SCIENCE AND HUMANITIES BLOCK - THREADS
    # ==================================================

    {
        "id": "snh_thread_1",
        "type": "thread",
        "block": "Faculty of Science and Humanities Block",
        "department": "Psychology",
        "title": "Research methods for student mental health studies",
        "author": "Research Scholar - Psychology",
        "description": "Discussion on survey design, ethical approval, counselling data, academic stress measurement, and wellbeing research methodology.",
        "tags": ["MentalHealth", "ResearchMethods", "Psychology", "StudentWellbeing", "Ethics"],
        "status": "Active Discussion",
        "likes": 64,
        "comments": 21
    },
    {
        "id": "snh_thread_2",
        "type": "thread",
        "block": "Faculty of Science and Humanities Block",
        "department": "English and Communication",
        "title": "Digital humanities tools for literature research",
        "author": "Research Scholar - English",
        "description": "Thread about text analysis, corpus tools, digital archives, media discourse, language models, and humanities research methods.",
        "tags": ["DigitalHumanities", "English", "TextAnalysis", "MediaStudies", "ResearchTools"],
        "status": "Active Discussion",
        "likes": 49,
        "comments": 17
    },
    {
        "id": "snh_thread_3",
        "type": "thread",
        "block": "Faculty of Science and Humanities Block",
        "department": "Mathematics",
        "title": "Using statistics for education data analysis",
        "author": "Research Scholar - Mathematics",
        "description": "Discussion about regression, classification, statistical tests, data preprocessing, and prediction models for academic datasets.",
        "tags": ["Statistics", "Education", "DataAnalysis", "Regression", "Prediction"],
        "status": "Active Discussion",
        "likes": 57,
        "comments": 18
    }
]


# --------------------------------------------------
# LOAD EMBEDDING MODEL
# --------------------------------------------------

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# --------------------------------------------------
# CHROMADB SETUP
# --------------------------------------------------

client = chromadb.PersistentClient(path="./curiousbees_scholar_chroma_db")

collection = client.get_or_create_collection(
    name="research_scholar_search"
)


# --------------------------------------------------
# STORE DATA IN CHROMADB
# --------------------------------------------------

for item in research_items:
    searchable_text = (
        item["type"] + " " +
        item["block"] + " " +
        item["department"] + " " +
        item["title"] + " " +
        item["author"] + " " +
        item["description"] + " " +
        " ".join(item["tags"]) + " " +
        item["status"]
    )

    embedding = model.encode(searchable_text).tolist()

    collection.upsert(
        ids=[item["id"]],
        documents=[searchable_text],
        embeddings=[embedding],
        metadatas=[{
            "type": item["type"],
            "block": item["block"],
            "department": item["department"],
            "title": item["title"],
            "author": item["author"],
            "description": item["description"],
            "tags": ", ".join(item["tags"]),
            "status": item["status"],
            "likes": item["likes"],
            "comments": item["comments"]
        }]
    )

print("Research Scholar test data stored successfully in ChromaDB!")


# --------------------------------------------------
# EXACT MATCH
# --------------------------------------------------

def exact_match(query):
    query = query.lower().strip()
    matched_items = []

    for item in research_items:
        tags = [tag.lower() for tag in item["tags"]]

        searchable_fields = [
            item["type"].lower(),
            item["block"].lower(),
            item["department"].lower()
        ]

        if query in tags or query in searchable_fields:
            matched_items.append(item)

    return matched_items


# --------------------------------------------------
# SEMANTIC SEARCH
# --------------------------------------------------

def semantic_search(query, top_k=8):
    query_embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    similar_items = []

    for i in range(len(results["ids"][0])):
        metadata = results["metadatas"][0][i]

        similar_items.append({
            "id": results["ids"][0][i],
            "type": metadata["type"],
            "block": metadata["block"],
            "department": metadata["department"],
            "title": metadata["title"],
            "author": metadata["author"],
            "description": metadata["description"],
            "tags": metadata["tags"].split(", "),
            "status": metadata["status"],
            "likes": metadata["likes"],
            "comments": metadata["comments"]
        })

    return similar_items


# --------------------------------------------------
# MAIN SEARCH FUNCTION
# --------------------------------------------------

def search_research_items(query):
    exact_results = exact_match(query)

    if exact_results:
        return {
            "search_type": "exact_match",
            "results": exact_results
        }

    semantic_results = semantic_search(query)

    return {
        "search_type": "semantic_search",
        "results": semantic_results
    }


# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


# --------------------------------------------------
# SEARCH API
# --------------------------------------------------

@app.route("/search", methods=["POST"])
def search():
    data = request.json
    query = data["query"]

    output = search_research_items(query)

    return jsonify(output)


# --------------------------------------------------
# RUN APP
# --------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)