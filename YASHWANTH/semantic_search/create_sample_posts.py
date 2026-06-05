import json
import random
from datetime import datetime, timedelta

# =========================
# CONFIG
# =========================

NUM_RECORDS = 1000

DOCUMENT_TYPES = [
    "Research Paper",
    "Technical Article",
    "Industry Report",
    "Case Study",
    "White Paper",
    "Policy Document",
    "Literature Review",
    "Conference Paper"
]

DOMAINS = [
    "Artificial Intelligence",
    "Law",
    "Management",
    "Health",
    "Science",
    "Finance",
    "Cybersecurity",
    "Education",
    "Environment",
    "Engineering",
    "Agriculture",
    "Psychology",
    "Biotechnology",
    "Energy",
    "Public Policy",
    "Supply Chain"
]

INDUSTRIES = [
    "Healthcare",
    "Banking",
    "Insurance",
    "Education",
    "Manufacturing",
    "Agriculture",
    "Retail",
    "Transportation",
    "Government",
    "Telecommunications",
    "E-commerce",
    "Energy",
    "Pharmaceuticals"
]

REGIONS = [
    "India",
    "United States",
    "Germany",
    "Japan",
    "United Kingdom",
    "Singapore",
    "Australia",
    "Canada",
    "South Korea",
    "Brazil"
]

METHODOLOGIES = [
    "survey analysis",
    "longitudinal study",
    "comparative evaluation",
    "systematic review",
    "case study analysis",
    "experimental research",
    "machine learning evaluation",
    "mixed-method research",
    "simulation modelling",
    "field investigation"
]

FINDINGS = [
    "improved operational efficiency",
    "reduced processing time",
    "enhanced decision quality",
    "higher stakeholder satisfaction",
    "improved sustainability metrics",
    "strong economic benefits",
    "better risk management",
    "improved scalability",
    "higher adoption rates",
    "improved regulatory compliance"
]

CHALLENGES = [
    "data quality limitations",
    "regulatory uncertainty",
    "high implementation costs",
    "organizational resistance",
    "privacy concerns",
    "infrastructure constraints",
    "technical complexity",
    "skill shortages",
    "security risks",
    "ethical considerations"
]

RECOMMENDATIONS = [
    "adopt phased implementation strategies",
    "strengthen governance frameworks",
    "increase stakeholder engagement",
    "expand workforce training",
    "improve data management practices",
    "invest in automation technologies",
    "establish monitoring mechanisms",
    "develop compliance programs",
    "encourage public-private partnerships",
    "promote interdisciplinary collaboration"
]

AUTHORS_FIRST = [
    "Aarav","Vihaan","Arjun","Aditya","Ananya",
    "Priya","Neha","Rahul","Rohan","Sneha",
    "Michael","Emma","Sophia","Daniel","James"
]

AUTHORS_LAST = [
    "Sharma","Patel","Nair","Iyer","Kumar",
    "Singh","Johnson","Brown","Miller","Davis",
    "Wilson","Taylor","Thomas","Moore"
]

TOPICS = [
    "predictive analytics",
    "large language models",
    "climate monitoring",
    "legal automation",
    "healthcare diagnostics",
    "smart agriculture",
    "supply chain optimization",
    "renewable energy planning",
    "cyber threat detection",
    "digital transformation",
    "knowledge management",
    "financial forecasting",
    "robotics deployment",
    "educational analytics",
    "public sector innovation"
]

# =========================
# HELPERS
# =========================

def random_author():
    return f"{random.choice(AUTHORS_FIRST)} {random.choice(AUTHORS_LAST)}"

def create_title(doc_type, domain, topic):
    patterns = [
        f"{topic.title()} in {domain}: A {doc_type}",
        f"Emerging Trends in {topic.title()}",
        f"Evaluating {topic.title()} Across Modern Organizations",
        f"{domain} Perspectives on {topic.title()}",
        f"Transforming Industry Through {topic.title()}",
        f"Future Directions for {topic.title()}",
        f"Strategic Applications of {topic.title()}",
        f"Challenges and Opportunities in {topic.title()}"
    ]
    return random.choice(patterns)

def create_abstract(
    doc_type,
    domain,
    topic,
    industry,
    region,
    methodology,
    finding,
    challenge,
    recommendation
):
    paragraphs = [

        f"This {doc_type.lower()} examines the role of "
        f"{topic} within the field of {domain.lower()}. "
        f"The study focuses on organizations operating in the "
        f"{industry.lower()} sector across {region}. "
        f"Growing technological and regulatory changes have increased "
        f"interest in understanding how modern approaches can improve "
        f"organizational performance and long-term sustainability.",

        f"Researchers employed a {methodology} approach to investigate "
        f"current practices, adoption patterns, operational constraints, "
        f"and stakeholder perceptions. Data was gathered from industry reports, "
        f"expert interviews, public datasets, surveys, and organizational records. "
        f"Multiple analytical techniques were used to identify trends, measure impact, "
        f"and compare performance across different implementation environments.",

        f"The findings revealed {finding}. "
        f"Organizations that successfully integrated advanced practices demonstrated "
        f"higher levels of productivity, resilience, innovation capacity, and strategic alignment. "
        f"Several high-performing organizations also reported measurable gains in service quality, "
        f"resource utilization, and customer engagement.",

        f"Despite these benefits, several barriers were identified including "
        f"{challenge}. The analysis showed that these factors significantly influence "
        f"implementation success and long-term adoption outcomes. Variations were observed "
        f"between large enterprises, public-sector organizations, and small businesses.",

        f"The report recommends that organizations {recommendation}. "
        f"Additional recommendations include establishing clear governance structures, "
        f"improving performance measurement systems, strengthening workforce capabilities, "
        f"and promoting collaborative innovation initiatives.",

        f"Overall, this document contributes practical insights for researchers, "
        f"industry professionals, regulators, and policymakers. The study highlights "
        f"future research opportunities and provides a foundation for evidence-based "
        f"decision-making in rapidly evolving environments."
    ]

    random.shuffle(paragraphs)

    return " ".join(paragraphs)

# =========================
# GENERATE
# =========================

records = []

start_date = datetime(2025, 1, 1)

for i in range(1, NUM_RECORDS + 1):

    doc_type = random.choice(DOCUMENT_TYPES)
    domain = random.choice(DOMAINS)
    topic = random.choice(TOPICS)
    industry = random.choice(INDUSTRIES)
    region = random.choice(REGIONS)
    methodology = random.choice(METHODOLOGIES)
    finding = random.choice(FINDINGS)
    challenge = random.choice(CHALLENGES)
    recommendation = random.choice(RECOMMENDATIONS)

    title = create_title(doc_type, domain, topic)

    abstract = create_abstract(
        doc_type,
        domain,
        topic,
        industry,
        region,
        methodology,
        finding,
        challenge,
        recommendation
    )

    date = start_date + timedelta(days=i)

    record = {
        "id": i,
        "title": title,
        "author": random_author(),
        "tag": f"{domain},{topic},{doc_type}",
        "abstract": abstract,
        "date": date.strftime("%Y-%m-%d"),
        "ts": int(date.timestamp())
    }

    records.append(record)

with open("research_dataset_1000_unique.json", "w", encoding="utf-8") as f:
    json.dump(records, f, indent=2, ensure_ascii=False)

print("Generated:", len(records))