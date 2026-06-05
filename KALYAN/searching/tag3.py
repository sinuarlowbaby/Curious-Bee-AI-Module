import sqlite3

# ==========================
# DATABASE CONNECTION
# ==========================

conn = sqlite3.connect("research.db")
cursor = conn.cursor()

# ==========================
# CREATE TABLES
# ==========================

cursor.execute("""
CREATE TABLE IF NOT EXISTS faculty(
    id INTEGER PRIMARY KEY,
    name TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS domains(
    faculty_id INTEGER,
    domain TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS papers(
    id INTEGER PRIMARY KEY,
    title TEXT,
    faculty_id INTEGER,
    domain TEXT
)
""")

conn.commit()

# ==========================
# SAMPLE DATA (ONLY IF EMPTY)
# ==========================

cursor.execute("SELECT COUNT(*) FROM faculty")
count = cursor.fetchone()[0]

if count == 0:

    # Faculty
    cursor.execute("INSERT INTO faculty VALUES(1,'Dr. Kumar')")
    cursor.execute("INSERT INTO faculty VALUES(2,'Dr. Ravi')")

    # Domains
    cursor.execute("INSERT INTO domains VALUES(1,'AI')")
    cursor.execute("INSERT INTO domains VALUES(1,'Machine Learning')")
    cursor.execute("INSERT INTO domains VALUES(1,'IoT')")

    cursor.execute("INSERT INTO domains VALUES(2,'AI')")
    cursor.execute("INSERT INTO domains VALUES(2,'Cyber Security')")

    # Papers
    cursor.execute("""
    INSERT INTO papers VALUES
    (1,'AI Chatbot for Students',1,'AI')
    """)

    cursor.execute("""
    INSERT INTO papers VALUES
    (2,'Disease Prediction Using ML',1,'Machine Learning')
    """)

    cursor.execute("""
    INSERT INTO papers VALUES
    (3,'Smart Sensor Network',1,'IoT')
    """)

    cursor.execute("""
    INSERT INTO papers VALUES
    (4,'AI Threat Detection',2,'AI')
    """)

    cursor.execute("""
    INSERT INTO papers VALUES
    (5,'Network Security Framework',2,'Cyber Security')
    """)

    conn.commit()

# ==========================
# SEARCH LOOP
# ==========================

while True:

    search_domain = input(
        "\nEnter Domain (or type exit): "
    ).strip()

    if search_domain.lower() == "exit":
        print("\nGoodbye!")
        break

    print("\nFACULTY FOUND")
    print("-" * 40)

    cursor.execute("""
    SELECT DISTINCT f.name
    FROM faculty f
    JOIN domains d
    ON f.id = d.faculty_id
    WHERE LOWER(d.domain)=LOWER(?)
    """, (search_domain,))

    faculty_results = cursor.fetchall()

    if faculty_results:
        for row in faculty_results:
            print(row[0])
    else:
        print("No faculty found")

    print("\nRESEARCH PAPERS FOUND")
    print("-" * 40)

    cursor.execute("""
    SELECT p.title, f.name
    FROM papers p
    JOIN faculty f
    ON p.faculty_id = f.id
    WHERE LOWER(p.domain)=LOWER(?)
    """, (search_domain,))

    paper_results = cursor.fetchall()

    if paper_results:
        for paper, author in paper_results:
            print(f"Paper : {paper}")
            print(f"Author: {author}")
            print("-" * 40)
    else:
        print("No papers found")

# ==========================
# CLOSE DATABASE
# ==========================

conn.close()