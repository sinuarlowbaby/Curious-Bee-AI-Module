# =========================================================
# SECURE EVENT EXTRACTION MAIL READER
# Uses GLiNER for entity extraction
# Uses keyword classifier for email type detection
# No LLM/SLM needed
# =========================================================

import imaplib
import email
import re
import json
import os
import datetime

from gliner import GLiNER
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
import dateparser

load_dotenv()

# =========================================================
# CONFIGURATION
# =========================================================

EMAIL_ADDRESS      = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD     = os.getenv("EMAIL_PASSWORD")
IMAP_SERVER        = "imap.gmail.com"
CALENDAR_FILE      = "calendar.json"
ANNOUNCEMENTS_FILE = "announcements.json"
PENDING_FILE       = "pending_emails.json"

# =========================================================
# AUTHORIZED SENDERS
# =========================================================

AUTHORIZED_SENDERS = [
    "dean@gmail.com",
    "hod@gmail.com",
    "lekhalokare.28@gmail.com"
]

# =========================================================
# KEYWORD LISTS FOR CLASSIFICATION
# =========================================================

RESCHEDULE_KEYWORDS = [
    "rescheduled", "postponed", "new date",
    "updated date", "date change", "changed to",
    "now scheduled", "moved to"
]

CANCELLATION_KEYWORDS = [
    "cancelled", "canceled", "called off",
    "will not be held", "stands cancelled",
    "has been cancelled"
]

ANNOUNCEMENT_KEYWORDS = [
    "notice", "circular", "policy", "holiday",
    "deadline", "submit", "reminder", "inform",
    "kindly note", "please note", "attention",
    "marks submission", "dress code", "fee payment",
    "result", "timetable", "schedule change"
]

EVENT_KEYWORDS = [
    "invited to attend", "cordially invited",
    "pleased to invite", "workshop", "seminar",
    "symposium", "conference", "fest", "expo",
    "guest lecture", "talk", "webinar", "meeting",
    "competition", "hackathon", "sports meet",
    "cultural", "event", "programme", "colloquium"
]

# =========================================================
# DEADLINE CONTEXT WORDS
# (dates near these words should be ignored)
# =========================================================

DEADLINE_CONTEXT_WORDS = [
    "deadline", "register by", "registration",
    "last date", "submit before", "submission",
    "apply before", "on or before", "no later than",
    "closing date"
]

# =========================================================
# LOAD MODELS
# =========================================================

print("Loading GLiNER model...")
gliner_model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")
print("GLiNER ready.")

print("Loading similarity model...")
similarity_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Similarity model ready.\n")

# =========================================================
# STEP 1 — EMAIL CLASSIFIER
# =========================================================

def classify_email(subject, body):
    """
    Classify email as event / reschedule / cancelled / announcement
    using keyword matching on subject + body.
    """
    text = (subject + " " + body).lower()

    # Check reschedule first — most specific
    for kw in RESCHEDULE_KEYWORDS:
        if kw in text:
            return "reschedule"

    # Check cancellation
    for kw in CANCELLATION_KEYWORDS:
        if kw in text:
            return "cancelled"

    # Check event keywords
    for kw in EVENT_KEYWORDS:
        if kw in text:
            return "event"

    # Check announcement keywords
    for kw in ANNOUNCEMENT_KEYWORDS:
        if kw in text:
            return "announcement"

    # Default
    return "announcement"


# =========================================================
# STEP 2 — GLINER ENTITY EXTRACTION
# =========================================================

def extract_entities(text):
    """
    Use GLiNER to extract named entities from email text.
    Returns grouped entities dict.
    """
    labels = [
        "event name",
        "event date",
        "old date",
        "new date",
        "start time",
        "end time",
        "venue",
        "department",
        "organizer name",
        "registration deadline"
    ]

    entities = gliner_model.predict_entities(text, labels, threshold=0.4)

    grouped = {}
    for entity in entities:
        label = entity["label"]
        value = entity["text"].strip()
        if label not in grouped:
            grouped[label] = []
        grouped[label].append(value)

    return grouped


# =========================================================
# STEP 3 — DATE PARSER
# =========================================================

def parse_dates(date_strings):
    """
    Convert list of raw date strings to YYYY-MM-DD format.
    Handles date ranges like 'August 3 to August 5'.
    """
    parsed_dates = []

    for raw in date_strings:

        # Handle date ranges — "August 3 to August 5 2026"
        range_match = re.search(
            r"(\w+\s+\d{1,2})\s+to\s+(\w+\s+\d{1,2},?\s*\d{4})",
            raw, re.IGNORECASE
        )
        if range_match:
            start_str = range_match.group(1)
            end_str   = range_match.group(2)

            year_match = re.search(r"\d{4}", end_str)
            year = year_match.group() if year_match else str(datetime.date.today().year)

            start = dateparser.parse(f"{start_str} {year}")
            end   = dateparser.parse(end_str)

            if start and end:
                current = start
                while current <= end:
                    parsed_dates.append(current.strftime("%Y-%m-%d"))
                    current += datetime.timedelta(days=1)
            continue

        # Handle DD-MM-YYYY or DD/MM/YYYY
        dmy_match = re.search(r"(\d{1,2})[-/](\d{1,2})[-/](\d{4})", raw)
        if dmy_match:
            d, m, y = dmy_match.groups()
            try:
                date_obj = datetime.date(int(y), int(m), int(d))
                parsed_dates.append(date_obj.strftime("%Y-%m-%d"))
                continue
            except:
                pass

        # Standard dateparser for everything else
        parsed = dateparser.parse(raw, settings={"PREFER_DATES_FROM": "future"})
        if parsed:
            parsed_dates.append(parsed.strftime("%Y-%m-%d"))

    # Remove duplicates while preserving order
    seen      = set()
    unique    = []
    for d in parsed_dates:
        if d not in seen:
            seen.add(d)
            unique.append(d)

    return unique


# =========================================================
# STEP 4 — CONTEXT FILTER
# (remove dates near deadline keywords)
# =========================================================

def filter_deadline_dates(body, all_dates):
    """
    Remove dates that appear near deadline-related keywords.
    Returns only actual event dates.
    """
    event_dates = []

    for date_str in all_dates:
        parsed = dateparser.parse(date_str)
        if not parsed:
            event_dates.append(date_str)
            continue

        date_patterns = [
            parsed.strftime("%B %d, %Y"),
            parsed.strftime("%B %d %Y"),
            parsed.strftime("%d %B %Y"),
            parsed.strftime("%d-%m-%Y"),
            parsed.strftime("%d/%m/%Y"),
        ]

        is_deadline = False

        for pattern in date_patterns:
            idx = body.lower().find(pattern.lower())
            if idx == -1:
                continue

            context_before = body[max(0, idx - 100):idx].lower()

            for kw in DEADLINE_CONTEXT_WORDS:
                if kw in context_before:
                    is_deadline = True
                    print(f"  Filtered deadline date: {date_str} (near '{kw}')")
                    break

            if is_deadline:
                break

        if not is_deadline:
            event_dates.append(date_str)

    return event_dates


# =========================================================
# STEP 5 — REGISTRATION LINK EXTRACTOR
# =========================================================

def extract_registration_link(body):
    """
    Extract registration/form links from email body.
    Looks for URLs near registration-related keywords.
    """
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    all_urls    = re.findall(url_pattern, body)

    if not all_urls:
        return None

    registration_keywords = [
        "register", "registration", "sign up", "signup",
        "enroll", "enrolment", "form", "apply", "booking"
    ]

    # Return URL found near a registration keyword
    for url in all_urls:
        idx     = body.lower().find(url.lower())
        context = body[max(0, idx - 150): idx + 150].lower()
        for kw in registration_keywords:
            if kw in context:
                return url

    # Fallback — return first URL if no registration context found
    return all_urls[0] if all_urls else None


# =========================================================
# STEP 6 — CATEGORY DETECTOR
# =========================================================

def detect_category(subject, body):
    """Detect event category from keywords."""
    text = (subject + " " + body).lower()

    categories = {
        "Symposiums":     ["symposium", "conference", "national conference"],
        "Workshops":      ["workshop", "fdp", "faculty development"],
        "Webinars":       ["webinar", "online session"],
        "Guest Lectures": ["guest lecture", "talk", "pre phd", "colloquium"],
        "Sports":         ["sports", "athletics", "tournament", "cricket", "football"],
        "Cultural":       ["cultural", "fest", "music", "dance", "arts"],
        "Policy Meet":    ["policy", "meeting", "administrative"],
        "Grants":         ["grant", "funding", "research proposal"]
    }

    for category, keywords in categories.items():
        for kw in keywords:
            if kw in text:
                return category

    return "Other"


# =========================================================
# MAIN ANALYZER
# =========================================================

def analyze_email(body, subject):
    """
    Full pipeline:
    1. Classify email type
    2. Extract entities with GLiNER
    3. Parse and clean dates
    4. Filter deadline dates
    5. Extract registration link
    6. Return structured result
    """

    full_text = f"{subject}. {body}"

    # Step 1 — classify
    email_type = classify_email(subject, body)
    print(f"  Classified as: {email_type}")

    # --------------------------------------------------
    # ANNOUNCEMENT — no entity extraction needed
    # --------------------------------------------------
    if email_type == "announcement":
        sentences   = re.split(r'(?<=[.!?])\s+', body.strip())
        description = " ".join(sentences[:2]) if sentences else body[:200]
        return {
            "type":              "announcement",
            "event":             None,
            "dates":             [],
            "old_dates":         [],
            "start_time":        None,
            "end_time":          None,
            "venue":             None,
            "department":        None,
            "organizer":         None,
            "description":       description,
            "category":          "announcement",
            "registration_link": None
        }

    # --------------------------------------------------
    # ALL OTHER TYPES — run GLiNER
    # --------------------------------------------------

    # Step 2 — extract entities
    entities = extract_entities(full_text)
    print(f"  Raw entities: {json.dumps(entities, indent=2)}")

    # Common fields
    start_time  = entities.get("start time",     [None])[0] if entities.get("start time")     else None
    end_time    = entities.get("end time",        [None])[0] if entities.get("end time")       else None
    venue       = entities.get("venue",           [None])[0] if entities.get("venue")          else None
    department  = entities.get("department",      [None])[0] if entities.get("department")     else None
    organizer   = entities.get("organizer name",  [None])[0] if entities.get("organizer name") else None
    event_name  = entities.get("event name",      [subject])[0] if entities.get("event name") else subject

    sentences   = re.split(r'(?<=[.!?])\s+', body.strip())
    description = " ".join(sentences[:2]) if sentences else body[:200]

    # Step 5 — registration link
    registration_link = extract_registration_link(body)

    # --------------------------------------------------
    # CANCELLED
    # --------------------------------------------------
    if email_type == "cancelled":
        return {
            "type":              "cancelled",
            "event":             event_name,
            "dates":             parse_dates(entities.get("event date", [])),
            "old_dates":         [],
            "start_time":        start_time,
            "end_time":          end_time,
            "venue":             venue,
            "department":        department,
            "organizer":         organizer,
            "description":       f"Event cancelled: {event_name}",
            "category":          detect_category(subject, body),
            "registration_link": None
        }

    # --------------------------------------------------
    # RESCHEDULE
    # --------------------------------------------------
    if email_type == "reschedule":
        old_dates = parse_dates(entities.get("old date", []))
        new_dates = parse_dates(entities.get("new date", []))

        # Fallback — if GLiNER didn't separate them use position
        if not old_dates or not new_dates:
            all_dates = parse_dates(entities.get("event date", []))
            if len(all_dates) >= 2:
                old_dates = [all_dates[0]]
                new_dates = [all_dates[-1]]
            elif len(all_dates) == 1:
                new_dates = all_dates
                old_dates = []

        return {
            "type":              "reschedule",
            "event":             event_name,
            "dates":             new_dates,
            "old_dates":         old_dates,
            "start_time":        start_time,
            "end_time":          end_time,
            "venue":             venue,
            "department":        department,
            "organizer":         organizer,
            "description":       description,
            "category":          detect_category(subject, body),
            "registration_link": registration_link
        }

    # --------------------------------------------------
    # EVENT
    # --------------------------------------------------
    # Step 3 — parse dates
    raw_dates  = entities.get("event date", [])
    all_dates  = parse_dates(raw_dates)

    # Step 4 — filter deadline dates
    event_dates = filter_deadline_dates(body, all_dates)

    return {
        "type":              "event",
        "event":             event_name,
        "dates":             event_dates,
        "old_dates":         [],
        "start_time":        start_time,
        "end_time":          end_time,
        "venue":             venue,
        "department":        department,
        "organizer":         organizer,
        "description":       description,
        "category":          detect_category(subject, body),
        "registration_link": registration_link
    }


# =========================================================
# CALENDAR STORE FUNCTIONS
# =========================================================

def load_json_file(filepath):
    """Safely load a JSON file. Returns empty list if missing or empty."""
    if not os.path.exists(filepath):
        return []
    with open(filepath, "r") as f:
        content = f.read().strip()
        return json.loads(content) if content else []


def save_json_file(filepath, data):
    """Save data to a JSON file."""
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)


def find_matching_event(event_name, old_dates=None, threshold=0.75):
    """
    Find existing calendar entry by semantic similarity.
    Boosts score if old_dates also match.
    """
    calendar = load_json_file(CALENDAR_FILE)
    if not calendar:
        return None

    query_embedding = similarity_model.encode(event_name, convert_to_tensor=True)
    best_match      = None
    best_score      = 0

    for entry in calendar:
        entry_embedding = similarity_model.encode(entry["title"], convert_to_tensor=True)
        score           = util.cos_sim(query_embedding, entry_embedding).item()

        if old_dates and entry["date"] in old_dates:
            score += 0.15

        if score > best_score:
            best_score = score
            best_match = entry

    print(f"  Similarity score: {best_score:.2f}")
    return best_match if best_score >= threshold else None


def add_event(entry):
    """Add a new event to calendar.json."""
    calendar = load_json_file(CALENDAR_FILE)
    calendar.append(entry)
    save_json_file(CALENDAR_FILE, calendar)
    print(f"  [ADDED]   '{entry['title']}' on {entry['date']}")


def update_event(old_entry, new_dates, new_time, new_venue, new_description):
    """Remove old event entries and insert updated ones."""
    calendar = load_json_file(CALENDAR_FILE)

    calendar = [
        e for e in calendar
        if not (
            e["title"] == old_entry["title"] and
            e["date"]  == old_entry["date"]
        )
    ]

    for date in new_dates:
        new_entry = {
            "title":             old_entry["title"],
            "date":              date,
            "start_time":        new_time  or old_entry.get("start_time"),
            "end_time":          old_entry.get("end_time"),
            "venue":             new_venue or old_entry.get("venue"),
            "department":        old_entry.get("department"),
            "organizer":         old_entry.get("organizer"),
            "category":          old_entry.get("category"),
            "description":       new_description,
            "registration_link": old_entry.get("registration_link")
        }
        calendar.append(new_entry)
        print(f"  [UPDATED] '{old_entry['title']}' — {old_entry['date']} → {date}")

    save_json_file(CALENDAR_FILE, calendar)


def save_announcement(sender_email, subject, description):
    """Save announcement to announcements.json."""
    announcements = load_json_file(ANNOUNCEMENTS_FILE)
    announcements.append({
        "sender":      sender_email,
        "subject":     subject,
        "description": description,
        "received_on": datetime.date.today().strftime("%Y-%m-%d")
    })
    save_json_file(ANNOUNCEMENTS_FILE, announcements)
    print(f"  [SAVED]   Announcement stored in '{ANNOUNCEMENTS_FILE}'")


def save_to_pending(body, subject, sender_email):
    """Save failed email to pending queue for retry."""
    pending = load_json_file(PENDING_FILE)
    pending.append({
        "subject":   subject,
        "sender":    sender_email,
        "body":      body,
        "failed_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    save_json_file(PENDING_FILE, pending)
    print(f"  [PENDING] Email saved to queue for later processing")


def process_pending_queue():
    """Re-process emails that failed in previous runs."""
    pending = load_json_file(PENDING_FILE)
    if not pending:
        return

    print(f"Found {len(pending)} pending email(s) — retrying...\n")
    recovered = []

    for item in pending:
        try:
            result = analyze_email(item["body"], item["subject"])
            if result["type"] not in ["unknown", "announcement"]:
                print(f"  [RECOVERED] '{item['subject']}'")
                recovered.append(item)

                if result["type"] == "event":
                    for date in result["dates"]:
                        add_event({
                            "title":             result["event"],
                            "date":              date,
                            "start_time":        result["start_time"],
                            "end_time":          result["end_time"],
                            "venue":             result["venue"],
                            "department":        result["department"],
                            "organizer":         result["organizer"],
                            "category":          result["category"],
                            "description":       result["description"],
                            "registration_link": result["registration_link"]
                        })
                elif result["type"] == "announcement":
                    save_announcement(
                        item["sender"],
                        item["subject"],
                        result.get("description")
                    )
        except Exception as e:
            print(f"  Still failing: {e}")

    remaining = [e for e in pending if e not in recovered]
    save_json_file(PENDING_FILE, remaining)

    if recovered:
        print(f"  Recovered {len(recovered)} email(s)\n")


# =========================================================
# CONNECT TO MAILBOX
# =========================================================

print("Connecting to mailbox...")
mail = imaplib.IMAP4_SSL(IMAP_SERVER)
mail.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
mail.select("inbox")
print("Connected.\n")

# =========================================================
# FETCH UNREAD EMAILS
# =========================================================

status, messages = mail.search(None, "UNSEEN")
email_ids = messages[0].split()

print(f"Found {len(email_ids)} unread email(s)\n")
print("=" * 50)

# Process any previously failed emails first
process_pending_queue()

# =========================================================
# PROCESS EACH EMAIL
# =========================================================

for email_id in email_ids:

    status, data = mail.fetch(email_id, "(RFC822)")
    raw_email    = data[0][1]
    msg          = email.message_from_bytes(raw_email)

    # --------------------------------------------------
    # EXTRACT METADATA
    # --------------------------------------------------
    subject  = msg["subject"] or ""
    sender   = msg["from"]    or ""
    receiver = msg["to"]      or ""

    sender_match = re.search(r"<(.+?)>", sender)
    sender_email = sender_match.group(1).strip().lower() if sender_match else sender.strip().lower()

    # --------------------------------------------------
    # AUTHORIZED SENDER CHECK
    # --------------------------------------------------
    if sender_email not in [s.strip().lower() for s in AUTHORIZED_SENDERS]:
        print(f"UNAUTHORIZED SENDER SKIPPED: {sender_email}\n")
        print("=" * 50)
        continue

    # --------------------------------------------------
    # EXTRACT EMAIL BODY
    # --------------------------------------------------
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            try:
                payload = part.get_payload(decode=True)
                if payload:
                    decoded = payload.decode(errors="ignore")
                    if content_type == "text/plain":
                        body += decoded
                    elif content_type == "text/html":
                        soup = BeautifulSoup(decoded, "lxml")
                        body += soup.get_text(separator=" ")
            except:
                pass
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            body = payload.decode(errors="ignore")

    body = re.sub(r"\s+", " ", body).strip()

    # --------------------------------------------------
    # PRINT EMAIL HEADER
    # --------------------------------------------------
    print(f"SENDER   : {sender_email}")
    print(f"RECEIVER : {receiver}")
    print(f"SUBJECT  : {subject}")
    print()

    # --------------------------------------------------
    # ANALYZE EMAIL
    # --------------------------------------------------
    try:
        result = analyze_email(body, subject)
    except Exception as e:
        print(f"  ERROR during analysis: {e}")
        save_to_pending(body, subject, sender_email)
        print("=" * 50)
        continue

    print(f"TYPE     : {result['type'].upper()}")
    print()
    print(json.dumps(result, indent=4))
    print()

    # ==================================================
    # HANDLE: NEW EVENT
    # ==================================================
    if result["type"] == "event":

        if not result["dates"]:
            print("  WARNING: No dates found — saving to pending")
            save_to_pending(body, subject, sender_email)
        else:
            print("CALENDAR ACTIONS:")
            for date in result["dates"]:
                add_event({
                    "title":             result["event"],
                    "date":              date,
                    "start_time":        result["start_time"],
                    "end_time":          result["end_time"],
                    "venue":             result["venue"],
                    "department":        result["department"],
                    "organizer":         result["organizer"],
                    "category":          result["category"],
                    "description":       result["description"],
                    "registration_link": result["registration_link"]
                })

    # ==================================================
    # HANDLE: RESCHEDULE
    # ==================================================
    elif result["type"] == "reschedule":

        print("RESCHEDULE DETECTED — searching for matching event...")
        match = find_matching_event(result["event"], result["old_dates"])
        print()
        print("CALENDAR ACTIONS:")

        if match:
            print(f"  Matched: '{match['title']}' on {match['date']}")
            update_event(
                old_entry       = match,
                new_dates       = result["dates"],
                new_time        = result["start_time"],
                new_venue       = result["venue"],
                new_description = result["description"]
            )
        else:
            print("  No match found — adding as new entry (flagged)")
            for date in result["dates"]:
                add_event({
                    "title":             result["event"],
                    "date":              date,
                    "start_time":        result["start_time"],
                    "end_time":          result["end_time"],
                    "venue":             result["venue"],
                    "department":        result["department"],
                    "organizer":         result["organizer"],
                    "category":          result["category"],
                    "description":       f"[POSSIBLY RESCHEDULED] {result['description']}",
                    "registration_link": result["registration_link"]
                })

    # ==================================================
    # HANDLE: CANCELLED
    # ==================================================
    elif result["type"] == "cancelled":

        print("CANCELLATION DETECTED — removing from calendar")
        match = find_matching_event(result["event"], result["dates"])

        if match:
            calendar = load_json_file(CALENDAR_FILE)
            calendar = [
                e for e in calendar
                if not (
                    e["title"] == match["title"] and
                    e["date"]  == match["date"]
                )
            ]
            save_json_file(CALENDAR_FILE, calendar)
            print(f"  [REMOVED] '{match['title']}' on {match['date']}")
            save_announcement(
                sender_email,
                subject,
                f"Event cancelled: {result['event']}"
            )
        else:
            print("  No matching event found in calendar")

    # ==================================================
    # HANDLE: ANNOUNCEMENT
    # ==================================================
    elif result["type"] == "announcement":

        print("ANNOUNCEMENT — not added to calendar")
        print(f"Summary  : {result.get('description')}")
        print()
        print("ANNOUNCEMENT ACTIONS:")
        save_announcement(sender_email, subject, result.get("description"))

    print()
    print("=" * 50)

# =========================================================
# LOGOUT
# =========================================================

mail.logout()
print("\nLogged out. Done.")
print(f"Events saved to       : {CALENDAR_FILE}")
print(f"Announcements saved to: {ANNOUNCEMENTS_FILE}")