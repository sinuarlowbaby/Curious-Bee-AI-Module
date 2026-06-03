# =========================================================
# SECURE EVENT EXTRACTION MAIL READER
# Uses Ollama (Phi-3) for classification + extraction
# Handles: new events, multi-day events, rescheduled events,
#          and announcements (saved separately)
# =========================================================

import imaplib
import email
import re
import json
import os
import datetime
import dateutil.parser

from dotenv import load_dotenv
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
from langchain_ollama import ChatOllama
# from groq import Groq

load_dotenv()
#USING API KEY 
# groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
# llm_client=ChatOllama(model="qwen2.5:1.5b")

from gliner import GLiNER

model = GLiNER.from_pretrained(
    "urchade/gliner_small-v2.1"
)

labels = [
    "event",
    "date",
    "time",
    "venue"
]

# =========================================================
# CONFIGURATION
# =========================================================

EMAIL_ADDRESS       = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD      = os.getenv("EMAIL_PASSWORD")
IMAP_SERVER         = "imap.gmail.com"
CALENDAR_FILE       = "calendar.json"
ANNOUNCEMENTS_FILE  = "announcements.json"

# =========================================================
# AUTHORIZED SENDERS
# ONLY THESE EMAILS CAN TRIGGER EVENT PROCESSING
# =========================================================

AUTHORIZED_SENDERS = [
    "dean@gmail.com",
    "hod@gmail.com",
    "lekhalokare.28@gmail.com"
]

# =========================================================
# LOAD SIMILARITY MODEL (for reschedule matching only)
# =========================================================

print("Loading similarity model...")
similarity_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model ready.\n")

# =========================================================
# CALENDAR STORE FUNCTIONS
# =========================================================

def load_calendar():
    """Load calendar entries from local JSON file."""
    if not os.path.exists(CALENDAR_FILE):
        return []
    with open(CALENDAR_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_calendar(entries):
    """Save calendar entries to local JSON file."""
    with open(CALENDAR_FILE, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=4, ensure_ascii=False)


def find_matching_event(event_name, old_dates=None, threshold=0.75):
    """
    Find an existing calendar entry that matches by:
    1. Semantic similarity of event name (SentenceTransformer)
    2. Optionally boost score if old_dates also match
    Returns the best matching entry or None.
    """
    calendar = load_calendar()
    if not calendar:
        return None

    query_embedding = similarity_model.encode(event_name, convert_to_tensor=True)

    best_match = None
    best_score = 0

    for entry in calendar:
        entry_embedding = similarity_model.encode(entry["title"], convert_to_tensor=True)
        score = util.cos_sim(query_embedding, entry_embedding).item()

        # Boost score if the old date also matches — stronger signal
        if old_dates and entry["date"] in old_dates:
            score += 0.15

        if score > best_score:
            best_score = score
            best_match = entry

    if best_score >= threshold:
        print(f"  Similarity score: {best_score:.2f} — matched '{best_match['title']}'")
        return best_match

    print(f"  Similarity score: {best_score:.2f} — no confident match found")
    return None


def add_event(entry):
    """Add a new event entry to the calendar."""
    calendar = load_calendar()
    calendar.append(entry)
    save_calendar(calendar)
    print(f"  [ADDED]   '{entry['title']}' on {entry['date']}")


def update_event(old_entry, new_dates, new_time, new_venue):
    """
    Remove all old calendar entries for this event title + old date,
    then insert new entries with updated dates.
    """
    calendar = load_calendar()

    # Remove old entries that match this event + old date
    calendar = [
        e for e in calendar
        if not (
            e["title"] == old_entry["title"] and
            e["date"]  == old_entry["date"]
        )
    ]

    # Insert new entries (one per new date)
    for date in new_dates:
        new_entry = {
            "title":       old_entry["title"],
            "date":        date,
            "time":        new_time        or old_entry.get("time"),
            "venue":       new_venue       or old_entry.get("venue")
        }
        calendar.append(new_entry)
        print(f"  [UPDATED] '{old_entry['title']}' — {old_entry['date']} → {date}")

    save_calendar(calendar)


def delete_event(old_entry):
    """
    Remove all old calendar entries for this event title + old date.
    """
    calendar = load_calendar()

    calendar = [
        e for e in calendar
        if not (
            e["title"] == old_entry["title"] and
            e["date"]  == old_entry["date"]
        )
    ]
    save_calendar(calendar)
    print(f"  [DELETED] '{old_entry['title']}' on {old_entry['date']}")


# =========================================================
# ANNOUNCEMENTS STORE FUNCTION
# =========================================================

def save_announcement(sender_email, subject, description):
    """
    Save announcement to a separate JSON file.
    This is read by the UI to display in the announcements section,
    completely separate from the calendar.
    """
    if not os.path.exists(ANNOUNCEMENTS_FILE):
        announcements = []
    else:
        with open(ANNOUNCEMENTS_FILE, "r", encoding="utf-8") as f:
            announcements = json.load(f)

    entry = {
        "sender":      sender_email,
        "subject":     re.sub(r"[\r\n\t]+", " ", subject).strip(),
        "description": description,
        "received_on": datetime.date.today().strftime("%Y-%m-%d")
    }

    announcements.append(entry)

    with open(ANNOUNCEMENTS_FILE, "w", encoding="utf-8") as f:
        json.dump(announcements, f, indent=4, ensure_ascii=False)

    print(f"  [SAVED]   Announcement stored in '{ANNOUNCEMENTS_FILE}'")


# =========================================================
# SLM EMAIL ANALYZER (Ollama Phi-3)
# =========================================================


def extract_main_body(body_text):
    """
    Strip greeting phrases from the start and sign-off phrases from the end
    of the email body, returning only the substantive content.
    Works on both multi-line and single-line (pre-flattened) text.
    """
    text = body_text.strip()

    # Remove greeting phrase(s) at the very beginning.
    # Matches patterns like "Dear All," / "Greetings!!!" / "Hi Team," etc.
    text = re.sub(
        r"^(dear\b[^.!?:,\n]*[:.,!]*\s*|"
        r"hi\b[^.!?:,\n]*[:.,!]*\s*|"
        r"hello\b[^.!?:,\n]*[:.,!]*\s*|"
        r"greetings[^.!?:,\n]*[:.,!]*\s*|"
        r"to\s+all\b[^.!?:,\n]*[:.,!]*\s*|"
        r"to\s+whomsoever\b[^.!?:,\n]*[:.,!]*\s*|"
        r"respected\b[^.!?:,\n]*[:.,!]*\s*)+",
        "", text, flags=re.IGNORECASE,
    ).strip()

    # Find the first sign-off keyword and cut everything from there to end.
    signoff = re.search(
        r"\b(thanks\b|thank you\b|with regards\b|best regards\b|"
        r"warm regards\b|regards\b|yours sincerely\b|yours faithfully\b|"
        r"sincerely\b|cheers\b)",
        text, flags=re.IGNORECASE,
    )
    if signoff:
        text = text[:signoff.start()].strip()

    # Clean up trailing punctuation/whitespace
    return text.strip(" ,;")


def analyze_email(body, subject):
    """
    Send email body + subject to GLiNER model.
    Returns a structured dict with type, event details,
    dates, old_dates, time, venue, description.
    """
    try:
        text = f"Subject: {subject}\n\n{body}"
        entities = model.predict_entities(text, labels)

        # Basic heuristic for classification
        lower_text = text.lower()
        if "cancel" in lower_text:
            email_type = "cancellation"
        elif "postpone" in lower_text or "reschedule" in lower_text:
            email_type = "reschedule"
        elif any(kw in lower_text for kw in ["event", "seminar", "workshop", "talk", "invite"]):
            email_type = "event"
        else:
            email_type = "announcement"

        result = {
            "type": email_type,
            "event": re.sub(r"[\u2013\u2014\u2015\u2212\ufe58\ufe63\uff0d]", "-", subject.strip()),  # normalize dashes
            "dates": [],
            "old_dates": [],
            "time": None,
            "venue": None,
            "description": None
        }

        if email_type == "announcement":
            # For general announcements, taking just the first sentence is usually 
            # the most concise and accurate description of the notice.
            full_body = extract_main_body(body)
            sentences = re.split(r'(?<=[.!?])\s+', full_body)
            result["description"] = sentences[0].strip() if sentences else full_body
            return result

        extracted_dates = []

        # Keywords that indicate a date is a DEADLINE, not an event date
        DEADLINE_KEYWORDS = [
            "deadline", "last date", "register before", "register by",
            "registration closes", "apply before", "apply by", "submit before",
            "submission deadline", "due by", "due date", "closing date",
            "last day to", "must register", "enroll before", "enroll by",
        ]

        for ent in entities:
            label = ent["label"]
            val = ent["text"]

            if label == "event" and result["event"] == subject:
                result["event"] = val
            elif label == "date":
                try:
                    dt = dateutil.parser.parse(val, fuzzy=True)
                    date_str = dt.strftime("%Y-%m-%d")

                    # ---- DEADLINE FILTER ----
                    # Split text into sentences and only check the sentence
                    # that contains this date — prevents cross-sentence bleed
                    # where a deadline phrase in sentence A wrongly flags a
                    # legitimate event date in sentence B.
                    sentences = re.split(r'(?<=[.!?\n])\s+', text)
                    date_sentence = ""
                    for sentence in sentences:
                        if val.lower() in sentence.lower():
                            date_sentence = sentence.lower()
                            break

                    is_deadline = any(kw in date_sentence for kw in DEADLINE_KEYWORDS)

                    if is_deadline:
                        print(f"  [SKIP]  '{val}' looks like a deadline — not an event date")
                    else:
                        extracted_dates.append(date_str)

                except Exception:
                    # Ignore dates that can't be parsed
                    pass
            elif label == "time" and not result["time"]:
                result["time"] = val
            elif label == "venue" and not result["venue"]:
                result["venue"] = val

        # --------------------------------------------------
        # EXPAND DATE RANGES (e.g. "July 10 to July 12" → all 3 days)
        # GLiNER only picks up explicitly written dates, so "from X to Y"
        # phrases leave out the intermediate dates. We detect range patterns
        # and fill them in.
        # --------------------------------------------------
        range_patterns = [
            r"from\s+(.+?)\s+to\s+(.+?)(?:\s+(?:at|in|starting|each)|[,\.\n]|$)",
            r"(\w+ \d{1,2}(?:st|nd|rd|th)?(?:,?\s*\d{4})?)\s+(?:to|through|till|until)\s+(\w+ \d{1,2}(?:st|nd|rd|th)?(?:,?\s*\d{4})?)",
        ]
        range_dates = []
        for pattern in range_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                start_str, end_str = match.group(1).strip(), match.group(2).strip()
                try:
                    start_dt = dateutil.parser.parse(start_str, fuzzy=True)
                    end_dt   = dateutil.parser.parse(end_str,   fuzzy=True)
                    if start_dt <= end_dt:
                        delta = (end_dt - start_dt).days
                        for i in range(delta + 1):
                            day = start_dt + datetime.timedelta(days=i)
                            range_dates.append(day.strftime("%Y-%m-%d"))
                        print(f"  [RANGE] Expanded '{start_str}' → '{end_str}' into {delta + 1} day(s)")
                except Exception:
                    pass

        # Merge GLiNER point-dates with range-expanded dates
        all_dates = extracted_dates + range_dates

        # Remove duplicates while maintaining order
        seen = set()
        unique_dates = [x for x in all_dates if not (x in seen or seen.add(x))]

        if email_type == "cancellation":
            result["old_dates"] = unique_dates

            # Use the actual email body (stripped of greeting/sign-off) as the
            # description — this preserves the real message including any
            # "new date will be announced" notices naturally present in the email.
            result["description"] = extract_main_body(body)
        elif email_type == "reschedule":
            # For reschedule, assume first date is old and rest are new
            if len(unique_dates) >= 2:
                result["old_dates"] = [unique_dates[0]]
                result["dates"] = unique_dates[1:]
            else:
                result["dates"] = unique_dates
        else:
            result["dates"] = unique_dates

        return result

    except Exception as e:
        print(f"  ERROR analyzing email with GLiNER: {e}")
        return {
            "type":        "unknown",
            "event":       subject,
            "dates":       [],
            "old_dates":   [],
            "time":        None,
            "venue":       None,
            "description": None
        }
# =========================================================
# CONNECT TO MAILBOX  [COMMENTED OUT — PROTOTYPE MODE]
# =========================================================

# print("Connecting to mailbox...")
# mail = imaplib.IMAP4_SSL(IMAP_SERVER)
# mail.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
# mail.select("inbox")
# print("Connected.\n")

# =========================================================
# FETCH UNREAD EMAILS  [COMMENTED OUT — PROTOTYPE MODE]
# =========================================================

# try:
#     status, messages = mail.search(None, "UNSEEN")
#     email_ids = messages[0].split()
# except imaplib.IMAP4.abort as e:
#     print(f"\n[!] IMAP connection aborted by server: {e}")
#     print("    This usually happens due to rate-limiting when running the script too frequently.")
#     import sys
#     sys.exit(1)
# print(f"Found {len(email_ids)} unread email(s)\n")
# print("=" * 50)

# =========================================================
# >>>  PROTOTYPE MAIL CONTENT — EDIT THIS SECTION  <<<
# =========================================================
# Change the three variables below to test a new email.
# MAIL_SENDER must be one of the AUTHORIZED_SENDERS above.

MAIL_SENDER   = "dean@gmail.com"          # who sent the email
MAIL_RECEIVER = "receiver@gmail.com"      # who received it
MAIL_SUBJECT  = """
Internal Assessment Marks Submission Deadline
"""
MAIL_BODY     = """
Dear Faculty,
This is to inform all faculty members that the Internal Assessment marks for the odd semester 2026 must be
submitted to the examination cell on or before July 10, 2026.
Faculty who fail to submit marks before the deadline will have their marks locked by the system automatically.
Kindly ensure timely submission to avoid any inconvenience.
Thanks and Regards,
Dr. A. Ramesh
Head of Department
Department of Computer Science and Engineering
SRMIST, Kattankulathur
"""

# =========================================================
# PROCESS EMAIL  (prototype — iterates over a single entry)
# =========================================================

for subject, sender, receiver, body in [
    (MAIL_SUBJECT, MAIL_SENDER, MAIL_RECEIVER, MAIL_BODY)
]:

    # --------------------------------------------------
    # EXTRACT CLEAN SENDER EMAIL ADDRESS
    # --------------------------------------------------

    sender_match = re.search(r"<(.+?)>", sender)
    if sender_match:
        sender_email = sender_match.group(1).strip().lower()
    else:
        sender_email = sender.strip().lower()

    # --------------------------------------------------
    # AUTHORIZED SENDER CHECK
    # --------------------------------------------------

    if sender_email not in [s.strip().lower() for s in AUTHORIZED_SENDERS]:
        print(f"UNAUTHORIZED SENDER SKIPPED: {sender_email}\n")
        print("=" * 50)
        continue

    # Clean up whitespace
    body = re.sub(r"\s+", " ", body).strip()

    # --------------------------------------------------
    # PRINT EMAIL HEADER
    # --------------------------------------------------

    print(f"SENDER   : {sender_email}")
    print(f"RECEIVER : {receiver}")
    print(f"SUBJECT  : {subject}")
    print()

    # --------------------------------------------------
    # ANALYZE EMAIL WITH LOCAL SLM
    # --------------------------------------------------

    print("Analyzing email")
    result = analyze_email(body, subject)

    print(f"TYPE     : {result['type'].upper()}")
    print()
    print(json.dumps(result, indent=4))
    print()

    # ==================================================
    # HANDLE: NEW EVENT → Save to calendar.json
    # ==================================================

    if result["type"] == "event":

        if not result["dates"]:
            print("  WARNING: No dates extracted — skipping calendar entry.\n")

        else:
            print("CALENDAR ACTIONS:")
            for date in result["dates"]:
                entry = {
                    "title":       result["event"],
                    "date":        date,
                    "time":        result["time"],
                    "venue":       result["venue"]
                }
                add_event(entry)

    # ==================================================
    # HANDLE: RESCHEDULED EVENT → Update calendar.json
    # ==================================================

    elif result["type"] == "reschedule":

        print("RESCHEDULE DETECTED — searching calendar for matching event...")

        match = find_matching_event(
            event_name = result["event"],
            old_dates  = result["old_dates"]
        )

        print()
        print("CALENDAR ACTIONS:")

        if match:
            print(f"  Matched: '{match['title']}' on {match['date']}")
            update_event(
                old_entry       = match,
                new_dates       = result["dates"],
                new_time        = result["time"],
                new_venue       = result["venue"]
            )
        else:
            # No confident match found — add as new entry but flag it
            print("  No existing event matched — adding as new entry (flagged)")
            for date in result["dates"]:
                entry = {
                    "title":       f"[POSSIBLY RESCHEDULED] {result['event']}",
                    "date":        date,
                    "time":        result["time"],
                    "venue":       result["venue"]
                }
                add_event(entry)

    # ==================================================
    # HANDLE: CANCELLED EVENT → Delete from calendar.json
    # ==================================================

    elif result["type"] == "cancellation":

        print("CANCELLATION DETECTED — searching calendar for matching event...")

        match = find_matching_event(
            event_name = result["event"],
            old_dates  = result["old_dates"]
        )

        print()
        print("CALENDAR ACTIONS:")

        if match:
            print(f"  Matched: '{match['title']}' on {match['date']} — deleting.")
            delete_event(match)
        else:
            print("  No existing event matched for cancellation.")
            
        print()
        print("ANNOUNCEMENT ACTIONS:")
        save_announcement(
            sender_email = sender_email,
            subject      = subject,
            description  = result.get("description")
        )

    # ==================================================
    # HANDLE: ANNOUNCEMENT → Save to announcements.json
    # NOT added to calendar
    # ==================================================

    elif result["type"] == "announcement":
        print("ANNOUNCEMENT — not added to calendar.")
        print(f"Summary  : {result.get('description')}")
        print()
        print("ANNOUNCEMENT ACTIONS:")
        save_announcement(
            sender_email = sender_email,
            subject      = subject,
            description  = result.get("description")
        )

    # ==================================================
    # HANDLE: UNKNOWN (SLM parse failure)
    # ==================================================

    else:
        print("UNKNOWN TYPE — could not process this email.")

    print()
    print("=" * 50)

# =========================================================
# LOGOUT  [COMMENTED OUT — PROTOTYPE MODE]
# =========================================================

# mail.logout()
print("\nPrototype run complete.")
print(f"\nEvents saved to       : {CALENDAR_FILE}")
print(f"Announcements saved to : {ANNOUNCEMENTS_FILE}")