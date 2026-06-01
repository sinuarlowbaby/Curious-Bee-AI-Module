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

from dotenv import load_dotenv
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
from langchain_ollama import ChatOllama
# from groq import Groq

load_dotenv()
#USING API KEY 
# groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
llm_client=ChatOllama(model="qwen2.5:1.5b")
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
    with open(CALENDAR_FILE, "r") as f:
        return json.load(f)


def save_calendar(entries):
    """Save calendar entries to local JSON file."""
    with open(CALENDAR_FILE, "w") as f:
        json.dump(entries, f, indent=4)


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
        with open(ANNOUNCEMENTS_FILE, "r") as f:
            announcements = json.load(f)

    entry = {
        "sender":      sender_email,
        "subject":     subject,
        "description": description,
        "received_on": datetime.date.today().strftime("%Y-%m-%d")
    }

    announcements.append(entry)

    with open(ANNOUNCEMENTS_FILE, "w") as f:
        json.dump(announcements, f, indent=4)

    print(f"  [SAVED]   Announcement stored in '{ANNOUNCEMENTS_FILE}'")


# =========================================================
# SLM EMAIL ANALYZER (Ollama Phi-3)
# =========================================================


def analyze_email(body, subject):
    """
    Send email body + subject to local Phi-3 model.
    Returns a structured dict with type, event details,
    dates, old_dates, time, venue, description.
    """
 
    prompt = f"""You are an intelligent email analyzer for a university.
 
Analyze the email below and classify it as exactly one of:
- "event"        → an invitation to attend something (talk, seminar, workshop, fest, sports event, meeting, competition)
- "reschedule"   → an existing event has been postponed or rescheduled to a new date
- "cancellation" → an existing event has been cancelled
- "announcement" → a general notice, policy update, holiday notice, result declaration, or information with no event to attend
 
────────────────────────────────────────
For "event" extract:
  - "event"       : full name/title of the event
  - "dates"       : list of ALL event dates in YYYY-MM-DD format
                    • If multi-day (e.g. "June 15-17"), include every date: ["2026-06-15","2026-06-16","2026-06-17"]
                    • Ignore registration deadlines, submission deadlines, abstract deadlines
                    • Only include actual dates the event takes place
  - "old_dates"   : [] (empty list — this is a new event)
  - "time"        : start time of the event e.g. "09:30 AM", or null if not mentioned
  - "venue"       : full venue name, or null if not mentioned
 
For "reschedule" extract:
  - "event"       : full name/title of the event (same as the original event name)
  - "dates"       : the NEW date(s) in YYYY-MM-DD (the updated schedule)
  - "old_dates"   : the PREVIOUS date(s) in YYYY-MM-DD that are being replaced
  - "time"        : new time if mentioned, else null
  - "venue"       : new venue if changed, else null
 
For "cancellation" extract:
  - "event"       : full name/title of the cancelled event
  - "dates"       : []
  - "old_dates"   : the original date(s) of the event being cancelled in YYYY-MM-DD
  - "time"        : null
  - "venue"       : null
  - "description" : one sentence explaining the cancellation
 
For "announcement" extract:
  - "event"       : null
  - "dates"       : []
  - "old_dates"   : []
  - "time"        : null
  - "venue"       : null
  - "description" : one sentence summarizing the announcement
────────────────────────────────────────
 
Return ONLY a raw JSON object. No explanation. No markdown fences. No extra text.
 
{{
  "type": "event",
  "event": "...",
  "dates": ["YYYY-MM-DD"],
  "old_dates": [],
  "time": "...",
  "venue": "...",
  "description": "..."
}}
 
Subject: {subject}
 
Body:
{body}
"""
 
    try:
        response = llm_client.invoke(prompt)
 
        raw   = response.content
        clean = re.sub(r"```json|```", "", raw).strip()
 
        result = json.loads(clean)
 
        # Sanitize — ensure lists are always lists
        if not isinstance(result.get("dates"), list):
            result["dates"] = []
        if not isinstance(result.get("old_dates"), list):
            result["old_dates"] = []
            
        # Sanitize - ensure string keys exist to prevent KeyErrors
        for key in ["event", "time", "venue", "description"]:
            if key not in result:
                result[key] = None
 
        return result
 
    except json.JSONDecodeError:
        print("  WARNING: SLM returned invalid JSON — using fallback.")
        return {
            "type":        "unknown",
            "event":       subject,
            "dates":       [],
            "old_dates":   [],
            "time":        None,
            "venue":       None,
            "description": None
        }
    except Exception as e:
        print(f"  ERROR calling Ollama: {e}")
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

# =========================================================
# PROCESS EACH EMAIL
# =========================================================

for email_id in email_ids:

    status, data = mail.fetch(email_id, "(RFC822)")
    raw_email    = data[0][1]
    msg          = email.message_from_bytes(raw_email)

    # --------------------------------------------------
    # EXTRACT BASIC EMAIL METADATA
    # --------------------------------------------------

    subject  = msg["subject"] or ""
    sender   = msg["from"]    or ""
    receiver = msg["to"]      or ""

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

    print("Analyzing email with Phi-3...")
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
# LOGOUT
# =========================================================

mail.logout()
print("\nLogged out. Done.")
print(f"\nEvents saved to      : {CALENDAR_FILE}")
print(f"Announcements saved to: {ANNOUNCEMENTS_FILE}")