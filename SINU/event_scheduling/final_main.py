# =========================================================
# SECURE EVENT EXTRACTION MAIL READER  (refactored)
# Uses GLiNER (urchade/gliner_small-v2.1) for entity extraction
# and SentenceTransformer (all-MiniLM-L6-v2) for reschedule matching.
# Handles: new events, multi-day events, rescheduled events,
#          cancellations, and announcements.
# =========================================================

import re
import json
import os
import datetime
import dateutil.parser

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
from gliner import GLiNER

load_dotenv()

# =========================================================
# CONFIGURATION
# =========================================================

EMAIL_ADDRESS      = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD     = os.getenv("EMAIL_PASSWORD")
IMAP_SERVER        = "imap.gmail.com"
CALENDAR_FILE      = "calendar.json"
ANNOUNCEMENTS_FILE = "announcements.json"

AUTHORIZED_SENDERS = {
    "dean@gmail.com",
    "hod@gmail.com",
    "lekhalokare.28@gmail.com",
}

GLINER_LABELS = ["event", "date", "time", "venue"]

# Module-level constants (previously buried inside functions)
DEADLINE_KEYWORDS = [
    "deadline", "last date", "register before", "register by",
    "registration closes", "apply before", "apply by", "submit before",
    "submission deadline", "due by", "due date", "closing date",
    "last day to", "must register", "enroll before", "enroll by",
]

VENUE_KEYWORDS = [
    "Hall", "Auditorium", "Seminar Hall", "Conference Hall", "Lab",
    "Laboratory", "Studio", "Room", "Block", "Centre", "Center",
    "Complex", "Ground", "Arena", "Classroom",
]

# Unicode dash variants — used in normalize_dashes() and time-splitting
DASH_CHARS = r"[\u2013\u2014\u2015\u2212\ufe58\ufe63\uff0d]"

# Pre-compiled regex patterns (were rebuilt on every call in the original)
_GREETING_RE = re.compile(
    r"^(dear\b[^.!?:,\n]*[:.,!]*\s*"
    r"|hi\b[^.!?:,\n]*[:.,!]*\s*"
    r"|hello\b[^.!?:,\n]*[:.,!]*\s*"
    r"|greetings[^.!?:,\n]*[:.,!]*\s*"
    r"|to\s+all\b[^.!?:,\n]*[:.,!]*\s*"
    r"|to\s+whomsoever\b[^.!?:,\n]*[:.,!]*\s*"
    r"|respected\b[^.!?:,\n]*[:.,!]*\s*)+",
    re.IGNORECASE,
)
_SIGNOFF_RE = re.compile(
    r"\b(thanks\b|thank you\b|with regards\b|best regards\b"
    r"|warm regards\b|regards\b|yours sincerely\b|yours faithfully\b"
    r"|sincerely\b|cheers\b)",
    re.IGNORECASE,
)

# =========================================================
# MODEL LOADING
# =========================================================

print("Loading models...")
gliner_model     = GLiNER.from_pretrained("urchade/gliner_small-v2.1")
similarity_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Models ready.\n")

# =========================================================
# JSON PERSISTENCE HELPERS
# =========================================================

def _load_json(path: str) -> list:
    """Load a JSON list from disk; returns [] if the file doesn't exist."""
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: str, data: list) -> None:
    """Persist a list as a JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

# =========================================================
# CALENDAR STORE
# =========================================================

def add_event(entry: dict) -> None:
    """Append a new event entry to the calendar."""
    cal = _load_json(CALENDAR_FILE)
    cal.append(entry)
    _save_json(CALENDAR_FILE, cal)
    print(f"  [ADDED]   '{entry['title']}' on {entry['date']}")


def _drop_entries(title: str, date: str) -> list:
    """
    Return a fresh calendar list with all entries matching
    title + date removed. Shared by update_event and delete_event
    to eliminate duplicated filter logic.
    """
    return [
        e for e in _load_json(CALENDAR_FILE)
        if not (e["title"] == title and e["date"] == date)
    ]


def update_event(old: dict, new_dates: list,
                 from_time, to_time, venue, link=None) -> None:
    """Replace old calendar entries for this event with entries on new dates."""
    cal = _drop_entries(old["title"], old["date"])
    for date in new_dates:
        cal.append({
            "title":     old["title"],
            "date":      date,
            "from_time": from_time or old.get("from_time"),
            "to_time":   to_time   or old.get("to_time"),
            "venue":     venue     or old.get("venue"),
            "link":      link      or old.get("link"),
            "status":    "reschedule",
        })
        print(f"  [UPDATED] '{old['title']}' — {old['date']} → {date}")
    _save_json(CALENDAR_FILE, cal)


def delete_event(old: dict) -> None:
    """Remove all calendar entries matching this event title + date."""
    _save_json(CALENDAR_FILE, _drop_entries(old["title"], old["date"]))
    print(f"  [DELETED] '{old['title']}' on {old['date']}")


def find_matching_event(event_name: str, old_dates: list = None,
                        threshold: float = 0.75) -> dict | None:
    """
    Semantic search over the calendar.

    Key improvement over the original: all calendar titles are batch-encoded
    in a single forward pass (one call to similarity_model.encode) rather than
    calling encode() individually for each entry, making this O(1) model calls
    regardless of calendar size.

    A 0.15 score bonus is applied when the candidate entry's date is also
    found in old_dates — acts as a strong disambiguation signal.
    """
    cal = _load_json(CALENDAR_FILE)
    if not cal:
        return None

    query_emb  = similarity_model.encode(event_name, convert_to_tensor=True)
    title_embs = similarity_model.encode([e["title"] for e in cal], convert_to_tensor=True)
    scores     = util.cos_sim(query_emb, title_embs)[0].tolist()

    best_score, best_match = 0.0, None
    for score, entry in zip(scores, cal):
        if old_dates and entry["date"] in old_dates:
            score += 0.15
        if score > best_score:
            best_score, best_match = score, entry

    if best_score >= threshold:
        print(f"  Similarity score: {best_score:.2f} — matched '{best_match['title']}'")
        return best_match

    print(f"  Similarity score: {best_score:.2f} — no confident match found")
    return None

# =========================================================
# ANNOUNCEMENTS STORE
# =========================================================

def save_announcement(sender_email: str, subject: str, description: str) -> None:
    """Append an announcement entry to announcements.json."""
    ann = _load_json(ANNOUNCEMENTS_FILE)
    ann.append({
        "sender":      sender_email,
        "subject":     re.sub(r"[\r\n\t]+", " ", subject).strip(),
        "description": description,
        "received_on": datetime.date.today().strftime("%Y-%m-%d"),
    })
    _save_json(ANNOUNCEMENTS_FILE, ann)
    print(f"  [SAVED]   Announcement stored in '{ANNOUNCEMENTS_FILE}'")

# =========================================================
# TEXT UTILITIES
# =========================================================

def extract_main_body(body: str) -> str:
    """Strip greetings and sign-offs, returning only substantive content."""
    text = _GREETING_RE.sub("", body.strip()).strip()
    m = _SIGNOFF_RE.search(text)
    return text[:m.start()].strip(" ,;") if m else text.strip(" ,;")


def normalize_dashes(text: str) -> str:
    """Replace Unicode dash variants with a plain ASCII hyphen."""
    return re.sub(DASH_CHARS, "-", text)


def is_likely_date(text: str) -> bool:
    """
    Guard against dateutil parsing non-date strings (e.g. 'GPT-2' → June 2).
    Only passes text containing a month abbreviation or a numeric separator.
    Bug fix: removed the unreachable duplicate 'return False' from the original.
    """
    lower = text.lower()
    months = ["jan", "feb", "mar", "apr", "may", "jun",
              "jul", "aug", "sep", "oct", "nov", "dec"]
    return any(m in lower for m in months) or bool(re.search(r'\d{1,2}[/-]\d{1,2}', text))


def is_likely_time(text: str) -> bool:
    """Return True if text contains digits or explicit time keywords."""
    lower = text.lower()
    return (
        any(c.isdigit() for c in lower) or
        any(kw in lower for kw in ["am", "pm", "noon", "midnight", "hours", "hrs"])
    )

# =========================================================
# PRIVATE EXTRACTION HELPERS
# =========================================================

def _classify(text: str) -> str:
    """Heuristic email classifier; returns one of: event, reschedule, cancellation, announcement."""
    lower = text.lower()
    if "cancel" in lower:
        return "cancellation"
    if "postpone" in lower or "reschedule" in lower:
        return "reschedule"
    if any(kw in lower for kw in ["event", "seminar", "workshop", "talk", "invite"]):
        return "event"
    return "announcement"


def _split_time(val: str) -> tuple[str, str | None]:
    """
    Split a time string like '10:00 AM to 12:00 PM' into (from_time, to_time).
    Returns (val, None) when no range separator is detected.
    """
    cleaned = re.sub(r'(?i)\s+onwards', '', val).strip()
    sep     = r'\s*(?:' + DASH_CHARS + r'|[-]|to|till|until)\s*'
    parts   = re.split(sep, cleaned, maxsplit=1, flags=re.IGNORECASE)
    return (parts[0].strip(), parts[1].strip()) if len(parts) == 2 else (cleaned, None)


def _expand_date_ranges(text: str) -> list[str]:
    """
    Find 'from X to Y' or 'X to Y' date range patterns and expand them
    into individual ISO date strings for every day in the range.
    """
    patterns = [
        r"from\s+(.+?)\s+to\s+(.+?)(?:\s+(?:at|in|starting|each)|[,\.\n]|$)",
        r"(\w+ \d{1,2}(?:st|nd|rd|th)?(?:,?\s*\d{4})?)\s+(?:to|through|till|until)\s+(\w+ \d{1,2}(?:st|nd|rd|th)?(?:,?\s*\d{4})?)",
    ]
    out = []
    for pat in patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            s_str, e_str = m.group(1).strip(), m.group(2).strip()
            if not (is_likely_date(s_str) and is_likely_date(e_str)):
                continue
            try:
                s_dt = dateutil.parser.parse(s_str, fuzzy=True)
                e_dt = dateutil.parser.parse(e_str, fuzzy=True)
                if s_dt <= e_dt:
                    delta = (e_dt - s_dt).days
                    out += [
                        (s_dt + datetime.timedelta(days=i)).strftime("%Y-%m-%d")
                        for i in range(delta + 1)
                    ]
                    print(f"  [RANGE] Expanded '{s_str}' → '{e_str}' into {delta + 1} day(s)")
            except Exception:
                pass
    return out


def _venue_fallback(text: str) -> str | None:
    """
    Regex fallback for venue extraction.
    Checks for an explicit 'Venue:' label first, then scans sentences for
    known venue keywords and extracts text following the word 'at'.
    """
    m = re.search(r"Venue\s*:\s*(.*?)(?:\n|$)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    for sentence in re.split(r"[.!?\n]+", text):
        sentence = sentence.strip()
        if any(w.lower() in sentence.lower() for w in VENUE_KEYWORDS):
            at_m = re.search(r"\bat\s+(.*)", sentence, re.IGNORECASE)
            return at_m.group(1).strip() if at_m else sentence
    return None


def _reg_link(text: str) -> str | None:
    """
    Return the first URL that looks like a registration/event link, or None.
    Checks both the URL itself and the surrounding context (40 chars before the link).
    """
    url_re  = r'https?://[^\s<>"\'\]\)]+|www\.[^\s<>"\'\]\)]+'
    url_kws = {"form", "unstop", "hackathon", "register", "apply", "ticket", "eventbrite"}
    ctx_kws = {"register", "registration", "apply", "join", "here", "link"}
    for link in dict.fromkeys(re.findall(url_re, text)):
        if any(kw in link.lower() for kw in url_kws):
            return link
        idx = text.find(link)
        if idx != -1 and any(kw in text[max(0, idx - 40):idx].lower() for kw in ctx_kws):
            return link
    return None


def _dedupe(seq: list) -> list:
    """Remove duplicates from a list while preserving insertion order."""
    seen: set = set()
    return [x for x in seq if not (x in seen or seen.add(x))]

# =========================================================
# EMAIL ANALYZER
# =========================================================

def analyze_email(body: str, subject: str) -> dict:
    """
    Extract structured event details from an email using GLiNER + regex fallbacks.
    Returns a dict with: type, event, dates, old_dates, time, from_time,
    to_time, venue, link, description.
    """
    try:
        # --- Normalize inputs ---
        subject = re.sub(r"^(Re:|Fwd:|FW:)\s*", "", subject, flags=re.IGNORECASE).strip()
        body    = re.sub(r"<[^>]+>", " ", body)   # strip HTML tags
        body    = re.sub(r"&nbsp;", " ", body)
        body    = re.sub(r"[ \t]+", " ", body)
        body    = re.sub(r"\n+", "\n", body).strip()

        text      = f"Subject: {subject}\n\n{body}"
        etype     = _classify(text)
        entities  = gliner_model.predict_entities(text, GLINER_LABELS)
        norm_subj = normalize_dashes(subject)      # used for event-name comparison (bug fix)

        result = {
            "type":        etype,
            "event":       norm_subj,
            "dates":       [],
            "old_dates":   [],
            "time":        None,
            "from_time":   None,
            "to_time":     None,
            "venue":       None,
            "link":        None,
            "description": None,
        }

        # --- Build description for non-event types (first substantive sentence) ---
        if etype in ("announcement", "cancellation", "reschedule"):
            body_clean = extract_main_body(body)
            sents      = re.split(r'(?<=[.!?])\s+', body_clean)
            result["description"] = sents[0].strip() if sents else body_clean
            if etype == "announcement":
                return result  # no date/venue extraction needed

        # --- Parse GLiNER entities ---
        extracted_dates = []
        # Pre-split once; avoids re-splitting inside the entity loop
        text_sentences = re.split(r'(?<=[.!?\n])\s+', text)

        for ent in entities:
            lbl, val = ent["label"], ent["text"]

            if lbl == "event" and result["event"] == norm_subj:
                # Only override the subject-derived name if GLiNER found a cleaner one
                # Bug fix: compare against norm_subj (not raw subject) to handle Unicode dashes
                result["event"] = val

            elif lbl == "date":
                if not is_likely_date(val):
                    continue
                try:
                    date_str = dateutil.parser.parse(val, fuzzy=True).strftime("%Y-%m-%d")
                    # Check only the sentence containing this date for deadline context —
                    # prevents cross-sentence bleed where a deadline phrase in sentence A
                    # wrongly flags a legitimate event date in sentence B.
                    ctx = next((s.lower() for s in text_sentences if val.lower() in s.lower()), "")
                    if any(kw in ctx for kw in DEADLINE_KEYWORDS):
                        print(f"  [SKIP]  '{val}' looks like a deadline — not an event date")
                    else:
                        extracted_dates.append(date_str)
                except Exception:
                    pass

            elif lbl == "time" and not result["from_time"]:
                if not is_likely_time(val):
                    continue
                result["time"] = val
                result["from_time"], result["to_time"] = _split_time(val)

            elif lbl == "venue" and not result["venue"]:
                # GLiNER often extracts only the first venue token (e.g. "Seminar Hall").
                # This pattern grabs all comma-separated address parts that immediately follow,
                # stopping at a period or newline — giving the full venue + address string.
                m = re.search(re.escape(val) + r"(?:,\s*[^.\n]+)*", text)
                result["venue"] = m.group(0).strip() if m else val

        # --- Time fallback (regex) ---
        if not result["from_time"]:
            range_t  = re.search(
                r"(\d{1,2}[:.]\d{2}\s?(?:AM|PM|am|pm))\s*(?:to|-)\s*(\d{1,2}[:.]\d{2}\s?(?:AM|PM|am|pm))",
                text, re.IGNORECASE,
            )
            single_t = re.search(r"\d{1,2}[:.]\d{2}\s?(?:AM|PM|am|pm)", text, re.IGNORECASE)
            if range_t:
                result["from_time"] = range_t.group(1)
                result["to_time"]   = range_t.group(2)
                result["time"]      = f"{result['from_time']} to {result['to_time']}"
            elif single_t:
                result["from_time"] = single_t.group()
                result["time"]      = result["from_time"]

        # --- Venue fallback (regex) ---
        if not result["venue"]:
            result["venue"] = _venue_fallback(text)

        # --- Deduplicate (GLiNER point-dates merged with range-expanded dates) ---
        all_dates = _dedupe(extracted_dates + _expand_date_ranges(text))

        # --- Assign dates by email type ---
        if etype == "cancellation":
            result["old_dates"]   = all_dates
            # Use the full cleaned body (not just first sentence) so that
            # any "new date will be announced" phrasing is preserved naturally.
            result["description"] = extract_main_body(body)

        elif etype == "reschedule":
            # Assumption: first extracted date is the old date; remaining are new dates.
            if len(all_dates) >= 2:
                result["old_dates"] = [all_dates[0]]
                result["dates"]     = all_dates[1:]
            else:
                result["dates"] = all_dates

        else:  # event
            result["dates"] = all_dates
            result["link"]  = _reg_link(text)

        return result

    except Exception as e:
        print(f"  ERROR analyzing email: {e}")
        return {
            "type": "unknown", "event": subject, "dates": [], "old_dates": [],
            "time": None, "venue": None, "description": None,
        }

# =========================================================
# EMAIL DISPATCHER
# =========================================================

def process_email(sender: str, receiver: str, subject: str, body: str) -> None:
    """
    Validate sender authorization, run analysis, and route the result
    to the correct calendar / announcement handler.
    Encapsulates the processing loop from the original script.
    """
    # --- Normalize sender address (handles "Name <email>" format) ---
    m = re.search(r"<(.+?)>", sender)
    sender_email = (m.group(1) if m else sender).strip().lower()

    if sender_email not in {s.lower() for s in AUTHORIZED_SENDERS}:
        print(f"UNAUTHORIZED SENDER SKIPPED: {sender_email}\n{'=' * 50}\n")
        return

    # Normalize body whitespace but preserve newlines (venue regex is newline-aware)
    body = re.sub(r"[ \t\r]+", " ", body).strip()

    print(f"SENDER   : {sender_email}")
    print(f"RECEIVER : {receiver}")
    print(f"SUBJECT  : {subject.strip()}\n")

    print("Analyzing email...")
    result = analyze_email(body, subject)
    etype  = result["type"]

    print(f"TYPE     : {etype.upper()}\n")
    print(json.dumps(result, indent=4), "\n")

    # ==================================================
    # HANDLE: NEW EVENT → add entries to calendar.json
    # ==================================================
    if etype == "event":
        if not result["dates"]:
            print("  WARNING: No dates extracted — skipping calendar entry.")
        else:
            print("CALENDAR ACTIONS:")
            for date in result["dates"]:
                add_event({
                    "title":     result["event"],
                    "date":      date,
                    "from_time": result["from_time"],
                    "to_time":   result["to_time"],
                    "venue":     result["venue"],
                    "link":      result.get("link"),
                    "status":    "schedule",
                })

    # ==================================================
    # HANDLE: RESCHEDULE → update entries in calendar.json
    # ==================================================
    elif etype == "reschedule":
        print("RESCHEDULE DETECTED — searching calendar for matching event...")
        match = find_matching_event(result["event"], result["old_dates"])
        print("\nCALENDAR ACTIONS:")
        if match:
            print(f"  Matched: '{match['title']}' on {match['date']}")
            update_event(
                old=match,
                new_dates=result["dates"],
                from_time=result["from_time"],
                to_time=result["to_time"],
                venue=result["venue"],
                link=result.get("link"),
            )
        else:
            print("  No existing event matched — adding as new entry (flagged)")
            for date in result["dates"]:
                add_event({
                    "title":     f"[POSSIBLY RESCHEDULED] {result['event']}",
                    "date":      date,
                    "from_time": result["from_time"],
                    "to_time":   result["to_time"],
                    "venue":     result["venue"],
                    "link":      result.get("link"),
                    "status":    "reschedule",
                })
        print("\nANNOUNCEMENT ACTIONS:")
        save_announcement(
            sender_email=sender_email,
            subject=subject,
            description=result.get("description") or f"Event '{result['event']}' has been rescheduled.",
        )

    # ==================================================
    # HANDLE: CANCELLATION → delete entries from calendar.json
    # ==================================================
    elif etype == "cancellation":
        print("CANCELLATION DETECTED — searching calendar for matching event...")
        match = find_matching_event(result["event"], result["old_dates"])
        print("\nCALENDAR ACTIONS:")
        if match:
            print(f"  Matched: '{match['title']}' on {match['date']} — deleting.")
            delete_event(match)
        else:
            print("  No existing event matched for cancellation.")
        print("\nANNOUNCEMENT ACTIONS:")
        save_announcement(
            sender_email=sender_email,
            subject=subject,
            description=result.get("description"),
        )

    # ==================================================
    # HANDLE: ANNOUNCEMENT → save to announcements.json only
    # ==================================================
    elif etype == "announcement":
        print("ANNOUNCEMENT — not added to calendar.")
        print(f"Summary  : {result.get('description')}\n")
        print("ANNOUNCEMENT ACTIONS:")
        save_announcement(
            sender_email=sender_email,
            subject=subject,
            description=result.get("description"),
        )

    else:
        print("UNKNOWN TYPE — could not process this email.")

    print(f"\n{'=' * 50}\n")

# =========================================================
# IMAP INTEGRATION  [commented out — prototype mode]
# =========================================================
# Uncomment when transitioning out of prototype mode:
#
# import imaplib, email
# from email.header import decode_header
# from bs4 import BeautifulSoup
#
# mail = imaplib.IMAP4_SSL(IMAP_SERVER)
# mail.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
# mail.select("inbox")
# status, messages = mail.search(None, "UNSEEN")
# for latest_email_id in messages[0].split():
#     status, msg_data = mail.fetch(latest_email_id, "(RFC822)")
#     for response_part in msg_data:
#         if not isinstance(response_part, tuple):
#             continue
#         msg = email.message_from_bytes(response_part[1])
#         raw_subject = msg["Subject"]
#         decoded_subject = decode_header(raw_subject)
#         subject = "".join(
#             part.decode(enc or "utf-8", errors="ignore") if isinstance(part, bytes) else part
#             for part, enc in decoded_subject
#         )
#         email_body = ""
#         if msg.is_multipart():
#             for part in msg.walk():
#                 ct = part.get_content_type()
#                 payload = part.get_payload(decode=True)
#                 if not payload:
#                     continue
#                 text = payload.decode(errors="ignore")
#                 if ct == "text/plain":
#                     email_body = text; break
#                 elif ct == "text/html" and not email_body:
#                     email_body = BeautifulSoup(text, "html.parser").get_text()
#         else:
#             payload = msg.get_payload(decode=True)
#             if payload:
#                 email_body = payload.decode(errors="ignore")
#         process_email(msg["From"], msg["To"], subject, email_body)
# mail.logout()

# =========================================================
# TEST EMAIL SUITE
# Each tuple: (sender, receiver, subject, body)
# Run in order — emails 3, 8, and 12 depend on prior entries.
# =========================================================

TEST_EMAILS = [
    # ── 1: Single Day Event ──────────────────────────────
    (
        "hod@gmail.com", "receiver@gmail.com",
        "Seminar on Artificial Intelligence and Future Technologies",
        """Dear All,
Greetings!!!
You are cordially invited to attend the Seminar on Artificial Intelligence and Future
Technologies organized by the Department of Computer Science and Engineering, SRMIST.
The seminar will be held on 14 June 2026 at 10:00 AM at the Seminar Hall, CSE Block A,
SRMIST, Kattankulathur.
All faculty members and students are requested to attend and make use of this opportunity.
Thanks and Regards,
Dr. A. Ramesh
Head of Department
Department of Computer Science and Engineering
SRMIST, Kattankulathur""",
    ),

    # ── 2: Multi-Day Event ───────────────────────────────
    (
        "dean@gmail.com", "receiver@gmail.com",
        "National Level Technical Symposium - Technovanza 2026",
        """Dear All,
Greetings!!!
We are pleased to invite you to Technovanza 2026, the National Level Technical Symposium
organized by SRMIST.
The symposium will be conducted from July 10 to July 12, 2026 starting at 09:00 AM each
day at the Main Auditorium, SRMIST, Kattankulathur.
The event will feature paper presentations, hackathons, project expos, and guest lectures.
All departments are requested to encourage maximum student participation.
Thanks and Regards,
Dr. P. Venkatesh
Dean, Faculty of Engineering and Technology
SRMIST, Kattankulathur""",
    ),

    # ── 3: Rescheduled Event (requires Email 1 in calendar) ──
    (
        "hod@gmail.com", "receiver@gmail.com",
        "Rescheduled - Seminar on Artificial Intelligence and Future Technologies",
        """Dear All,
Greetings!!!
This is to inform you that the Seminar on Artificial Intelligence and Future Technologies
which was originally scheduled on 14 June 2026 has been rescheduled due to unavoidable
circumstances.
The seminar will now be held on 20 June 2026 at 10:00 AM at the same venue, Seminar Hall,
CSE Block A, SRMIST, Kattankulathur.
We regret the inconvenience caused and request all to update your calendars accordingly.
Thanks and Regards,
Dr. A. Ramesh
Head of Department
Department of Computer Science and Engineering
SRMIST, Kattankulathur""",
    ),

    # ── 4: Announcement (No Event) ───────────────────────
    (
        "dean@gmail.com", "receiver@gmail.com",
        "Holiday Notice - Institution Closed on June 17, 2026",
        """Dear All,
This is to inform all faculty, staff, and students that the institution will remain
closed on June 17, 2026 on account of a public holiday.
All scheduled activities, classes, and lab sessions for that day stand cancelled.
Please plan accordingly.
For any urgent matters, kindly contact your respective department offices.
Thanks and Regards,
Administrative Office
SRMIST, Kattankulathur""",
    ),

    # ── 5: Two Dates (Deadline + Event Date) ─────────────
    (
        "lekhalokare.28@gmail.com", "receiver@gmail.com",
        "Workshop on Cyber Security and Ethical Hacking",
        """Dear All,
Greetings!!!
The Department of Information Technology is organizing a Workshop on Cyber Security and
Ethical Hacking for final year students.
Registration Deadline: June 5, 2026 — Interested students must register before this date
via the department office.
The workshop will be conducted on June 18, 2026 at 09:30 AM at the IT Seminar Hall,
IT Block, SRMIST, Kattankulathur.
Seats are limited. Early registration is encouraged.
Thanks and Regards,
Dr. S. Kavitha
Department of Information Technology
SRMIST, Kattankulathur""",
    ),

    # ── 6: Unauthorized Sender (should be skipped entirely) ──
    (
        "randomstudent@gmail.com", "receiver@gmail.com",
        "Party at Hostel Block C",
        """Hey everyone,
There is a party happening at Hostel Block C on June 20, 2026 at 08:00 PM.
Everyone is welcome. Bring snacks!
Cheers""",
    ),

    # ── 7: Event with No Venue Mentioned ─────────────────
    (
        "hod@gmail.com", "receiver@gmail.com",
        "Guest Lecture on Quantum Computing",
        """Dear All,
Greetings!!!
You are cordially invited to attend a Guest Lecture on Quantum Computing by Dr. Rajesh
Sharma, Senior Scientist, ISRO.
The lecture will be held on July 5, 2026 at 02:00 PM.
All faculty and students of the Department of Physics and Computer Science are requested
to attend.
Thanks and Regards,
Dr. M. Krishnan
Department of Physics
SRMIST, Kattankulathur""",
    ),

    # ── 8: Cancellation (requires Email 5 in calendar) ───
    (
        "dean@gmail.com", "receiver@gmail.com",
        "Cancellation - Workshop on Cyber Security and Ethical Hacking",
        """Dear All,
Greetings!!!
We regret to inform you that the Workshop on Cyber Security and Ethical Hacking scheduled
on June 18, 2026 has been cancelled due to unavoidable circumstances.
We apologize for the inconvenience caused. A new date will be announced shortly.
Thanks and Regards,
Dr. P. Venkatesh
Dean, Faculty of Engineering and Technology
SRMIST, Kattankulathur""",
    ),

    # ── 9: Three-Day Event ───────────────────────────────
    (
        "dean@gmail.com", "receiver@gmail.com",
        "Annual Sports Meet - Sportanza 2026",
        """Dear All,
Greetings!!!
We are excited to announce the Annual Sports Meet - Sportanza 2026 organized by the Sports
and Physical Education Department, SRMIST.
The event will be conducted from August 3 to August 5, 2026 starting at 08:00 AM each day
at the University Sports Complex, SRMIST, Kattankulathur.
Events include athletics, basketball, volleyball, cricket, badminton, and chess.
All students and faculty are encouraged to participate and support their departments.
Thanks and Regards,
Dr. R. Subramaniam
Director, Sports and Physical Education
SRMIST, Kattankulathur""",
    ),

    # ── 10: Announcement with a Date (not an event) ──────
    (
        "hod@gmail.com", "receiver@gmail.com",
        "Internal Assessment Marks Submission Deadline",
        """Dear Faculty,
This is to inform all faculty members that the Internal Assessment marks for the odd
semester 2026 must be submitted to the examination cell on or before July 10, 2026.
Faculty who fail to submit marks before the deadline will have their marks locked by
the system automatically.
Kindly ensure timely submission to avoid any inconvenience.
Thanks and Regards,
Dr. A. Ramesh
Head of Department
Department of Computer Science and Engineering
SRMIST, Kattankulathur""",
    ),

    # ── 11: Vague Email — "tomorrow" resolves to today + 1 day ──
    (
        "lekhalokare.28@gmail.com", "receiver@gmail.com",
        "Meeting Tomorrow",
        """Dear All,
There will be a departmental meeting tomorrow at 11:00 AM at the Conference Room,
Admin Block.
Attendance is mandatory for all faculty members.
Regards,
Dr. S. Kavitha""",
    ),

    # ── 12: Reschedule + Venue Change (requires Email 9) ─
    (
        "dean@gmail.com", "receiver@gmail.com",
        "Rescheduled and Venue Changed - Annual Sports Meet Sportanza 2026",
        """Dear All,
Greetings!!!
Please note that the Annual Sports Meet - Sportanza 2026 originally scheduled from
August 3 to August 5, 2026 at the University Sports Complex has been rescheduled.
The event will now be held from August 10 to August 12, 2026 at 08:00 AM at the
SRMIST Cricket Ground and Open Arena, Kattankulathur.
We apologize for the inconvenience and request everyone to update their schedules.
Thanks and Regards,
Dr. R. Subramaniam
Director, Sports and Physical Education
SRMIST, Kattankulathur""",
    ),
]

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    for sender, receiver, subject, body in TEST_EMAILS:
        process_email(sender, receiver, subject, body)

    print("Prototype run complete.")
    print(f"Events saved to        : {CALENDAR_FILE}")
    print(f"Announcements saved to : {ANNOUNCEMENTS_FILE}")
