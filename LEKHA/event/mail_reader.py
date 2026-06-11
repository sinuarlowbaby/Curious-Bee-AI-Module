# =========================================================
# SECURE EVENT EXTRACTION MAIL READER — FULL EDGE CASE VERSION
# No LLM/SLM required. Uses:
#   - GLiNER          (NER: event/date/time/venue)
#   - SentenceTransformer (reschedule matching)
#   - dateutil        (date parsing)
#   - BeautifulSoup   (HTML email parsing)
#   - pytesseract     (OCR for image-only emails)  [optional]
#   - regex           (all structural patterns)
# =========================================================

import imaplib
import email
import re
import json
import os
import sys
import datetime
import dateutil.parser
import dateutil.relativedelta
import hashlib
import logging
from email.header import decode_header
from email.utils import parseaddr

from dotenv import load_dotenv
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
from gliner import GLiNER

# Google Calendar
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pickle

# Optional OCR (image-only emails)
try:
    import pytesseract
    from PIL import Image
    import io
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

load_dotenv()

# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("mail_reader.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger(__name__)

# =========================================================
# CONFIGURATION
# =========================================================

SCOPES              = ["https://www.googleapis.com/auth/calendar"]
EMAIL_ADDRESS       = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD      = os.getenv("EMAIL_PASSWORD")
IMAP_SERVER         = "imap.gmail.com"
CALENDAR_FILE       = "calendar.json"
ANNOUNCEMENTS_FILE  = "announcements.json"
PROCESSED_IDS_FILE  = "processed_ids.json"   # duplicate guard
TIMEZONE            = "Asia/Kolkata"

AUTHORIZED_SENDERS = [
    "dean@gmail.com",
    "hod@gmail.com",
    "lekhalokare.28@gmail.com",
    "yashwanthkumar0812@gmail.com"
]

# Vague time-of-day fallbacks
VAGUE_TIME_MAP = {
    "morning":   "09:00 AM",
    "afternoon": "02:00 PM",
    "evening":   "05:00 PM",
    "noon":      "12:00 PM",
    "midnight":  "12:00 AM",
    "night":     "07:00 PM",
}

# =========================================================
# MODELS  (loaded once at startup)
# =========================================================

log.info("Loading GLiNER model...")
gliner_model = GLiNER.from_pretrained("urchade/gliner_small-v2.1")
GLINER_LABELS = ["event", "date", "time", "venue"]

log.info("Loading SentenceTransformer model...")
sim_model = SentenceTransformer("all-MiniLM-L6-v2")
log.info("Models ready.\n")

# =========================================================
# PROCESSED-IDS  (duplicate email guard)
# =========================================================

def load_processed_ids():
    if not os.path.exists(PROCESSED_IDS_FILE):
        return set()
    with open(PROCESSED_IDS_FILE, "r", encoding="utf-8") as f:
        return set(json.load(f))

def save_processed_id(uid):
    ids = load_processed_ids()
    ids.add(uid)
    with open(PROCESSED_IDS_FILE, "w", encoding="utf-8") as f:
        json.dump(list(ids), f)

def email_fingerprint(sender, subject, body):
    """Hash sender+subject+first 300 chars of body — catches re-delivered duplicates."""
    raw = f"{sender}|{subject}|{body[:300]}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()

# =========================================================
# CALENDAR / ANNOUNCEMENTS STORE
# =========================================================

def load_calendar():
    if not os.path.exists(CALENDAR_FILE):
        return []
    with open(CALENDAR_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_calendar(entries):
    with open(CALENDAR_FILE, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=4, ensure_ascii=False)

def calendar_event_exists(title, date):
    """Return True if an event with this exact title+date is already stored."""
    return any(
        e["title"].strip().lower() == title.strip().lower() and e["date"] == date
        for e in load_calendar()
    )

def add_event(entry):
    entry["title"] = _clean_text(entry["title"])
    if calendar_event_exists(entry["title"], entry["date"]):
        log.info(f"  [SKIP-DUP] '{entry['title']}' on {entry['date']} already exists.")
        return False
    cal = load_calendar()
    cal.append(entry)
    save_calendar(cal)
    log.info(f"  [ADDED]   '{entry['title']}' on {entry['date']}")
    return True

def update_event(old_entry, old_dates, new_dates, new_from_time, new_to_time, new_venue, new_link=None):
    cal = load_calendar()
    cal = [
        e for e in cal
        if not (e["title"] == old_entry["title"] and e["date"] in old_dates)
    ]
    for date in new_dates:
        new_entry = {
            "title":     old_entry["title"],
            "date":      date,
            "from_time": new_from_time or old_entry.get("from_time"),
            "to_time":   new_to_time   or old_entry.get("to_time"),
            "venue":     new_venue     or old_entry.get("venue"),
            "link":      new_link      or old_entry.get("link"),
            "status":    "reschedule"
        }
        cal.append(new_entry)
        log.info(f"  [UPDATED] '{old_entry['title']}' → {date}")
    save_calendar(cal)

def delete_event(old_entry):
    cal = load_calendar()
    cal = [
        e for e in cal
        if not (e["title"] == old_entry["title"] and e["date"] == old_entry["date"])
    ]
    save_calendar(cal)
    log.info(f"  [DELETED] '{old_entry['title']}' on {old_entry['date']}")

def save_announcement(sender_email, subject, description):
    anns = []
    if os.path.exists(ANNOUNCEMENTS_FILE):
        with open(ANNOUNCEMENTS_FILE, "r", encoding="utf-8") as f:
            anns = json.load(f)
    anns.append({
        "sender":      sender_email,
        "subject":     _clean_text(subject),
        "description": description,
        "received_on": datetime.date.today().strftime("%Y-%m-%d")
    })
    with open(ANNOUNCEMENTS_FILE, "w", encoding="utf-8") as f:
        json.dump(anns, f, indent=4, ensure_ascii=False)
    log.info(f"  [SAVED]   Announcement stored.")

def find_matching_event(event_name, old_dates=None, threshold=0.75):
    cal = load_calendar()
    if not cal:
        return None
    q_emb = sim_model.encode(event_name, convert_to_tensor=True)
    best, best_score = None, 0
    for e in cal:
        e_emb = sim_model.encode(e["title"], convert_to_tensor=True)
        score = util.cos_sim(q_emb, e_emb).item()
        if old_dates and e["date"] in old_dates:
            score += 0.15
        if score > best_score:
            best_score, best = score, e
    log.info(f"  Similarity: {best_score:.2f} — {'matched: ' + best['title'] if best_score >= threshold else 'no confident match'}")
    return best if best_score >= threshold else None

# =========================================================
# TEXT UTILITIES
# =========================================================

def _clean_text(text):
    text = re.sub(r"[\r\n\t]+", " ", text or "")
    return re.sub(r"\s+", " ", text).strip()

def _normalize_dashes(text):
    return re.sub(r"[\u2013\u2014\u2015\u2212\ufe58\ufe63\uff0d]", "-", text)

def strip_reply_chain(body):
    """
    Remove quoted/forwarded blocks from reply/forward chains.
    Keeps only the topmost (latest) message.
    """
    # Common quoted-block starters
    patterns = [
        r"\n-{3,}.*?Original Message.*?-{3,}",
        r"\nOn .+? wrote:",
        r"\n>.*",                            # "> quoted line"
        r"\n_{3,}",                          # ___ divider
        r"\nFrom:.*\nSent:.*\nTo:",          # Outlook forward header
        r"\n-{3,} Forwarded message -{3,}",
    ]
    for p in patterns:
        body = re.split(p, body, maxsplit=1, flags=re.IGNORECASE | re.DOTALL)[0]
    return body.strip()

def extract_main_body(body_text):
    text = body_text.strip()
    text = re.sub(
        r"^(dear\b[^.!?:,\n]*[:.,!]*\s*|hi\b[^.!?:,\n]*[:.,!]*\s*|"
        r"hello\b[^.!?:,\n]*[:.,!]*\s*|greetings[^.!?:,\n]*[:.,!]*\s*|"
        r"to\s+all\b[^.!?:,\n]*[:.,!]*\s*|respected\b[^.!?:,\n]*[:.,!]*\s*)+",
        "", text, flags=re.IGNORECASE
    ).strip()
    signoff = re.search(
        r"\b(thanks\b|thank you\b|with regards\b|best regards\b|"
        r"warm regards\b|regards\b|yours sincerely\b|sincerely\b|cheers\b)",
        text, flags=re.IGNORECASE
    )
    if signoff:
        text = text[:signoff.start()].strip()
    return text.strip(" ,;")

# =========================================================
# EMAIL BODY EXTRACTION
# =========================================================

def extract_body_from_message(msg):
    """
    Extract plain-text body from email.message.Message.
    Handles: multipart, HTML-only, image-only (OCR), forwarded chains.
    """
    plain_body = ""
    html_body  = ""
    image_parts = []

    if msg.is_multipart():
        for part in msg.walk():
            ctype    = part.get_content_type()
            charset  = part.get_content_charset() or "utf-8"
            payload  = part.get_payload(decode=True)
            if not payload:
                continue
            try:
                decoded = payload.decode(charset, errors="ignore")
            except Exception:
                decoded = payload.decode("utf-8", errors="ignore")

            if ctype == "text/plain" and not plain_body:
                plain_body = decoded
            elif ctype == "text/html" and not html_body:
                html_body = decoded
            elif ctype.startswith("image/") and OCR_AVAILABLE:
                image_parts.append(payload)
    else:
        payload = msg.get_payload(decode=True)
        ctype   = msg.get_content_type()
        charset = msg.get_content_charset() or "utf-8"
        if payload:
            try:
                text = payload.decode(charset, errors="ignore")
            except Exception:
                text = payload.decode("utf-8", errors="ignore")
            if ctype == "text/html":
                html_body = text
            else:
                plain_body = text

    # Prefer plain text; fall back to HTML→text conversion
    if plain_body:
        body = plain_body
    elif html_body:
        soup = BeautifulSoup(html_body, "html.parser")
        # Extract href links before stripping tags
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.startswith("http"):
                a.replace_with(f" {a.get_text()} {href} ")
        # Extract text from table cells with newlines
        for tag in soup.find_all(["td", "th", "tr", "br", "p", "li", "div"]):
            tag.append("\n")
        body = soup.get_text(separator=" ")
    elif image_parts and OCR_AVAILABLE:
        # OCR fallback for image-only emails
        body_parts = []
        for img_bytes in image_parts:
            try:
                img = Image.open(io.BytesIO(img_bytes))
                text = pytesseract.image_to_string(img)
                body_parts.append(text)
            except Exception:
                pass
        body = "\n".join(body_parts)
    else:
        body = ""

    # Normalize whitespace
    body = re.sub(r"[ \t\r]+", " ", body)
    body = re.sub(r"\n{3,}", "\n\n", body)

    # Strip reply/forward chains
    body = strip_reply_chain(body)

    return body.strip()

# =========================================================
# DATE UTILITIES
# =========================================================

MONTH_NAMES = (
    "jan", "feb", "mar", "apr", "may", "jun",
    "jul", "aug", "sep", "oct", "nov", "dec"
)

def is_likely_date(text):
    t = text.lower().strip()
    if any(m in t for m in MONTH_NAMES):
        return True
    if re.search(r"\d{1,2}[/\-]\d{1,2}", t):
        return True
    return False

def is_likely_time(text):
    t = text.lower()
    if any(c.isdigit() for c in t):
        return True
    if any(kw in t for kw in ["am", "pm", "noon", "midnight", "hrs", "hours"]):
        return True
    return False

def resolve_relative_date(text, reference_date=None):
    """
    Convert relative date expressions to absolute datetime.date.
    e.g. 'tomorrow', 'next Monday', 'this Friday'
    """
    ref = reference_date or datetime.date.today()
    text = text.lower().strip()

    if text in ("today",):
        return ref
    if text in ("tomorrow",):
        return ref + datetime.timedelta(days=1)
    if text in ("day after tomorrow",):
        return ref + datetime.timedelta(days=2)
    if text in ("yesterday",):
        return ref - datetime.timedelta(days=1)

    # "next <weekday>" / "this <weekday>"
    weekdays = ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"]
    for i, wd in enumerate(weekdays):
        if wd in text:
            days_ahead = (i - ref.weekday()) % 7
            if "next" in text:
                days_ahead = days_ahead if days_ahead > 0 else 7
            elif "this" in text:
                days_ahead = days_ahead if days_ahead >= 0 else days_ahead + 7
            else:
                days_ahead = days_ahead if days_ahead > 0 else days_ahead + 7
            return ref + datetime.timedelta(days=days_ahead)

    return None

def parse_date_safe(text, fuzzy=True):
    """
    Try to parse a date string. Returns datetime.date or None.
    First tries relative, then dateutil.
    """
    rel = resolve_relative_date(text)
    if rel:
        return rel
    if not is_likely_date(text):
        return None
    try:
        return dateutil.parser.parse(text, fuzzy=fuzzy).date()
    except Exception:
        return None

def expand_date_range(start_str, end_str):
    """Return list of 'YYYY-MM-DD' strings for every day in [start, end]."""
    try:
        s = parse_date_safe(start_str)
        e = parse_date_safe(end_str)
        if not s or not e or s > e:
            return []
        out = []
        cur = s
        while cur <= e:
            out.append(cur.strftime("%Y-%m-%d"))
            cur += datetime.timedelta(days=1)
        return out
    except Exception:
        return []

def parse_date_list(text):
    """
    Parse comma/ampersand separated date lists.
    e.g. 'June 20, 21 & 22' → ['2026-06-20','2026-06-21','2026-06-22']
    Infers the month/year from context for bare day numbers.
    """
    results = []

    # Normalize separators
    text = re.sub(r"&|and", ",", text, flags=re.IGNORECASE)
    parts = [p.strip() for p in text.split(",") if p.strip()]

    # Find a "base" month/year from the first part that has one
    base_dt = None
    for part in parts:
        if is_likely_date(part):
            try:
                base_dt = dateutil.parser.parse(part, fuzzy=True)
                break
            except Exception:
                pass

    for part in parts:
        part = part.strip().rstrip(".")
        # Bare day number e.g. "21", "22nd"
        bare = re.match(r"^(\d{1,2})(?:st|nd|rd|th)?$", part)
        if bare and base_dt:
            try:
                dt = base_dt.replace(day=int(bare.group(1)))
                results.append(dt.strftime("%Y-%m-%d"))
            except Exception:
                pass
            continue
        # Full or partial date
        d = parse_date_safe(part)
        if d:
            results.append(d.strftime("%Y-%m-%d"))

    return list(dict.fromkeys(results))   # preserve order, remove duplicates

# =========================================================
# TIME UTILITIES
# =========================================================

_DASH_PAT = r"[\u2013\u2014\u2015\u2212\ufe58\ufe63\uff0d\-]"

def normalize_time_str(t):
    """Standardize time strings: '10.00AM' → '10:00 AM', '1000 hrs' → '10:00 AM'"""
    t = t.strip()
    # Military: '1030 hrs' / '1030 to 1230'
    mil = re.match(r"^(\d{3,4})\s*(?:hrs?|h)?$", t, re.IGNORECASE)
    if mil:
        raw = mil.group(1).zfill(4)
        h, m = int(raw[:2]), int(raw[2:])
        try:
            return datetime.time(h, m).strftime("%I:%M %p")
        except Exception:
            pass
    # Dot separator: '10.30 AM'
    t = re.sub(r"(\d{1,2})\.(\d{2})\s*(AM|PM)", r"\1:\2 \3", t, flags=re.IGNORECASE)
    # Missing space before AM/PM
    t = re.sub(r"(\d)(AM|PM)", r"\1 \2", t, flags=re.IGNORECASE)
    # Uppercase AM/PM
    t = re.sub(r"\b(am|pm)\b", lambda m: m.group(0).upper(), t)
    return t.strip()

def split_time_range(val):
    """
    Split 'HH:MM AM to HH:MM PM' or 'HH:MM AM - HH:MM PM' into (from, to).
    Returns (from_time, to_time) — to_time may be None.
    """
    val_clean = re.sub(r"(?i)\s+onwards", "", val).strip()
    split_pat = r"\s*(?:" + _DASH_PAT + r"|to|till|until|–)\s*"
    parts = re.split(split_pat, val_clean, maxsplit=1, flags=re.IGNORECASE)
    if len(parts) == 2:
        ft = normalize_time_str(parts[0].strip())
        tt = normalize_time_str(parts[1].strip())
        return ft, tt
    return normalize_time_str(val_clean), None

def resolve_vague_time(text):
    """Return a clock time for vague words like 'morning', 'afternoon'."""
    for kw, t in VAGUE_TIME_MAP.items():
        if kw in text.lower():
            return t
    return None

# =========================================================
# REPORTING TIME
# =========================================================

# Time-value pattern reused across all reporting patterns
_TIME_VAL = r"\d{1,2}[:.]\d{2}\s*(?:AM|PM)|\d{1,2}\s*(?:AM|PM)"

REPORTING_TIME_PATTERNS = [
    # "Reporting Time: 9:00 AM"  /  "Reporting time - 9 AM"
    rf"reporting\s+time\s*[:–\-]\s*({_TIME_VAL})",
    # "Report by / Report at 9:00 AM"
    rf"\breport\s+(?:by|at)\s+({_TIME_VAL})",
    # "Please report at / by 9:00 AM"
    rf"please\s+report\s+(?:by|at)\s+({_TIME_VAL})",
    # "Arrive by / Arrive at 9:30 AM"
    rf"\barrive\s+(?:by|at)\s+({_TIME_VAL})",
    # "Be present by / at 9:00 AM"
    rf"be\s+present\s+(?:by|at)\s+({_TIME_VAL})",
    # "Assembly at / by 9:00 AM"
    rf"\bassembly\s+(?:at|by)\s+({_TIME_VAL})",
    # "Entry at 9:00 AM"
    rf"\bentry\s+(?:at|by)\s+({_TIME_VAL})",
    # "Gates open at 9:00 AM"
    rf"gates?\s+open\s+(?:at|by)\s+({_TIME_VAL})",
]

def extract_reporting_time(text):
    """
    Extract reporting/arrival time from email body or a sub-event block.
    Returns a normalized time string (e.g. '09:00 AM') or None.

    Deliberately does NOT raise — safe to call on any string.
    """
    try:
        for pat in REPORTING_TIME_PATTERNS:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                return normalize_time_str(m.group(1).strip())
    except Exception:
        pass
    return None

# =========================================================
# DEADLINE DETECTION
# =========================================================

DEADLINE_KEYWORDS = [
    # explicit deadline labels
    "deadline", "registration deadline", "submission deadline",
    "application deadline", "last date", "closing date", "due date",
    # "register/apply/enroll/submit by|before|on or before"
    "register before", "register by", "register on or before",
    "registration before", "registration by", "registration closes",
    "registration ends", "registration open until", "registration till",
    "apply before", "apply by", "apply on or before",
    "submit before", "submit by",
    "enroll before", "enroll by",
    # fill/complete/send form by
    "fill.*?form.*?by", "complete.*?form.*?by", "send.*?form.*?by",
    # generic "due by / due on"
    "due by", "due on",
    # "last day to register/apply/submit"
    "last day to register", "last day to apply", "last day to submit",
    "last day for registration",
    # "must register/apply by"
    "must register", "must apply", "must enroll",
    # "interested students.*register"  (common in college emails)
    "interested students.*register", "interested candidates.*register",
]

def sentence_containing(val, text):
    """Return the sentence from text that contains val."""
    sentences = re.split(r"(?<=[.!?\n])\s+", text)
    for s in sentences:
        if val.lower() in s.lower():
            return s.lower()
    return ""

def is_deadline_date(val, text):
    """
    Return True if the date string `val` appears in a sentence that looks
    like a registration/submission deadline rather than an event date.
    Uses both plain substring matching and regex for pattern-based keywords.
    Safe — never raises.
    """
    try:
        sentence = sentence_containing(val, text)
        if not sentence:
            return False
        for kw in DEADLINE_KEYWORDS:
            # keywords containing '.*' need regex; others use plain 'in'
            if ".*" in kw:
                if re.search(kw, sentence, re.IGNORECASE):
                    return True
            else:
                if kw in sentence:
                    return True
    except Exception:
        pass
    return False

# =========================================================
# SUBJECT LINE CLEANING
# =========================================================

def clean_subject(subject):
    # Strip Re:/Fwd:/FW: prefixes (multiple levels)
    subject = re.sub(r"^(?:Re|Fwd?|FW)\s*:\s*", "", subject, flags=re.IGNORECASE)
    subject = re.sub(r"^(?:Re|Fwd?|FW)\s*:\s*", "", subject, flags=re.IGNORECASE)
    # Strip [TAG] prefixes like [IMPORTANT], [URGENT]
    subject = re.sub(r"^\[.*?\]\s*", "", subject)
    # Strip pipe-separated trailing date/venue info commonly appended in subject
    subject = re.sub(r"\s*\|.*$", "", subject)
    return _clean_text(subject)

# =========================================================
# EMAIL CLASSIFICATION
# =========================================================

CANCEL_WORDS     = ["cancel", "called off", "will not be held", "not taking place", "postponed indefinitely"]
RESCHEDULE_WORDS = ["postpone", "reschedule", "rescheduled to", "new date", "shifted to", "moved to"]
REMINDER_WORDS   = ["reminder", "don't forget", "gentle reminder", "upcoming event", "as a reminder"]
EVENT_WORDS      = ["event", "seminar", "workshop", "talk", "symposium", "conference",
                    "hackathon", "competition", "exhibition", "session", "programme",
                    "invited", "organizing", "announcing", "schedule"]
UPDATE_WORDS     = ["venue changed", "venue has been changed", "time changed", "time has been updated",
                    "location changed", "new venue", "new time", "updated venue"]

def classify_email(text):
    t = text.lower()
    if any(w in t for w in CANCEL_WORDS):
        return "cancellation"
    if any(w in t for w in RESCHEDULE_WORDS):
        return "reschedule"
    if any(w in t for w in REMINDER_WORDS):
        return "reminder"
    if any(w in t for w in UPDATE_WORDS):
        return "partial_update"
    if any(w in t for w in EVENT_WORDS):
        return "event"
    return "announcement"

# =========================================================
# MULTI-EVENT / TABLE DETECTION
# =========================================================

def extract_html_table_events(html_body, subject=""):
    """
    Parse HTML <table> formatted event emails.
    Returns list of sub_event dicts or empty list.
    """
    if not html_body:
        return []
    soup = BeautifulSoup(html_body, "html.parser")
    sub_events = []

    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        if len(rows) < 2:
            continue
        headers = [td.get_text(strip=True).lower() for td in rows[0].find_all(["th", "td"])]
        key_map = {}
        for i, h in enumerate(headers):
            if any(k in h for k in ["event", "name", "title", "program"]):
                key_map["title"] = i
            elif any(k in h for k in ["date"]):
                key_map["date"] = i
            elif any(k in h for k in ["time", "from", "start"]):
                key_map["from_time"] = i
            elif any(k in h for k in ["to", "end", "till"]):
                key_map["to_time"] = i
            elif any(k in h for k in ["venue", "location", "place"]):
                key_map["venue"] = i

        if not key_map:
            continue

        for row in rows[1:]:
            cells = row.find_all(["td", "th"])
            if not cells:
                continue
            sub = {
                "title":     None,
                "date":      None,
                "from_time": None,
                "to_time":   None,
                "venue":     None
            }
            for field, idx in key_map.items():
                if idx < len(cells):
                    val = _clean_text(cells[idx].get_text())
                    if field == "date":
                        d = parse_date_safe(val)
                        sub["date"] = d.strftime("%Y-%m-%d") if d else None
                    elif field in ("from_time", "to_time"):
                        sub[field] = normalize_time_str(val) if val else None
                    else:
                        sub[field] = val

            if sub.get("title") and sub.get("date"):
                sub_events.append(sub)

    return sub_events

def extract_numbered_sub_events(text):
    """
    Extract numbered event blocks:
    'Event 1: Coding Competition\nDate: ...\nTime: ...\nVenue: ...'
    Also handles 'Session 1:', 'Day 1:', 'Programme 1:'
    """
    block_header = r"\n(?=(?:Event|Session|Day|Programme|Program|Activity|Round|Phase)\s*\d+\s*[:–\-])"
    blocks = re.split(block_header, text, flags=re.IGNORECASE)

    sub_events = []
    for block in blocks:
        if not re.match(r"(?:Event|Session|Day|Programme|Program|Activity|Round|Phase)\s*\d+\s*[:–\-]",
                        block.strip(), re.IGNORECASE):
            continue
        sub = _parse_labeled_block(block)
        if sub.get("title") and sub.get("date"):
            sub_events.append(sub)

    return sub_events

def extract_bullet_sub_events(text):
    """
    Detect bullet/dash-listed event blocks:
    '• Coding Competition — 12 Aug, 10 AM, Lab 1'
    """
    pattern = re.compile(
        r"^[\•\-\*\u2022]\s*(.+?)(?:\s*[–\-—]\s*|\s*,\s*)(\d.+?)(?:\s*[–\-—,]\s*(.+?))?(?:\s*[–\-—,]\s*(.+?))?$",
        re.MULTILINE
    )
    sub_events = []
    for m in pattern.finditer(text):
        title      = _clean_text(m.group(1))
        date_chunk = _clean_text(m.group(2)) if m.group(2) else ""
        time_chunk = _clean_text(m.group(3)) if m.group(3) else ""
        venue      = _clean_text(m.group(4)) if m.group(4) else None

        d = parse_date_safe(date_chunk)
        if not title or not d:
            continue

        from_t, to_t = None, None
        if time_chunk:
            from_t, to_t = split_time_range(time_chunk)

        sub_events.append({
            "title":     title,
            "date":      d.strftime("%Y-%m-%d"),
            "from_time": from_t,
            "to_time":   to_t,
            "venue":     venue
        })

    return sub_events

def _parse_labeled_block(block):
    """
    Parse a single labeled block with Date:/Time:/Venue: fields.
    Returns a sub_event dict.
    """
    sub = {"title": None, "date": None, "from_time": None, "to_time": None, "venue": None}

    # Title: first line after the block header
    title_m = re.match(
        r"(?:Event|Session|Day|Programme|Program|Activity|Round|Phase)\s*\d+\s*[:–\-]\s*(.+)",
        block.strip(), re.IGNORECASE
    )
    if title_m:
        sub["title"] = _clean_text(title_m.group(1))

    # Date
    date_m = re.search(r"(?:Date|On)\s*[:–\-]\s*(.+?)(?:\n|$)", block, re.IGNORECASE)
    if date_m:
        raw = date_m.group(1).strip()
        # Could be a range: "12-14 Aug"
        if re.search(r"\d\s*(?:to|–|\-)\s*\d", raw, re.IGNORECASE):
            parts = re.split(r"\s*(?:to|–|\-)\s*", raw, maxsplit=1, flags=re.IGNORECASE)
            if len(parts) == 2:
                dates = expand_date_range(parts[0], parts[1])
                if dates:
                    sub["date"] = dates[0]   # first day of range
                    sub["_all_dates"] = dates
        else:
            d = parse_date_safe(raw)
            if d:
                sub["date"] = d.strftime("%Y-%m-%d")

    # Time
    time_m = re.search(
        r"Time\s*[:–\-]\s*(.+?)(?:\n|$)", block, re.IGNORECASE
    )
    if time_m:
        raw_t = time_m.group(1).strip()
        sub["from_time"], sub["to_time"] = split_time_range(raw_t)
    else:
        # Military time fallback
        mil_m = re.search(
            r"(\d{3,4})\s*(?:to|–|\-)\s*(\d{3,4})\s*(?:hrs?)?",
            block, re.IGNORECASE
        )
        if mil_m:
            sub["from_time"] = normalize_time_str(mil_m.group(1))
            sub["to_time"]   = normalize_time_str(mil_m.group(2))
        else:
            # Vague time
            for kw in VAGUE_TIME_MAP:
                if kw in block.lower():
                    sub["from_time"] = VAGUE_TIME_MAP[kw]
                    break

    # Venue
    venue_m = re.search(r"(?:Venue|Location|Place|At|Held at)\s*[:–\-]\s*(.+?)(?:\n|$)", block, re.IGNORECASE)
    if venue_m:
        sub["venue"] = _clean_text(venue_m.group(1))

    # Reporting time (per sub-event block)
    rt = extract_reporting_time(block)
    if rt:
        sub["reporting_time"] = rt

    return sub

# =========================================================
# PARTIAL UPDATE HANDLER
# =========================================================

def parse_partial_update(text):
    """
    Detect emails that only update venue or time for an existing event.
    Returns dict with changed fields, or None.
    """
    update = {}

    # Venue change
    vm = re.search(
        r"(?:venue|location)\s+(?:has been\s+)?(?:changed|updated|shifted)\s+to\s+[:\-]?\s*(.+?)(?:\.|$|\n)",
        text, re.IGNORECASE
    )
    if vm:
        update["venue"] = _clean_text(vm.group(1))

    # New venue stated directly
    nv = re.search(r"new\s+venue\s*[:\-]\s*(.+?)(?:\.|$|\n)", text, re.IGNORECASE)
    if nv:
        update["venue"] = _clean_text(nv.group(1))

    # Time change
    tm = re.search(
        r"(?:time)\s+(?:has been\s+)?(?:changed|updated)\s+to\s+[:\-]?\s*(.+?)(?:\.|$|\n)",
        text, re.IGNORECASE
    )
    if tm:
        raw_t = tm.group(1).strip()
        ft, tt = split_time_range(raw_t)
        update["from_time"] = ft
        update["to_time"]   = tt

    return update if update else None

# =========================================================
# URL / LINK EXTRACTION
# =========================================================

REG_LINK_KEYWORDS = ["form", "unstop", "hackathon", "register", "apply", "ticket", "eventbrite", "devfolio", "lu.ma"]
REG_CONTEXT_KEYWORDS = ["register", "registration", "apply", "join", "here", "link", "enroll"]

def extract_registration_link(text, html_body=""):
    """
    Extract registration/form links from plain text and HTML href attributes.
    """
    url_pat = r'https?://[^\s<>\"\')\]()]+'
    links = list(dict.fromkeys(re.findall(url_pat, text)))

    # Also grab hrefs from HTML
    if html_body:
        soup = BeautifulSoup(html_body, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if href.startswith("http"):
                links.append(href)

    links = list(dict.fromkeys(links))

    for link in links:
        ll = link.lower()
        if any(kw in ll for kw in REG_LINK_KEYWORDS):
            return link
        idx = text.find(link)
        if idx != -1:
            ctx = text[max(0, idx - 50):idx].lower()
            if any(kw in ctx for kw in REG_CONTEXT_KEYWORDS):
                return link

    return None

# =========================================================
# CORE EMAIL ANALYZER
# =========================================================

def analyze_email(body, subject, html_body=""):
    """
    Full email analysis. Returns structured dict with:
    type, event, dates, old_dates, from_time, to_time, venue,
    link, description, sub_events (optional).
    """
    try:
        subject = clean_subject(subject)
        body    = _normalize_dashes(body)

        # Combined text for NER + classification
        text = f"Subject: {subject}\n\n{body}"

        email_type = classify_email(text)

        result = {
            "type":           email_type,
            "event":          subject,
            "dates":          [],
            "old_dates":      [],
            "from_time":      None,
            "to_time":        None,
            "reporting_time": None,
            "venue":          None,
            "link":           None,
            "description":    None,
            "sub_events":     []
        }

        # ==================================================
        # STEP 1 — GLiNER NER pass
        # ==================================================
        entities = gliner_model.predict_entities(text[:2000], GLINER_LABELS)

        extracted_date_entities = []

        for ent in entities:
            label = ent["label"]
            val   = ent["text"]

            if label == "event" and result["event"] == subject:
                result["event"] = val

            elif label == "date":
                if is_likely_date(val) and not is_deadline_date(val, text):
                    extracted_date_entities.append(ent)

            elif label == "time" and not result["from_time"]:
                if is_likely_time(val):
                    val = _clean_text(val)
                    result["from_time"], result["to_time"] = split_time_range(val)

            elif label == "venue" and not result["venue"]:
                venue_pat = re.escape(val) + r"(?:,\s*[^.\n]+)*"
                m = re.search(venue_pat, text)
                result["venue"] = _clean_text(m.group(0)) if m else val

        # ==================================================
        # STEP 2 — Explicit labeled field fallbacks
        # ==================================================

        # Time fallback (regex)
        if not result["from_time"]:
            # Standard range: 10:00 AM to 12:30 PM
            tm = re.search(
                r"(\d{1,2}[:.]\d{2}\s*(?:AM|PM))\s*(?:to|–|\-)\s*(\d{1,2}[:.]\d{2}\s*(?:AM|PM))",
                text, re.IGNORECASE
            )
            if tm:
                result["from_time"] = normalize_time_str(tm.group(1))
                result["to_time"]   = normalize_time_str(tm.group(2))
            else:
                # Military range: 1000 to 1230 hrs
                mil = re.search(
                    r"(\d{3,4})\s*(?:to|–|\-)\s*(\d{3,4})\s*(?:hrs?)?",
                    text, re.IGNORECASE
                )
                if mil:
                    result["from_time"] = normalize_time_str(mil.group(1))
                    result["to_time"]   = normalize_time_str(mil.group(2))
                else:
                    # Single time
                    st = re.search(r"\d{1,2}[:.]\d{2}\s*(?:AM|PM)", text, re.IGNORECASE)
                    if st:
                        result["from_time"] = normalize_time_str(st.group())
                    else:
                        # Labeled: "Time: 10 AM"
                        lt = re.search(r"Time\s*[:–\-]\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
                        if lt:
                            raw_t = lt.group(1).strip()
                            vt = resolve_vague_time(raw_t)
                            if vt:
                                result["from_time"] = vt
                            else:
                                result["from_time"], result["to_time"] = split_time_range(raw_t)

        # Venue fallback
        if not result["venue"]:
            vm = re.search(r"Venue\s*[:–\-]\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
            if vm:
                result["venue"] = _clean_text(vm.group(1))
            else:
                VENUE_WORDS = [
                    "Hall", "Auditorium", "Seminar", "Conference", "Lab", "Laboratory",
                    "Studio", "Room", "Block", "Centre", "Center", "Complex",
                    "Ground", "Arena", "Classroom", "Amphitheatre"
                ]
                for sent in re.split(r"[.!?\n]+", text):
                    for vw in VENUE_WORDS:
                        if vw.lower() in sent.lower():
                            at_m = re.search(r"\bat\s+(.+)", sent, re.IGNORECASE)
                            result["venue"] = _clean_text(at_m.group(1) if at_m else sent)
                            break
                    if result["venue"]:
                        break

        # Reporting time (top-level — applies when there are no sub-events
        # or as a fallback for sub-events that don't have their own)
        result["reporting_time"] = extract_reporting_time(text)

        # ==================================================
        # STEP 3 — Date collection & expansion
        # ==================================================
        search_text = re.sub(r"\s+", " ", text)
        date_references = []

        # A. Explicit date ranges: "from X to Y", "X through Y"
        range_pats = [
            r"from\s+(.+?)\s+to\s+(.+?)(?=\s+(?:at|in|starting|each)|[,.\n]|$)",
            r"(\w+\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s*\d{4})?)\s+(?:to|through|till|until)\s+(\w+\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s*\d{4})?)",
        ]
        for pat in range_pats:
            for m in re.finditer(pat, search_text, re.IGNORECASE):
                s_str, e_str = m.group(1).strip(), m.group(2).strip()
                if not is_likely_date(s_str) or not is_likely_date(e_str):
                    continue
                expanded = expand_date_range(s_str, e_str)
                if expanded:
                    date_references.append({
                        "start": m.start(), "end": m.end(),
                        "type": "range", "dates": expanded
                    })

        # B. Comma/ampersand lists: "June 20, 21 & 22"
        list_pat = re.compile(
            r"(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+)?"
            r"\d{1,2}(?:st|nd|rd|th)?(?:\s*[,&]\s*\d{1,2}(?:st|nd|rd|th)?)+(?:\s+\w+)?",
            re.IGNORECASE
        )
        for m in list_pat.finditer(search_text):
            if is_deadline_date(m.group(0), text):
                continue
            dates = parse_date_list(m.group(0))
            if len(dates) > 1:
                date_references.append({
                    "start": m.start(), "end": m.end(),
                    "type": "list", "dates": dates
                })

        # C. Relative dates: "tomorrow", "next Monday"
        rel_pats = [
            r"\b(tomorrow|today|day after tomorrow|yesterday)\b",
            r"\b(next|this)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b"
        ]
        for pat in rel_pats:
            for m in re.finditer(pat, search_text, re.IGNORECASE):
                if is_deadline_date(m.group(0), text):
                    continue
                d = resolve_relative_date(m.group(0))
                if d:
                    date_references.append({
                        "start": m.start(), "end": m.end(),
                        "type": "relative", "dates": [d.strftime("%Y-%m-%d")]
                    })

        # D. GLiNER point dates (non-overlapping)
        for ent in extracted_date_entities:
            val = ent["text"]
            em = re.search(re.escape(val), search_text)
            if not em:
                continue
            # Skip if overlaps with any range/list already found
            if any(em.start() < r["end"] and em.end() > r["start"] for r in date_references):
                continue
            d = parse_date_safe(val)
            if d:
                date_references.append({
                    "start": em.start(), "end": em.end(),
                    "type": "point", "dates": [d.strftime("%Y-%m-%d")]
                })

        # Sort + deduplicate overlapping references
        date_references.sort(key=lambda x: x["start"])
        deduped = []
        for ref in date_references:
            if not any(ref["start"] < a["end"] and ref["end"] > a["start"] for a in deduped):
                deduped.append(ref)

        all_dates = sorted({d for r in deduped for d in r["dates"]})

        # ==================================================
        # STEP 4 — Multi-event detection
        # ==================================================
        sub_events = []

        # Priority order: HTML table > numbered blocks > bullet list
        if html_body:
            sub_events = extract_html_table_events(html_body, subject)

        if not sub_events:
            sub_events = extract_numbered_sub_events(text)

        if not sub_events:
            sub_events = extract_bullet_sub_events(text)

        if sub_events:
            result["sub_events"] = sub_events
            # Override top-level dates with union of all sub-event dates
            all_dates = sorted({s["date"] for s in sub_events if s.get("date")})
            # Top-level time/venue = first sub-event (fallback)
            if not result["from_time"]:
                result["from_time"] = sub_events[0].get("from_time")
                result["to_time"]   = sub_events[0].get("to_time")
            if not result["venue"]:
                result["venue"] = sub_events[0].get("venue")

        # ==================================================
        # STEP 5 — Assign dates & type-specific fields
        # ==================================================

        if email_type == "cancellation":
            result["old_dates"]   = all_dates
            result["description"] = extract_main_body(body)

        elif email_type == "reschedule":
            if len(deduped) >= 2:
                ref_a, ref_b = deduped[0], deduped[1]
                scores = [0, 0]
                for i, ref in enumerate([ref_a, ref_b]):
                    ctx = search_text[max(0, ref["start"] - 80):ref["start"]].lower()
                    if any(kw in ctx for kw in ["instead of", "originally", "scheduled from", "postponed from", "previously"]):
                        scores[i] -= 2
                    if any(kw in ctx for kw in ["now", "rescheduled to", "new date", "will be held", "conducted on"]):
                        scores[i] += 2
                if scores[0] >= scores[1]:
                    result["dates"]     = ref_a["dates"]
                    result["old_dates"] = ref_b["dates"]
                else:
                    result["old_dates"] = ref_a["dates"]
                    result["dates"]     = ref_b["dates"]
            elif len(deduped) == 1:
                result["dates"] = deduped[0]["dates"]
            else:
                result["dates"] = all_dates

            result["description"] = extract_main_body(body)

        elif email_type == "partial_update":
            result["dates"]  = all_dates
            result["_delta"] = parse_partial_update(text)
            result["description"] = extract_main_body(body)

        elif email_type == "reminder":
            result["dates"]       = all_dates
            result["description"] = extract_main_body(body)

        else:
            # event / announcement
            result["dates"] = all_dates
            result["link"]  = extract_registration_link(text, html_body)
            if email_type == "announcement":
                sentences = re.split(r"(?<=[.!?])\s+", extract_main_body(body))
                result["description"] = sentences[0].strip() if sentences else body[:200]

        return result

    except Exception as e:
        log.error(f"  ERROR in analyze_email: {e}", exc_info=True)
        return {
            "type": "unknown", "event": subject,
            "dates": [], "old_dates": [],
            "from_time": None, "to_time": None,
            "venue": None, "link": None, "description": None, "sub_events": []
        }

# =========================================================
# GOOGLE CALENDAR
# =========================================================

def get_calendar_service():
    creds = None
    base_dir   = os.path.dirname(os.path.abspath(__file__))
    token_path = os.path.join(base_dir, "token.pickle")
    creds_path = os.path.join(base_dir, "credentials.json")

    if os.path.exists(token_path):
        with open(token_path, "rb") as f:
            creds = pickle.load(f)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            search = [
                creds_path,
                os.path.abspath(os.path.join(base_dir, "..", "..", "credentials.json")),
                os.path.join(os.getcwd(), "credentials.json")
            ]
            found = next((p for p in search if os.path.exists(p)), None)
            if not found:
                raise FileNotFoundError(f"credentials.json not found in: {search}")
            flow  = InstalledAppFlow.from_client_secrets_file(found, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(token_path, "wb") as f:
            pickle.dump(creds, f)

    return build("calendar", "v3", credentials=creds)

def _gcal_delete_event(service, title, dates):
    """Delete Google Calendar events matching title + any of the given dates."""
    tmin = (datetime.datetime.utcnow() - datetime.timedelta(days=365)).isoformat() + "Z"
    tmax = (datetime.datetime.utcnow() + datetime.timedelta(days=730)).isoformat() + "Z"
    items = service.events().list(
        calendarId="primary", timeMin=tmin, timeMax=tmax,
        q=title, singleEvents=True
    ).execute().get("items", [])
    clean = title.strip().lower()
    for ev in items:
        if ev.get("summary", "").strip().lower() != clean:
            continue
        ev_date = (ev.get("start", {}).get("dateTime") or ev.get("start", {}).get("date", ""))[:10]
        if not dates or ev_date in dates:
            service.events().delete(calendarId="primary", eventId=ev["id"]).execute()
            log.info(f"  [GCal DELETED] '{ev.get('summary')}' on {ev_date}")

def add_event_to_gcal(event_data):
    service = get_calendar_service()

    raw_from = event_data.get("from_time") or "09:00 AM"
    norm_from = re.sub(r"(\d{1,2})\.(\d{2})", r"\1:\2", str(raw_from))
    start_dt  = dateutil.parser.parse(f"{event_data['date']} {norm_from}")

    raw_to = event_data.get("to_time")
    if raw_to:
        norm_to = re.sub(r"(\d{1,2})\.(\d{2})", r"\1:\2", str(raw_to))
        end_dt  = dateutil.parser.parse(f"{event_data['date']} {norm_to}")
        if end_dt <= start_dt:
            end_dt += datetime.timedelta(days=1)
    else:
        end_dt = start_dt + datetime.timedelta(hours=1)

    # Dedup check before inserting
    _gcal_delete_event(service, event_data["title"], [event_data["date"]])

    desc_parts = [event_data.get("description", "Added by Curious Bees")]
    if event_data.get("reporting_time"):
        desc_parts.append(f"⚠️ Reporting Time: {event_data['reporting_time']}")

    body = {
        "summary":  event_data["title"],
        "location": event_data.get("venue", ""),
        "description": "\n".join(desc_parts),
        "start": {"dateTime": start_dt.isoformat(), "timeZone": TIMEZONE},
        "end":   {"dateTime": end_dt.isoformat(),   "timeZone": TIMEZONE},
    }
    created = service.events().insert(calendarId="primary", body=body).execute()
    return created.get("id")

# =========================================================
# PER-EMAIL PROCESSOR
# =========================================================

def process_result(result, sender_email, subject):
    """
    Route a parsed result dict to the correct calendar action.
    Handles: event, reschedule, cancellation, partial_update,
             reminder, announcement, unknown.
    """
    etype = result["type"]

    # ---- Helper: save to GCal + local for one entry ----
    def _save_entry(entry):
        added = add_event(entry)
        if added:
            try:
                eid = add_event_to_gcal(entry)
                log.info(f"  [GCal ADDED] ID: {eid}")
            except Exception as e:
                log.error(f"  [GCal ERROR] {e}")

    # --------------------------------------------------
    if etype == "event":
        if not result["dates"]:
            log.warning("  No dates extracted — skipping.")
            return

        sub_events = result.get("sub_events")
        if sub_events:
            log.info(f"  Multi-event email — {len(sub_events)} sub-events detected.")
            for sub in sub_events:
                if not sub.get("date"):
                    continue
                entry = {
                    "title":          sub.get("title") or result["event"],
                    "date":           sub["date"],
                    "from_time":      sub.get("from_time"),
                    "to_time":        sub.get("to_time"),
                    "reporting_time": sub.get("reporting_time") or result.get("reporting_time"),
                    "venue":          sub.get("venue"),
                    "link":           result.get("link"),
                    "status":         "schedule",
                    "parent_event":   result["event"]
                }
                _save_entry(entry)
                # If sub-event spans multiple days, save those too
                for extra_date in sub.get("_all_dates", [])[1:]:
                    extra = dict(entry, date=extra_date)
                    _save_entry(extra)
        else:
            # Build per-day venue map from the full email text
            per_day_venues = extract_per_day_venues(
                f"Subject: {result['event']}\n\n"   # reuse what analyze_email had
            )
            # Re-extract from result description if available
            _pdv_source = result.get("description") or result["event"]
            per_day_venues = extract_per_day_venues(_pdv_source)

            sorted_dates = sorted(result["dates"])
            for i, date in enumerate(sorted_dates, start=1):
                # Per-day venue if available, else fall back to top-level
                day_venue = per_day_venues.get(i) or result["venue"]

                # For the last date, use to_time; for intermediate days
                # with no specific time, leave to_time as None so GCal
                # doesn't show a wrong end time
                is_last  = (i == len(sorted_dates))
                is_first = (i == 1)
                day_from = result["from_time"] if is_first else None
                day_to   = result["to_time"]   if is_last  else None

                entry = {
                    "title":          result["event"],
                    "date":           date,
                    "from_time":      day_from,
                    "to_time":        day_to,
                    "reporting_time": result.get("reporting_time"),
                    "venue":          day_venue,
                    "link":           result.get("link"),
                    "status":         "schedule"
                }
                _save_entry(entry)

    # --------------------------------------------------
    elif etype == "reschedule":
        clean_name = _clean_text(result["event"])
        match = find_matching_event(clean_name, result["old_dates"])
        if match:
            update_event(
                old_entry=match,
                old_dates=result["old_dates"],
                new_dates=result["dates"],
                new_from_time=result["from_time"],
                new_to_time=result["to_time"],
                new_venue=result["venue"],
                new_link=result.get("link")
            )
            try:
                svc = get_calendar_service()
                _gcal_delete_event(svc, match["title"], result["old_dates"] or [match["date"]])
            except Exception as e:
                log.error(f"  [GCal DELETE ERROR] {e}")
            for date in result["dates"]:
                try:
                    eid = add_event_to_gcal({
                        "title":          match["title"],
                        "date":           date,
                        "from_time":      result["from_time"] or match.get("from_time") or "09:00 AM",
                        "to_time":        result["to_time"]   or match.get("to_time"),
                        "venue":          result["venue"]      or match.get("venue"),
                        "reporting_time": result.get("reporting_time") or match.get("reporting_time"),
                    })
                    log.info(f"  [GCal RESCHEDULED] ID: {eid}")
                except Exception as e:
                    log.error(f"  [GCal ERROR] {e}")
        else:
            for date in result["dates"]:
                entry = {
                    "title":     f"[POSSIBLY RESCHEDULED] {result['event']}",
                    "date":      date,
                    "from_time": result["from_time"],
                    "to_time":   result["to_time"],
                    "venue":     result["venue"],
                    "link":      result.get("link"),
                    "status":    "reschedule"
                }
                _save_entry(entry)

        save_announcement(sender_email, subject,
            result.get("description") or f"'{result['event']}' has been rescheduled.")

    # --------------------------------------------------
    elif etype == "cancellation":
        clean_name = _clean_text(result["event"])
        match = find_matching_event(clean_name, result["old_dates"])
        if match:
            delete_event(match)
            try:
                svc = get_calendar_service()
                _gcal_delete_event(svc, match["title"], None)
            except Exception as e:
                log.error(f"  [GCal ERROR] {e}")
        else:
            log.warning("  No matching event found for cancellation.")

        save_announcement(sender_email, subject, result.get("description"))

    # --------------------------------------------------
    elif etype == "partial_update":
        clean_name = _clean_text(result["event"])
        match = find_matching_event(clean_name, result["dates"])
        delta = result.get("_delta") or {}
        if match and delta:
            update_event(
                old_entry=match,
                old_dates=[match["date"]],
                new_dates=result["dates"] or [match["date"]],
                new_from_time=delta.get("from_time") or match.get("from_time"),
                new_to_time=delta.get("to_time")   or match.get("to_time"),
                new_venue=delta.get("venue")        or match.get("venue")
            )
            try:
                svc = get_calendar_service()
                _gcal_delete_event(svc, match["title"], [match["date"]])
                add_event_to_gcal({
                    "title":     match["title"],
                    "date":      result["dates"][0] if result["dates"] else match["date"],
                    "from_time": delta.get("from_time") or match.get("from_time") or "09:00 AM",
                    "to_time":   delta.get("to_time")   or match.get("to_time"),
                    "venue":     delta.get("venue")      or match.get("venue")
                })
            except Exception as e:
                log.error(f"  [GCal ERROR] {e}")
        else:
            log.warning("  Partial update: no matching event or no delta — skipping.")

        save_announcement(sender_email, subject,
            result.get("description") or f"Update for '{result['event']}'.")

    # --------------------------------------------------
    elif etype == "reminder":
        log.info("  REMINDER — no calendar change needed.")
        save_announcement(sender_email, subject,
            result.get("description") or f"Reminder for '{result['event']}'.")

    # --------------------------------------------------
    elif etype == "announcement":
        log.info("  ANNOUNCEMENT — not added to calendar.")
        save_announcement(sender_email, subject, result.get("description"))

    # --------------------------------------------------
    else:
        log.warning("  UNKNOWN TYPE — could not process.")

# =========================================================
# MAIN LOOP
# =========================================================

def main():
    processed_ids = load_processed_ids()

    log.info("Connecting to mailbox...")
    mail = imaplib.IMAP4_SSL(IMAP_SERVER)
    mail.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
    mail.select("inbox")
    log.info("Connected.\n")

    try:
        status, messages = mail.search(None, "UNSEEN")
        email_ids = messages[0].split()
    except imaplib.IMAP4.abort as e:
        log.error(f"IMAP connection aborted: {e}")
        sys.exit(1)

    log.info(f"Found {len(email_ids)} unread email(s)\n")
    log.info("=" * 60)

    for eid in email_ids:
        status, msg_data = mail.fetch(eid, "(RFC822)")
        for part in msg_data:
            if not isinstance(part, tuple):
                continue

            msg = email.message_from_bytes(part[1])

            # ---- Decode subject ----
            raw_sub = msg["Subject"] or ""
            dec     = decode_header(raw_sub)
            subject = "".join(
                p.decode(enc or "utf-8", errors="ignore") if isinstance(p, bytes) else p
                for p, enc in dec
            )
            subject = _clean_text(subject)

            # ---- Extract sender ----
            _, sender_email = parseaddr(msg.get("from", ""))
            sender_email = sender_email.strip().lower()

            receiver = msg.get("to", "")

            # ---- Authorization check ----
            if sender_email not in [s.strip().lower() for s in AUTHORIZED_SENDERS]:
                log.info(f"UNAUTHORIZED SKIP: {sender_email}")
                log.info("=" * 60)
                continue

            # ---- Extract body ----
            html_body  = ""
            plain_body = ""
            if msg.is_multipart():
                for p in msg.walk():
                    ctype   = p.get_content_type()
                    charset = p.get_content_charset() or "utf-8"
                    payload = p.get_payload(decode=True)
                    if not payload:
                        continue
                    text = payload.decode(charset, errors="ignore")
                    if ctype == "text/plain" and not plain_body:
                        plain_body = text
                    elif ctype == "text/html" and not html_body:
                        html_body = text
            else:
                payload = msg.get_payload(decode=True)
                if payload:
                    charset = msg.get_content_charset() or "utf-8"
                    text    = payload.decode(charset, errors="ignore")
                    if msg.get_content_type() == "text/html":
                        html_body = text
                    else:
                        plain_body = text

            # Convert HTML to plain if needed
            if not plain_body and html_body:
                soup = BeautifulSoup(html_body, "html.parser")
                for tag in soup.find_all(["td","th","tr","br","p","li","div"]):
                    tag.append("\n")
                plain_body = soup.get_text(separator=" ")

            body = re.sub(r"[ \t\r]+", " ", plain_body).strip()
            body = strip_reply_chain(body)

            # ---- Duplicate fingerprint check ----
            fp = email_fingerprint(sender_email, subject, body)
            if fp in processed_ids:
                log.info(f"DUPLICATE SKIP: {subject}")
                log.info("=" * 60)
                continue

            log.info(f"SENDER   : {sender_email}")
            log.info(f"RECEIVER : {receiver}")
            log.info(f"SUBJECT  : {subject}")
            log.info("")

            result = analyze_email(body, subject, html_body)

            log.info(f"TYPE     : {result['type'].upper()}")
            log.info(json.dumps(result, indent=4, default=str))
            log.info("")
            log.info("CALENDAR ACTIONS:")

            process_result(result, sender_email, subject)

            save_processed_id(fp)

            log.info("")
            log.info("=" * 60)

    mail.logout()
    log.info("\nRun complete.")
    log.info(f"Events saved to        : {CALENDAR_FILE}")
    log.info(f"Announcements saved to : {ANNOUNCEMENTS_FILE}")

if __name__ == "__main__":
    main()