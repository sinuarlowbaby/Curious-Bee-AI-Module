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

import imaplib, email
from email.header import decode_header
from bs4 import BeautifulSoup

env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(env_path)

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

# Unicode dash variants Ã¢â‚¬â€ used in normalize_dashes() and time-splitting
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
    """Load a JSON list from disk; returns [] if the file doesn't exist or is empty."""
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if not content:          # file exists but is empty
            return []
        return json.loads(content)


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


def update_event(old_entries: list[dict], new_dates: list,
                 from_time, to_time, venue, link=None) -> None:
    """Replace ALL old calendar entries for this event with entries on new dates."""
    # Use the first entry as the template for title/time/venue defaults
    template = old_entries[0]
    title    = template["title"]

    # Drop every existing entry for this event title in one pass
    cal = [e for e in _load_json(CALENDAR_FILE) if e["title"] != title]

    for date in new_dates:
        cal.append({
            "title":     title,
            "date":      date,
            "from_time": from_time or template.get("from_time"),
            "to_time":   to_time   or template.get("to_time"),
            "venue":     venue     or template.get("venue"),
            "link":      link      or template.get("link"),
            "status":    "reschedule",
        })
        old_dates_str = ", ".join(e["date"] for e in old_entries)
        print(f"  [UPDATED] '{title}' Ã¢â‚¬â€ [{old_dates_str}] Ã¢â€ â€™ {date}")
    _save_json(CALENDAR_FILE, cal)


def delete_event(old_entries: list[dict]) -> None:
    """Remove ALL calendar entries matching this event title."""
    title = old_entries[0]["title"]
    cal   = [e for e in _load_json(CALENDAR_FILE) if e["title"] != title]
    _save_json(CALENDAR_FILE, cal)
    dates_str = ", ".join(e["date"] for e in old_entries)
    print(f"  [DELETED] '{title}' on [{dates_str}]")


def find_matching_event(event_name: str, old_dates: list = None,
                        threshold: float = 0.75) -> list[dict]:
    """
    Semantic search over the calendar.

    Returns a list of ALL calendar entries whose title matches event_name above
    the threshold (so multi-day events return one entry per day).  Returns an
    empty list when nothing matches confidently.

    Key improvement over the original: all calendar titles are batch-encoded
    in a single forward pass (one call to similarity_model.encode) rather than
    calling encode() individually for each entry, making this O(1) model calls
    regardless of calendar size.

    A 0.15 score bonus is applied when the candidate entry's date is also
    found in old_dates Ã¢â‚¬â€ acts as a strong disambiguation signal for finding
    the best title, but ALL entries sharing that best title are returned so
    that multi-day reschedules and cancellations remove every day, not only
    the first one.
    """
    cal = _load_json(CALENDAR_FILE)
    if not cal:
        return []

    query_emb  = similarity_model.encode(event_name, convert_to_tensor=True)
    title_embs = similarity_model.encode([e["title"] for e in cal], convert_to_tensor=True)
    scores     = util.cos_sim(query_emb, title_embs)[0].tolist()

    best_score, best_title = 0.0, None
    for score, entry in zip(scores, cal):
        boosted = score + (0.15 if old_dates and entry["date"] in old_dates else 0.0)
        if boosted > best_score:
            best_score, best_title = boosted, entry["title"]

    if best_score >= threshold:
        matched = [e for e in cal if e["title"] == best_title]
        print(f"  Similarity score: {best_score:.2f} Ã¢â‚¬â€ matched '{best_title}' "
              f"({len(matched)} entr{'y' if len(matched) == 1 else 'ies'})")
        return matched

    print(f"  Similarity score: {best_score:.2f} Ã¢â‚¬â€ no confident match found")
    return []

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
    Guard against dateutil parsing non-date strings (e.g. 'GPT-2' Ã¢â€ â€™ June 2).
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
    if any(kw in lower for kw in [
        "event", "seminar", "workshop", "talk", "invite", "invited",
        "lecture", "symposium", "meet", "fest", "hackathon", "conference",
    ]):
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
                    print(f"  [RANGE] Expanded '{s_str}' Ã¢â€ â€™ '{e_str}' into {delta + 1} day(s)")
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
        # Venue label may span multiple comma-separated lines
        venue_start = m.group(1).strip()
        full_m = re.search(re.escape(venue_start) + r"(?:,\s*[^.]+)*", text)
        raw = full_m.group(0) if full_m else venue_start
        return re.sub(r"\s*\n\s*", " ", raw).strip().rstrip(".")
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

        # --- Build description for non-event types ---
        if etype in ("announcement", "cancellation", "reschedule"):
            body_clean = extract_main_body(body)
            sents      = re.split(r'(?<=[.!?])\s+', body_clean)
            if etype == "reschedule":
                # Use the full cleaned body so the new date/venue info is captured
                result["description"] = body_clean
            else:
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
                    # Check only the sentence containing this date for deadline context Ã¢â‚¬â€
                    # prevents cross-sentence bleed where a deadline phrase in sentence A
                    # wrongly flags a legitimate event date in sentence B.
                    ctx = next((s.lower() for s in text_sentences if val.lower() in s.lower()), "")
                    if any(kw in ctx for kw in DEADLINE_KEYWORDS):
                        print(f"  [SKIP]  '{val}' looks like a deadline Ã¢â‚¬â€ not an event date")
                    else:
                        extracted_dates.append(date_str)
                except Exception:
                    pass

            elif lbl == "time" and not result["from_time"]:
                if not is_likely_time(val):
                    continue
                result["time"] = val
                result["from_time"], result["to_time"] = _split_time(val)

            elif lbl == "venue":
                # If reschedule explicitly says "same venue", we skip extraction
                # so it falls back to the old venue in update_event.
                if etype == "reschedule" and re.search(r"same\s+venue", text, re.IGNORECASE):
                    continue

                # For reschedule emails, we only want the NEW venue (after the
                # reschedule keyword).  For all other types, first-found wins.
                if etype == "reschedule":
                    # Locate the reschedule keyword in the full text
                    rsplit = re.search(
                        r"\b(reschedule[d]?|postpone[d]?|rescheduling)\b",
                        text, re.IGNORECASE,
                    )
                    # Only store a venue entity that appears AFTER the keyword
                    entity_start = ent.get("start", text.find(val))
                    if rsplit and entity_start < rsplit.start():
                        continue  # old-venue mention Ã¢â‚¬â€ skip
                if not result["venue"] or etype == "reschedule":
                    # Extend forward from GLiNER entity, crossing newlines
                    m = re.search(re.escape(val) + r"(?:,\s*[^.]+)*", text)
                    raw_venue = m.group(0).strip() if m else val
                    # Collapse embedded newlines and trim
                    result["venue"] = re.sub(r"\s*\n\s*", " ", raw_venue).strip().rstrip(".")

        # --- Venue fallback (regex) ---
        if not result["venue"]:
            if etype == "reschedule" and re.search(r"same\s+venue", text, re.IGNORECASE):
                pass  # leave as None to inherit old venue
            else:
                result["venue"] = _venue_fallback(text)

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

        # --- Deduplicate (GLiNER point-dates merged with range-expanded dates) ---
        all_dates = _dedupe(extracted_dates + _expand_date_ranges(text))

        # --- Assign dates by email type ---
        if etype == "cancellation":
            result["old_dates"]   = all_dates
            # Use the full cleaned body (not just first sentence) so that
            # any "new date will be announced" phrasing is preserved naturally.
            result["description"] = extract_main_body(body)

        elif etype == "reschedule":
            # Split the email at the reschedule keyword so that date ranges in
            # each half are expanded independently.  This prevents old dates
            # from being misidentified as new dates in a multi-day reschedule.
            reschedule_re = re.compile(
                r"\b(reschedule[d]?|postpone[d]?|rescheduling)\b", re.IGNORECASE
            )
            split_m = reschedule_re.search(text)
            if split_m:
                pre_text  = text[:split_m.start()]
                post_text = text[split_m.end():]
                # Extract point-dates from each half
                pre_dates  = [d for d in extracted_dates
                               if any(d in pre_text for d in [d])]
                post_dates = [d for d in extracted_dates
                               if d not in pre_dates]
                # Expand date ranges independently from each half
                old_range  = _expand_date_ranges(pre_text)
                new_range  = _expand_date_ranges(post_text)
                old_part   = _dedupe(pre_dates  + old_range)
                new_part   = _dedupe(post_dates + new_range)
                if old_part or new_part:
                    result["old_dates"] = old_part
                    result["dates"]     = new_part
                else:
                    # Fallback: if split produced nothing, use original heuristic
                    if len(all_dates) >= 2:
                        result["old_dates"] = [all_dates[0]]
                        result["dates"]     = all_dates[1:]
                    else:
                        result["dates"] = all_dates
            else:
                # No reschedule keyword found in text Ã¢â‚¬â€ use original heuristic
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
    # HANDLE: NEW EVENT Ã¢â€ â€™ add entries to calendar.json
    # ==================================================
    if etype == "event":
        if not result["dates"]:
            print("  WARNING: No dates extracted Ã¢â‚¬â€ skipping calendar entry.")
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
    # HANDLE: RESCHEDULE Ã¢â€ â€™ update entries in calendar.json
    # ==================================================
    elif etype == "reschedule":
        print("RESCHEDULE DETECTED Ã¢â‚¬â€ searching calendar for matching event...")
        matches = find_matching_event(result["event"], result["old_dates"])
        print("\nCALENDAR ACTIONS:")
        if matches:
            print(f"  Matched: '{matches[0]['title']}' Ã¢â‚¬â€ {len(matches)} entr{'y' if len(matches)==1 else 'ies'}")
            update_event(
                old_entries=matches,
                new_dates=result["dates"],
                from_time=result["from_time"],
                to_time=result["to_time"],
                venue=result["venue"],
                link=result.get("link"),
            )
        else:
            print("  No existing event matched Ã¢â‚¬â€ adding as new entry (flagged)")
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
    # HANDLE: CANCELLATION Ã¢â€ â€™ delete entries from calendar.json
    # ==================================================
    elif etype == "cancellation":
        print("CANCELLATION DETECTED Ã¢â‚¬â€ searching calendar for matching event...")
        matches = find_matching_event(result["event"], result["old_dates"])
        print("\nCALENDAR ACTIONS:")
        if matches:
            print(f"  Matched: '{matches[0]['title']}' Ã¢â‚¬â€ deleting {len(matches)} entr{'y' if len(matches)==1 else 'ies'}.")
            delete_event(matches)
        else:
            print("  No existing event matched for cancellation.")
        print("\nANNOUNCEMENT ACTIONS:")
        save_announcement(
            sender_email=sender_email,
            subject=subject,
            description=result.get("description"),
        )

    # ==================================================
    # HANDLE: ANNOUNCEMENT Ã¢â€ â€™ save to announcements.json only
    # ==================================================
    elif etype == "announcement":
        print("ANNOUNCEMENT Ã¢â‚¬â€ not added to calendar.")
        print(f"Summary  : {result.get('description')}\n")
        print("ANNOUNCEMENT ACTIONS:")
        save_announcement(
            sender_email=sender_email,
            subject=subject,
            description=result.get("description"),
        )

    else:
        print("UNKNOWN TYPE Ã¢â‚¬â€ could not process this email.")

    print(f"\n{'=' * 50}\n")



# =========================================================
# IMAP INTEGRATION
# =========================================================

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    mail = imaplib.IMAP4_SSL(IMAP_SERVER)
    mail.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
    mail.select("inbox")

    status, messages = mail.search(None, "UNSEEN")
    email_ids = messages[0].split()

    if not email_ids:
        print("No unread emails found.")
    else:
        print(f"Found {len(email_ids)} unread email(s).\n")

    for latest_email_id in email_ids:
        status, msg_data = mail.fetch(latest_email_id, "(RFC822)")
        for response_part in msg_data:
            if not isinstance(response_part, tuple):
                continue
            msg = email.message_from_bytes(response_part[1])

            raw_subject = msg["Subject"]
            decoded_subject = decode_header(raw_subject)
            subject = "".join(
                part.decode(enc or "utf-8", errors="ignore") if isinstance(part, bytes) else part
                for part, enc in decoded_subject
            )

            email_body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    ct = part.get_content_type()
                    payload = part.get_payload(decode=True)
                    if not payload:
                        continue
                    text = payload.decode(errors="ignore")
                    if ct == "text/plain":
                        email_body = text
                        break
                    elif ct == "text/html" and not email_body:
                        email_body = BeautifulSoup(text, "html.parser").get_text()
            else:
                payload = msg.get_payload(decode=True)
                if payload:
                    email_body = payload.decode(errors="ignore")

            process_email(msg["From"], msg["To"], subject, email_body)

    mail.logout()
    print("Done.")
    print(f"Events saved to        : {CALENDAR_FILE}")
    print(f"Announcements saved to : {ANNOUNCEMENTS_FILE}")
