"""
Pure text / regex helpers and the email classifier.

Everything in this module is a deterministic function of its input
strings — no model calls, no I/O. That makes the parsing logic
straightforward to reason about and test in isolation.
"""

import datetime
import re

import dateutil.parser

from config import (
    DASH_CHARS,
    DEADLINE_KEYWORDS,
    GREETING_RE,
    SIGNOFF_RE,
    VENUE_KEYWORDS,
)


# ---------------------------------------------------------------
# Body cleaning
# ---------------------------------------------------------------

def extract_main_body(body: str) -> str:
    """Strip greetings and sign-offs, returning only substantive content."""
    text = GREETING_RE.sub("", body.strip()).strip()
    m = SIGNOFF_RE.search(text)
    return text[:m.start()].strip(" ,;") if m else text.strip(" ,;")


def normalize_dashes(text: str) -> str:
    """Replace Unicode dash variants with a plain ASCII hyphen."""
    return re.sub(DASH_CHARS, "-", text)


# ---------------------------------------------------------------
# Date / time sanity checks
# ---------------------------------------------------------------

def is_likely_date(text: str) -> bool:
    """
    Guard against dateutil parsing non-date strings (e.g. 'GPT-2' → June 2).
    Only passes text containing a month abbreviation or a numeric separator.
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


# ---------------------------------------------------------------
# Email classifier
# ---------------------------------------------------------------

def classify_email(text: str) -> str:
    """
    Heuristic email classifier. Returns one of:
    ``event``, ``reschedule``, ``cancellation``, ``announcement``.
    """
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


# ---------------------------------------------------------------
# Time / date parsing
# ---------------------------------------------------------------

def split_time_range(val: str) -> tuple[str, str | None]:
    """
    Split a time string like '10:00 AM to 12:00 PM' into (from_time, to_time).
    Returns (val, None) when no range separator is detected.
    """
    cleaned = re.sub(r'(?i)\s+onwards', '', val).strip()
    sep     = r'\s*(?:' + DASH_CHARS + r'|[-]|to|till|until)\s*'
    parts   = re.split(sep, cleaned, maxsplit=1, flags=re.IGNORECASE)
    return (parts[0].strip(), parts[1].strip()) if len(parts) == 2 else (cleaned, None)


def expand_date_ranges(text: str) -> list[str]:
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


# ---------------------------------------------------------------
# Venue / link extraction
# ---------------------------------------------------------------

def venue_fallback(text: str) -> str | None:
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


def find_registration_link(text: str) -> str | None:
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


# ---------------------------------------------------------------
# Misc
# ---------------------------------------------------------------

def dedupe(seq: list) -> list:
    """Remove duplicates from a list while preserving insertion order."""
    seen: set = set()
    return [x for x in seq if not (x in seen or seen.add(x))]
