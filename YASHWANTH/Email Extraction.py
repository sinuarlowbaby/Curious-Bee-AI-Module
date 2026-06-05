import os, re, json, imaplib, email
from email import policy
from email.header import decode_header
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv
# pyrefly: ignore [missing-import]
from bs4 import BeautifulSoup

load_dotenv();

EMAIL = os.getenv("EMAIL")
APP_PASSWORD = os.getenv("APP_PASSWORD")
IMAP_SERVER = os.getenv("IMAP_SERVER", "imap.gmail.com")
MISSING = "Not provided in email"

MONTH = r"(?:Jan|January|Feb|February|Mar|March|Apr|April|May|Jun|June|Jul|July|Aug|August|Sep|September|Oct|October|Nov|November|Dec|December)"
DATE = rf"(?:\d{{1,2}}\s+{MONTH}\s+\d{{4}}|{MONTH}\s+\d{{1,2}},?\s+\d{{4}}|\d{{1,2}}[/-]\d{{1,2}}[/-]\d{{2,4}}|\d{{4}}-\d{{2}}-\d{{2}})"
TIME = r"(?:\d{1,2}[:.]\d{2}\s*(?:AM|PM|am|pm)|\d{1,2}\s*(?:AM|PM|am|pm))"

DATE_L = ["Main Event Date", "Date of Event", "Event Date", "Visit Date", "Date", "Pass Collection Date", "Collection Date"]
SDATE_L = ["New Start Date", "Event Start Date", "Event opening date", "Camp Opening Date", "Start Date"]
EDATE_L = ["New Closing Date", "Event End Date", "Event closing date", "Camp Closing Date", "End Date", "Closing Date"]
TIME_L = ["Main Event Time", "Program Time", "Visitor timing", "Event timing", "Workshop Timing", "Camp Timing", "Collection Time", "Time"]
STIME_L = ["Start Time", "Departure Time"]
ETIME_L = ["End Time", "Return Time"]
VENUE_L = ["New Venue", "Main Event Venue", "Venue", "Location", "Place"]
OFFLINE_L = ["Offline Venue"]
ONLINE_L = ["Online Venue"]


def decode_subject(subject):
    if not subject:
        return ""
    return "".join(
        p.decode(e or "utf-8", "ignore") if isinstance(p, bytes) else p
        for p, e in decode_header(subject)
    ).strip()


def html_to_text(html):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return "\n".join(x.strip() for x in soup.get_text("\n").splitlines() if x.strip())


def get_body(msg):
    for kind in ("plain", "html"):
        part = msg.get_body(preferencelist=(kind,))
        if part:
            data = part.get_content()
            return html_to_text(data) if kind == "html" else data.strip()

    for part in msg.walk():
        data = part.get_payload(decode=True)
        if not data:
            continue
        data = data.decode(errors="ignore")
        if part.get_content_type() == "text/plain":
            return data.strip()
        if part.get_content_type() == "text/html":
            return html_to_text(data)
    return ""


def fetch_latest_email():
    if not EMAIL or not APP_PASSWORD:
        raise ValueError("EMAIL or APP_PASSWORD missing in .env file")

    imap = imaplib.IMAP4_SSL(IMAP_SERVER)

    try:
        imap.login(EMAIL, APP_PASSWORD)
        print("Login Successful")

        imap.select("inbox")
        status, data = imap.search(None, "ALL")
        if status != "OK":
            raise Exception("Could not search inbox")

        ids = data[0].split()
        if not ids:
            return None

        status, msg_data = imap.fetch(ids[-1], "(RFC822)")
        if status != "OK":
            raise Exception("Could not fetch email")

        msg = email.message_from_bytes(msg_data[0][1], policy=policy.default)

        return {
            "subject": decode_subject(msg["subject"]),
            "body": get_body(msg)
        }

    finally:
        try:
            imap.logout()
        except Exception:
            pass


def clean_text(text):
    text = text or ""
    text = text.replace("*", "").replace("–", "-").replace("—", "-").replace("\xa0", " ")
    return re.sub(r"[ \t]+", " ", text).strip()


def one_line(text):
    return re.sub(r"\s+", " ", clean_text(text))


def lines(text):
    return [x.strip() for x in clean_text(text).splitlines() if x.strip()]


def label(line, labels):
    pat = "|".join(map(re.escape, labels))
    m = re.match(rf"^(?:{pat})\s*[:\-]\s*(.+)$", line, re.I)
    return m.group(1).strip(" .,-") if m else None


def first(lines_, labels):
    return next((v for x in lines_ if (v := label(x, labels))), None)


def findp(text, patterns):
    for p in patterns:
        m = re.search(p, text, re.I)
        if m:
            return f"{m.group(1).strip()} to {m.group(2).strip()}" if m.lastindex and m.lastindex >= 2 and m.group(2) else m.group(1).strip()
    return MISSING


def clean_venue(v):
    if not v:
        return v
    v = v.strip(" .,-")
    v = re.sub(r"^(the|same\s+venue|venue)\s*[-:]?\s*", "", v, flags=re.I)
    return v.strip(" .,-")


def clean_event(e):
    if not e:
        return None

    e = e.strip(" .,-:").replace("–", "-").replace("—", "-")
    e = re.sub(r"^(a|an|the)\s+", "", e, flags=re.I)
    e = re.sub(r"\s+which\s+was.*$", "", e, flags=re.I)
    e = re.sub(r"\s+for\s+(students|faculty|participants|final year students|all students|students and staff members).*$", "", e, flags=re.I)
    e = re.sub(r"\s+to\s+(connect|showcase|encourage|educate).*$", "", e, flags=re.I)
    e = re.sub(r"\b(has been|will be|is)\b.*$", "", e, flags=re.I).strip()

    return None if e.lower() in {
        "announcement", "schedule announcement", "notice",
        "event invitation", "registration details", "important update", "schedule"
    } else e


def event_from_subject(subject):
    if not subject:
        return None

    subject = re.sub(r"^subject\s*:\s*", "", subject.strip(), flags=re.I)
    subject = subject.replace("–", "-").replace("—", "-")
    subject = re.sub(r"^(Circular|Reminder|Important Update|Cancellation Notice|Schedule Announcement|Rescheduled)\s*[:\-]\s*", "", subject, flags=re.I)
    subject = re.sub(r"^(Rescheduled Date for|Venue Change for|Agenda for|Permission Required for|Entry Pass Collection for|Certificate Distribution for|Result Announcement Ceremony for|Invitation to|Invitation for)\s+", "", subject, flags=re.I)

    if "-" in subject:
        left, right = [x.strip() for x in subject.split("-", 1)]
        if re.search(r"(schedule|announcement|notice|notification|registration details|event invitation|project display)", right, re.I):
            return clean_event(left)

    subject = re.sub(r"\s+(notice|announcement|schedule|notification|reminder).*$", "", subject, flags=re.I)
    return clean_event(subject)


def extract_event(subject, body):
    text = one_line(body)

    patterns = [
        r"\bthat\s+(?:the\s+)?(.+?)\s+which\s+was\s+originally\s+scheduled\b",
        r"\b(?:announce the schedule for|excited to announce the schedule for|pleased to announce)\s+(?:a|an|the)?\s*(.+?)(?=\.|,)",
        r"\b(?:will conduct|is organizing|are organizing|is arranging|is conducting|are conducting)\s+(?:a|an|the)?\s*(.+?)(?=\s+for\b|\s+on\b|\.|,)",
        r"\b(?:invite you to participate in|invite you to|delighted to invite you to)\s+(.+?)(?=\.|,)",
        r"\bThe\s+(.+?)\s+will be (?:organized|conducted|held)\b",
        r"\bthe\s+(.+?)\s+(?:has been scheduled|is scheduled|will now be conducted|planned for|originally scheduled)\b",
        r"\bfor(?: the)?\s+(.+?)\s+will be (?:distributed|conducted)\b",
        r"\b(?:event|program|programme|title)\s*[:\-]\s*(.+?)(?=\.|$)"
    ]

    for p in patterns:
        m = re.search(p, text, re.I)
        if m and (e := clean_event(m.group(1))):
            return e

    return event_from_subject(subject) or MISSING


def extract_date(body):
    ls, text = lines(body), one_line(body)

    if v := first(ls, ["Main Event Date"]):
        return v

    s, e = first(ls, SDATE_L), first(ls, EDATE_L)
    if s and e:
        return f"{s} to {e}"

    if v := first(ls, DATE_L):
        return v

    return findp(text, [
        rf"\bwill\s+now\s+be\s+held\s+on\s+({DATE})",
        rf"\bnow\s+be\s+held\s+on\s+({DATE})",
        rf"\bwill\s+now\s+be\s+conducted\s+on\s+({DATE})",
        rf"\bnow\s+be\s+conducted\s+on\s+({DATE})",
        rf"\brescheduled.*?\b(?:to|on)\s+({DATE})",
        rf"\b(?:will begin|begins|begin) on\s+({DATE}).*?\b(?:will close|closes|close) on\s+({DATE})",
        rf"\b(?:event start date is)\s+({DATE}).*?\b(?:event end date is)\s+({DATE})",
        rf"\b(?:commence|commences) on\s+({DATE}).*?\b(?:conclude|concludes) on\s+({DATE})",
        rf"\b(?:starts from|start from)\s+({DATE}).*?\b(?:ends on|end on)\s+({DATE})",
        rf"\b(?:will start|start) on\s+({DATE}).*?\b(?:will end|end) on\s+({DATE})",
        rf"\bfrom\s+({MONTH}\s+\d{{1,2}}\s+to\s+{MONTH}\s+\d{{1,2}},?\s+\d{{4}})",
        rf"\bfrom\s+({MONTH}\s+\d{{1,2}}\s+to\s+\d{{1,2}},?\s+\d{{4}})",
        rf"\bfrom\s+({DATE})\s+to\s+({DATE})",
        rf"\b(?:scheduled for|scheduled on|held on|conducted on|planned for|will be distributed on|will be held on|on)\s+({DATE})",
        rf"\b({DATE})"
    ])


def extract_time(body):
    ls, text = lines(body), one_line(body)

    s, e = first(ls, STIME_L), first(ls, ETIME_L)
    if s and e:
        return f"{s} to {e}"

    if v := first(ls, TIME_L):
        return v

    if re.search(r"\bAgenda\s*:", body, re.I):
        times = re.findall(TIME, body, re.I)
        close = re.search(rf"\b(?:event|program|programme|session)\s+will\s+close\s+at\s+({TIME})", text, re.I)
        if times:
            return f"{times[0].strip()} to {(close.group(1) if close else times[-1]).strip()}"

    return findp(text, [
        rf"\bfrom\s+({TIME})\s+to\s+({TIME})",
        rf"\bbetween\s+({TIME})\s+and\s+({TIME})",
        rf"\b(?:commence|commences|start|starts|begin|begins)\b.*?\bat\s+({TIME}).*?\b(?:conclude|concludes|end|ends|close|closes)\b.*?\bat\s+({TIME})",
        rf"\b(?:event|program|programme|session|ceremony)\s+will\s+(?:begin|start)\s+at\s+({TIME}).*?\b(?:end|conclude|close)\s+(?:at|by)\s+({TIME})",
        rf"\b(?:starting at|start at|begin at|at)\s+({TIME})",
        rf"\b({TIME})"
    ])


def extract_venue(body):
    ls, text = lines(body), one_line(body)

    if v := first(ls, ["New Venue", "Main Event Venue"]):
        return clean_venue(v)

    off, on = first(ls, OFFLINE_L), first(ls, ONLINE_L)
    if off or on:
        return " | ".join(x for x in [
            f"Offline: {clean_venue(off)}" if off else "",
            f"Online: {clean_venue(on)}" if on else ""
        ] if x)

    day = []
    for x in ls:
        if m := re.match(r"^Venue\s+for\s+(.+?)\s*[:\-]\s*(.+)$", x, re.I):
            day.append(f"{m.group(1).strip()}: {clean_venue(m.group(2))}")
    if day:
        return " | ".join(day)

    if v := first(ls, ["Venue", "Location", "Place"]):
        return clean_venue(v)

    m = re.search(
        r"\b(?:Venue|Location|Place)\s*[:\-]\s*(.+?)(?=(?:\.|\s+The event\b|\s+The competition\b|\s+The program\b|\s+The session\b|\s+All\b|\s+Participants\b|\s+Students\b|\s+Interested\b|\s+Thanks\b|\s+Regards\b|$))",
        text, re.I
    )
    if m:
        return clean_venue(m.group(1))

    for v in re.findall(
        r"\bat\s+(?!\d{1,2}[:.]\d{2}\s*(?:AM|PM|am|pm))(.+?)(?=(?:\.|\s+We\b|\s+The event\b|\s+The competition\b|\s+The program\b|\s+The session\b|\s+All\b|\s+Participants\b|\s+Students\b|\s+Interested\b|\s+Thanks\b|\s+Regards\b|$))",
        text, re.I
    ):
        v = clean_venue(v)
        if len(v) > 3 and not re.search(r"\b(?:AM|PM)\b", v, re.I):
            return v

    return MISSING


def event_start(line):
    m = re.match(r"^(?:Event\s*\d*|Program(?:me)?|Title)\s*[:\-]\s*(.+)$", line, re.I)
    return m.group(1).strip() if m else None


def add_data(ev, line):
    for key, labels in [
        ("date", DATE_L), ("_sd", SDATE_L), ("_ed", EDATE_L),
        ("time", TIME_L), ("_st", STIME_L), ("_et", ETIME_L),
        ("venue", VENUE_L), ("_off", OFFLINE_L), ("_on", ONLINE_L)
    ]:
        if v := label(line, labels):
            ev[key] = clean_venue(v) if key in ("venue", "_off", "_on") else v

    if m := re.match(r"^Venue\s+for\s+(.+?)\s*[:\-]\s*(.+)$", line, re.I):
        ev.setdefault("_day", []).append(f"{m.group(1).strip()}: {clean_venue(m.group(2))}")


def final(ev):
    if "_sd" in ev and "_ed" in ev:
        ev["date"] = f"{ev['_sd']} to {ev['_ed']}"
    if "_st" in ev and "_et" in ev:
        ev["time"] = f"{ev['_st']} to {ev['_et']}"
    if "_off" in ev or "_on" in ev:
        ev["venue"] = " | ".join(x for x in [
            f"Offline: {ev['_off']}" if "_off" in ev else "",
            f"Online: {ev['_on']}" if "_on" in ev else ""
        ] if x)
    if "_day" in ev:
        ev["venue"] = " | ".join(ev["_day"])

    return {
        "event": clean_event(ev.get("event")) or MISSING,
        "date": ev.get("date", MISSING),
        "time": ev.get("time", MISSING),
        "venue": clean_venue(ev.get("venue", MISSING))
    }


def block_events(body):
    out, cur = [], None

    for x in lines(body):
        if start := event_start(x):
            if cur:
                out.append(final(cur))
            parts = re.split(r"\s*[|;]\s*", start)
            cur = {"event": parts[0].strip()}
            for p in parts[1:]:
                add_data(cur, p)
        elif cur:
            add_data(cur, x)

    if cur:
        out.append(final(cur))

    return out


def table_events(body):
    out = []

    for x in lines(body):
        m = re.match(rf"^\s*\d+[\).\s]+\s*(.+?)\s+-\s+({DATE})\s+-\s+(.+?)\s+-\s+(.+)$", x, re.I)
        if m:
            out.append({
                "event": clean_event(m.group(1)) or MISSING,
                "date": m.group(2).strip(),
                "time": m.group(3).strip(),
                "venue": clean_venue(m.group(4))
            })

    return out


def extract_all(subject, body):
    b = block_events(body)

    if b:
        if len(b) == 1:
            b[0]["date"] = b[0]["date"] if b[0]["date"] != MISSING else extract_date(body)
            b[0]["time"] = b[0]["time"] if b[0]["time"] != MISSING else extract_time(body)
            b[0]["venue"] = b[0]["venue"] if b[0]["venue"] != MISSING else extract_venue(body)
        return b

    t = table_events(body)
    if t:
        return t

    return [{
        "event": extract_event(subject, body),
        "date": extract_date(body),
        "time": extract_time(body),
        "venue": extract_venue(body)
    }]


def extract_event_details(subject, body):
    events = extract_all(subject, body)
    return events[0] if len(events) == 1 else events


if __name__ == "__main__":
    mail = fetch_latest_email()

    if not mail:
        print("No email found.")
    else:
        result = extract_event_details(mail["subject"], mail["body"])
        print(json.dumps(result, indent=4, ensure_ascii=False))