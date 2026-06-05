import os
import re
import json
import imaplib
import email
from email import policy
from email.header import decode_header
from dotenv import load_dotenv
from bs4 import BeautifulSoup


# --------------------------------------------------
# LOAD ENV VARIABLES
# --------------------------------------------------

load_dotenv()

EMAIL = os.getenv("srmdean02@gmail.com")
APP_PASSWORD = os.getenv("xewwthmnlnctavwe")
IMAP_SERVER = os.getenv("IMAP_SERVER", "imap.gmail.com")

MISSING = "Not provided in email"


# --------------------------------------------------
# DECODE EMAIL SUBJECT
# --------------------------------------------------

def decode_email_subject(subject):
    if subject is None:
        return ""

    decoded_parts = decode_header(subject)
    decoded_subject = ""

    for part, encoding in decoded_parts:
        if isinstance(part, bytes):
            decoded_subject += part.decode(encoding or "utf-8", errors="ignore")
        else:
            decoded_subject += part

    return decoded_subject.strip()


# --------------------------------------------------
# CONVERT HTML EMAIL TO TEXT
# --------------------------------------------------

def html_to_text(html_content):
    soup = BeautifulSoup(html_content, "html.parser")

    for tag in soup(["script", "style"]):
        tag.decompose()

    text = soup.get_text(separator="\n")

    lines = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            lines.append(line)

    return "\n".join(lines)


# --------------------------------------------------
# EXTRACT EMAIL BODY
# --------------------------------------------------

def extract_email_body(msg):
    text_part = msg.get_body(preferencelist=("plain",))

    if text_part:
        return text_part.get_content().strip()

    html_part = msg.get_body(preferencelist=("html",))

    if html_part:
        return html_to_text(html_part.get_content()).strip()

    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            payload = part.get_payload(decode=True)

            if not payload:
                continue

            if content_type == "text/plain":
                return payload.decode(errors="ignore").strip()

            if content_type == "text/html":
                return html_to_text(payload.decode(errors="ignore")).strip()

    return ""


# --------------------------------------------------
# FETCH LATEST EMAIL
# --------------------------------------------------

def fetch_latest_email():
    if not EMAIL or not APP_PASSWORD:
        raise ValueError("EMAIL or APP_PASSWORD is missing in .env file")

    imap = imaplib.IMAP4_SSL(IMAP_SERVER)

    imap.login(EMAIL, APP_PASSWORD)
    print("Login Successful")

    imap.select("inbox")

    status, messages = imap.search(None, "ALL")

    if status != "OK":
        imap.logout()
        raise Exception("Could not search inbox")

    mail_ids = messages[0].split()

    if not mail_ids:
        imap.logout()
        return None

    latest_email_id = mail_ids[-1]

    status, msg_data = imap.fetch(latest_email_id, "(RFC822)")

    if status != "OK":
        imap.logout()
        raise Exception("Could not fetch email")

    raw_email = msg_data[0][1]
    msg = email.message_from_bytes(raw_email, policy=policy.default)

    subject = decode_email_subject(msg["subject"])
    body = extract_email_body(msg)

    imap.logout()

    return {
        "subject": subject,
        "body": body
    }


# --------------------------------------------------
# TEXT CLEANING
# --------------------------------------------------

def normalize_text(text):
    if not text:
        return ""

    text = text.replace("*", "")
    text = text.replace("–", "-")
    text = text.replace("—", "-")
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def get_clean_lines(text):
    if not text:
        return []

    lines = []

    for line in text.splitlines():
        line = line.replace("*", "").strip()
        if line:
            lines.append(line)

    return lines


# --------------------------------------------------
# EVENT CLEANING
# --------------------------------------------------

def clean_event_name(event):
    if not event:
        return None

    event = event.strip()
    event = event.replace("–", "-")
    event = event.replace("—", "-")
    event = event.strip(" .,-:")

    event = re.sub(r"^(a|an|the)\s+", "", event, flags=re.IGNORECASE)

    unwanted_phrases = [
        "schedule announcement",
        "announcement",
        "invitation",
        "reminder",
        "notification"
    ]

    if event.lower() in unwanted_phrases:
        return None

    return event


# --------------------------------------------------
# EVENT FROM SUBJECT
# --------------------------------------------------

def extract_event_from_subject(subject):
    if not subject:
        return None

    subject = subject.replace("–", "-").replace("—", "-").strip()

    # Example:
    # National Robotics League 2026 - Schedule Announcement
    if "-" in subject:
        left, right = subject.split("-", 1)
        left = left.strip()
        right = right.strip()

        if any(word in right.lower() for word in ["schedule", "announcement", "notification"]):
            return clean_event_name(left)

        if any(word in left.lower() for word in ["invitation", "invite", "regarding"]):
            return clean_event_name(right)

    subject = re.sub(r"^subject\s*:\s*", "", subject, flags=re.IGNORECASE)
    subject = re.sub(r"^invitation\s+to\s+", "", subject, flags=re.IGNORECASE)
    subject = re.sub(r"^invitation\s+for\s+", "", subject, flags=re.IGNORECASE)
    subject = re.sub(r"\s+for\s+students.*$", "", subject, flags=re.IGNORECASE)
    subject = re.sub(r"\s+for\s+final\s+year\s+students.*$", "", subject, flags=re.IGNORECASE)

    return clean_event_name(subject)


# --------------------------------------------------
# EVENT EXTRACTION
# --------------------------------------------------

def extract_event(subject, body):
    text = normalize_text(body)

    event_patterns = [
        r"\binvite you to\s+(.+?)(?=,|\.)",
        r"\bdelighted to invite you to\s+(.+?)(?=,|\.)",
        r"\bhappy to inform you that .*? organizing\s+(?:a|an|the)?\s*(.+?)(?=,|\.)",
        r"\borganizing\s+(?:a|an|the)?\s*(.+?)(?=\s+for\b|,|\.)",
        r"\borganising\s+(?:a|an|the)?\s*(.+?)(?=\s+for\b|,|\.)",
        r"\bconducting\s+(?:a|an|the)?\s*(.+?)(?=\s+for\b|,|\.)",
        r"\bhosting\s+(?:a|an|the)?\s*(.+?)(?=,|\.)",
        r"\bannounce the schedule for\s+(.+?)(?=,|\.)",
        r"\bexcited to announce the schedule for\s+(.+?)(?=,|\.)",
        r"\bevent\s*[:\-]\s*(.+?)(?=\.|$)",
        r"\bprogram\s*[:\-]\s*(.+?)(?=\.|$)",
        r"\bprogramme\s*[:\-]\s*(.+?)(?=\.|$)",
        r"\btitle\s*[:\-]\s*(.+?)(?=\.|$)"
    ]

    for pattern in event_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            event = clean_event_name(match.group(1))
            if event:
                return event

    event_from_subject = extract_event_from_subject(subject)

    if event_from_subject:
        return event_from_subject

    return MISSING


# --------------------------------------------------
# DATE EXTRACTION
# --------------------------------------------------

def extract_date(body):
    text = normalize_text(body)

    month = r"(?:Jan|January|Feb|February|Mar|March|Apr|April|May|Jun|June|Jul|July|Aug|August|Sep|September|Oct|October|Nov|November|Dec|December)"

    # 21 November 2026
    day_month_year = rf"\d{{1,2}}\s+{month}\s+\d{{4}}"

    # November 21, 2026
    month_day_year = rf"{month}\s+\d{{1,2}},?\s+\d{{4}}"

    # Date range: commence on 21 November 2026 ... conclude on 23 November 2026
    range_patterns = [
        rf"\bcommence on\s+({day_month_year}).*?\bconclude on\s+({day_month_year})",
        rf"\bstart on\s+({day_month_year}).*?\bend on\s+({day_month_year})",
        rf"\bfrom\s+({month_day_year})\s+to\s+({month_day_year})",
        rf"\bfrom\s+({day_month_year})\s+to\s+({day_month_year})",

        # May 4 to May 9, 2027
        rf"\bfrom\s+({month}\s+\d{{1,2}}\s+to\s+{month}\s+\d{{1,2}},?\s+\d{{4}})",

        # May 4 to 9, 2027
        rf"\bfrom\s+({month}\s+\d{{1,2}}\s+to\s+\d{{1,2}},?\s+\d{{4}})"
    ]

    for pattern in range_patterns:
        match = re.search(pattern, text, re.IGNORECASE)

        if match:
            if len(match.groups()) == 2:
                return match.group(1).strip() + " to " + match.group(2).strip()
            return match.group(1).strip()

    single_date_patterns = [
        rf"\bon\s+({month_day_year})",
        rf"\bon\s+({day_month_year})",
        rf"\bheld on\s+({month_day_year})",
        rf"\bheld on\s+({day_month_year})",
        rf"\bconducted on\s+({month_day_year})",
        rf"\bconducted on\s+({day_month_year})",
        rf"\b({month_day_year})",
        rf"\b({day_month_year})",
        r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b",
        r"\b(\d{4}-\d{2}-\d{2})\b"
    ]

    for pattern in single_date_patterns:
        match = re.search(pattern, text, re.IGNORECASE)

        if match:
            return match.group(1).strip()

    return MISSING


# --------------------------------------------------
# TIME EXTRACTION
# --------------------------------------------------

def extract_time(body):
    text = normalize_text(body)

    time_value = r"\d{1,2}[:.]\d{2}\s*(?:AM|PM|am|pm)|\d{1,2}\s*(?:AM|PM|am|pm)"

    range_patterns = [
        rf"\bfrom\s+({time_value})\s+to\s+({time_value})",
        rf"\bbetween\s+({time_value})\s+and\s+({time_value})",
        rf"\bcommence\b.*?\bat\s+({time_value}).*?\bconclude\b.*?\bat\s+({time_value})",
        rf"\bstart\b.*?\bat\s+({time_value}).*?\bend\b.*?\bat\s+({time_value})"
    ]

    for pattern in range_patterns:
        match = re.search(pattern, text, re.IGNORECASE)

        if match:
            return match.group(1).strip() + " to " + match.group(2).strip()

    single_time_patterns = [
        rf"\btime\s*[:\-]\s*({time_value})",
        rf"\bstarting\s+at\s+({time_value})",
        rf"\bat\s+({time_value})",
        rf"\b({time_value})"
    ]

    for pattern in single_time_patterns:
        match = re.search(pattern, text, re.IGNORECASE)

        if match:
            return match.group(1).strip()

    return MISSING


# --------------------------------------------------
# VENUE EXTRACTION
# --------------------------------------------------

def extract_venue(body):
    lines = get_clean_lines(body)
    venues = []

    # First priority: line-based venue extraction
    for line in lines:
        match = re.search(
            r"^venue(?:\s+for\s+[^:]+)?\s*[:\-]\s*(.+)$",
            line,
            re.IGNORECASE
        )

        if match:
            venue = match.group(1).strip(" .,-")
            if venue:
                venues.append(venue)

    if venues:
        return " | ".join(venues)

    text = normalize_text(body)

    # Second priority: labeled venue in paragraph
    label_patterns = [
        r"\bvenue\s*[:\-]\s*(.+?)(?=(?:\.|\s+The event\b|\s+The competition\b|\s+All\b|\s+Participants\b|\s+Interested\b|\s+Thanks\b|\s+Regards\b|$))",
        r"\blocation\s*[:\-]\s*(.+?)(?=(?:\.|\s+The event\b|\s+The competition\b|\s+All\b|\s+Participants\b|\s+Interested\b|\s+Thanks\b|\s+Regards\b|$))",
        r"\bplace\s*[:\-]\s*(.+?)(?=(?:\.|\s+The event\b|\s+The competition\b|\s+All\b|\s+Participants\b|\s+Interested\b|\s+Thanks\b|\s+Regards\b|$))"
    ]

    for pattern in label_patterns:
        match = re.search(pattern, text, re.IGNORECASE)

        if match:
            venue = match.group(1).strip(" .,-")
            if venue:
                return venue

    # Third priority: "at <venue>" but avoid "at 09:00 AM"
    at_patterns = [
        r"\bat\s+(?!\d{1,2}[:.]\d{2}\s*(?:AM|PM|am|pm))(.+?)(?=(?:\.|\s+The event\b|\s+The competition\b|\s+All\b|\s+Participants\b|\s+Interested\b|\s+Thanks\b|\s+Regards\b|$))"
    ]

    for pattern in at_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)

        for venue in matches:
            venue = venue.strip(" .,-")

            # Remove accidental time/date fragments if present
            venue = re.sub(
                r"^\d{1,2}[:.]\d{2}\s*(?:AM|PM|am|pm)\s*(?:and|to)?\s*",
                "",
                venue,
                flags=re.IGNORECASE
            ).strip()

            # Ignore bad venue values
            if re.search(r"\bAM\b|\bPM\b", venue, re.IGNORECASE):
                continue

            if venue:
                return venue

    return MISSING


# --------------------------------------------------
# FINAL EVENT DETAILS EXTRACTION
# --------------------------------------------------

def extract_event_details(subject, body):
    return {
        "event": extract_event(subject, body),
        "date": extract_date(body),
        "time": extract_time(body),
        "venue": extract_venue(body)
    }


# --------------------------------------------------
# MAIN PROGRAM
# --------------------------------------------------

if __name__ == "__main__":
    email_data = fetch_latest_email()

    if email_data is None:
        print("No email found.")
    else:
        result = extract_event_details(
            email_data["subject"],
            email_data["body"]
        )

        print(json.dumps(result, indent=4))