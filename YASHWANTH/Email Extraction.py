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

EMAIL = os.getenv("EMAIL")
APP_PASSWORD = os.getenv("APP_PASSWORD")
IMAP_SERVER = os.getenv("IMAP_SERVER", "imap.gmail.com")

MISSING = "Not provided in email"


# --------------------------------------------------
# EMAIL FETCHING FUNCTIONS
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
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


def one_line(text):
    text = normalize_text(text)
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
# COMMON REGEX VALUES
# --------------------------------------------------

MONTH = (
    r"(?:Jan|January|Feb|February|Mar|March|Apr|April|May|Jun|June|"
    r"Jul|July|Aug|August|Sep|September|Oct|October|Nov|November|Dec|December)"
)

DAY_MONTH_YEAR = rf"\d{{1,2}}\s+{MONTH}\s+\d{{4}}"
MONTH_DAY_YEAR = rf"{MONTH}\s+\d{{1,2}},?\s+\d{{4}}"

DATE_ANY = (
    rf"(?:{DAY_MONTH_YEAR}|{MONTH_DAY_YEAR}|"
    r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|"
    r"\d{4}-\d{2}-\d{2})"
)

TIME_VAL = (
    r"(?:\d{1,2}[:.]\d{2}\s*(?:AM|PM|am|pm)|"
    r"\d{1,2}\s*(?:AM|PM|am|pm))"
)


def find_line_label(lines, labels):
    values = []

    for line in lines:
        for label in labels:
            pattern = rf"^{label}\s*[:\-]\s*(.+)$"
            match = re.search(pattern, line, re.IGNORECASE)

            if match:
                values.append(match.group(1).strip(" .,-"))

    return values


# --------------------------------------------------
# EVENT EXTRACTION
# --------------------------------------------------

def clean_event_name(event):
    if not event:
        return None

    event = event.strip()
    event = event.replace("–", "-")
    event = event.replace("—", "-")
    event = event.strip(" .,-:")

    event = re.sub(r"^(a|an|the)\s+", "", event, flags=re.IGNORECASE)

    event = re.sub(
        r"\s+for\s+(students|faculty|participants|final year students|"
        r"all students|students and staff members).*$",
        "",
        event,
        flags=re.IGNORECASE
    )

    event = re.sub(
        r"\s+to\s+(connect|showcase|encourage|educate).*$",
        "",
        event,
        flags=re.IGNORECASE
    )

    event = re.sub(
        r"\b(has been|will be|is)\b.*$",
        "",
        event,
        flags=re.IGNORECASE
    ).strip()

    unwanted = {
        "announcement",
        "schedule announcement",
        "notice",
        "event invitation",
        "registration details",
        "important update"
    }

    if event.lower() in unwanted:
        return None

    return event if event else None


def extract_event_from_subject(subject):
    if not subject:
        return None

    subject = subject.strip()
    subject = re.sub(r"^subject\s*:\s*", "", subject, flags=re.IGNORECASE)
    subject = subject.replace("–", "-").replace("—", "-").strip()

    prefixes = [
        r"^Circular\s*:\s*",
        r"^Reminder\s*:\s*",
        r"^Important Update\s*:\s*",
        r"^Cancellation Notice\s*:\s*",
        r"^Rescheduled Date for\s+",
        r"^Venue Change for\s+",
        r"^Agenda for\s+",
        r"^Permission Required for\s+",
        r"^Entry Pass Collection for\s+",
        r"^Certificate Distribution for\s+",
        r"^Result Announcement Ceremony for\s+",
        r"^Invitation to\s+",
        r"^Invitation for\s+",
    ]

    for pattern in prefixes:
        subject = re.sub(pattern, "", subject, flags=re.IGNORECASE).strip()

    if "-" in subject:
        left, right = subject.split("-", 1)
        left = left.strip()
        right = right.strip()

        if re.search(
            r"(schedule|announcement|notice|notification|registration details|"
            r"event invitation|project display)",
            right,
            re.IGNORECASE
        ):
            return clean_event_name(left)

        if (
            re.search(r"(challenge|fest|competition|hackathon|exhibition)\b", left, re.IGNORECASE)
            and re.search(r"\d{4}", right)
        ):
            return clean_event_name(right)

        return clean_event_name(subject)

    subject = re.sub(
        r"\s+(notice|announcement|schedule|notification|reminder).*$",
        "",
        subject,
        flags=re.IGNORECASE
    ).strip()

    return clean_event_name(subject)


def extract_event(subject, body):
    text = one_line(body)

    patterns = [
        r"\bannounce the schedule for\s+(.+?)(?=\.|,)",
        r"\bexcited to announce the schedule for\s+(.+?)(?=\.|,)",
        r"\bpleased to announce\s+(?:a|an|the)?\s*(.+?)(?=\.|,)",

        r"\bwill conduct\s+(?:a|an|the)?\s*(.+?)(?=\s+on\b|\.|,)",

        r"\bis organizing\s+(?:a|an|the)?\s*(.+?)(?=\s+for\b|\.|,)",
        r"\bare organizing\s+(?:a|an|the)?\s*(.+?)(?=\s+for\b|\.|,)",
        r"\bis arranging\s+(?:a|an|the)?\s*(.+?)(?=\s+for\b|\.|,)",
        r"\bis conducting\s+(?:a|an|the)?\s*(.+?)(?=\.|,)",
        r"\bare conducting\s+(?:a|an|the)?\s*(.+?)(?=\.|,)",

        r"\binvite you to participate in\s+(.+?)(?=\.|,)",
        r"\binvite you to\s+(.+?)(?=\.|,)",
        r"\bdelighted to invite you to\s+(.+?)(?=\.|,)",

        r"\bThe\s+(.+?)\s+will be organized\b",
        r"\bThe\s+(.+?)\s+will be conducted\b",
        r"\bThe\s+(.+?)\s+will be held\b",

        r"\bthe\s+(.+?)\s+has been scheduled\b",
        r"\bthe\s+(.+?)\s+is scheduled\b",
        r"\bthe\s+(.+?)\s+will now be conducted\b",
        r"\bthe\s+(.+?)\s+planned for\b",
        r"\bthe\s+(.+?)\s+originally scheduled\b",

        r"\bfor the\s+(.+?)\s+will be distributed\b",
        r"\bfor\s+(.+?)\s+will be conducted\b",

        r"\bevent\s*[:\-]\s*(.+?)(?=\.|$)",
        r"\bprogram\s*[:\-]\s*(.+?)(?=\.|$)",
        r"\bprogramme\s*[:\-]\s*(.+?)(?=\.|$)",
        r"\btitle\s*[:\-]\s*(.+?)(?=\.|$)",
    ]

    for pattern in patterns:
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
    text = one_line(body)
    lines = get_clean_lines(body)

    main_event_date = find_line_label(lines, [r"Main Event Date"])

    if main_event_date:
        return main_event_date[0]

    date_range_pairs = [
        (r"New Start Date", r"New Closing Date"),
        (r"Event Start Date", r"Event End Date"),
        (r"Event opening date", r"Event closing date"),
        (r"Camp Opening Date", r"Camp Closing Date"),
    ]

    for start_label, end_label in date_range_pairs:
        start_value = find_line_label(lines, [start_label])
        end_value = find_line_label(lines, [end_label])

        if start_value and end_value:
            return start_value[0] + " to " + end_value[0]

    single_date_labels = [
        r"Date of Event",
        r"Event Date",
        r"Visit Date",
        r"Date",
        r"Pass Collection Date",
        r"Collection Date",
    ]

    single_dates = find_line_label(lines, single_date_labels)

    if single_dates:
        return single_dates[0]

    patterns = [
        rf"\b(?:will begin|begins|begin) on\s+({DATE_ANY}).*?\b(?:will close|closes|close) on\s+({DATE_ANY})",
        rf"\b(?:event start date is)\s+({DATE_ANY}).*?\b(?:event end date is)\s+({DATE_ANY})",
        rf"\b(?:commence|commences) on\s+({DATE_ANY}).*?\b(?:conclude|concludes) on\s+({DATE_ANY})",
        rf"\b(?:starts from|start from)\s+({DATE_ANY}).*?\b(?:ends on|end on)\s+({DATE_ANY})",
        rf"\b(?:will start|start) on\s+({DATE_ANY}).*?\b(?:will end|end) on\s+({DATE_ANY})",

        rf"\b(?:will be conducted|conducted|take place|takes place|will take place)\s+from\s+({MONTH}\s+\d{{1,2}}\s+to\s+{MONTH}\s+\d{{1,2}},?\s+\d{{4}})",
        rf"\b(?:will be conducted|conducted|take place|takes place|will take place)\s+from\s+({MONTH}\s+\d{{1,2}}\s+to\s+\d{{1,2}},?\s+\d{{4}})",

        rf"\bfrom\s+({MONTH}\s+\d{{1,2}}\s+to\s+{MONTH}\s+\d{{1,2}},?\s+\d{{4}})",
        rf"\bfrom\s+({MONTH}\s+\d{{1,2}}\s+to\s+\d{{1,2}},?\s+\d{{4}})",
        rf"\bfrom\s+({DATE_ANY})\s+to\s+({DATE_ANY})",

        rf"\bnow be conducted on\s+({DATE_ANY})",
        rf"\bscheduled for\s+({DATE_ANY})",
        rf"\bscheduled on\s+({DATE_ANY})",
        rf"\bheld on\s+({DATE_ANY})",
        rf"\bconducted on\s+({DATE_ANY})",
        rf"\bplanned for\s+({DATE_ANY})",
        rf"\bwill be distributed on\s+({DATE_ANY})",
        rf"\bwill be held on\s+({DATE_ANY})",
        rf"\bon\s+({DATE_ANY})",

        rf"\b({DATE_ANY})",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)

        if match:
            if len(match.groups()) >= 2 and match.group(2):
                return match.group(1).strip() + " to " + match.group(2).strip()

            return match.group(1).strip()

    return MISSING


# --------------------------------------------------
# TIME EXTRACTION
# --------------------------------------------------

def extract_time(body):
    text = one_line(body)
    lines = get_clean_lines(body)

    main_event_time = find_line_label(lines, [r"Main Event Time"])

    if main_event_time:
        return main_event_time[0]

    start_time = find_line_label(lines, [r"Start Time", r"Departure Time"])
    end_time = find_line_label(lines, [r"End Time", r"Return Time"])

    if start_time and end_time:
        return start_time[0] + " to " + end_time[0]

    direct_time_labels = [
        r"Program Time",
        r"Visitor timing",
        r"Event timing",
        r"Workshop Timing",
        r"Camp Timing",
        r"Collection Time",
        r"Time",
    ]

    direct_times = find_line_label(lines, direct_time_labels)

    if direct_times:
        return direct_times[0]

    if re.search(r"\bAgenda\s*:", body, re.IGNORECASE):
        times = re.findall(TIME_VAL, body, re.IGNORECASE)

        close_match = re.search(
            rf"\b(?:event|program|programme|session)\s+will\s+close\s+at\s+({TIME_VAL})",
            text,
            re.IGNORECASE
        )

        if times:
            if close_match:
                return times[0].strip() + " to " + close_match.group(1).strip()

            return times[0].strip() + " to " + times[-1].strip()

    patterns = [
        rf"\bfrom\s+({TIME_VAL})\s+to\s+({TIME_VAL})",
        rf"\bbetween\s+({TIME_VAL})\s+and\s+({TIME_VAL})",

        rf"\b(?:commence|commences|start|starts|begin|begins)\b.*?\bat\s+({TIME_VAL}).*?\b(?:conclude|concludes|end|ends|close|closes)\b.*?\bat\s+({TIME_VAL})",

        rf"\b(?:event|program|programme|session|ceremony)\s+will\s+(?:begin|start)\s+at\s+({TIME_VAL}).*?\b(?:end|conclude|close)\s+(?:at|by)\s+({TIME_VAL})",

        rf"\bstarting at\s+({TIME_VAL})",
        rf"\bstart at\s+({TIME_VAL})",
        rf"\bbegin at\s+({TIME_VAL})",
        rf"\bat\s+({TIME_VAL})",
        rf"\b({TIME_VAL})",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)

        if match:
            if len(match.groups()) >= 2 and match.group(2):
                return match.group(1).strip() + " to " + match.group(2).strip()

            return match.group(1).strip()

    return MISSING


# --------------------------------------------------
# VENUE EXTRACTION
# --------------------------------------------------

def extract_venue(body):
    lines = get_clean_lines(body)

    new_venue = find_line_label(lines, [r"New Venue"])

    if new_venue:
        return new_venue[0]

    main_event_venue = find_line_label(lines, [r"Main Event Venue"])

    if main_event_venue:
        return main_event_venue[0]

    offline_venue = find_line_label(lines, [r"Offline Venue"])
    online_venue = find_line_label(lines, [r"Online Venue"])

    if offline_venue or online_venue:
        venues = []

        if offline_venue:
            venues.append("Offline: " + offline_venue[0])

        if online_venue:
            venues.append("Online: " + online_venue[0])

        return " | ".join(venues)

    day_venues = []
    normal_venues = []

    for line in lines:
        day_match = re.search(
            r"^Venue\s+for\s+(.+?)\s*[:\-]\s*(.+)$",
            line,
            re.IGNORECASE
        )

        if day_match:
            day_venues.append(
                day_match.group(1).strip() + ": " + day_match.group(2).strip(" .,-")
            )
            continue

        normal_match = re.search(
            r"^Venue\s*[:\-]\s*(.+)$",
            line,
            re.IGNORECASE
        )

        if normal_match:
            normal_venues.append(normal_match.group(1).strip(" .,-"))

    if day_venues:
        return " | ".join(day_venues)

    if normal_venues:
        return normal_venues[0]

    location_values = find_line_label(lines, [r"Location", r"Place"])

    if location_values:
        return location_values[0]

    text = one_line(body)

    label_match = re.search(
        r"\b(?:Venue|Location|Place)\s*[:\-]\s*(.+?)"
        r"(?=(?:\.|\s+The event\b|\s+The competition\b|\s+The program\b|"
        r"\s+The session\b|\s+All\b|\s+Participants\b|\s+Students\b|"
        r"\s+Interested\b|\s+Thanks\b|\s+Regards\b|$))",
        text,
        re.IGNORECASE
    )

    if label_match:
        return label_match.group(1).strip(" .,-")

    at_matches = re.findall(
        r"\bat\s+"
        r"(?!\d{1,2}[:.]\d{2}\s*(?:AM|PM|am|pm))"
        r"(.+?)"
        r"(?=(?:\.|\s+The event\b|\s+The competition\b|\s+The program\b|"
        r"\s+The session\b|\s+All\b|\s+Participants\b|\s+Students\b|"
        r"\s+Interested\b|\s+Thanks\b|\s+Regards\b|$))",
        text,
        re.IGNORECASE
    )

    for venue in at_matches:
        venue = venue.strip(" .,-")

        if re.search(r"\b(?:AM|PM)\b", venue, re.IGNORECASE):
            continue

        if len(venue) > 3:
            return venue

    return MISSING


# --------------------------------------------------
# FINAL EXTRACTION
# --------------------------------------------------

def extract_event_details(subject, body):
    return {
        "subject": subject if subject else MISSING,
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