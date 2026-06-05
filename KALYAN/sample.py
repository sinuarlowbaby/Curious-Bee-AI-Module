# ==========================
# IMPORTS
# ==========================

import imaplib
import email
import re
from email.header import decode_header


# ==========================
# VENUE EXTRACTOR
# ==========================

def extract_venue(email_body):

    venue_words = [
        "Hall",
        "Auditorium",
        "Seminar Hall",
        "Conference Hall",
        "Lab",
        "Laboratory",
        "Studio",
        "Room",
        "Block",
        "Centre",
        "Center",
        "Complex",
        "Ground",
        "Arena",
        "Classroom"
    ]

    venue_match = re.search(
        r"Venue\s*:\s*(.*?)(?:\n|$)",
        email_body,
        re.IGNORECASE
    )

    if venue_match:
        return venue_match.group(1).strip()

    sentences = re.split(r"[.!?\n]+", email_body)

    for sentence in sentences:

        sentence = sentence.strip()

        for word in venue_words:

            if word.lower() in sentence.lower():

                at_match = re.search(
                    r"\bat\s+(.*)",
                    sentence,
                    re.IGNORECASE
                )

                if at_match:
                    return at_match.group(1).strip()

                return sentence

    return "Not Found"
# ==========================
# GMAIL LOGIN
# ==========================

import imaplib

EMAIL_ID = "srmdean02@gmail.com"
APP_PASSWORD = "xewwthmnlnctavwe"

try:

    imap = imaplib.IMAP4_SSL(
        "imap.gmail.com"
    )

    imap.login(
        EMAIL_ID,
        APP_PASSWORD
    )

    print("Connected Successfully")

except Exception as e:

    print("Login Failed:", e)
    exit()
# ==========================
# OPEN INBOX
# ==========================

imap.select("inbox")

status, messages = imap.search(
    None,
    "ALL"
)

mail_ids = messages[0].split()

if not mail_ids:

    print("No Emails Found")
    exit()

latest_email_id = mail_ids[-1]

status, msg_data = imap.fetch(
    latest_email_id,
    "(RFC822)"
)


# ==========================
# READ EMAIL
# ==========================

email_body = ""
subject = ""

for response_part in msg_data:

    if isinstance(response_part, tuple):

        msg = email.message_from_bytes(
            response_part[1]
        )

        raw_subject = msg["Subject"]

        decoded_subject = decode_header(
            raw_subject
        )

        subject_parts = []

        for part, encoding in decoded_subject:

            if isinstance(part, bytes):

                subject_parts.append(
                    part.decode(
                        encoding if encoding else "utf-8",
                        errors="ignore"
                    )
                )

            else:

                subject_parts.append(part)

        subject = "".join(subject_parts)

        if msg.is_multipart():

            for part in msg.walk():

                content_type = part.get_content_type()

                try:

                    payload = part.get_payload(
                        decode=True
                    )

                    if payload:

                        text = payload.decode(
                            errors="ignore"
                        )

                        if content_type == "text/plain":

                            email_body = text
                            break

                        elif (
                            content_type == "text/html"
                            and not email_body
                        ):

                            email_body = text

                except:
                    pass

        else:

            payload = msg.get_payload(
                decode=True
            )

            if payload:

                email_body = payload.decode(
                    errors="ignore"
                )


# ==========================
# CLEAN EMAIL BODY
# ==========================

email_body = re.sub(
    r"<[^>]+>",
    " ",
    email_body
)

email_body = re.sub(
    r"&nbsp;",
    " ",
    email_body
)

email_body = re.sub(
    r"[ \t]+",
    " ",
    email_body
)

email_body = re.sub(
    r"\n+",
    "\n",
    email_body
)

email_body = email_body.strip()


# ==========================
# EVENT NAME
# ==========================

event_name = subject.strip()

event_name = re.sub(
    r"^(Re:|Fwd:|FW:)\s*",
    "",
    event_name,
    flags=re.IGNORECASE
)


# ==========================
# STATUS
# ==========================

event_status = "Active"

if re.search(
    r"\bcancelled\b|\bcanceled\b",
    email_body,
    re.IGNORECASE
):

    event_status = "Cancelled"

elif re.search(
    r"\brescheduled\b",
    email_body,
    re.IGNORECASE
):

    event_status = "Rescheduled"


# ==========================
# DATE EXTRACTION
# ==========================

event_date = "Not Found"

date_patterns = [

    r"(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}\s+to\s+(?:(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+)?\d{1,2},\s+\d{4}",

    r"\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}",

    r"(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}",

    r"\d{1,2}-\d{1,2}-\d{4}",

    r"\d{1,2}/\d{1,2}/\d{4}"
]

for pattern in date_patterns:

    match = re.search(
        pattern,
        email_body,
        re.IGNORECASE
    )

    if match:

        event_date = match.group().strip()
        break


# ==========================
# TIME EXTRACTION
# ==========================

start_time = "Not Found"
end_time = "Not Found"

time_match = re.search(
    r"(\d{1,2}[:.]\d{2}\s?(?:AM|PM|am|pm))\s*(?:to|-)\s*(\d{1,2}[:.]\d{2}\s?(?:AM|PM|am|pm))",
    email_body,
    re.IGNORECASE
)

if time_match:

    start_time = time_match.group(1)
    end_time = time_match.group(2)

else:

    single_time = re.search(
        r"\d{1,2}[:.]\d{2}\s?(?:AM|PM|am|pm)",
        email_body
    )

    if single_time:

        start_time = single_time.group()


# ==========================
# VENUE EXTRACTION
# ==========================

venue = extract_venue(
    email_body
)


# ==========================
# OUTPUT
# ==========================

print("\nEVENT DETAILS")
print("--------------------------------")
print("Event Name :", event_name)
print("Date       :", event_date)

if start_time != "Not Found":
    print("Start Time :", start_time)

if end_time != "Not Found":
    print("End Time   :", end_time)

print("Venue      :", venue)
print("Status     :", event_status)


# ==========================
# LOGOUT
# ==========================

imap.logout()