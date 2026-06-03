import imaplib
import email
import re
from email.header import decode_header

# ==========================
# GMAIL LOGIN
# ==========================

imap = imaplib.IMAP4_SSL("imap.gmail.com")
imap.login(
    "srmdean02@gmail.com",
    "xewwthmnlnctavwe"
)

# ==========================
# OPEN INBOX
# ==========================

imap.select("inbox")

# ==========================
# GET LATEST EMAIL
# ==========================

status, messages = imap.search(None, "ALL")

mail_ids = messages[0].split()

if not mail_ids:
    print("No Emails Found")
    exit()

latest_email_id = mail_ids[-1]

status, msg_data = imap.fetch(
    latest_email_id,
    "(RFC822)"
)

email_body = ""
subject = ""

# ==========================
# READ EMAIL
# ==========================

for response_part in msg_data:

    if isinstance(response_part, tuple):

        msg = email.message_from_bytes(
            response_part[1]
        )

        # --------------------------
        # DECODE SUBJECT
        # --------------------------

        raw_subject = msg["Subject"]

        decoded_subject = decode_header(raw_subject)

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

        # --------------------------
        # GET EMAIL BODY
        # --------------------------

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

                        # Prefer plain text
                        if content_type == "text/plain":
                            email_body = text
                            break

                        # Fallback HTML
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
# REMOVE HTML TAGS
# ==========================

email_body = re.sub(
    r"<[^>]+>",
    " ",
    email_body
)

email_body = re.sub(
    r"\s+",
    " ",
    email_body
)

# ==========================
# EVENT NAME
# ==========================

event_name = subject

# ==========================
# DATE EXTRACTION
# ==========================

date = "Not Found"

date_patterns = [

    # July 10 to July 12, 2026
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}\s+to\s+(?:(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+)?\d{1,2},\s+\d{4}",

    # July 12, 2026
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}",

    # 10/07/2026
    r"\d{1,2}/\d{1,2}/\d{4}",

    # 10-07-2026
    r"\d{1,2}-\d{1,2}-\d{4}"
]

for pattern in date_patterns:

    match = re.search(
        pattern,
        email_body,
        re.IGNORECASE
    )

    if match:
        date = match.group()
        break

# ==========================
# TIME EXTRACTION
# ==========================

time = "Not Found"

time_patterns = [
    r"\d{1,2}:\d{2}\s?(?:AM|PM|am|pm)",
    r"\d{1,2}:\d{2}",
    r"\d{1,2}\s?(?:AM|PM|am|pm)"
]

for pattern in time_patterns:

    match = re.search(
        pattern,
        email_body
    )

    if match:
        time = match.group()
        break

# ==========================
# VENUE EXTRACTION
# ==========================

venue = "Not Found"

venue_patterns = [

    r"Venue\s*:\s*([^\.]+)",

    r"Location\s*:\s*([^\.]+)",

    r"Venue\s*-\s*([^\.]+)",

    r"held at\s+([^\.]+)",

    r"conducted at\s+([^\.]+)"
]

for pattern in venue_patterns:

    match = re.search(
        pattern,
        email_body,
        re.IGNORECASE
    )

    if match:

        venue = match.group(1).strip()

        venue = re.sub(
            r"\s+",
            " ",
            venue
        )

        break

# ==========================
# OUTPUT
# ==========================

print("\nEVENT DETAILS")
print("-----------------------------")
print("Event Name :", event_name)
print("Date       :", date)
print("Time       :", time)
print("Venue      :", venue)

# ==========================
# LOGOUT
# ==========================

imap.logout()