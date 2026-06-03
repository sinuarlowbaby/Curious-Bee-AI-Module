import imaplib
import email
import re

# Gmail Login
imap = imaplib.IMAP4_SSL("imap.gmail.com")
imap.login("srmdean02@gmail.com", "xewwthmnlnctavwe")

# Open Inbox
imap.select("inbox")

# Get Emails
status, messages = imap.search(None, "ALL")

mail_ids = messages[0].split()

if len(mail_ids) == 0:
    print("No Emails Found")
    exit()

latest_email_id = mail_ids[-1]

status, msg_data = imap.fetch(latest_email_id, "(RFC822)")

email_body = ""
subject = ""

for response_part in msg_data:

    if isinstance(response_part, tuple):

        msg = email.message_from_bytes(response_part[1])

        subject = msg["Subject"]

        if msg.is_multipart():

            for part in msg.walk():

                if part.get_content_type() == "text/plain":

                    email_body = part.get_payload(
                        decode=True
                    ).decode(errors="ignore")

                    break

        else:

            email_body = msg.get_payload(
                decode=True
            ).decode(errors="ignore")

# -------------------------
# EVENT NAME
# -------------------------

event_name = subject

# -------------------------
# DATE
# -------------------------

date_patterns = [
    r"\d{1,2}/\d{1,2}/\d{4}",
    r"\d{1,2}-\d{1,2}-\d{4}"
]

date = "Not Found"

for pattern in date_patterns:

    match = re.search(pattern, email_body)

    if match:
        date = match.group()
        break

# -------------------------
# TIME
# -------------------------

time_pattern = r"\d{1,2}:\d{2}\s?(AM|PM|am|pm)?"

time_match = re.search(time_pattern, email_body)

time = time_match.group() if time_match else "Not Found"

# -------------------------
# VENUE
# -------------------------

venue = "Not Found"

venue_patterns = [
    r"Venue\s*:\s*(.*)",
    r"Location\s*:\s*(.*)"
]

for pattern in venue_patterns:

    match = re.search(
        pattern,
        email_body,
        re.IGNORECASE
    )

    if match:
        venue = match.group(1).strip()
        break

# -------------------------
# OUTPUT
# -------------------------

print("\nEVENT DETAILS")
print("----------------------")
print("Event Name :", event_name)
print("Date       :", date)
print("Time       :", time)
print("Venue      :", venue)

imap.logout()