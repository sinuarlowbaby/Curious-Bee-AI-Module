import imaplib
import email
import ollama

# Gmail Login
imap = imaplib.IMAP4_SSL("imap.gmail.com")
imap.login("srmdean02@gmail.com", "xewwthmnlnctavwe")

# Open Inbox
imap.select("inbox")

# Read Latest Email
status, messages = imap.search(None, "ALL")

mail_ids = messages[0].split()
latest_email_id = mail_ids[-1]

status, msg_data = imap.fetch(latest_email_id, "(RFC822)")

email_body = ""

for response_part in msg_data:
    if isinstance(response_part, tuple):

        msg = email.message_from_bytes(response_part[1])

        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    email_body = part.get_payload(
                        decode=True
                    ).decode()
                    break
        else:
            email_body = msg.get_payload(
                decode=True
            ).decode()

# Send Email Body to SLM
prompt = f"""
Extract only:

1. Event Name
2. Date
3. Time
4. Venue

Return ONLY JSON.

Email:
{email_body}
"""

response = ollama.chat(
    model="phi3",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

print(response["message"]["content"])

imap.logout()