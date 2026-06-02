import imaplib
import email
import ollama
import json

# ---------------------------
# READ EMAIL FROM GMAIL
# ---------------------------

imap = imaplib.IMAP4_SSL("imap.gmail.com")

imap.login("srmdean02@gmail.com", "xewwthmnlnctavwe")

imap.select("inbox")

status, messages = imap.search(None, "ALL")

mail_ids = messages[0].split()

latest_email_id = mail_ids[-1]

status, msg_data = imap.fetch(latest_email_id, "(RFC822)")

email_body = ""

for response_part in msg_data:
    if isinstance(response_part, tuple):
        msg = email.message_from_bytes(response_part[1])

        print("Subject:", msg["subject"])
        print("From:", msg["from"])

        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    email_body = part.get_payload(decode=True).decode()
                    break
        else:
            email_body = msg.get_payload(decode=True).decode()

print("\nEMAIL CONTENT:")
print(email_body)

# ---------------------------
# SLM ANALYSIS USING OLLAMA
# ---------------------------

prompt = f"""
You are an AI Email Analyzer.

Analyze the email and return ONLY JSON.

Extract:

1. Named Entities:
   - Event Name
   - Date
   - Time
   - Venue
   - Organization

2. Intent Detection:
   - New Event
   - Event Cancellation
   - Event Reschedule
   - General Announcement

3. Text Classification:
   - Workshop
   - Seminar
   - Meeting
   - Examination
   - Placement Drive

4. Keywords

5. Priority:
   - High
   - Medium
   - Low

Email:

{email_body}

Return format:

{{
    "event_name":"",
    "date":"",
    "time":"",
    "venue":"",
    "organization":"",
    "intent":"",
    "classification":"",
    "keywords":[],
    "priority":""
}}
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

result = response["message"]["content"]

print("\nAI ANALYSIS:")
print(result)

imap.logout()