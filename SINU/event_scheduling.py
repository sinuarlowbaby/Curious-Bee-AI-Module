from openai import AsyncOpenAI
from dotenv import load_dotenv
load_dotenv()
import os
import json
from langsmith import traceable
import imaplib
import email

client = AsyncOpenAI(
    base_url="http://localhost:11434/v1", # Point to local Ollama server
    api_key="ollama",                     # Required field, but its value is ignored by Ollama
)

@traceable(name="retrive email", run_type="tool")
def read_mail():

    EMAIL = os.getenv("GMAIL_EMAIL")
    APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD")
    imap = imaplib.IMAP4_SSL("imap.gmail.com")
    imap.login(EMAIL, APP_PASSWORD)

    imap.select("inbox")

    status, messages = imap.search(None, "ALL")

    mail_ids = messages[0].split()

    latest_email_id = mail_ids[-1]

    status, msg_data = imap.fetch(latest_email_id, "(RFC822)")

    # -----------------------------
    # READ EMAIL CONTENT
    # -----------------------------
    
    body = ""

    for response_part in msg_data:

        if isinstance(response_part, tuple):

            # Convert bytes into email object
            msg = email.message_from_bytes(response_part[1])


            if msg.is_multipart():

                for part in msg.walk():

                    content_type = part.get_content_type()

                    if content_type == "text/plain":

                        body = part.get_payload(decode=True).decode(errors="ignore")

                        # print(body)

            else:

                body = msg.get_payload(decode=True).decode(errors="ignore")

    imap.logout()
    return body

@traceable(name="generate_summary", run_type="tool", metadata={"llm":"qwen2.5:1.5b"})
async def generate_summary(text = None):


    response = await client.chat.completions.create(
        model="qwen2.5:1.5b",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
            "content": """ You are an information extraction assistant.

            Extract event details from the given text and return ONLY valid JSON.

            Rules:
            - Extract:
            - event,
            - date (YYYY-MM-DD)
            - time
            - venue
            - If a field is not found, return null.
            - Do not include explanations.
            - Do not include markdown.
            - Return only a JSON object.

            Example:
            {
  "event": "Staff Meeting",
  "date": "2026-06-15",
  "time": "10 AM",
  "venue": "Seminar Hall"
}
"""
            },
            {
                "role": "user",
                "content": text
            }
        ]
    )

    return response.choices[0].message.content

import asyncio

def main():
    if __name__ == "__main__":
        email = read_mail()
        data = json.loads(asyncio.run(generate_summary(email)))
        print(data)
        
main()