import asyncio
import email
import imaplib
import json
import time
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

client = AsyncOpenAI(
    base_url="http://localhost:11434/v1", api_key="ollama"
)


def read_mail():
    print("Connecting to Gmail IMAP...")
    imap = imaplib.IMAP4_SSL("imap.gmail.com")
    imap.login("srmdean02@gmail.com", "xewwthmnlnctavwe")
    print("Login Successful ✅")

    imap.select("inbox")
    status, search_data = imap.search(None, "ALL")

    # The absolute fix: target the byte string index explicitly
    mail_ids = search_data.split()

    if not mail_ids:
        print("No emails found.")
        imap.logout()
        return ""

    status, msg_data = imap.fetch(mail_ids[-1], "(RFC822)")

    body = ""
    for response_part in msg_data:
        if isinstance(response_part, tuple):
            msg = email.message_from_bytes(response_part)
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        body = part.get_payload(decode=True).decode(
                            errors="ignore"
                        )
                        break
            else:
                body = msg.get_payload(decode=True).decode(errors="ignore")

    imap.logout()
    print("Logged out successfully ✅")
    return body


async def generate_summary(text=None):
    if not text:
        return "{}"
    start_time = time.perf_counter()
    response = await client.chat.completions.create(
        model="qwen2.5:1.5b",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "Extract event, date (YYYY-MM-DD), time, venue into clean JSON. No markdown.",
            },
            {"role": "user", "content": text},
        ],
    )
    print(
        f"\n⏱️ Model Action Time: {time.perf_counter() - start_time:.2f} seconds"
    )
    return response.choices.message.content


def main():
    email_body = read_mail()
    if email_body:
        raw_output = asyncio.run(generate_summary(email_body))
        print("\nExtracted Result:\n", json.dumps(json.loads(raw_output), indent=2))


if __name__ == "__main__":
    main()