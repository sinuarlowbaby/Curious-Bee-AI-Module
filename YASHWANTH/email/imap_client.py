"""
IMAP client.

The only IMAP-specific code in the project. Exposes one function —
``fetch_unread_emails`` — that connects, pulls every UNSEEN message,
decodes the subject + body, and returns plain dicts the dispatcher
can consume. Always logs out before returning, even on error.
"""

import email
import imaplib
from email.header import decode_header

from bs4 import BeautifulSoup


def _decode_subject(raw_subject: str) -> str:
    """Decode a (possibly RFC2047-encoded) Subject header into a plain str."""
    decoded = decode_header(raw_subject)
    return "".join(
        part.decode(enc or "utf-8", errors="ignore") if isinstance(part, bytes) else part
        for part, enc in decoded
    )


def _decode_body(msg) -> str:
    """
    Walk a ``email.message.Message`` and return the best plain-text body
    we can find. Prefers ``text/plain``; falls back to ``text/html``
    stripped via BeautifulSoup.
    """
    if msg.is_multipart():
        body = ""
        for part in msg.walk():
            ct = part.get_content_type()
            payload = part.get_payload(decode=True)
            if not payload:
                continue
            text = payload.decode(errors="ignore")
            if ct == "text/plain":
                body = text
                break
            elif ct == "text/html" and not body:
                body = BeautifulSoup(text, "html.parser").get_text()
        return body

    payload = msg.get_payload(decode=True)
    return payload.decode(errors="ignore") if payload else ""


def fetch_unread_emails(server: str, address: str, password: str) -> list[dict]:
    """
    Connect to IMAP, fetch all UNSEEN messages, decode subject + body,
    and return a list of ``{"sender", "receiver", "subject", "body"}`` dicts.

    The connection is always closed before returning (success or error).
    """
    out: list[dict] = []
    mail = imaplib.IMAP4_SSL(server)
    try:
        mail.login(address, password)
        mail.select("inbox")

        status, messages = mail.search(None, "UNSEEN")
        if status != "OK":
            raise RuntimeError(f"IMAP search failed with status: {status!r}")
        email_ids = messages[0].split()

        for latest_email_id in email_ids:
            status, msg_data = mail.fetch(latest_email_id, "(RFC822)")
            if status != "OK" or not msg_data:
                print(f"WARNING: Failed to fetch email ID {latest_email_id} — skipping.")
                continue

            for response_part in msg_data:
                if not isinstance(response_part, tuple):
                    continue
                msg = email.message_from_bytes(response_part[1])

                # msg.get() returns None for missing headers; default to ""
                # so decode_header() and process_email() never receive None.
                subject     = _decode_subject(msg.get("Subject") or "")
                email_body  = _decode_body(msg)
                sender      = msg.get("From") or ""
                receiver    = msg.get("To")   or ""
                out.append({
                    "sender":   sender,
                    "receiver": receiver,
                    "subject":  subject,
                    "body":     email_body,
                })
    finally:
        # Always log out, even if an exception is raised during processing.
        mail.logout()

    return out
