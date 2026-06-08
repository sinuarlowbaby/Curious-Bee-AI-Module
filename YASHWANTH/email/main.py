"""
CLI entry point for the email extractor.

Glues together:
- ``config`` for credentials + output file paths
- ``imap_client`` for fetching unread messages
- ``dispatcher`` for analyzing + routing each message

Run with: ``python main.py``
"""

import sys

from config import (
    ANNOUNCEMENTS_FILE,
    CALENDAR_FILE,
    EMAIL_ADDRESS,
    EMAIL_PASSWORD,
    IMAP_SERVER,
)
from dispatcher import process_email
from imap_client import fetch_unread_emails


def main() -> int:
    # Fail fast if credentials are missing rather than getting a cryptic IMAP error.
    if not EMAIL_ADDRESS or not EMAIL_PASSWORD:
        print("ERROR: EMAIL_ADDRESS or EMAIL_PASSWORD not set in environment / .env file.")
        return 1

    emails = fetch_unread_emails(IMAP_SERVER, EMAIL_ADDRESS, EMAIL_PASSWORD)

    if not emails:
        print("No unread emails found.")
    else:
        print(f"Found {len(emails)} unread email(s).\n")
        for em in emails:
            process_email(em["sender"], em["receiver"], em["subject"], em["body"])

    print("Done.")
    print(f"Events saved to        : {CALENDAR_FILE}")
    print(f"Announcements saved to : {ANNOUNCEMENTS_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
