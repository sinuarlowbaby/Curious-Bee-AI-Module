"""
Announcements persistence.

Announcements are independent of the calendar — they capture notices,
cancellations, and reschedule notices that should appear in a feed
without driving calendar mutations.
"""

import datetime
import re

from config import ANNOUNCEMENTS_FILE
from storage import load_json, save_json


def save_announcement(sender_email: str, subject: str, description: str) -> None:
    """Append an announcement entry to announcements.json."""
    ann = load_json(ANNOUNCEMENTS_FILE)
    ann.append({
        "sender":      sender_email,
        "subject":     re.sub(r"[\r\n\t]+", " ", subject).strip(),
        "description": description,
        "received_on": datetime.date.today().strftime("%Y-%m-%d"),
    })
    save_json(ANNOUNCEMENTS_FILE, ann)
    print(f"  [SAVED]   Announcement stored in '{ANNOUNCEMENTS_FILE}'")
