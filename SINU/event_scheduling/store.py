import json
import os
import datetime
import re

from config import CALENDAR_FILE, ANNOUNCEMENTS_FILE

# =========================================================
# CALENDAR STORE FUNCTIONS
# =========================================================

def load_calendar():
    """Load calendar entries from local JSON file."""
    if not os.path.exists(CALENDAR_FILE):
        return []
    with open(CALENDAR_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_calendar(entries):
    """Save calendar entries to local JSON file."""
    with open(CALENDAR_FILE, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=4, ensure_ascii=False)


def add_event(entry):
    """Add a new event entry to the calendar."""
    calendar = load_calendar()
    calendar.append(entry)
    save_calendar(calendar)
    print(f"  [ADDED]   '{entry['title']}' on {entry['date']}")


def update_event(old_entry, new_dates, new_from_time, new_to_time, new_venue, new_link=None):
    """
    Remove all old calendar entries for this event title + old date,
    then insert new entries with updated dates.
    """
    calendar = load_calendar()

    # Remove old entries that match this event + old date
    calendar = [
        e for e in calendar
        if not (
            e["title"] == old_entry["title"] and
            e["date"]  == old_entry["date"]
        )
    ]

    # Insert new entries (one per new date)
    for date in new_dates:
        new_entry = {
            "title":       old_entry["title"],
            "date":        date,
            "from_time":   new_from_time   or old_entry.get("from_time"),
            "to_time":     new_to_time     or old_entry.get("to_time"),
            "venue":       new_venue       or old_entry.get("venue"),
            "link":        new_link        or old_entry.get("link")
        }
        calendar.append(new_entry)
        print(f"  [UPDATED] '{old_entry['title']}' — {old_entry['date']} → {date}")

    save_calendar(calendar)


def delete_event(old_entry):
    """
    Remove all old calendar entries for this event title + old date.
    """
    calendar = load_calendar()

    calendar = [
        e for e in calendar
        if not (
            e["title"] == old_entry["title"] and
            e["date"]  == old_entry["date"]
        )
    ]
    save_calendar(calendar)
    print(f"  [DELETED] '{old_entry['title']}' on {old_entry['date']}")


# =========================================================
# ANNOUNCEMENTS STORE FUNCTION
# =========================================================

def save_announcement(sender_email, subject, description):
    """
    Save announcement to a separate JSON file.
    This is read by the UI to display in the announcements section,
    completely separate from the calendar.
    """
    if not os.path.exists(ANNOUNCEMENTS_FILE):
        announcements = []
    else:
        with open(ANNOUNCEMENTS_FILE, "r", encoding="utf-8") as f:
            announcements = json.load(f)

    entry = {
        "sender":      sender_email,
        "subject":     re.sub(r"[\r\n\t]+", " ", subject).strip(),
        "description": description,
        "received_on": datetime.date.today().strftime("%Y-%m-%d")
    }

    announcements.append(entry)

    with open(ANNOUNCEMENTS_FILE, "w", encoding="utf-8") as f:
        json.dump(announcements, f, indent=4, ensure_ascii=False)

    print(f"  [SAVED]   Announcement stored in '{ANNOUNCEMENTS_FILE}'")
