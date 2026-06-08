"""
Email dispatcher.

Routes an analyzed email to the correct handler based on its type:
``event``        → add to calendar
``reschedule``   → find + update (or add a flagged entry) + announcement
``cancellation`` → find + delete + announcement
``announcement`` → save to announcements only

The dispatcher owns sender authorization and the print-output contract
that the original script's process loop produced.
"""

import json
import re

from analyzer import analyze_email
from announcement_store import save_announcement
from calendar_store import add_event, delete_event, find_matching_event, update_event
from config import AUTHORIZED_SENDERS


def process_email(sender: str, receiver: str, subject: str, body: str) -> None:
    """
    Validate sender authorization, run analysis, and route the result
    to the correct calendar / announcement handler.
    """
    # --- Normalize sender address (handles "Name <email>" format) ---
    m = re.search(r"<(.+?)>", sender)
    sender_email = (m.group(1) if m else sender).strip().lower()

    if sender_email not in {s.lower() for s in AUTHORIZED_SENDERS}:
        print(f"UNAUTHORIZED SENDER SKIPPED: {sender_email}\n{'=' * 50}\n")
        return

    # Normalize body whitespace but preserve newlines (venue regex is newline-aware)
    body = re.sub(r"[ \t\r]+", " ", body).strip()

    print(f"SENDER   : {sender_email}")
    print(f"RECEIVER : {receiver}")
    print(f"SUBJECT  : {subject.strip()}\n")

    print("Analyzing email...")
    result = analyze_email(body, subject)
    etype  = result["type"]

    print(f"TYPE     : {etype.upper()}\n")
    print(json.dumps(result, indent=4), "\n")

    # ==================================================
    # HANDLE: NEW EVENT → add entries to calendar.json
    # ==================================================
    if etype == "event":
        if not result["dates"]:
            print("  WARNING: No dates extracted — skipping calendar entry.")
        else:
            print("CALENDAR ACTIONS:")
            for date in result["dates"]:
                add_event({
                    "title":     result["event"],
                    "date":      date,
                    "from_time": result["from_time"],
                    "to_time":   result["to_time"],
                    "venue":     result["venue"],
                    "link":      result.get("link"),
                    "status":    "schedule",
                })

    # ==================================================
    # HANDLE: RESCHEDULE → update entries in calendar.json
    # ==================================================
    elif etype == "reschedule":
        print("RESCHEDULE DETECTED — searching calendar for matching event...")
        matches = find_matching_event(result["event"], result["old_dates"])
        print("\nCALENDAR ACTIONS:")
        if matches:
            print(f"  Matched: '{matches[0]['title']}' — {len(matches)} entr{'y' if len(matches)==1 else 'ies'}")
            update_event(
                old_entries=matches,
                new_dates=result["dates"],
                from_time=result["from_time"],
                to_time=result["to_time"],
                venue=result["venue"],
                link=result.get("link"),
            )
        else:
            print("  No existing event matched — adding as new entry (flagged)")
            for date in result["dates"]:
                add_event({
                    "title":     f"[POSSIBLY RESCHEDULED] {result['event']}",
                    "date":      date,
                    "from_time": result["from_time"],
                    "to_time":   result["to_time"],
                    "venue":     result["venue"],
                    "link":      result.get("link"),
                    "status":    "reschedule",
                })
        print("\nANNOUNCEMENT ACTIONS:")
        save_announcement(
            sender_email=sender_email,
            subject=subject,
            description=result.get("description") or f"Event '{result['event']}' has been rescheduled.",
        )

    # ==================================================
    # HANDLE: CANCELLATION → delete entries from calendar.json
    # ==================================================
    elif etype == "cancellation":
        print("CANCELLATION DETECTED — searching calendar for matching event...")
        matches = find_matching_event(result["event"], result["old_dates"])
        print("\nCALENDAR ACTIONS:")
        if matches:
            print(f"  Matched: '{matches[0]['title']}' — deleting {len(matches)} entr{'y' if len(matches)==1 else 'ies'}.")
            delete_event(matches)
        else:
            print("  No existing event matched for cancellation.")
        print("\nANNOUNCEMENT ACTIONS:")
        save_announcement(
            sender_email=sender_email,
            subject=subject,
            description=result.get("description"),
        )

    # ==================================================
    # HANDLE: ANNOUNCEMENT → save to announcements.json only
    # ==================================================
    elif etype == "announcement":
        print("ANNOUNCEMENT — not added to calendar.")
        print(f"Summary  : {result.get('description')}\n")
        print("ANNOUNCEMENT ACTIONS:")
        save_announcement(
            sender_email=sender_email,
            subject=subject,
            description=result.get("description"),
        )

    else:
        print("UNKNOWN TYPE — could not process this email.")

    print(f"\n{'=' * 50}\n")
