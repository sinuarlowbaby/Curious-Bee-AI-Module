# =========================================================
# SECURE EVENT EXTRACTION MAIL READER
# Uses Ollama (Phi-3) for classification + extraction
# Handles: new events, multi-day events, rescheduled events,
#          and announcements (saved separately)
# =========================================================

import imaplib
import email
import re
import json

from config import (
    EMAIL_ADDRESS,
    EMAIL_PASSWORD,
    IMAP_SERVER,
    AUTHORIZED_SENDERS,
    CALENDAR_FILE,
    ANNOUNCEMENTS_FILE
)
from store import (
    add_event,
    update_event,
    delete_event,
    save_announcement,
    load_calendar
)
from analyzer import analyze_email, find_matching_event

# =========================================================
# CONNECT TO MAILBOX  [COMMENTED OUT — PROTOTYPE MODE]
# =========================================================

# print("Connecting to mailbox...")
# mail = imaplib.IMAP4_SSL(IMAP_SERVER)
# mail.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
# mail.select("inbox")
# print("Connected.\n")

# =========================================================
# FETCH UNREAD EMAILS  [COMMENTED OUT — PROTOTYPE MODE]
# =========================================================

# try:
#     status, messages = mail.search(None, "UNSEEN")
#     email_ids = messages[0].split()
# except imaplib.IMAP4.abort as e:
#     print(f"\n[!] IMAP connection aborted by server: {e}")
#     print("    This usually happens due to rate-limiting when running the script too frequently.")
#     import sys
#     sys.exit(1)
# print(f"Found {len(email_ids)} unread email(s)\n")
# print("=" * 50)

# =========================================================
# >>>  PROTOTYPE MAIL CONTENT — EDIT THIS SECTION  <<<
# =========================================================
# Change the three variables below to test a new email.
# MAIL_SENDER must be one of the AUTHORIZED_SENDERS above.

MAIL_SENDER   = "dean@gmail.com"          # who sent the email
MAIL_RECEIVER = "receiver@gmail.com"      # who received it
MAIL_SUBJECT  = """
Holiday Notice – Institution Closed on June 17, 2026
"""
MAIL_BODY     = """
Dear All,
This is to inform all faculty, staff, and students that the institution will remain closed on June 17, 2026 on account of
a public holiday.
All scheduled activities, classes, and lab sessions for that day stand cancelled. Please plan accordingly.
For any urgent matters, kindly contact your respective department offices.
Thanks and Regards,
Administrative Office
SRMIST, Kattankulathur
"""

# =========================================================
# PROCESS EMAIL  (prototype — iterates over a single entry)
# =========================================================

for subject, sender, receiver, body in [
    (MAIL_SUBJECT, MAIL_SENDER, MAIL_RECEIVER, MAIL_BODY)
]:

    # --------------------------------------------------
    # EXTRACT CLEAN SENDER EMAIL ADDRESS
    # --------------------------------------------------

    sender_match = re.search(r"<(.+?)>", sender)
    if sender_match:
        sender_email = sender_match.group(1).strip().lower()
    else:
        sender_email = sender.strip().lower()

    # --------------------------------------------------
    # AUTHORIZED SENDER CHECK
    # --------------------------------------------------

    if sender_email not in [s.strip().lower() for s in AUTHORIZED_SENDERS]:
        print(f"UNAUTHORIZED SENDER SKIPPED: {sender_email}\n")
        print("=" * 50)
        continue

    # Clean up whitespace but preserve newlines so our line-based regexes (like venue extraction) still work
    body = re.sub(r"[ \t\r]+", " ", body).strip()

    # --------------------------------------------------
    # PRINT EMAIL HEADER
    # --------------------------------------------------

    print(f"SENDER   : {sender_email}")
    print(f"RECEIVER : {receiver}")
    print(f"SUBJECT  : {subject}")
    print()

    # --------------------------------------------------
    # ANALYZE EMAIL WITH LOCAL SLM
    # --------------------------------------------------

    print("Analyzing email")
    result = analyze_email(body, subject)

    print(f"TYPE     : {result['type'].upper()}")
    print()
    print(json.dumps(result, indent=4))
    print()

    # ==================================================
    # HANDLE: NEW EVENT → Save to calendar.json
    # ==================================================

    if result["type"] == "event":

        if not result["dates"]:
            print("  WARNING: No dates extracted — skipping calendar entry.\n")

        else:
            print("CALENDAR ACTIONS:")
            for date in result["dates"]:
                entry = {
                    "title":       result["event"],
                    "date":        date,
                    "from_time":   result["from_time"],
                    "to_time":     result["to_time"],
                    "venue":       result["venue"],
                    "link":        result.get("link")
                }
                add_event(entry)

    # ==================================================
    # HANDLE: RESCHEDULED EVENT → Update calendar.json
    # ==================================================

    elif result["type"] == "reschedule":

        print("RESCHEDULE DETECTED — searching calendar for matching event...")

        # Load calendar here to pass to the matching logic
        calendar_data = load_calendar()
        match = find_matching_event(
            event_name = result["event"],
            old_dates  = result["old_dates"],
            calendar   = calendar_data
        )

        print()
        print("CALENDAR ACTIONS:")

        if match:
            print(f"  Matched: '{match['title']}' on {match['date']}")
            update_event(
                old_entry       = match,
                new_dates       = result["dates"],
                new_from_time   = result["from_time"],
                new_to_time     = result["to_time"],
                new_venue       = result["venue"],
                new_link        = result.get("link")
            )
        else:
            # No confident match found — add as new entry but flag it
            print("  No existing event matched — adding as new entry (flagged)")
            for date in result["dates"]:
                entry = {
                    "title":       f"[POSSIBLY RESCHEDULED] {result['event']}",
                    "date":        date,
                    "from_time":   result["from_time"],
                    "to_time":     result["to_time"],
                    "venue":       result["venue"],
                    "link":        result.get("link")
                }
                add_event(entry)

        print()
        print("ANNOUNCEMENT ACTIONS:")
        save_announcement(
            sender_email = sender_email,
            subject      = subject,
            description  = result.get("description") or f"Event '{result['event']}' has been rescheduled."
        )

    # ==================================================
    # HANDLE: CANCELLED EVENT → Delete from calendar.json
    # ==================================================

    elif result["type"] == "cancellation":

        print("CANCELLATION DETECTED — searching calendar for matching event...")
        
        calendar_data = load_calendar()
        match = find_matching_event(
            event_name = result["event"],
            old_dates  = result["old_dates"],
            calendar   = calendar_data
        )

        print()
        print("CALENDAR ACTIONS:")

        if match:
            print(f"  Matched: '{match['title']}' on {match['date']} — deleting.")
            delete_event(match)
        else:
            print("  No existing event matched for cancellation.")
            
        print()
        print("ANNOUNCEMENT ACTIONS:")
        save_announcement(
            sender_email = sender_email,
            subject      = subject,
            description  = result.get("description")
        )

    # ==================================================
    # HANDLE: ANNOUNCEMENT → Save to announcements.json
    # NOT added to calendar
    # ==================================================

    elif result["type"] == "announcement":
        print("ANNOUNCEMENT — not added to calendar.")
        print(f"Summary  : {result.get('description')}")
        print()
        print("ANNOUNCEMENT ACTIONS:")
        save_announcement(
            sender_email = sender_email,
            subject      = subject,
            description  = result.get("description")
        )

    # ==================================================
    # HANDLE: UNKNOWN (SLM parse failure)
    # ==================================================

    else:
        print("UNKNOWN TYPE — could not process this email.")

    print()
    print("=" * 50)

# =========================================================
# LOGOUT  [COMMENTED OUT — PROTOTYPE MODE]
# =========================================================

# mail.logout()
print("\nPrototype run complete.")
print(f"\nEvents saved to       : {CALENDAR_FILE}")
print(f"Announcements saved to : {ANNOUNCEMENTS_FILE}")