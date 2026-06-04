import os
from dotenv import load_dotenv

load_dotenv()

# =========================================================
# CONFIGURATION
# =========================================================

EMAIL_ADDRESS       = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD      = os.getenv("EMAIL_PASSWORD")
IMAP_SERVER         = "imap.gmail.com"

# Force JSON files to be saved inside the event_scheduling folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CALENDAR_FILE       = os.path.join(BASE_DIR, "calendar.json")
ANNOUNCEMENTS_FILE  = os.path.join(BASE_DIR, "announcements.json")

# =========================================================
# AUTHORIZED SENDERS
# ONLY THESE EMAILS CAN TRIGGER EVENT PROCESSING
# =========================================================

AUTHORIZED_SENDERS = [
    "dean@gmail.com",
    "hod@gmail.com",
    "lekhalokare.28@gmail.com"
]

LABELS = [
    "event",
    "date",
    "time",
    "venue"
]
