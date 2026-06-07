"""
Calendar persistence: add, update, delete, and semantic search.

All operations go through the JSON file at ``config.CALENDAR_FILE`` via
``storage.load_json`` / ``storage.save_json`` so there is exactly one
read-modify-write convention.
"""

from config import CALENDAR_FILE, similarity_model
from sentence_transformers import util
from storage import load_json, save_json


# ---------------------------------------------------------------
# Add
# ---------------------------------------------------------------

def add_event(entry: dict) -> None:
    """Append a new event entry to the calendar."""
    cal = load_json(CALENDAR_FILE)
    cal.append(entry)
    save_json(CALENDAR_FILE, cal)
    print(f"  [ADDED]   '{entry['title']}' on {entry['date']}")


# ---------------------------------------------------------------
# Internal: drop entries matching (title, date)
# ---------------------------------------------------------------

def _drop_entries(title: str, date: str) -> list:
    """
    Return a fresh calendar list with all entries matching
    title + date removed. Shared by update_event and delete_event
    to eliminate duplicated filter logic.
    """
    return [
        e for e in load_json(CALENDAR_FILE)
        if not (e["title"] == title and e["date"] == date)
    ]


# ---------------------------------------------------------------
# Update / delete
# ---------------------------------------------------------------

def update_event(old_entries: list[dict], new_dates: list,
                 from_time, to_time, venue, link=None) -> None:
    """Replace ALL old calendar entries for this event with entries on new dates."""
    # Use the first entry as the template for title/time/venue defaults
    template = old_entries[0]
    title    = template["title"]

    # Drop every existing entry for this event title in one pass
    cal = [e for e in load_json(CALENDAR_FILE) if e["title"] != title]

    for date in new_dates:
        cal.append({
            "title":     title,
            "date":      date,
            "from_time": from_time or template.get("from_time"),
            "to_time":   to_time   or template.get("to_time"),
            "venue":     venue     or template.get("venue"),
            "link":      link      or template.get("link"),
            "status":    "reschedule",
        })
        old_dates_str = ", ".join(e["date"] for e in old_entries)
        print(f"  [UPDATED] '{title}' — [{old_dates_str}] → {date}")
    save_json(CALENDAR_FILE, cal)


def delete_event(old_entries: list[dict]) -> None:
    """Remove ALL calendar entries matching this event title."""
    title = old_entries[0]["title"]
    cal   = [e for e in load_json(CALENDAR_FILE) if e["title"] != title]
    save_json(CALENDAR_FILE, cal)
    dates_str = ", ".join(e["date"] for e in old_entries)
    print(f"  [DELETED] '{title}' on [{dates_str}]")


# ---------------------------------------------------------------
# Semantic search
# ---------------------------------------------------------------

def find_matching_event(event_name: str, old_dates: list = None,
                        threshold: float = 0.75) -> list[dict]:
    """
    Semantic search over the calendar.

    Returns a list of ALL calendar entries whose title matches event_name above
    the threshold (so multi-day events return one entry per day).  Returns an
    empty list when nothing matches confidently.

    Key improvement over the original: all calendar titles are batch-encoded
    in a single forward pass (one call to similarity_model.encode) rather than
    calling encode() individually for each entry, making this O(1) model calls
    regardless of calendar size.

    A 0.15 score bonus is applied when the candidate entry's date is also
    found in old_dates — acts as a strong disambiguation signal for finding
    the best title, but ALL entries sharing that best title are returned so
    that multi-day reschedules and cancellations remove every day, not only
    the first one.
    """
    cal = load_json(CALENDAR_FILE)
    if not cal:
        return []

    query_emb  = similarity_model.encode(event_name, convert_to_tensor=True)
    title_embs = similarity_model.encode([e["title"] for e in cal], convert_to_tensor=True)
    scores     = util.cos_sim(query_emb, title_embs)[0].tolist()

    best_score, best_title = 0.0, None
    for score, entry in zip(scores, cal):
        boosted = score + (0.15 if old_dates and entry["date"] in old_dates else 0.0)
        if boosted > best_score:
            best_score, best_title = boosted, entry["title"]

    if best_score >= threshold:
        matched = [e for e in cal if e["title"] == best_title]
        print(f"  Similarity score: {best_score:.2f} — matched '{best_title}' "
              f"({len(matched)} entr{'y' if len(matched) == 1 else 'ies'})")
        return matched

    print(f"  Similarity score: {best_score:.2f} — no confident match found")
    return []
