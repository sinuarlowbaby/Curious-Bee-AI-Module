"""
Email analyzer.

Takes a raw email ``body`` + ``subject`` and produces a structured dict
describing what the email means (``event`` / ``reschedule`` /
``cancellation`` / ``announcement``) plus the extracted fields the
dispatcher needs to act on.

The function is intentionally self-contained: it does not touch the
filesystem or the calendar; the dispatcher does that. It also does not
log any state — it only ``print()``s the same status lines the original
script emitted so the user-visible output is unchanged.
"""

import re

import dateutil.parser

from config import DEADLINE_KEYWORDS, GLINER_LABELS, gliner_model
from text_utils import (
    classify_email,
    dedupe,
    expand_date_ranges,
    extract_main_body,
    find_registration_link,
    is_likely_date,
    is_likely_time,
    normalize_dashes,
    split_time_range,
    venue_fallback,
)


def analyze_email(body: str, subject: str) -> dict:
    """
    Extract structured event details from an email using GLiNER + regex fallbacks.
    Returns a dict with: type, event, dates, old_dates, time, from_time,
    to_time, venue, link, description.
    """
    try:
        # --- Normalize inputs ---
        subject = re.sub(r"^(Re:|Fwd:|FW:)\s*", "", subject, flags=re.IGNORECASE).strip()
        body    = re.sub(r"<[^>]+>", " ", body)   # strip HTML tags
        body    = re.sub(r"&nbsp;", " ", body)
        body    = re.sub(r"[ \t]+", " ", body)
        body    = re.sub(r"\n+", "\n", body).strip()

        text      = f"Subject: {subject}\n\n{body}"
        etype     = classify_email(text)
        entities  = gliner_model.predict_entities(text, GLINER_LABELS)
        norm_subj = normalize_dashes(subject)      # used for event-name comparison (bug fix)

        result = {
            "type":        etype,
            "event":       norm_subj,
            "dates":       [],
            "old_dates":   [],
            "time":        None,
            "from_time":   None,
            "to_time":     None,
            "venue":       None,
            "link":        None,
            "description": None,
        }

        # --- Build description for non-event types ---
        if etype in ("announcement", "cancellation", "reschedule"):
            body_clean = extract_main_body(body)
            sents      = re.split(r'(?<=[.!?])\s+', body_clean)
            if etype == "reschedule":
                # Use the full cleaned body so the new date/venue info is captured
                result["description"] = body_clean
            else:
                result["description"] = sents[0].strip() if sents else body_clean
            if etype == "announcement":
                return result  # no date/venue extraction needed

        # --- Parse GLiNER entities ---
        extracted_dates = []
        # Pre-split once; avoids re-splitting inside the entity loop
        text_sentences = re.split(r'(?<=[.!?\n])\s+', text)

        for ent in entities:
            lbl, val = ent["label"], ent["text"]

            if lbl == "event" and result["event"] == norm_subj:
                # Only override the subject-derived name if GLiNER found a cleaner one
                # Bug fix: compare against norm_subj (not raw subject) to handle Unicode dashes
                result["event"] = val

            elif lbl == "date":
                if not is_likely_date(val):
                    continue
                try:
                    date_str = dateutil.parser.parse(val, fuzzy=True).strftime("%Y-%m-%d")
                    # Check only the sentence containing this date for deadline context —
                    # prevents cross-sentence bleed where a deadline phrase in sentence A
                    # wrongly flags a legitimate event date in sentence B.
                    ctx = next((s.lower() for s in text_sentences if val.lower() in s.lower()), "")
                    if any(kw in ctx for kw in DEADLINE_KEYWORDS):
                        print(f"  [SKIP]  '{val}' looks like a deadline — not an event date")
                    else:
                        extracted_dates.append(date_str)
                except Exception:
                    pass

            elif lbl == "time" and not result["from_time"]:
                if not is_likely_time(val):
                    continue
                result["time"] = val
                result["from_time"], result["to_time"] = split_time_range(val)

            elif lbl == "venue":
                # If reschedule explicitly says "same venue", we skip extraction
                # so it falls back to the old venue in update_event.
                if etype == "reschedule" and re.search(r"same\s+venue", text, re.IGNORECASE):
                    continue

                # For reschedule emails, we only want the NEW venue (after the
                # reschedule keyword).  For all other types, first-found wins.
                if etype == "reschedule":
                    # Locate the reschedule keyword in the full text
                    rsplit = re.search(
                        r"\b(reschedule[d]?|postpone[d]?|rescheduling)\b",
                        text, re.IGNORECASE,
                    )
                    # Only store a venue entity that appears AFTER the keyword
                    entity_start = ent.get("start", text.find(val))
                    if rsplit and entity_start < rsplit.start():
                        continue  # old-venue mention — skip
                if not result["venue"] or etype == "reschedule":
                    # Extend forward from GLiNER entity, crossing newlines
                    m = re.search(re.escape(val) + r"(?:,\s*[^.]+)*", text)
                    raw_venue = m.group(0).strip() if m else val
                    # Collapse embedded newlines and trim
                    result["venue"] = re.sub(r"\s*\n\s*", " ", raw_venue).strip().rstrip(".")

        # --- Venue fallback (regex) ---
        if not result["venue"]:
            if etype == "reschedule" and re.search(r"same\s+venue", text, re.IGNORECASE):
                pass  # leave as None to inherit old venue
            else:
                result["venue"] = venue_fallback(text)

        # --- Time fallback (regex) ---
        if not result["from_time"]:
            range_t  = re.search(
                r"(\d{1,2}[:.]\d{2}\s?(?:AM|PM|am|pm))\s*(?:to|-)\s*(\d{1,2}[:.]\d{2}\s?(?:AM|PM|am|pm))",
                text, re.IGNORECASE,
            )
            single_t = re.search(r"\d{1,2}[:.]\d{2}\s?(?:AM|PM|am|pm)", text, re.IGNORECASE)
            if range_t:
                result["from_time"] = range_t.group(1)
                result["to_time"]   = range_t.group(2)
                result["time"]      = f"{result['from_time']} to {result['to_time']}"
            elif single_t:
                result["from_time"] = single_t.group()
                result["time"]      = result["from_time"]

        # --- Deduplicate (GLiNER point-dates merged with range-expanded dates) ---
        all_dates = dedupe(extracted_dates + expand_date_ranges(text))

        # --- Assign dates by email type ---
        if etype == "cancellation":
            result["old_dates"]   = all_dates
            # Use the full cleaned body (not just first sentence) so that
            # any "new date will be announced" phrasing is preserved naturally.
            result["description"] = extract_main_body(body)

        elif etype == "reschedule":
            # Split the email at the reschedule keyword so that date ranges in
            # each half are expanded independently.  This prevents old dates
            # from being misidentified as new dates in a multi-day reschedule.
            reschedule_re = re.compile(
                r"\b(reschedule[d]?|postpone[d]?|rescheduling)\b", re.IGNORECASE
            )
            split_m = reschedule_re.search(text)
            if split_m:
                pre_text  = text[:split_m.start()]
                post_text = text[split_m.end():]
                # Extract point-dates from each half
                pre_dates  = [d for d in extracted_dates if d in pre_text]
                post_dates = [d for d in extracted_dates
                               if d not in pre_dates]
                # Expand date ranges independently from each half
                old_range  = expand_date_ranges(pre_text)
                new_range  = expand_date_ranges(post_text)
                old_part   = dedupe(pre_dates  + old_range)
                new_part   = dedupe(post_dates + new_range)
                if old_part or new_part:
                    result["old_dates"] = old_part
                    result["dates"]     = new_part
                else:
                    # Fallback: if split produced nothing, use original heuristic
                    if len(all_dates) >= 2:
                        result["old_dates"] = [all_dates[0]]
                        result["dates"]     = all_dates[1:]
                    else:
                        result["dates"] = all_dates
            else:
                # No reschedule keyword found in text — use original heuristic
                if len(all_dates) >= 2:
                    result["old_dates"] = [all_dates[0]]
                    result["dates"]     = all_dates[1:]
                else:
                    result["dates"] = all_dates

        else:  # event
            result["dates"] = all_dates
            result["link"]  = find_registration_link(text)

        return result

    except Exception as e:
        print(f"  ERROR analyzing email: {e}")
        return {
            "type": "unknown", "event": subject, "dates": [], "old_dates": [],
            "time": None, "venue": None, "description": None,
        }
