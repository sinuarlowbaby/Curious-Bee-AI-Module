"""
Activity Score Updater
──────────────────────
Run daily via cron to refresh all user activity scores.

Uses exponential decay:
    score = e^( -seconds_since_last_active / HALF_LIFE_SECONDS )

Half-life = 10 days:
  Active today       → ~1.0
  Active 7 days ago  → ~0.50
  Active 20 days ago → ~0.14
  Active 30 days ago → ~0.05

Crontab (runs at 2am daily):
    0 2 * * * cd /your/project && python -m jobs.update_activity_scores
"""

import logging
from db.connection import get_conn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

HALF_LIFE_SECONDS = 864_000   # 10 days — lower = faster decay


def update_activity_scores():
    logger.info("Updating activity scores...")

    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            UPDATE user_activity
            SET activity_score = GREATEST(
                0.0,
                EXP(
                    -EXTRACT(EPOCH FROM (NOW() - last_active_at))
                    / %s
                )
            )
        """, (float(HALF_LIFE_SECONDS),))
        updated = cur.rowcount
        conn.commit()
        cur.close()

    logger.info(f"Updated {updated} users.")


def touch_user_activity(user_id: str, conn):
    """
    Call this on any user action — login, publish paper, comment, etc.
    Resets last_active_at to NOW and score to 1.0.
    """
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO user_activity (user_id, last_active_at, activity_score)
        VALUES (%s, NOW(), 1.0)
        ON CONFLICT (user_id) DO UPDATE
            SET last_active_at = NOW(),
                activity_score = 1.0
    """, (user_id,))
    cur.close()


if __name__ == "__main__":
    update_activity_scores()