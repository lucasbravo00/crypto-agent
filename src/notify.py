"""
notify.py
----------
Notification dispatcher. The agent produces text; where that text goes
(console, email, Telegram) is a configuration decision, not logic.
Choose the channel with NOTIFY_CHANNEL in .env:

  NOTIFY_CHANNEL=console   -> print to terminal only (default, zero setup)
  NOTIFY_CHANNEL=email     -> also send to your mailbox (e.g. Outlook)
  NOTIFY_CHANNEL=telegram  -> also send via Telegram
  NOTIFY_CHANNEL=all       -> email + telegram
"""
import os
from datetime import date
from typing import Optional


def notify(text: str, subject_prefix: str = "Crypto report",
           image: Optional[bytes] = None) -> None:
    """`image` is an optional PNG (see report_chart.py) that channels able
    to show it will attach. It stays optional per call rather than being
    fetched here, because only the daily report has something worth
    illustrating -- a bullet alert or a state-mismatch warning is urgent
    and short, and a chart would just slow it down."""
    channel = os.environ.get("NOTIFY_CHANNEL", "console").lower()
    subject = f"{subject_prefix} — {date.today().isoformat()}"

    # The console always shows the result, no matter what.
    print("--- Generated report ---")
    print(text)

    errors = []
    if channel in ("email", "all"):
        try:
            from . import email_notifier
            email_notifier.send_email(subject, text, image=image)
            print("Sent by email ✅")
        except Exception as exc:
            errors.append(f"email: {exc}")

    if channel in ("telegram", "all"):
        try:
            from . import telegram_notifier
            telegram_notifier.send_message(text, image=image)
            print("Sent to Telegram ✅")
        except Exception as exc:
            errors.append(f"telegram: {exc}")

    for err in errors:
        print(f"⚠️ Delivery failed for {err}")
