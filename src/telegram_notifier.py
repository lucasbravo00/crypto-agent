"""
telegram_notifier.py
---------------------
Optional output channel. Deliberately separated from the "brain"
(agent.py) so the channel can change (console, email, Telegram) without
touching decision logic.
"""
import os
import requests


def send_message(text: str) -> None:
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    chat_id = os.environ["TELEGRAM_CHAT_ID"]
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    resp = requests.post(
        url,
        data={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"},
        timeout=10,
    )
    resp.raise_for_status()
