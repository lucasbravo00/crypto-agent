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
    """Plain text, deliberately no parse_mode. `text` is free-form LLM
    output -- it's never guaranteed to be well-formed Markdown (e.g. an
    odd number of "*" bullet points is enough to break Telegram's legacy
    Markdown parser with a 400 "can't parse entities" error, confirmed
    2026-07-24). Losing bold/italic formatting is a fair trade for never
    having a report silently fail to deliver over something as small as
    list punctuation."""
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    chat_id = os.environ["TELEGRAM_CHAT_ID"]
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    resp = requests.post(
        url,
        data={"chat_id": chat_id, "text": text},
        timeout=10,
    )
    resp.raise_for_status()
