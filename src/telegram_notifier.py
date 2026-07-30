"""
telegram_notifier.py
---------------------
Optional output channel. Deliberately separated from the "brain"
(agent.py) so the channel can change (console, email, Telegram) without
touching decision logic.
"""
import os
from typing import Optional

import requests

# Telegram rejects sendPhoto with a caption longer than 1024 characters
# (sendMessage allows 4096). Today's report is far shorter, but a prompt
# change or a chatty model would silently start failing to deliver, so
# the overflow path below is not hypothetical insurance -- it is the
# difference between "report arrives without a picture" and "no report".
CAPTION_LIMIT = 1024


def send_message(text: str, image: Optional[bytes] = None) -> None:
    """Plain text, deliberately no parse_mode. `text` is free-form LLM
    output -- it's never guaranteed to be well-formed Markdown (e.g. an
    odd number of "*" bullet points is enough to break Telegram's legacy
    Markdown parser with a 400 "can't parse entities" error, confirmed
    2026-07-24). Losing bold/italic formatting is a fair trade for never
    having a report silently fail to deliver over something as small as
    list punctuation.

    With `image`, the text rides along as the photo's caption so it
    arrives as a single message. If the text does not fit in a caption,
    the photo is sent bare and the text follows as its own message --
    never truncated, because a cut-off market report is worse than one
    split in two."""
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    chat_id = os.environ["TELEGRAM_CHAT_ID"]
    base = f"https://api.telegram.org/bot{token}"

    if image:
        caption = text if len(text) <= CAPTION_LIMIT else ""
        resp = requests.post(
            f"{base}/sendPhoto",
            data={"chat_id": chat_id, "caption": caption},
            files={"photo": ("chart.png", image, "image/png")},
            timeout=30,   # uploading a PNG needs more room than a text POST
        )
        resp.raise_for_status()
        if caption:
            return
        # Fall through: caption was too long, so the text still owes a message.

    resp = requests.post(
        f"{base}/sendMessage",
        data={"chat_id": chat_id, "text": text},
        timeout=10,
    )
    resp.raise_for_status()
