"""
creators.py
------------
Watches a short list of YouTube channels you follow and, when one of them
publishes a NEW video that is actually about crypto, adds a one-or-two
sentence synthesis of it to the daily report.

Design decisions worth knowing:

1. NO YouTube API key. Uploads are detected through each channel's public
   Atom/RSS feed (youtube.com/feeds/videos.xml?channel_id=...), the same
   "free public endpoint, no auth" approach already used for Binance
   klines and alternative.me in market_data.py. Transcripts come from
   youtube-transcript-api, which reads the same captions the web player
   uses.

2. TWO-STAGE crypto filter, cheap one first. Stage 1 is a deterministic
   keyword scan over title + transcript: it costs nothing and throws out
   videos that are obviously off-topic. Stage 2 is the LLM, which is the
   only thing that can actually judge "is this really about crypto" for
   an ambiguous case (a macro/Fed video may or may not be). The LLM is
   given an explicit escape hatch (NOT_CRYPTO) rather than being forced
   to summarize something irrelevant. Stage 1 is deliberately GENEROUS
   -- it is there to save tokens, not to make the call.

3. The raw transcript NEVER reaches the report or the database. It is
   thousands of words, it belongs to the creator, and the point is a
   synthesis in the agent's own words. Only the short summary is stored.

4. Every video seen is recorded, INCLUDING non-crypto ones. Otherwise
   each run would re-download and re-judge the same off-topic video
   forever.

5. Best-effort by contract: get_creator_digest() returns None instead of
   raising. A channel being down, a video without captions, YouTube
   rate-limiting the transcript endpoint -- none of that may cost the
   daily report, which works fine without this section.

KNOWN RISK, read before trusting this in production: YouTube throttles
and sometimes blocks transcript requests coming from datacenter IPs, and
the daily report runs on GitHub Actions. The library surfaces this as
IpBlocked / RequestBlocked / PoTokenRequired, which this module treats
like any other failure (skip, report goes out without the section). See
README's "YouTube creator digest" section for how to verify it on your
own runner.
"""
from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests

from . import db

FEED_URL = "https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
CHANNEL_PAGE_URL = "https://www.youtube.com/{handle}"

# Atom + YouTube's own namespace, needed to read the feed. Parsed with a
# real XML parser rather than regex on purpose: the feed carries a
# FEED-level <published> element as well as one per entry, so a naive
# regex zip of ids/titles/dates silently shifts every date by one entry
# (confirmed while building this -- it dated the newest video to 2019).
NS = {"a": "http://www.w3.org/2005/Atom", "yt": "http://www.youtube.com/xml/schemas/2015"}

# How far back a video can be and still be worth mentioning.
#
# Two weeks, not two days, BECAUSE Shorts are excluded (see is_short).
# Creators post Shorts far more often than full videos: measured on
# CriptoNorber, the newest full video was 11 days old while four Shorts
# had gone up in the meantime. A 2-day window plus a no-Shorts rule left
# the channel permanently silent. The dedup table is what stops repeats;
# this window only stops the digest dredging up genuinely stale content
# when a channel goes quiet.
MAX_VIDEO_AGE_DAYS = 14

# Only ONE video per channel per run -- the newest full one. Shorts are
# recaps of the full videos, so they add nothing but noise and duplicate
# the same ideas across the report.
#
# How many feed entries to walk per channel while looking for that newest
# full video. Each check is one HEAD request, and a channel can post a
# run of Shorts before its last real video (CriptoNorber had four in a
# row), so this needs headroom -- but bounded, so a Shorts-only channel
# can't cost an unbounded number of requests.
MAX_FEED_ENTRIES_SCANNED = 10
# Transcripts run to tens of thousands of characters; only the first
# slice is sent to the model. The opening minutes are where these videos
# state their thesis, and this bounds both cost and context use.
TRANSCRIPT_CHAR_LIMIT = 6000

# Stage-1 prefilter. Generous on purpose (see docstring point 2): its job
# is only to skip clearly-unrelated uploads before spending LLM tokens.
CRYPTO_KEYWORDS = (
    "bitcoin", "btc", "crypto", "cripto", "criptomoneda", "ethereum", "eth",
    "altcoin", "blockchain", "defi", "stablecoin", "satoshi", "binance",
    "coinbase", "solana", "cardano", "xrp", "ripple", "halving", "on-chain",
    "onchain", "hodl", "satoshis", "tether", "usdt", "memecoin", "token",
    "wallet", "exchange", "mining", "minería", "etf de bitcoin", "bitcoin etf",
)
# How many DISTINCT keywords must appear before a video is worth an LLM
# call. 2 rather than 1 for a full-length transcript, so a single passing
# mention ("...unlike bitcoin, anyway...") in an unrelated video doesn't
# qualify.
MIN_KEYWORD_HITS = 2
# ...but that bar is wrong for SHORTS. A 600-character Short has no room
# for a passing mention: whatever it names is what it is about. Measured
# on two real Shorts published the same day -- a CriptoNorber one about
# Saylor and BlackRock hoarding bitcoin scored exactly ONE distinct
# keyword and was wrongly thrown out, while a genuinely off-topic one
# about an Argentine mining-investment regime scored ZERO. One keyword
# separates them cleanly; two loses the crypto one.
#
# Being generous here is safe by construction: stage 1 only decides
# whether to SPEND TOKENS, and stage 2's classifier -- which fails closed
# -- is what actually decides whether a video reaches the report.
SHORT_TRANSCRIPT_CHARS = 2000
MIN_KEYWORD_HITS_SHORT = 1


def is_enabled() -> bool:
    """The digest is opt-in and needs somewhere to remember seen videos."""
    if os.environ.get("YOUTUBE_DIGEST_ENABLED", "false").lower() not in ("1", "true", "yes"):
        return False
    return bool(_configured_channels()) and db.is_enabled()


def _configured_channels() -> list[str]:
    """Channels to watch, from YOUTUBE_CHANNELS (comma-separated).
    Accepts either raw channel ids (UC...) or @handles."""
    raw = os.environ.get("YOUTUBE_CHANNELS", "")
    return [c.strip() for c in raw.split(",") if c.strip()]


def resolve_channel_id(handle_or_id: str) -> Optional[str]:
    """Turn an @handle into a channel id, or pass a channel id through.

    Handles are what a human actually knows ("@CoinBureau"); the RSS feed
    only accepts the UC... id. Resolution scrapes the public channel page
    for the id it embeds, since there is no keyless API for this.
    """
    value = handle_or_id.strip()
    if value.startswith("UC") and len(value) == 24:
        return value

    handle = value if value.startswith("@") else f"@{value}"
    try:
        resp = requests.get(
            CHANNEL_PAGE_URL.format(handle=handle),
            timeout=15,
            headers={"User-Agent": "Mozilla/5.0"},   # plain requests UA gets a consent wall
        )
        resp.raise_for_status()
        return _channel_id_from_page(resp.text)
    except Exception as exc:
        print(f"⚠️ Could not resolve YouTube channel '{handle_or_id}': {exc}")
        return None


def _channel_id_from_page(html: str) -> Optional[str]:
    """Pull a channel page's OWN id out of its HTML.

    Reads <link rel="canonical">, and ONLY that. The obvious-looking
    alternative -- the first "channelId":"UC..." in the embedded JSON --
    is wrong: those entries are the sidebar's RECOMMENDED channels, in
    YouTube's own order, and the page's own id may not appear among them
    at all. Confirmed the hard way: resolving @CoinBureau that way
    returned UCnThE8FLrlN-tYvZhZL0uaA, which is "Finance Bureau", a
    different channel entirely -- the digest would have quietly watched
    the wrong creator forever. The canonical link gave the right one
    (UCqK_GSMbpiV8spgD3ZGloSw, "Coin Bureau").

    Returns None rather than guessing if the canonical link is missing:
    silently watching an unknown channel is worse than not resolving.
    """
    match = re.search(
        r'<link\s+rel="canonical"\s+href="https://www\.youtube\.com/channel/(UC[\w-]{22})"',
        html,
    )
    return match.group(1) if match else None


def fetch_channel_videos(channel_id: str) -> list[dict]:
    """Recent uploads for a channel, newest first, from its public feed."""
    resp = requests.get(FEED_URL.format(channel_id=channel_id), timeout=15)
    resp.raise_for_status()
    root = ET.fromstring(resp.text)

    channel_name_el = root.find("a:title", NS)
    channel_name = channel_name_el.text if channel_name_el is not None else channel_id

    videos = []
    for entry in root.findall("a:entry", NS):
        video_id_el = entry.find("yt:videoId", NS)
        title_el = entry.find("a:title", NS)
        published_el = entry.find("a:published", NS)
        if video_id_el is None or video_id_el.text is None:
            continue
        videos.append({
            "video_id": video_id_el.text,
            "title": title_el.text if title_el is not None else "",
            "published_at": published_el.text if published_el is not None else None,
            "channel_id": channel_id,
            "channel_name": channel_name,
        })
    return videos


def _is_recent(published_at: Optional[str]) -> bool:
    if not published_at:
        return False
    try:
        published = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
    except ValueError:
        return False
    return published >= datetime.now(timezone.utc) - timedelta(days=MAX_VIDEO_AGE_DAYS)


def is_short(video_id: str) -> bool:
    """True if the video is a YouTube Short.

    The RSS feed carries no duration and there is no keyless API for it,
    so this uses YouTube's own routing: request /shorts/<id> WITHOUT
    following redirects. A real Short stays there (200); a full-length
    video is redirected (303) to /watch?v=<id>. Verified against a real
    channel feed -- it separated four Shorts from a full video correctly,
    and agreed with two independently-known full videos.

    Fails OPEN (returns False, "not a Short") when the check itself
    fails. A network blip should not silently make every video look like
    a Short and empty the digest; the worst case of the other direction
    is one recap video in the report.
    """
    try:
        resp = requests.head(
            f"https://www.youtube.com/shorts/{video_id}",
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=15,
            allow_redirects=False,
        )
        return resp.status_code == 200
    except Exception as exc:
        print(f"⚠️ Could not check whether {video_id} is a Short: {exc}")
        return False


def latest_full_video(channel_id: str) -> Optional[dict]:
    """The channel's newest NON-Short video inside the age window, or None.

    Deliberately returns at most one: Shorts are recaps of the full
    videos, so pulling several items from one channel would repeat the
    same ideas across the report.
    """
    try:
        videos = fetch_channel_videos(channel_id)
    except Exception as exc:
        print(f"⚠️ Could not read feed for channel {channel_id}: {exc}")
        return None

    for video in videos[:MAX_FEED_ENTRIES_SCANNED]:
        if not _is_recent(video["published_at"]):
            break   # feed is newest-first, so everything after is older too
        if is_short(video["video_id"]):
            continue
        return video
    return None


def fetch_transcript(video_id: str) -> Optional[str]:
    """Full caption text for a video in WHATEVER language it exists in,
    or None if unavailable.

    Language handling is the whole point of this function's shape.
    YouTubeTranscriptApi().fetch() defaults to English only and raises
    NoTranscriptFound for anything else -- which silently killed the
    feature for every Spanish-speaking creator tested (all three had
    perfectly good auto-generated 'es' captions). So: ask for a preferred
    language first, then accept ANY transcript the video has. The
    summarizer writes in REPORT_LANGUAGE regardless of the source
    language, so a Spanish video still yields an English summary if
    that's how the report is configured, and vice versa.

    Returns None (never raises) for the many ordinary reasons this fails:
    captions genuinely disabled, members-only/age-restricted video, or
    YouTube blocking the request -- see this module's docstring on
    datacenter IPs.
    """
    try:
        from youtube_transcript_api import YouTubeTranscriptApi

        transcript_list = YouTubeTranscriptApi().list(video_id)
        # Preferred order, de-duplicated: the report's own language first
        # (cheapest for the model to work with), then the two languages
        # this project actually sees, then anything at all.
        preferred = list(dict.fromkeys(
            [os.environ.get("REPORT_LANGUAGE", "en").lower(), "es", "en"]
        ))
        try:
            transcript = transcript_list.find_transcript(preferred)
        except Exception:
            transcript = next(iter(transcript_list), None)
        if transcript is None:
            print(f"⚠️ No transcript for video {video_id}: no tracks at all")
            return None

        fetched = transcript.fetch()
        return " ".join(snippet.text for snippet in fetched).strip() or None
    except Exception as exc:
        print(f"⚠️ No transcript for video {video_id}: {type(exc).__name__}")
        return None


def _distinct_keyword_hits(text: str) -> int:
    lowered = (text or "").lower()
    return len({kw for kw in CRYPTO_KEYWORDS if kw in lowered})


def _mentions_crypto(text: str) -> bool:
    """One keyword is enough. For SHORT text only (a summary) -- see the
    note at the stage-3 backstop in _summarize_with_llm()."""
    return _distinct_keyword_hits(text) >= 1


def looks_crypto(title: str, transcript: str) -> bool:
    """Stage-1 deterministic prefilter -- see this module's docstring.

    The bar scales with length: a short video (a Short) needs one keyword,
    a full-length one needs two. See MIN_KEYWORD_HITS_SHORT for the real
    measurements behind that.
    """
    text = f"{title or ''} {transcript or ''}"
    needed = (MIN_KEYWORD_HITS_SHORT if len(transcript or "") < SHORT_TRANSCRIPT_CHARS
              else MIN_KEYWORD_HITS)
    return _distinct_keyword_hits(text) >= needed


# Stage 2a. A DEDICATED binary classifier, deliberately kept as its own
# LLM call rather than folded into the summarizer prompt.
#
# The first version did fold them together ("if it's not crypto, reply
# NOT_CRYPTO, otherwise summarize"). Tested against the local Ollama
# model with a laptop review that name-drops bitcoin and ethereum, it
# ignored the rule outright and cheerfully summarized the laptop -- the
# same class of failure that got the LLM "portfolio manager" sub-agent
# deleted from this project. A combined prompt offers an easy path:
# summarizing is the more natural task, so the judgment silently gets
# skipped. Asked the yes/no question ALONE, the same model answered all
# three probe cases correctly, including the laptop review (NO) and a
# genuinely-crypto video whose title contains no crypto word (YES).
_CLASSIFIER_PROMPT = """You are a strict binary classifier. Answer with ONE word only.

Question: is this YouTube video PRIMARILY about cryptocurrency, crypto
markets, crypto assets, or the crypto industry?

Answer YES only if crypto is the MAIN SUBJECT of the video.
Answer NO if crypto is merely mentioned, used as an example, or is
incidental to a video that is really about something else (a laptop
review, gaming, general tech, general news, etc).

Reply with exactly one word: YES or NO. No explanation."""


def _agent_module():
    """The configured LLM backend, chosen the same way main.py does."""
    if os.environ.get("LLM_BACKEND", "claude").lower() == "ollama":
        from . import agent_ollama as module
    else:
        from . import agent as module
    return module


def _video_context(title: str, channel_name: str, transcript: str) -> str:
    return (
        f"Channel: {channel_name}\n"
        f"Video title: {title}\n\n"
        f"Transcript (may be truncated):\n{transcript[:TRANSCRIPT_CHAR_LIMIT]}"
    )


def classify_is_crypto(title: str, channel_name: str, transcript: str) -> bool:
    """Stage 2a: does this video actually have crypto as its subject?

    FAILS CLOSED. Anything that isn't a clear YES -- an ambiguous answer,
    an empty one, a crashed call -- is treated as "not crypto", because
    the requirement is that non-crypto videos never reach the report. The
    cost of a false negative is a missed mention; the cost of a false
    positive is a laptop review in a crypto report.
    """
    try:
        answer = _agent_module().summarize(
            _CLASSIFIER_PROMPT, _video_context(title, channel_name, transcript), max_tokens=5
        )
    except Exception as exc:
        print(f"⚠️ Could not classify video: {exc}")
        return False
    # Strict: must actually START with yes. A model that starts explaining
    # itself ("YES, because...") still passes; one that hedges does not.
    return (answer or "").strip().upper().startswith("YES")


def _summarizer_prompt() -> str:
    language = os.environ.get("REPORT_LANGUAGE", "en")
    return f"""You are summarizing a crypto YouTube video for a daily
market report. You get the video's title and part of its transcript.

1. Write ONE OR TWO short sentences capturing the video's central
   argument -- the thesis, not a list of everything it covers. Write it
   in your own words as a synthesis; never quote the transcript at length.
2. Attribute it as the creator's view, not as fact and not as your own
   analysis (e.g. "X argues that...", "según X...").
3. NEVER turn it into advice: no buy/sell recommendation, no price
   target, no "this means you should...". You are reporting what someone
   said, nothing more.
4. No preamble, no "here is the summary", no markdown. Just the sentences.

Write in this language (ISO code): {language}."""


def _summarize_with_llm(title: str, channel_name: str, transcript: str) -> Optional[str]:
    """Stage 2: classify first, summarize only if it passed, then check
    the result. Returns None whenever the video shouldn't be reported."""
    if not classify_is_crypto(title, channel_name, transcript):
        return None

    try:
        summary = (_agent_module().summarize(
            _summarizer_prompt(), _video_context(title, channel_name, transcript)
        ) or "").strip()
    except Exception as exc:
        print(f"⚠️ Could not summarize video: {exc}")
        return None

    if not summary:
        return None

    # Stage 3, code-level backstop: the summary itself must still read as
    # crypto. Cheap insurance against a summarizer that drifts onto the
    # video's side topics -- and, unlike the prompt rules above, this one
    # cannot be talked out of. Same principle as everywhere else here: if
    # code can guarantee it, don't buy it from a prompt.
    #
    # Note this uses a ONE-keyword bar, not looks_crypto()'s two. That
    # function scans a whole transcript, where a lone passing mention
    # means nothing; here the text is one or two sentences, where a single
    # "bitcoin" is a strong signal and demanding two would throw away
    # perfectly good summaries (the first real one produced by this code
    # mentioned only Bitcoin).
    if not _mentions_crypto(f"{title} {summary}"):
        print(f"⚠️ Discarding summary that doesn't read as crypto: {summary[:80]!r}")
        return None
    return summary


def _seen_video_ids(video_ids: list[str]) -> set[str]:
    client = db.get_client()
    if not client or not video_ids:
        return set()
    rows = client.table("creator_videos").select("video_id").in_("video_id", video_ids).execute().data
    return {r["video_id"] for r in rows or []}


def _record_video(video: dict, is_crypto: bool, summary: Optional[str]) -> None:
    client = db.get_client()
    if not client:
        return
    client.table("creator_videos").insert({
        "channel_id": video["channel_id"],
        "channel_name": video["channel_name"],
        "video_id": video["video_id"],
        "title": video["title"],
        "published_at": video["published_at"],
        "is_crypto": is_crypto,
        "summary": summary,
        "reported_at": datetime.now(timezone.utc).isoformat() if is_crypto else None,
    }).execute()


def get_creator_digest() -> Optional[str]:
    """Short synthesis of any NEW crypto video from the watched channels,
    ready to append to the daily report -- or None when there's nothing
    to say (no new video, none of them crypto, feature off, or anything
    at all went wrong).

    Never raises: see this module's docstring, point 5.
    """
    if not is_enabled():
        return None
    try:
        return _build_digest()
    except Exception as exc:
        print(f"⚠️ Could not build the creator digest: {exc}")
        return None


def _build_digest() -> Optional[str]:
    # Exactly one candidate per channel: its newest full video. If that
    # one was already covered on a previous run, this channel simply has
    # nothing new to say -- the digest does NOT walk further back for a
    # second-newest video it hasn't reported yet.
    candidates: list[dict] = []
    for entry in _configured_channels():
        channel_id = resolve_channel_id(entry)
        if not channel_id:
            continue
        video = latest_full_video(channel_id)
        if video:
            candidates.append(video)

    if not candidates:
        return None

    already_seen = _seen_video_ids([v["video_id"] for v in candidates])
    fresh = [v for v in candidates if v["video_id"] not in already_seen]

    lines = []
    for video in fresh:
        transcript = fetch_transcript(video["video_id"])
        if not transcript:
            # Deliberately NOT recorded: a missing transcript is usually
            # temporary (captions still processing right after upload),
            # so leaving it unrecorded lets tomorrow's run retry it.
            continue

        if not looks_crypto(video["title"], transcript):
            _record_video(video, is_crypto=False, summary=None)
            continue

        summary = _summarize_with_llm(video["title"], video["channel_name"], transcript)
        if not summary:
            _record_video(video, is_crypto=False, summary=None)
            continue

        _record_video(video, is_crypto=True, summary=summary)
        lines.append(f"📺 {video['channel_name']}: {summary}")

    return "\n".join(lines) if lines else None
