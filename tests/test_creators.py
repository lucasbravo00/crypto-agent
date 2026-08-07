"""
Tests for the YouTube creator digest (src/creators.py). No network, no
LLM, no Supabase: the feed, the transcript fetch, the summarizer and the
database are all monkeypatched.
"""
import pytest

from src import creators


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch):
    """Force the feature ON with a known channel list, and force Supabase
    OFF at the db level so no test can touch a real project. Individual
    tests re-patch creators' db helpers where they need them."""
    monkeypatch.setenv("YOUTUBE_DIGEST_ENABLED", "true")
    monkeypatch.setenv("YOUTUBE_CHANNELS", "UCqK_GSMbpiV8spgD3ZGloSw")
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_KEY", raising=False)
    yield


# A trimmed copy of a real feed, including the FEED-LEVEL <published>
# element that broke a regex-based parse while this was being built (it
# shifted every video's date by one entry).
FEED_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns:yt="http://www.youtube.com/xml/schemas/2015" xmlns="http://www.w3.org/2005/Atom">
 <title>Coin Bureau</title>
 <published>2019-01-16T00:00:00+00:00</published>
 <entry>
  <yt:videoId>vid_new</yt:videoId>
  <title>Bitcoin Just PROVED It Doesn't Need Saylor</title>
  <published>2026-08-04T12:00:00+00:00</published>
 </entry>
 <entry>
  <yt:videoId>vid_old</yt:videoId>
  <title>An old video</title>
  <published>2020-01-01T12:00:00+00:00</published>
 </entry>
</feed>"""


class _Resp:
    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        pass


# --- feed parsing ---

def test_fetch_channel_videos_does_not_shift_dates(monkeypatch):
    """Regression guard for a real bug: YouTube's feed carries a
    FEED-level <published> as well as one per entry, so zipping regex
    matches dated the newest video to 2019."""
    monkeypatch.setattr(creators.requests, "get", lambda *a, **k: _Resp(FEED_XML))
    videos = creators.fetch_channel_videos("UC123")
    assert [v["video_id"] for v in videos] == ["vid_new", "vid_old"]
    assert videos[0]["published_at"].startswith("2026-08-04")
    assert videos[0]["title"].startswith("Bitcoin Just PROVED")
    assert videos[0]["channel_name"] == "Coin Bureau"


def test_is_recent_rejects_old_videos():
    assert creators._is_recent("2020-01-01T00:00:00+00:00") is False
    assert creators._is_recent(None) is False
    assert creators._is_recent("not-a-date") is False


# --- channel id resolution ---

def test_resolve_channel_id_passes_through_a_real_id():
    cid = "UCqK_GSMbpiV8spgD3ZGloSw"
    assert creators.resolve_channel_id(cid) == cid
    # ...and does NOT hit the network to do it (no requests patch here;
    # a network call would fail the test run).


def test_resolve_channel_id_extracts_from_a_handle_page(monkeypatch):
    page = '<link rel="canonical" href="https://www.youtube.com/channel/UCqK_GSMbpiV8spgD3ZGloSw">'
    monkeypatch.setattr(creators.requests, "get", lambda *a, **k: _Resp(page))
    assert creators.resolve_channel_id("@CoinBureau") == "UCqK_GSMbpiV8spgD3ZGloSw"


def test_resolve_channel_id_ignores_recommended_channels_in_the_json():
    """Regression guard for a real bug. A channel page embeds the
    sidebar's RECOMMENDED channels as "channelId":"UC..." entries, listed
    BEFORE (and without) the page's own id. Taking the first match
    resolved @CoinBureau to "Finance Bureau" -- the digest would have
    silently watched the wrong creator. Only the canonical link counts."""
    page = (
        '{"channelId":"UCnThE8FLrlN-tYvZhZL0uaA"}'          # recommended, WRONG
        '{"channelId":"UCKVtBpONPhVWez_0X-yVCOw"}'          # recommended, WRONG
        '<link rel="canonical" href="https://www.youtube.com/channel/UCqK_GSMbpiV8spgD3ZGloSw">'
    )
    assert creators._channel_id_from_page(page) == "UCqK_GSMbpiV8spgD3ZGloSw"


def test_resolve_channel_id_returns_none_rather_than_guessing():
    """No canonical link -> None. Watching an unknown channel silently is
    worse than failing to resolve."""
    assert creators._channel_id_from_page('{"channelId":"UCnThE8FLrlN-tYvZhZL0uaA"}') is None


def test_resolve_channel_id_returns_none_on_failure(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("network down")

    monkeypatch.setattr(creators.requests, "get", boom)
    assert creators.resolve_channel_id("@Whoever") is None


# --- stage 1: the cheap keyword prefilter ---

def test_looks_crypto_needs_more_than_one_passing_mention():
    # In a FULL-LENGTH video, a single keyword is not enough: an unrelated
    # video that name-drops bitcoin once must not cost an LLM call. (The
    # bar is deliberately looser for Shorts -- see the tests below.)
    long_unrelated = ("hoy horneamos pan de masa madre en casa. " * 200) + " unlike bitcoin"
    assert len(long_unrelated) > creators.SHORT_TRANSCRIPT_CHARS
    assert creators.looks_crypto("My sourdough recipe", long_unrelated) is False


def test_looks_crypto_accepts_an_obviously_crypto_video():
    assert creators.looks_crypto(
        "Bitcoin and Ethereum update",
        "the bitcoin halving and ethereum staking are the topic",
    ) is True


def test_looks_crypto_handles_empty_input():
    assert creators.looks_crypto("", "") is False


def test_looks_crypto_accepts_a_short_video_with_one_keyword():
    """Regression guard, from real Shorts published the same day. A
    600-char Short has no room for a passing mention: whatever it names
    is what it's about. A CriptoNorber Short about Saylor and BlackRock
    hoarding bitcoin scored exactly ONE distinct keyword and the flat
    two-keyword bar wrongly threw it out."""
    short = ("Tenés dos entidades que tienen el 5%. Michael Saylor tiene casi 900.000 "
             "bitcoins y BlackRock otros 900.000. De los 21 millones tienen casi un "
             "palo cada uno. Si quieren salir a vender, mueven el precio.")
    assert len(short) < creators.SHORT_TRANSCRIPT_CHARS
    assert creators.looks_crypto("El problema de Saylor y Blackrock", short) is True


def test_looks_crypto_still_rejects_a_short_video_with_no_keywords():
    """The other real Short from that same test: off-topic (an Argentine
    mining-investment regime) and scoring zero. One keyword separates the
    two cleanly; two loses the crypto one."""
    short = ("El gobierno sumó un proyecto en Catamarca por 709 millones. Estiman "
             "4.500 empleos y exportaciones de 400 millones de dólares al año.")
    assert creators.looks_crypto("NUEVO RIGI EN CATAMARCA", short) is False


def test_looks_crypto_keeps_the_stricter_bar_for_long_transcripts():
    """A single passing mention buried in a full-length video must still
    not qualify -- that's what the two-keyword bar is for."""
    long_text = ("hoy hablamos de velas japonesas y análisis técnico. " * 200) + " bitcoin "
    assert len(long_text) > creators.SHORT_TRANSCRIPT_CHARS
    assert creators.looks_crypto("Trading con velas japonesas", long_text) is False


def test_keyword_hits_use_word_boundaries_not_bare_substrings():
    """Regression guard, from a real Alex Ruiz transcript. The old plain
    `kw in text` check matched "defi" inside "en definitiva" (Spanish for
    "in short") and "eth" inside "something" -- neither is a crypto
    mention. Word-boundary matching must reject both."""
    assert creators._distinct_keyword_hits("en definitiva no puede dedicar más tiempo") == 0
    assert creators._distinct_keyword_hits("this is definitely something else entirely") == 0


def test_keyword_hits_still_match_plural_forms():
    """The word-boundary fix must not lose real hits: the same Alex Ruiz
    video that exposed the "defi" false positive genuinely says
    "criptomonedas" and "criptos" -- plurals of the singular keywords in
    CRYPTO_KEYWORDS -- and those must still count."""
    assert creators._distinct_keyword_hits("hablamos del mercado de las criptomonedas") == 1
    assert creators._distinct_keyword_hits("si tú quieres operar criptos, o acciones") == 1


# --- stage 2 + assembly ---

def _patch_pipeline(monkeypatch, *, transcript, summary, recorded=None, shorts=()):
    """Wire the whole pipeline with stubs; collect what got recorded.
    `shorts` lists video ids to treat as Shorts (everything else is a
    full video). Stubbed because the real check is an HTTP HEAD."""
    monkeypatch.setattr(creators.requests, "get", lambda *a, **k: _Resp(FEED_XML))
    monkeypatch.setattr(creators, "is_short", lambda vid: vid in shorts)
    monkeypatch.setattr(creators, "_is_recent", lambda p: p is not None and p.startswith("2026"))
    monkeypatch.setattr(creators, "fetch_transcript", lambda vid: transcript)
    monkeypatch.setattr(creators, "_summarize_with_llm", lambda *a, **k: summary)
    monkeypatch.setattr(creators, "_seen_video_ids", lambda ids: set())
    if recorded is not None:
        monkeypatch.setattr(creators, "_record_video",
                            lambda v, is_crypto, summary: recorded.append((v["video_id"], is_crypto, summary)))
    else:
        monkeypatch.setattr(creators, "_record_video", lambda *a, **k: None)
    monkeypatch.setattr(creators.db, "is_enabled", lambda: True)


def test_digest_includes_a_new_crypto_video(monkeypatch):
    _patch_pipeline(monkeypatch,
                    transcript="bitcoin ethereum halving talk",
                    summary="Coin Bureau sostiene que el ciclo aún no terminó.")
    digest = creators.get_creator_digest()
    assert digest is not None
    assert "Coin Bureau" in digest
    assert "el ciclo aún no terminó" in digest


def test_digest_skips_a_video_the_llm_judged_not_crypto(monkeypatch):
    recorded = []
    # Passes the keyword prefilter but the LLM returns NOT_CRYPTO -> None.
    _patch_pipeline(monkeypatch,
                    transcript="bitcoin ethereum mentioned in passing",
                    summary=None, recorded=recorded)
    assert creators.get_creator_digest() is None
    # Still recorded, so tomorrow's run doesn't re-download and re-judge it.
    assert recorded == [("vid_new", False, None)]


def test_digest_skips_and_records_a_video_that_fails_the_keyword_filter(monkeypatch):
    """Uses a feed whose TITLE is also off-topic: looks_crypto() scans
    title+transcript together, so a crypto title alone would carry it
    past stage 1 (correctly -- stage 2 would then be the one to reject)."""
    off_topic_feed = FEED_XML.replace(
        "Bitcoin Just PROVED It Doesn't Need Saylor", "My sourdough recipe")
    recorded = []
    _patch_pipeline(monkeypatch, transcript="today we bake bread in a cast iron pot",
                    summary="should never be used", recorded=recorded)
    monkeypatch.setattr(creators.requests, "get", lambda *a, **k: _Resp(off_topic_feed))
    assert creators.get_creator_digest() is None
    assert recorded == [("vid_new", False, None)]


def test_digest_does_not_record_a_video_with_no_transcript(monkeypatch):
    """Captions are often still processing right after upload -- leaving
    it unrecorded is what lets tomorrow's run retry it."""
    recorded = []
    _patch_pipeline(monkeypatch, transcript=None, summary="unused", recorded=recorded)
    assert creators.get_creator_digest() is None
    assert recorded == []


def test_digest_skips_videos_already_seen(monkeypatch):
    _patch_pipeline(monkeypatch, transcript="bitcoin ethereum halving",
                    summary="a summary")
    monkeypatch.setattr(creators, "_seen_video_ids", lambda ids: {"vid_new"})
    assert creators.get_creator_digest() is None


def test_digest_takes_only_one_video_per_channel(monkeypatch):
    """One line per channel at most: Shorts recap the full videos, so
    pulling several items from the same creator repeats the same ideas."""
    many = "".join(
        f"<entry><yt:videoId>v{i}</yt:videoId><title>Bitcoin {i}</title>"
        f"<published>2026-08-04T12:00:00+00:00</published></entry>"
        for i in range(10)
    )
    feed = ('<?xml version="1.0" encoding="UTF-8"?>'
            '<feed xmlns:yt="http://www.youtube.com/xml/schemas/2015" '
            'xmlns="http://www.w3.org/2005/Atom"><title>C</title>' + many + "</feed>")
    monkeypatch.setattr(creators.requests, "get", lambda *a, **k: _Resp(feed))
    monkeypatch.setattr(creators, "is_short", lambda vid: False)
    monkeypatch.setattr(creators, "_is_recent", lambda p: True)
    monkeypatch.setattr(creators, "fetch_transcript", lambda vid: "bitcoin ethereum halving")
    monkeypatch.setattr(creators, "_summarize_with_llm", lambda *a, **k: "s")
    monkeypatch.setattr(creators, "_seen_video_ids", lambda ids: set())
    monkeypatch.setattr(creators, "_record_video", lambda *a, **k: None)
    monkeypatch.setattr(creators.db, "is_enabled", lambda: True)

    digest = creators.get_creator_digest()
    assert len(digest.splitlines()) == 1, "one configured channel -> at most one line"


# --- Shorts exclusion ---
# Shorts are recaps of the full videos, so they add no new ideas and would
# duplicate content across the report.

def test_latest_full_video_skips_over_shorts(monkeypatch):
    """Real shape: CriptoNorber had four Shorts in a row before its last
    full video, so the scan has to walk past them, not give up at the
    newest entry."""
    monkeypatch.setattr(creators.requests, "get", lambda *a, **k: _Resp(FEED_XML))
    monkeypatch.setattr(creators, "_is_recent", lambda p: True)
    monkeypatch.setattr(creators, "is_short", lambda vid: vid == "vid_new")
    got = creators.latest_full_video("UC123")
    assert got["video_id"] == "vid_old"


def test_latest_full_video_returns_none_when_everything_is_a_short(monkeypatch):
    monkeypatch.setattr(creators.requests, "get", lambda *a, **k: _Resp(FEED_XML))
    monkeypatch.setattr(creators, "_is_recent", lambda p: True)
    monkeypatch.setattr(creators, "is_short", lambda vid: True)
    assert creators.latest_full_video("UC123") is None


def test_latest_full_video_stops_at_the_age_window(monkeypatch):
    """The feed is newest-first, so the first too-old entry means every
    entry after it is older too -- no point checking (or paying a HEAD
    request for) any of them."""
    checked = []
    monkeypatch.setattr(creators.requests, "get", lambda *a, **k: _Resp(FEED_XML))
    monkeypatch.setattr(creators, "_is_recent", lambda p: False)
    monkeypatch.setattr(creators, "is_short", lambda vid: checked.append(vid) or False)
    assert creators.latest_full_video("UC123") is None
    assert checked == [], "must not check Shorts for videos outside the window"


def test_digest_skips_a_channel_whose_latest_full_video_was_already_covered(monkeypatch):
    """'Only the latest video, as long as it wasn't taken already' -- it
    does NOT walk further back looking for an older unreported one."""
    _patch_pipeline(monkeypatch, transcript="bitcoin ethereum halving",
                    summary="a summary", shorts=())
    monkeypatch.setattr(creators, "_seen_video_ids", lambda ids: {"vid_new"})
    assert creators.get_creator_digest() is None


def test_is_short_fails_open_on_a_network_error(monkeypatch):
    """A blip must not make every video look like a Short and silently
    empty the digest."""
    def boom(*a, **k):
        raise RuntimeError("network down")

    monkeypatch.setattr(creators.requests, "head", boom)
    assert creators.is_short("vid") is False


def test_is_short_reads_the_redirect(monkeypatch):
    """YouTube keeps a real Short on /shorts/ (200) and redirects a
    full-length video (303) to /watch?v=."""
    class _Head:
        def __init__(self, code):
            self.status_code = code

    monkeypatch.setattr(creators.requests, "head", lambda *a, **k: _Head(200))
    assert creators.is_short("a_short") is True
    monkeypatch.setattr(creators.requests, "head", lambda *a, **k: _Head(303))
    assert creators.is_short("a_full_video") is False


# --- guardrails ---

def test_digest_is_none_when_disabled(monkeypatch):
    monkeypatch.setenv("YOUTUBE_DIGEST_ENABLED", "false")
    called = []
    monkeypatch.setattr(creators.requests, "get", lambda *a, **k: called.append(1))
    assert creators.get_creator_digest() is None
    assert not called, "must not touch the network when the feature is off"


def test_digest_is_none_without_configured_channels(monkeypatch):
    monkeypatch.setenv("YOUTUBE_CHANNELS", "")
    monkeypatch.setattr(creators.db, "is_enabled", lambda: True)
    assert creators.get_creator_digest() is None


def test_digest_is_none_without_supabase(monkeypatch):
    """Dedup needs somewhere to remember seen videos; without it the same
    video would be re-reported every day."""
    monkeypatch.setattr(creators.db, "is_enabled", lambda: False)
    assert creators.get_creator_digest() is None


def test_digest_never_raises(monkeypatch):
    """Best-effort by contract: the daily report must go out regardless."""
    monkeypatch.setattr(creators.db, "is_enabled", lambda: True)

    def boom(*a, **k):
        raise RuntimeError("youtube exploded")

    monkeypatch.setattr(creators, "_build_digest", boom)
    assert creators.get_creator_digest() is None


class _FakeSnippet:
    def __init__(self, text):
        self.text = text


class _FakeTranscript:
    def __init__(self, lang, text):
        self.language_code = lang
        self._text = text

    def fetch(self):
        return [_FakeSnippet(self._text)]


class _FakeTranscriptList:
    """Mimics the library's TranscriptList: iterable, plus find_transcript()
    which raises when none of the requested languages exist."""
    def __init__(self, transcripts):
        self._transcripts = transcripts

    def __iter__(self):
        return iter(self._transcripts)

    def find_transcript(self, languages):
        for lang in languages:
            for t in self._transcripts:
                if t.language_code == lang:
                    return t
        raise RuntimeError("NoTranscriptFound")


def _fake_transcript_api(monkeypatch, transcripts):
    import youtube_transcript_api

    class _Api:
        def list(self, video_id):
            return _FakeTranscriptList(transcripts)

    monkeypatch.setattr(youtube_transcript_api, "YouTubeTranscriptApi", _Api)


def test_fetch_transcript_falls_back_to_any_available_language(monkeypatch):
    """Regression guard for the bug that made this feature useless for
    every Spanish-speaking creator: the library's fetch() defaults to
    English only and raises NoTranscriptFound otherwise. All three real
    channels tested had perfectly good auto-generated 'es' captions and
    every one of them came back empty."""
    monkeypatch.setenv("REPORT_LANGUAGE", "en")
    _fake_transcript_api(monkeypatch, [_FakeTranscript("es", "hola bitcoin")])
    assert creators.fetch_transcript("vid") == "hola bitcoin"


def test_fetch_transcript_prefers_the_report_language(monkeypatch):
    monkeypatch.setenv("REPORT_LANGUAGE", "es")
    _fake_transcript_api(monkeypatch, [
        _FakeTranscript("en", "english text"),
        _FakeTranscript("es", "texto en espanol"),
    ])
    assert creators.fetch_transcript("vid") == "texto en espanol"


def test_fetch_transcript_returns_none_when_there_are_no_tracks(monkeypatch):
    _fake_transcript_api(monkeypatch, [])
    assert creators.fetch_transcript("vid") is None


def test_fetch_transcript_returns_none_instead_of_raising(monkeypatch):
    """Covers the blocked-datacenter-IP case the module docstring warns
    about, plus captions-disabled and every other ordinary failure."""
    import youtube_transcript_api

    class _Boom:
        def fetch(self, vid):
            raise youtube_transcript_api.IpBlocked("blocked")

    monkeypatch.setattr(youtube_transcript_api, "YouTubeTranscriptApi", _Boom)
    assert creators.fetch_transcript("whatever") is None


def _fake_backend(monkeypatch, answers):
    """Stub the LLM backend. `answers` is a list consumed in call order:
    the classifier call first, then the summarizer call."""
    calls = []

    class _FakeAgent:
        @staticmethod
        def summarize(system_prompt, user_text, max_tokens=300):
            calls.append(user_text)
            return answers[len(calls) - 1] if len(calls) <= len(answers) else ""

    monkeypatch.setattr(creators, "_agent_module", lambda: _FakeAgent)
    return calls


def test_transcript_is_truncated_before_reaching_the_llm(monkeypatch):
    """The raw transcript is thousands of words; only a bounded slice is
    sent, and the full text is never stored or reported."""
    calls = _fake_backend(monkeypatch, ["YES", "Bitcoin sigue fuerte, dice el creador."])
    huge = "bitcoin " * 20000
    creators._summarize_with_llm("Bitcoin update", "c", huge)
    assert calls, "the LLM was never called"
    for sent in calls:
        assert len(sent) < len(huge)
        assert len(sent) <= creators.TRANSCRIPT_CHAR_LIMIT + 500   # + the header lines


# --- stage 2a: the dedicated binary classifier ---
# Split out of the summarizer prompt after the combined version failed a
# real probe: asked to reply NOT_CRYPTO for a laptop review that name-drops
# bitcoin, the local Ollama model ignored the rule and summarized the
# laptop. Asked the yes/no question alone, it answered correctly.

def test_classifier_accepts_a_clear_yes(monkeypatch):
    _fake_backend(monkeypatch, ["YES"])
    assert creators.classify_is_crypto("t", "c", "transcript") is True


@pytest.mark.parametrize("answer", ["NO", "no", "Maybe", "", "I think so", "NOPE"])
def test_classifier_fails_closed_on_anything_but_yes(monkeypatch, answer):
    """A non-crypto video reaching the report is the failure that matters,
    so anything short of an explicit YES is treated as NO."""
    _fake_backend(monkeypatch, [answer])
    assert creators.classify_is_crypto("t", "c", "transcript") is False


def test_classifier_fails_closed_when_the_call_raises(monkeypatch):
    class _Boom:
        @staticmethod
        def summarize(*a, **k):
            raise RuntimeError("model down")

    monkeypatch.setattr(creators, "_agent_module", lambda: _Boom)
    assert creators.classify_is_crypto("t", "c", "transcript") is False


def test_no_summary_call_is_made_when_the_classifier_says_no(monkeypatch):
    calls = _fake_backend(monkeypatch, ["NO"])
    assert creators._summarize_with_llm("t", "c", "bitcoin ethereum") is None
    assert len(calls) == 1, "must not pay for a summary of a rejected video"


# --- stage 3: the code-level backstop on the produced summary ---

def test_backstop_discards_a_summary_that_is_not_about_crypto(monkeypatch):
    """Even if the classifier said YES, a summary that reads as something
    else must not reach the report. Unlike a prompt rule, this one can't
    be ignored by the model."""
    _fake_backend(monkeypatch, ["YES", "El creador dice que el teclado se siente bien."])
    assert creators._summarize_with_llm("A laptop review", "c", "transcript") is None


def test_backstop_accepts_a_summary_with_a_single_crypto_mention(monkeypatch):
    """Regression guard for a bug caught in manual testing: the backstop
    first reused looks_crypto()'s two-distinct-keyword bar, which is right
    for a whole transcript but wrong for a one-sentence summary -- it
    rejected the first real, correct summary this code produced, which
    mentioned only Bitcoin."""
    real_shape = "Según Coin Bureau, Michael Saylor dejó de comprar Bitcoin."
    _fake_backend(monkeypatch, ["YES", real_shape])
    assert creators._summarize_with_llm("Saylor stopped buying", "Coin Bureau", "t") == real_shape


def test_backstop_discards_an_empty_summary(monkeypatch):
    _fake_backend(monkeypatch, ["YES", "   "])
    assert creators._summarize_with_llm("Bitcoin news", "c", "t") is None
