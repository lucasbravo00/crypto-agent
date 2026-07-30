"""
Tests for the report chart (src/report_chart.py) and for the two channels
that carry it. No network and no real SMTP/Telegram call: candles are
handed in directly and `requests` is monkeypatched.
"""
import pytest

from src import market_data, report_chart, telegram_notifier, email_notifier


# ---------------------------------------------------------------- series

def test_sma_series_is_none_until_the_window_is_full():
    out = report_chart._sma_series([1, 2, 3, 4, 5], 3)
    assert out[0] is None and out[1] is None
    assert out[2] == pytest.approx(2.0)   # (1+2+3)/3
    assert out[3] == pytest.approx(3.0)
    assert out[4] == pytest.approx(4.0)


def test_sma_series_slides_the_window_rather_than_growing_it():
    # A running-sum bug that forgets to subtract the outgoing value shows
    # up here as an ever-increasing average.
    out = report_chart._sma_series([10] * 5 + [20] * 5, 5)
    assert out[4] == pytest.approx(10.0)
    assert out[-1] == pytest.approx(20.0)


def test_rsi_series_matches_market_datas_own_rsi():
    """The whole point of the chart is to illustrate what the text says.
    If these two definitions ever diverge (e.g. someone "improves" one of
    them to Wilder smoothing), the last plotted RSI point would contradict
    the RSI quoted in the report."""
    closes = [100 + ((i * 7919) % 23) - 11 for i in range(80)]   # deterministic zig-zag
    series = report_chart._rsi_series(closes, 14)
    for i in range(20, len(closes)):
        expected = market_data._rsi(closes[: i + 1], 14)
        assert series[i] == pytest.approx(expected, abs=0.01), f"mismatch at index {i}"


def test_rsi_series_is_none_before_enough_history():
    series = report_chart._rsi_series([100, 101, 102], 14)
    assert all(v is None for v in series)


def test_rsi_series_is_100_when_there_are_no_losses():
    closes = list(range(100, 130))       # monotonically rising
    assert report_chart._rsi_series(closes, 14)[-1] == pytest.approx(100.0)


# ------------------------------------------------------------ build/guard

def test_build_report_chart_returns_none_when_disabled(monkeypatch):
    monkeypatch.setenv("REPORT_CHART_ENABLED", "false")
    called = []
    monkeypatch.setattr(market_data, "get_ohlcv", lambda *a, **k: called.append(1) or [])
    assert report_chart.build_report_chart("BTC/USDT") is None
    assert not called, "should not even fetch candles when disabled"


def test_build_report_chart_swallows_failures(monkeypatch):
    """A chart is a garnish: it must never take the report down with it."""
    monkeypatch.delenv("REPORT_CHART_ENABLED", raising=False)

    def boom(*a, **k):
        raise RuntimeError("exchange unreachable")

    monkeypatch.setattr(market_data, "get_ohlcv", boom)
    assert report_chart.build_report_chart("BTC/USDT") is None


def test_build_report_chart_produces_a_png(monkeypatch):
    monkeypatch.delenv("REPORT_CHART_ENABLED", raising=False)
    candles = [
        [1_700_000_000_000 + i * 86_400_000, 0, 0, 0, 100 + ((i * 13) % 40), 0]
        for i in range(260)
    ]
    monkeypatch.setattr(market_data, "get_ohlcv", lambda *a, **k: candles)
    # The header chips are best-effort extras; stub them so the test never
    # reaches the network.
    monkeypatch.setattr(market_data, "get_cycle_metrics",
                        lambda *a, **k: {"mayer_multiple": 1.1, "pct_distance_to_sma200w": 12.0})
    monkeypatch.setattr(market_data, "get_fear_greed_index",
                        lambda *a, **k: {"value": 55, "classification": "Greed"})

    png = report_chart.build_report_chart("BTC/USDT")
    assert png is not None
    assert png[:8] == b"\x89PNG\r\n\x1a\n", "should be a real PNG"
    assert len(png) > 5000


def test_build_report_chart_survives_a_failing_header_lookup(monkeypatch):
    """Fear & Greed is rate-limited often enough that this matters: losing
    a chip must not lose the chart."""
    monkeypatch.delenv("REPORT_CHART_ENABLED", raising=False)
    candles = [
        [1_700_000_000_000 + i * 86_400_000, 0, 0, 0, 100 + ((i * 13) % 40), 0]
        for i in range(260)
    ]
    monkeypatch.setattr(market_data, "get_ohlcv", lambda *a, **k: candles)
    monkeypatch.setattr(market_data, "get_cycle_metrics",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("429")))
    monkeypatch.setattr(market_data, "get_fear_greed_index",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("429")))
    assert report_chart.build_report_chart("BTC/USDT") is not None


# ------------------------------------------------------------- telegram

class _FakeResp:
    def raise_for_status(self):
        pass


@pytest.fixture
def telegram_env(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")


def _record_posts(monkeypatch):
    posts = []

    def fake_post(url, **kwargs):
        posts.append({"url": url, **kwargs})
        return _FakeResp()

    monkeypatch.setattr(telegram_notifier.requests, "post", fake_post)
    return posts


def test_telegram_without_image_sends_a_plain_message(telegram_env, monkeypatch):
    posts = _record_posts(monkeypatch)
    telegram_notifier.send_message("hola")
    assert len(posts) == 1
    assert posts[0]["url"].endswith("/sendMessage")
    assert "files" not in posts[0]


def test_telegram_with_image_sends_one_photo_carrying_the_text(telegram_env, monkeypatch):
    posts = _record_posts(monkeypatch)
    telegram_notifier.send_message("informe corto", image=b"PNGDATA")
    assert len(posts) == 1, "short text should ride along as the caption"
    assert posts[0]["url"].endswith("/sendPhoto")
    assert posts[0]["data"]["caption"] == "informe corto"
    assert posts[0]["files"]["photo"][1] == b"PNGDATA"


def test_telegram_splits_when_the_text_exceeds_the_caption_limit(telegram_env, monkeypatch):
    posts = _record_posts(monkeypatch)
    long_text = "x" * (telegram_notifier.CAPTION_LIMIT + 1)
    telegram_notifier.send_message(long_text, image=b"PNGDATA")

    assert len(posts) == 2, "photo first, then the full text as its own message"
    assert posts[0]["url"].endswith("/sendPhoto")
    assert posts[0]["data"]["caption"] == ""
    assert posts[1]["url"].endswith("/sendMessage")
    # Never truncated -- that was the whole reason for the split.
    assert posts[1]["data"]["text"] == long_text


# ---------------------------------------------------------------- email

def test_email_html_escapes_the_report_text():
    """report_text is LLM output; it is never trusted as markup."""
    html = email_notifier._html_body('<script>alert("x")</script>', "cid1")
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert 'src="cid:cid1"' in html


def test_email_without_image_stays_plain_text(monkeypatch):
    sent = _capture_sent_message(monkeypatch)
    email_notifier.send_email("asunto", "cuerpo")
    msg = sent[0]
    assert not msg.is_multipart()
    assert msg.get_content().strip() == "cuerpo"


def test_email_with_image_embeds_it_inline(monkeypatch):
    sent = _capture_sent_message(monkeypatch)
    email_notifier.send_email("asunto", "cuerpo", image=b"\x89PNG-data")
    msg = sent[0]

    assert msg.is_multipart()
    # The text/plain part must survive for clients that refuse HTML.
    assert msg.get_body(preferencelist=("plain",)).get_content().strip() == "cuerpo"

    html_part = msg.get_body(preferencelist=("html",))
    assert html_part is not None
    assert "cid:report-chart" in html_part.get_content()

    images = [p for p in msg.walk() if p.get_content_type() == "image/png"]
    assert len(images) == 1
    assert images[0].get_payload(decode=True) == b"\x89PNG-data"
    # A bare "related" sibling of the root would leave cid: unresolvable.
    assert images[0].get("Content-ID") == "<report-chart>"


def _capture_sent_message(monkeypatch):
    sent = []

    class _FakeSMTP:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def starttls(self, **k):
            pass

        def login(self, *a):
            pass

        def send_message(self, msg):
            sent.append(msg)

    monkeypatch.setattr(email_notifier.smtplib, "SMTP", _FakeSMTP)
    for key, val in [
        ("SMTP_HOST", "smtp.test"), ("SMTP_PORT", "587"), ("SMTP_USER", "u"),
        ("SMTP_PASSWORD", "p"), ("EMAIL_TO", "to@test"), ("EMAIL_FROM", "from@test"),
    ]:
        monkeypatch.setenv(key, val)
    return sent
