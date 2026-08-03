"""
market_data.py
----------------
"Tool" functions the agent can invoke to fetch market data.
Every function is pure and testable in isolation (no LLM dependency),
which is the recommended pattern for agent tools: deterministic logic
kept separate from LLM reasoning.
"""
from __future__ import annotations
import time
import requests
import ccxt
from typing import Callable, Optional


def _with_retries(fn: Callable, attempts: int = 3, wait_seconds: float = 2.0):
    """Retry a network call up to `attempts` times with a growing wait.
    Public APIs fail sporadically (rate limits, brief outages): without
    this, a transient failure would take down the whole report."""
    last_exc = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if i < attempts - 1:
                time.sleep(wait_seconds * (i + 1))
    raise last_exc


def _exchange() -> ccxt.bingx:
    """Create a BingX client in public mode (no API key required to read
    market data)."""
    return ccxt.bingx({"enableRateLimit": True})


def get_price(symbol: str = "BTC/USDT") -> dict:
    """Return the current price and 24h change for a symbol on BingX.

    Args:
        symbol: Trading pair in BASE/QUOTE format, e.g. "BTC/USDT".
    """
    ex = _exchange()
    ticker = _with_retries(lambda: ex.fetch_ticker(symbol))
    return {
        "symbol": symbol,
        "last_price": ticker.get("last"),
        "change_24h_pct": ticker.get("percentage"),
        "high_24h": ticker.get("high"),
        "low_24h": ticker.get("low"),
        "volume_24h": ticker.get("baseVolume"),
    }


def get_ohlcv(symbol: str = "BTC/USDT", timeframe: str = "1d", limit: int = 220) -> list:
    """Fetch OHLCV candles (used to compute indicators). limit=220 is
    enough for a daily SMA200."""
    ex = _exchange()
    return _with_retries(lambda: ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit))


def _sma(values: list[float], period: int) -> Optional[float]:
    if len(values) < period:
        return None
    return sum(values[-period:]) / period


def _rsi(closes: list[float], period: int = 14) -> Optional[float]:
    if len(closes) < period + 1:
        return None
    gains, losses = [], []
    for i in range(-period, 0):
        delta = closes[i] - closes[i - 1]
        gains.append(max(delta, 0))
        losses.append(max(-delta, 0))
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 2)


def get_indicators(symbol: str = "BTC/USDT", timeframe: str = "1d") -> dict:
    """Compute objective technical indicators from real candles.
    No interpretation or prediction: it only returns verifiable numbers
    for the agent (or the user) to use as context.

    Args:
        symbol: Trading pair in BASE/QUOTE format, e.g. "BTC/USDT".
        timeframe: Candle interval, e.g. "1d", "4h", "1h".
    """
    candles = get_ohlcv(symbol, timeframe=timeframe, limit=220)
    closes = [c[4] for c in candles]
    last_close = closes[-1]

    sma50 = _sma(closes, 50)
    sma200 = _sma(closes, 200)
    rsi14 = _rsi(closes, 14)

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "last_close": last_close,
        "sma50": round(sma50, 2) if sma50 else None,
        "sma200": round(sma200, 2) if sma200 else None,
        "pct_distance_to_sma200": (
            round((last_close - sma200) / sma200 * 100, 2) if sma200 else None
        ),
        "rsi_14": rsi14,
    }


def get_intraday_rsi(symbol: str = "BTC/USDT", timeframe: str = "15m", period: int = 14) -> Optional[float]:
    """Current RSI on `timeframe` candles (default 15m) -- used by
    bullets.auto_trade() to time WHEN in the day it opens a bullet, not
    just whether to. A rolling calculation over recent candles (not reset
    at midnight), matching backtest.py's _rsi_series() and how a chart
    would actually show it -- see bullets.py's RSI_ENTRY_* constants for
    the decision this feeds."""
    candles = get_ohlcv(symbol, timeframe=timeframe, limit=100)
    closes = [c[4] for c in candles]
    return _rsi(closes, period)


def get_cycle_metrics(symbol: str = "BTC/USDT") -> dict:
    """Objective cycle-position metrics, useful for an accumulation
    strategy during a bear market. They do not predict the bottom: they
    describe where price sits today relative to historical cycle anchors.

    - sma_200w: 200-week moving average. Historically BTC spent little
      time below it, and areas near/below it coincided with bottoms of
      previous cycles (2015, 2018-19, 2022). Past behavior is no
      guarantee it repeats.
    - mayer_multiple: price / 200-day SMA. Values < 1 mean price below
      its long annual mean; historically < 0.8 coincided with
      capitulation zones.
    - drawdown_from_high_pct: percentage drop from the highest price in
      the exchange's available history (may not be the true ATH if the
      exchange has limited history; history_weeks tells you how much).
    - weekly_rsi_14: 14-period RSI on weekly candles; in previous cycles
      bottoms coincided with weekly readings < 30.

    Args:
        symbol: Trading pair in BASE/QUOTE format, e.g. "BTC/USDT".
    """
    weekly = get_ohlcv(symbol, timeframe="1w", limit=210)
    w_closes = [c[4] for c in weekly]
    w_highs = [c[2] for c in weekly]
    last = w_closes[-1]

    daily = get_ohlcv(symbol, timeframe="1d", limit=220)
    d_closes = [c[4] for c in daily]

    sma200w = _sma(w_closes, 200)
    sma200d = _sma(d_closes, 200)
    high = max(w_highs)

    return {
        "symbol": symbol,
        "last_close": last,
        "history_weeks": len(weekly),
        "sma_200w": round(sma200w, 2) if sma200w else None,
        "pct_distance_to_sma200w": (
            round((last - sma200w) / sma200w * 100, 2) if sma200w else None
        ),
        "mayer_multiple": round(last / sma200d, 3) if sma200d else None,
        "high_available_history": high,
        "drawdown_from_high_pct": round((last - high) / high * 100, 2),
        "weekly_rsi_14": _rsi(w_closes, 14),
    }

def get_trailing_high_drawdown(symbol: str = "BTC/USDT", lookback_days: int = 90) -> dict:
    """How far the current price sits below its trailing N-day high --
    decision-support data for a human's own de-risking judgment, NOT a
    signal this codebase acts on. Used by bullets.get_daily_alert() to
    surface round-depth context in the daily report (see README's
    Roadmap item 4).

    lookback_days defaults to 90 to match the ONLY lookback/threshold
    combinations that survived BOTH real bull-cycle backtests in
    backtest.py's `derisk_mode="drawdown"` sweep (30d/90d/180d, all at
    -5%) -- see README's "Reactive de-risking" section. This is not a
    coincidence: reusing that exact, already-validated framing means this
    number means the same thing here as it does in the backtest, instead
    of introducing a second, uncalibrated "recent high" definition.

    The trailing high EXCLUDES today's own (still-incomplete) daily
    candle, for the same reason backtest.py's trailing-high does: today's
    high isn't fully known yet, and including it would let an intraday
    spike inflate the number being measured against on the very day it's
    measured.

    Args:
        symbol: Trading pair in BASE/QUOTE format, e.g. "BTC/USDT".
        lookback_days: Size of the trailing window, in days.
    """
    candles = get_ohlcv(symbol, timeframe="1d", limit=lookback_days + 2)
    complete_days = candles[:-1]  # drop today's still-forming candle
    if len(complete_days) < 2:
        return {
            "symbol": symbol, "lookback_days": lookback_days,
            "trailing_high": None, "current_price": None,
            "drawdown_from_trailing_high_pct": None,
        }

    window = complete_days[-lookback_days:]
    trailing_high = max(c[2] for c in window)  # candle[2] = high
    current_price = get_price(symbol)["last_price"]

    return {
        "symbol": symbol,
        "lookback_days": lookback_days,
        "trailing_high": round(trailing_high, 2),
        "current_price": current_price,
        "drawdown_from_trailing_high_pct": round((current_price - trailing_high) / trailing_high * 100, 2),
    }


def get_current_date(**_ignored) -> dict:
    """Return today's real date. The model has no reliable notion of
    'today' on its own — if a report needs a date, it must come from
    this tool, never be guessed from training data."""
    from datetime import date
    today = date.today()
    return {"iso_date": today.isoformat(), "human_readable": today.strftime("%B %d, %Y")}

def get_fear_greed_index(**_ignored) -> dict:
    """Crypto market Fear & Greed index (public source: alternative.me)."""
    def _call():
        resp = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10)
        resp.raise_for_status()
        return resp.json()["data"][0]
    data = _with_retries(_call)
    return {"value": int(data["value"]), "classification": data["value_classification"]}


def get_btc_dominance(**_ignored) -> dict:
    """BTC dominance over total crypto market cap (CoinGecko, public)."""
    def _call():
        resp = requests.get("https://api.coingecko.com/api/v3/global", timeout=10)
        resp.raise_for_status()
        return resp.json()["data"]["market_cap_percentage"]["btc"]
    pct = _with_retries(_call)
    return {"btc_dominance_pct": round(pct, 2)}


def _true_range(candles: list) -> list[Optional[float]]:
    """True Range per candle: candles[i] = [ts, open, high, low, close, volume].
    First candle has no previous close, so its TR is undefined (None)."""
    trs: list[Optional[float]] = [None]
    for i in range(1, len(candles)):
        high, low = candles[i][2], candles[i][3]
        prev_close = candles[i - 1][4]
        trs.append(max(high - low, abs(high - prev_close), abs(low - prev_close)))
    return trs


def _rma(values: list[Optional[float]], period: int) -> list[Optional[float]]:
    """Wilder's moving average (what Pine Script's ta.atr uses under the
    hood): seeded with a plain SMA of the first `period` values, then
    recursively smoothed. `values[0]` is expected to be None (see
    _true_range) and is skipped, matching Pine's bar-indexing."""
    usable = values[1:]
    result: list[Optional[float]] = [None]
    if len(usable) < period:
        return result + [None] * len(usable)
    seed = sum(usable[:period]) / period
    result += [None] * (period - 1) + [seed]
    prev = seed
    for v in usable[period:]:
        prev = (prev * (period - 1) + v) / period
        result.append(prev)
    return result


def get_predictive_ranges(
    symbol: str = "BTC/USDT", timeframe: str = "1d", length: int = 200, mult: float = 6.0,
) -> dict:
    """Predictive Ranges [LuxAlgo] (CC BY-NC-SA 4.0), ported from the
    original Pine Script. An ATR-based step function: a central "average"
    line that only jumps when price strays further than `atr * mult` from
    it, plus two bands above/below (R1/R2 resistance, S1/S2 support) sized
    from half the ATR at the last jump.

    APPROXIMATION CAVEAT: this indicator is recursive -- every bar's
    result depends on the previous bar's, all the way back to the first
    candle it's given. Pine computes it over a chart's FULL history;
    we're limited to BingX's max of 1000 daily candles (~2.7 years) per
    request. The ATR itself needs `length` (default 200) bars just to
    warm up, leaving ~800 bars of real step behavior to converge --
    plenty to be directionally useful, but the exact price levels may
    not match TradingView bit-for-bit. Sanity-check against your own
    chart before trusting it for anything precise.
    """
    candles = get_ohlcv(symbol, timeframe=timeframe, limit=1000)
    closes = [c[4] for c in candles]
    atr_series = _rma(_true_range(candles), length)

    avg = closes[0]
    hold_atr = 0.0
    for i in range(1, len(candles)):
        atr = (atr_series[i] or 0.0) * mult  # nz(ta.atr(length)) -- 0 during warmup
        src = closes[i]
        prev_avg = avg
        if src - avg > atr:
            avg = avg + atr
        elif avg - src > atr:
            avg = avg - atr
        if avg != prev_avg:
            hold_atr = atr / 2

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "length": length,
        "mult": mult,
        "current_price": round(closes[-1], 2),
        "resistance_2": round(avg + hold_atr * 2, 2),
        "resistance_1": round(avg + hold_atr, 2),
        "average": round(avg, 2),
        "support_1": round(avg - hold_atr, 2),
        "support_2": round(avg - hold_atr * 2, 2),
    }
