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
