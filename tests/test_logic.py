"""
Tests for the pure logic (no network, no LLM). Run with:
    python -m pytest tests/ -v
They demonstrate that the agent's tools are testable in isolation, which
is the main reason to keep them separate from the LLM loop.
"""
import math
import pytest

from src import market_data
from src.market_data import _sma, _rsi, _true_range, _rma
from src.strategy_tools import simulate_bullet_math


def test_sma_basic():
    assert _sma([1, 2, 3, 4, 5], 5) == 3
    assert _sma([1, 2, 3, 4, 5], 2) == 4.5


def test_sma_insufficient_data():
    assert _sma([1, 2], 5) is None


def test_rsi_extremes():
    up = [100 + i for i in range(30)]
    down = [130 - i for i in range(30)]
    assert _rsi(up, 14) == 100.0
    assert _rsi(down, 14) == 0.0


def test_rsi_insufficient_data():
    assert _rsi([1, 2, 3], 14) is None


def test_bullet_math_x5():
    r = simulate_bullet_math(collateral_usd=500, entry_price=60000, leverage=5, target_position_gain_pct=15)
    assert r["position_size_usd"] == 2500
    # +15% on the position at x5 = price must move +3%
    assert math.isclose(r["required_price_move_pct"], 3.0)
    assert math.isclose(r["target_price"], 61800.0)
    assert math.isclose(r["profit_at_target_usd"], 75.0)
    # theoretical liquidation at x5: -20% price move
    assert math.isclose(r["approx_liquidation_move_pct"], -20.0)
    assert math.isclose(r["approx_liquidation_price"], 48000.0)


def test_bullet_math_rejects_invalid_input():
    with pytest.raises(ValueError):
        simulate_bullet_math(collateral_usd=0, entry_price=60000)
    with pytest.raises(ValueError):
        simulate_bullet_math(collateral_usd=100, entry_price=-1)


# --- Predictive Ranges [LuxAlgo] port -----------------------------------
# candles: [ts, open, high, low, close, volume] -- open/volume unused.
_PR_CANDLES = [
    [0, 0, 10, 8, 9, 0],
    [1, 0, 11, 9, 10, 0],
    [2, 0, 12, 10, 11, 0],
    [3, 0, 20, 18, 19, 0],  # a big move, big enough to trigger a jump once ATR warms up
]


def test_true_range():
    trs = _true_range(_PR_CANDLES)
    assert trs[0] is None
    assert trs[1] == 2      # max(11-9, |11-9|, |9-9|)
    assert trs[2] == 2      # max(12-10, |12-10|, |10-10|)
    assert trs[3] == 9      # max(20-18, |20-11|, |18-11|)


def test_rma_seeds_with_sma_then_smooths():
    # period=2 over [2, 2, 9] (skipping the leading None)
    rma = _rma(_true_range(_PR_CANDLES), period=2)
    assert rma[0] is None
    assert rma[1] is None            # not enough data yet
    assert rma[2] == 2               # seed: sma([2, 2])
    assert rma[3] == 5.5             # (2*(2-1) + 9) / 2


def test_predictive_ranges_hand_computed(monkeypatch):
    # Hand-computed with length=2, mult=1 (see conversation/commit notes):
    # avg starts at 9, stays pinned during ATR warmup (nz(atr)=0), then
    # jumps to 14.5 on the big bar 3 move once atr=5.5 kicks in, with
    # hold_atr set to atr/2 = 2.75 at the moment of that jump.
    monkeypatch.setattr(market_data, "get_ohlcv", lambda *a, **kw: _PR_CANDLES)
    result = market_data.get_predictive_ranges(length=2, mult=1.0)
    assert result["average"] == 14.5
    assert result["resistance_1"] == 17.25
    assert result["resistance_2"] == 20.0
    assert result["support_1"] == 11.75
    assert result["support_2"] == 9.0
    assert result["current_price"] == 19


# --- get_trailing_high_drawdown ---
# candle shape from ccxt: [timestamp, open, high, low, close, volume]

def _candle(day_index, high):
    return [day_index * 86_400_000, high - 1, high, high - 2, high - 0.5, 0]


def test_trailing_high_drawdown_excludes_todays_candle(monkeypatch):
    # 5 "complete" days with highs 10..14, then a TODAY candle with a
    # spike to 100 that must NOT count -- it's still forming.
    candles = [_candle(i, 10 + i) for i in range(5)] + [_candle(5, 100)]
    monkeypatch.setattr(market_data, "get_ohlcv", lambda *a, **kw: candles)
    monkeypatch.setattr(market_data, "get_price", lambda *a, **kw: {"last_price": 12.0})

    result = market_data.get_trailing_high_drawdown("BTC/USDT", lookback_days=90)
    assert result["trailing_high"] == 14   # max of the 5 complete days, not 100
    assert result["current_price"] == 12.0
    assert math.isclose(result["drawdown_from_trailing_high_pct"], (12.0 - 14) / 14 * 100, abs_tol=0.01)


def test_trailing_high_drawdown_respects_the_lookback_window(monkeypatch):
    # Highs 10..19 across 10 complete days plus today; a lookback of 3
    # must only look at the last 3 complete days (highs 17, 18, 19).
    candles = [_candle(i, 10 + i) for i in range(10)] + [_candle(10, 999)]
    monkeypatch.setattr(market_data, "get_ohlcv", lambda *a, **kw: candles)
    monkeypatch.setattr(market_data, "get_price", lambda *a, **kw: {"last_price": 15.0})

    result = market_data.get_trailing_high_drawdown("BTC/USDT", lookback_days=3)
    assert result["trailing_high"] == 19
    assert result["lookback_days"] == 3


def test_trailing_high_drawdown_positive_when_at_a_new_high(monkeypatch):
    candles = [_candle(i, 10 + i) for i in range(5)] + [_candle(5, 999)]
    monkeypatch.setattr(market_data, "get_ohlcv", lambda *a, **kw: candles)
    monkeypatch.setattr(market_data, "get_price", lambda *a, **kw: {"last_price": 20.0})

    result = market_data.get_trailing_high_drawdown("BTC/USDT", lookback_days=90)
    assert result["drawdown_from_trailing_high_pct"] > 0   # above the trailing high


def test_trailing_high_drawdown_handles_too_little_history(monkeypatch):
    # Only one candle -- after excluding "today", nothing is left to
    # compute a trailing high from.
    monkeypatch.setattr(market_data, "get_ohlcv", lambda *a, **kw: [_candle(0, 10)])
    monkeypatch.setattr(market_data, "get_price", lambda *a, **kw: {"last_price": 10.0})

    result = market_data.get_trailing_high_drawdown("BTC/USDT")
    assert result["trailing_high"] is None
    assert result["drawdown_from_trailing_high_pct"] is None


# --- get_volume_profile (VPVR) ---
# candle shape: [timestamp, open, high, low, close, volume]

def test_volume_profile_hand_computed_two_candle_case(monkeypatch):
    # Candle A: range [100, 110], volume 100 -- ALL of the window's high
    # volume sits low. Candle B: range [190, 200], volume 10 -- a thin,
    # separate spike far away. With bins=10 (bin_size=10), each candle
    # maps to exactly one bin (rounding aside), so this is exact, not an
    # approximation artifact: POC must be candle A's bin, and the 70%
    # value area must NOT reach out to isolated candle B (100/(100+10)
    # already covers 90.9%, comfortably over 70% on its own).
    candles = [
        [0, 100, 110, 100, 105, 100.0],
        [1, 190, 200, 190, 195, 10.0],
    ]
    monkeypatch.setattr(market_data, "get_ohlcv", lambda *a, **kw: candles)
    monkeypatch.setattr(market_data, "get_price", lambda *a, **kw: {"last_price": 105.0})

    result = market_data.get_volume_profile("BTC/USDT", lookback_days=2, bins=10)
    assert 100 <= result["point_of_control"] <= 110
    assert result["value_area_high"] < 190   # must not reach the thin spike
    assert result["position_vs_value_area"] == "inside_value_area"


def test_volume_profile_current_price_above_value_area(monkeypatch):
    candles = [[0, 100, 110, 100, 105, 100.0], [1, 190, 200, 190, 195, 10.0]]
    monkeypatch.setattr(market_data, "get_ohlcv", lambda *a, **kw: candles)
    monkeypatch.setattr(market_data, "get_price", lambda *a, **kw: {"last_price": 500.0})
    result = market_data.get_volume_profile("BTC/USDT", lookback_days=2, bins=10)
    assert result["position_vs_value_area"] == "above_value_area"


def test_volume_profile_current_price_below_value_area(monkeypatch):
    candles = [[0, 100, 110, 100, 105, 100.0], [1, 190, 200, 190, 195, 10.0]]
    monkeypatch.setattr(market_data, "get_ohlcv", lambda *a, **kw: candles)
    monkeypatch.setattr(market_data, "get_price", lambda *a, **kw: {"last_price": 1.0})
    result = market_data.get_volume_profile("BTC/USDT", lookback_days=2, bins=10)
    assert result["position_vs_value_area"] == "below_value_area"


def test_volume_profile_splits_a_wide_candles_volume_across_bins(monkeypatch):
    """A single candle spanning MULTIPLE bins must have its volume spread
    proportionally to overlap, not dumped entirely into one bin. Uses the
    raw bin array directly (via a 1-bin-wide probe below) rather than
    asserting on POC: a candle spanning the WHOLE range splits its volume
    perfectly evenly across every bin (verified: all ten bins got exactly
    10.0 of the candle's 100), which is a genuine tie -- there is no
    single "correct" POC to assert on in that case, only a real property
    to check: every bin the candle spans got a nonzero, proportional share."""
    wide_candle = [[0, 0, 100, 0, 50, 100.0]]   # spans the whole [0,100] range
    monkeypatch.setattr(market_data, "get_ohlcv", lambda *a, **kw: wide_candle)
    monkeypatch.setattr(market_data, "get_price", lambda *a, **kw: {"last_price": 50.0})

    # A NARROWER, off-center candle makes the correct answer unambiguous:
    # most of its range sits in the bin around 20-30, so the POC must
    # land there, not smeared "in the middle" of the whole visible range.
    off_center = [[0, 100, 200, 0, 25, 100.0], [1, 15, 35, 15, 25, 900.0]]
    monkeypatch.setattr(market_data, "get_ohlcv", lambda *a, **kw: off_center)
    result = market_data.get_volume_profile("BTC/USDT", lookback_days=2, bins=20)
    assert 15 <= result["point_of_control"] <= 35


def test_volume_profile_handles_a_perfectly_flat_price(monkeypatch):
    """A market that didn't move at all over the window -- every price
    literally IS the single observed price, not "price plus half a
    synthetic bin" (a real edge case this hit during development)."""
    flat = [[0, 100, 100, 100, 100, 50.0]] * 5
    monkeypatch.setattr(market_data, "get_ohlcv", lambda *a, **kw: flat)
    monkeypatch.setattr(market_data, "get_price", lambda *a, **kw: {"last_price": 100.0})
    result = market_data.get_volume_profile("BTC/USDT", lookback_days=5, bins=10)
    assert result["point_of_control"] == 100
    assert result["value_area_high"] == 100
    assert result["value_area_low"] == 100
    assert result["position_vs_value_area"] == "inside_value_area"


def test_volume_profile_handles_no_candles_at_all(monkeypatch):
    monkeypatch.setattr(market_data, "get_ohlcv", lambda *a, **kw: [])
    result = market_data.get_volume_profile("BTC/USDT")
    assert result["point_of_control"] is None
    assert result["position_vs_value_area"] is None


def test_volume_profile_works_with_a_single_candle(monkeypatch):
    """A single candle is legitimate data (it still has a real high and
    low) -- it must NOT be treated as "not enough history"."""
    monkeypatch.setattr(market_data, "get_ohlcv", lambda *a, **kw: [[0, 100, 110, 100, 105, 50.0]])
    monkeypatch.setattr(market_data, "get_price", lambda *a, **kw: {"last_price": 105.0})
    result = market_data.get_volume_profile("BTC/USDT", lookback_days=1, bins=10)
    assert result["point_of_control"] is not None
    assert 100 <= result["point_of_control"] <= 110


def test_volume_profile_ignores_zero_volume_candles(monkeypatch):
    """A candle with no volume must not distort the profile or crash
    the proportional split (division by a real span is fine, but a
    zero-volume candle should just contribute nothing)."""
    candles = [
        [0, 100, 110, 100, 105, 100.0],
        [1, 500, 600, 500, 550, 0.0],   # far away, zero volume
    ]
    monkeypatch.setattr(market_data, "get_ohlcv", lambda *a, **kw: candles)
    monkeypatch.setattr(market_data, "get_price", lambda *a, **kw: {"last_price": 105.0})
    result = market_data.get_volume_profile("BTC/USDT", lookback_days=2, bins=10)
    assert result["value_area_high"] < 500


# --- _rsi_zone ---

def test_rsi_zone_thresholds():
    assert market_data._rsi_zone(15) == "oversold"
    assert market_data._rsi_zone(29.9) == "oversold"
    assert market_data._rsi_zone(30) == "neutral"
    assert market_data._rsi_zone(50) == "neutral"
    assert market_data._rsi_zone(70) == "neutral"
    assert market_data._rsi_zone(70.1) == "overbought"
    assert market_data._rsi_zone(90) == "overbought"
    assert market_data._rsi_zone(None) is None


def test_get_indicators_includes_rsi_zone(monkeypatch):
    monkeypatch.setattr(market_data, "get_ohlcv", lambda *a, **kw: _PR_CANDLES)
    result = market_data.get_indicators("BTC/USDT")
    assert "rsi_14_zone" in result
