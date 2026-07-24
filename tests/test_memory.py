"""
Tests for src/memory.py. No network, no LLM: market_data and state are
monkeypatched with fixed values/fake snapshots.
"""
from datetime import datetime, timedelta, timezone

import pytest

from src import memory, market_data
from src import state as state_module


def _snapshot(days_ago: int, **fields) -> dict:
    created_at = (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()
    return {"created_at": created_at, **fields}


@pytest.fixture(autouse=True)
def fixed_current(monkeypatch):
    """Pin every 'current' value memory.py fetches, so tests only vary
    the historical snapshots."""
    monkeypatch.setattr(market_data, "get_price", lambda symbol: {"last_price": 100.0})
    monkeypatch.setattr(market_data, "get_indicators", lambda symbol: {"rsi_14": 60.0})
    monkeypatch.setattr(market_data, "get_fear_greed_index", lambda: {"value": 70})
    monkeypatch.setattr(market_data, "get_cycle_metrics", lambda symbol: {"mayer_multiple": 1.5})
    monkeypatch.setattr(market_data, "get_btc_dominance", lambda: {"btc_dominance_pct": 55.0})


def test_no_snapshots_returns_sin_dato_everywhere(monkeypatch):
    monkeypatch.setattr(state_module, "get_snapshots", lambda limit=60: [])
    result = memory.get_market_memory()
    for window in result["windows"].values():
        for metric in window.values():
            assert metric["trend"] == "sin_dato"
            assert metric["value_then"] is None


def test_price_trend_labeled_subiendo_above_threshold(monkeypatch):
    # price was 90 a day ago -> current 100 is +11.1%, above the 3% threshold
    monkeypatch.setattr(state_module, "get_snapshots",
                         lambda limit=60: [_snapshot(1, price=90.0)])
    result = memory.get_market_memory()
    price_1d = result["windows"]["1d"]["price"]
    assert price_1d["trend"] == "subiendo"
    assert price_1d["value_then"] == 90.0
    assert price_1d["delta"] == 10.0


def test_price_trend_labeled_estable_within_threshold(monkeypatch):
    # 100 -> 101 is +1%, below the 3% threshold
    monkeypatch.setattr(state_module, "get_snapshots",
                         lambda limit=60: [_snapshot(1, price=101.0)])
    result = memory.get_market_memory()
    assert result["windows"]["1d"]["price"]["trend"] == "estable"


def test_rsi_uses_absolute_delta_not_percent(monkeypatch):
    # current rsi14=60. 60-46=14 > 5-point threshold -> subiendo, even
    # though 14/46 as a percent would be a different (larger) number --
    # RSI is compared on its own 0-100 points scale, not as a percent.
    monkeypatch.setattr(state_module, "get_snapshots",
                         lambda limit=60: [_snapshot(7, rsi14=46.0)])
    result = memory.get_market_memory()
    rsi_7d = result["windows"]["7d"]["rsi14"]
    assert rsi_7d["delta"] == 14.0
    assert rsi_7d["trend"] == "subiendo"


def test_bajando_label_for_negative_change(monkeypatch):
    monkeypatch.setattr(state_module, "get_snapshots",
                         lambda limit=60: [_snapshot(30, fear_greed_value=90)])
    result = memory.get_market_memory()
    # current fear_greed=70, was 90 -> delta -20, below -10 threshold
    assert result["windows"]["30d"]["fear_greed"]["trend"] == "bajando"


def test_picks_closest_snapshot_within_tolerance(monkeypatch):
    # Two candidates near the 7-day mark: 6 days ago and 9 days ago.
    # 6 is within the +-1 day tolerance of 7, 9 is not -- must pick the 6.
    monkeypatch.setattr(state_module, "get_snapshots", lambda limit=60: [
        _snapshot(6, price=95.0),
        _snapshot(9, price=50.0),
    ])
    result = memory.get_market_memory()
    assert result["windows"]["7d"]["price"]["value_then"] == 95.0


def test_no_snapshot_within_tolerance_is_sin_dato(monkeypatch):
    # Only a snapshot from 20 days ago -- too far from both 7d (tolerance
    # +-1) and 30d (tolerance +-1) windows.
    monkeypatch.setattr(state_module, "get_snapshots",
                         lambda limit=60: [_snapshot(20, price=80.0)])
    result = memory.get_market_memory()
    assert result["windows"]["7d"]["price"]["trend"] == "sin_dato"
    assert result["windows"]["30d"]["price"]["trend"] == "sin_dato"
