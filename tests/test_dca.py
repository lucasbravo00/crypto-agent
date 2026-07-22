"""
Tests for DCA sync against real BingX spot trades (src/dca.py). No
network, no LLM -- bingx_client is monkeypatched, never called for real.
"""
import pytest

from src import state as state_module
from src import dca
from src import bingx_client


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    temp_state = tmp_path / "portfolio_state.json"
    monkeypatch.setattr(state_module, "STATE_PATH", str(temp_state))
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_KEY", raising=False)
    monkeypatch.delenv("BINGX_API_KEY", raising=False)
    monkeypatch.delenv("BINGX_API_SECRET", raising=False)
    yield


def test_sync_not_synced_when_bingx_disabled(monkeypatch):
    monkeypatch.setattr(bingx_client, "is_enabled", lambda: False)
    result = dca.sync_with_bingx()
    assert result["synced"] is False
    assert result["imported"] == []


def test_sync_imports_buy_fills(monkeypatch):
    monkeypatch.setattr(bingx_client, "is_enabled", lambda: True)
    monkeypatch.setattr(bingx_client, "get_real_spot_trades", lambda **_: [
        {"id": "t1", "side": "buy", "cost": 50.0, "price": 61719.46, "datetime": "2026-07-15T12:00:00Z"},
        {"id": "t2", "side": "buy", "cost": 30.0, "price": 62000.0, "datetime": "2026-07-16T12:00:00Z"},
    ])
    result = dca.sync_with_bingx()
    assert result["synced"] is True
    assert len(result["imported"]) == 2
    purchases = state_module.get_dca_purchases()
    assert len(purchases) == 2
    assert {p["bingx_trade_id"] for p in purchases} == {"t1", "t2"}


def test_sync_skips_sell_fills(monkeypatch):
    monkeypatch.setattr(bingx_client, "is_enabled", lambda: True)
    monkeypatch.setattr(bingx_client, "get_real_spot_trades", lambda **_: [
        {"id": "t1", "side": "sell", "cost": 50.0, "price": 61719.46, "datetime": "2026-07-15T12:00:00Z"},
    ])
    result = dca.sync_with_bingx()
    assert result["imported"] == []
    assert state_module.get_dca_purchases() == []


def test_sync_is_idempotent(monkeypatch):
    monkeypatch.setattr(bingx_client, "is_enabled", lambda: True)
    trades = [
        {"id": "t1", "side": "buy", "cost": 50.0, "price": 61719.46, "datetime": "2026-07-15T12:00:00Z"},
    ]
    monkeypatch.setattr(bingx_client, "get_real_spot_trades", lambda **_: trades)

    first = dca.sync_with_bingx()
    second = dca.sync_with_bingx()

    assert len(first["imported"]) == 1
    assert len(second["imported"]) == 0
    assert len(state_module.get_dca_purchases()) == 1
