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


# --- Exchange fees --------------------------------------------------------
# Real fees, read from BingX's own trade record. Spot fees on this
# account come in BTC (the asset bought), which reduces net BTC received
# rather than costing extra USD -- folded into an EFFECTIVE price instead
# of added to amount_usd, so quantity (amount_usd / price) stays correct.

def test_fee_adjusted_no_fee_returns_raw_values():
    trade = {"price": 60000.0, "cost": 600.0, "amount": 0.01}
    amount_usd, price, fee_usd = dca._fee_adjusted(trade)
    assert amount_usd == 600.0
    assert price == 60000.0
    assert fee_usd == 0.0


def test_fee_adjusted_btc_fee_raises_effective_price():
    # Bought 0.01 BTC for $600, fee 0.0001 BTC (1% of the trade, exaggerated
    # for a clean hand-computation) -> net 0.0099 BTC actually received.
    trade = {"price": 60000.0, "cost": 600.0, "amount": 0.01,
              "fee": {"currency": "BTC", "cost": 0.0001}}
    amount_usd, price, fee_usd = dca._fee_adjusted(trade)
    assert amount_usd == 600.0                     # no extra USD spent
    assert price == pytest.approx(600.0 / 0.0099)   # same $ / fewer net BTC
    assert fee_usd == pytest.approx(0.0001 * 60000.0)
    # The resulting quantity (amount_usd / price) must equal the NET BTC
    # actually received, not the gross fill amount.
    assert amount_usd / price == pytest.approx(0.0099)


def test_fee_adjusted_quote_currency_fee_adds_to_cost():
    trade = {"price": 60000.0, "cost": 600.0, "amount": 0.01,
              "fee": {"currency": "USDT", "cost": 0.6}}
    amount_usd, price, fee_usd = dca._fee_adjusted(trade)
    assert amount_usd == 600.6      # real extra USD spent
    assert price == 60000.0         # fill price unaffected
    assert fee_usd == 0.6


def test_sync_imports_fee_adjusted_purchase(monkeypatch):
    monkeypatch.setattr(bingx_client, "is_enabled", lambda: True)
    monkeypatch.setattr(bingx_client, "get_real_spot_trades", lambda **_: [
        {"id": "t1", "side": "buy", "price": 60000.0, "cost": 600.0, "amount": 0.01,
         "datetime": "2026-07-15T12:00:00Z", "fee": {"currency": "BTC", "cost": 0.0001}},
    ])
    result = dca.sync_with_bingx()
    purchase = result["imported"][0]
    assert purchase["amount_usd"] == 600.0
    assert purchase["price"] == pytest.approx(600.0 / 0.0099)
    assert purchase["fee_usd"] == pytest.approx(6.0)
