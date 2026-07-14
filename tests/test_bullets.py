"""
Tests for the bullet state machine (src/bullets.py). No network, no LLM.
Run with:  python -m pytest tests/ -v

Every test is isolated from the user's real state: the autouse fixture (1)
monkeypatches state.STATE_PATH to a fresh temp file per test, and (2)
force-disables the Supabase backend by unsetting SUPABASE_URL/SUPABASE_KEY
for the duration of the test, regardless of what's in the developer's
shell environment or .env. Without step (2), a developer who has sourced
their real .env could have tests silently write test data into their
production Supabase project instead of the local temp file.
"""
import math
import pytest

from src import state as state_module
from src import bullets


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    """Point state persistence at a per-test temp file and force the local
    JSON backend (never Supabase), no matter the ambient environment."""
    temp_state = tmp_path / "portfolio_state.json"
    monkeypatch.setattr(state_module, "STATE_PATH", str(temp_state))
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_KEY", raising=False)
    yield


# --- Regression test for the shared-mutable-default bug in state.py ---

def test_default_state_does_not_share_nested_lists():
    """Two fresh states must not alias their nested lists (the bug that
    silently leaked entries between supposedly independent instances)."""
    a = state_module._default_state()
    b = state_module._default_state()
    assert a["bullets"] is not b["bullets"]
    assert a["dca_purchases"] is not b["dca_purchases"]
    assert a["notes"] is not b["notes"]
    # Mutating one must not touch the other.
    a["bullets"].append("x")
    assert b["bullets"] == []


# --- open_bullet ---

def test_open_bullet_sets_expected_fields():
    b = bullets.open_bullet(collateral_usd=500, entry_price=60000, leverage=5, target_position_gain_pct=15)
    assert b["id"] == 1
    assert b["status"] == "open"
    assert b["collateral_usd"] == 500
    assert b["entry_price"] == 60000
    assert b["leverage"] == 5
    # Reused from strategy_tools.simulate_bullet_math:
    assert math.isclose(b["target_price"], 61800.0)
    assert math.isclose(b["approx_liquidation_price"], 48000.0)
    assert math.isclose(b["position_size_usd"], 2500.0)
    assert b["outcome"] is None
    assert b["realized_pnl_usd"] is None


def test_open_second_bullet_while_active_raises():
    bullets.open_bullet(collateral_usd=500, entry_price=60000)
    with pytest.raises(RuntimeError):
        bullets.open_bullet(collateral_usd=300, entry_price=59000)


def test_open_bullet_persists_across_reload():
    bullets.open_bullet(collateral_usd=500, entry_price=60000)
    # Fresh read from disk (new load) still sees the active bullet.
    assert bullets.get_open_bullet() is not None


# --- check_bullet ---

def test_check_bullet_transitions_open_to_tracking():
    bullets.open_bullet(collateral_usd=500, entry_price=60000)
    assert bullets.get_open_bullet()["status"] == "open"
    result = bullets.check_bullet(current_price=60000)
    assert result["status"] == "tracking"
    # Transition is persisted, not just returned.
    assert bullets.get_open_bullet()["status"] == "tracking"


def test_check_bullet_detects_target_reached():
    bullets.open_bullet(collateral_usd=500, entry_price=60000, leverage=5, target_position_gain_pct=15)
    result = bullets.check_bullet(current_price=61800)  # exactly the target
    assert result["target_reached"] is True
    # +3% price move at x5 = +15% on the position.
    assert math.isclose(result["price_move_pct"], 3.0)
    assert math.isclose(result["position_gain_pct"], 15.0)
    assert math.isclose(result["unrealized_pnl_usd"], 75.0)
    assert result["near_liquidation"] is False


def test_check_bullet_detects_near_liquidation():
    bullets.open_bullet(collateral_usd=500, entry_price=60000, leverage=5)
    # Liquidation approx at 48000; threshold = 48000 * 1.05 = 50400.
    assert bullets.check_bullet(current_price=50000)["near_liquidation"] is True
    # Just above the threshold is NOT near.
    bullets.close_bullet("manual", 50000)  # free the slot to reopen cleanly
    bullets.open_bullet(collateral_usd=500, entry_price=60000, leverage=5)
    assert bullets.check_bullet(current_price=51000)["near_liquidation"] is False


def test_check_bullet_without_active_raises():
    with pytest.raises(RuntimeError):
        bullets.check_bullet(current_price=60000)


def test_check_bullet_rejects_bad_price():
    bullets.open_bullet(collateral_usd=500, entry_price=60000)
    with pytest.raises(ValueError):
        bullets.check_bullet(current_price=0)


# --- close_bullet ---

def test_close_bullet_computes_realized_pnl_and_frees_slot():
    bullets.open_bullet(collateral_usd=500, entry_price=60000, leverage=5)
    closed = bullets.close_bullet("tp", closing_price=61800)
    assert closed["status"] == "closed_tp"
    assert closed["outcome"] == "tp"
    # +3% move at x5 = +15% => 500 * 0.15 = 75 USD.
    assert math.isclose(closed["realized_pnl_usd"], 75.0)
    # Slot is free again: a new bullet can be opened.
    assert bullets.get_open_bullet() is None
    b2 = bullets.open_bullet(collateral_usd=200, entry_price=59000)
    assert b2["id"] == 2


def test_close_bullet_manual_negative_pnl():
    bullets.open_bullet(collateral_usd=500, entry_price=60000, leverage=5)
    closed = bullets.close_bullet("manual", closing_price=59400)  # -1% move
    assert closed["status"] == "closed_manual"
    # -1% at x5 = -5% => -25 USD.
    assert math.isclose(closed["realized_pnl_usd"], -25.0)


def test_close_bullet_invalid_outcome_raises():
    bullets.open_bullet(collateral_usd=500, entry_price=60000)
    with pytest.raises(ValueError):
        bullets.close_bullet("whatever", closing_price=60000)


def test_close_bullet_without_active_raises():
    with pytest.raises(RuntimeError):
        bullets.close_bullet("tp", closing_price=60000)


# --- get_cycle_summary ---

def test_cycle_summary_aggregates_across_bullets():
    # Bullet 1: TP win (+75).
    bullets.open_bullet(collateral_usd=500, entry_price=60000, leverage=5)
    bullets.close_bullet("tp", closing_price=61800)
    # Bullet 2: manual loss (-25).
    bullets.open_bullet(collateral_usd=500, entry_price=60000, leverage=5)
    bullets.close_bullet("manual", closing_price=59400)
    # Bullet 3: still open.
    bullets.open_bullet(collateral_usd=100, entry_price=58000, leverage=5)

    summary = bullets.get_cycle_summary()
    assert summary["max_bullets"] == 30
    assert summary["bullets_used"] == 3
    assert summary["bullets_remaining"] == 27
    assert summary["closed"] == 2
    assert summary["tp_wins"] == 1
    assert math.isclose(summary["total_realized_pnl_usd"], 50.0)  # 75 - 25
    assert summary["has_open_bullet"] is True
    assert summary["open_bullet_id"] == 3
