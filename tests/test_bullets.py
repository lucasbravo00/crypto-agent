"""
Tests for the bullet state machine (src/bullets.py). No network, no LLM.
Run with:  python -m pytest tests/ -v

Every test is isolated from the user's real state: the autouse fixture (1)
monkeypatches state.STATE_PATH to a fresh temp file per test, and (2)
force-disables the Supabase AND BingX backends by unsetting their env
vars for the duration of the test, regardless of what's in the
developer's shell environment or .env. Without step (2), a developer
who has sourced their real .env could have tests silently write test
data into their production Supabase project, or hit the real BingX API.

STRATEGY UNDER TEST: bullets accumulate within a ROUND (at most one NEW
bullet per calendar day, previous ones stay open), the +15% target is
evaluated on the COMBINED position across every active bullet, and
hitting it closes ALL active bullets together, ending the round.
MAX_BULLETS_PER_ROUND (30) is a PER-ROUND cap, not a lifetime one: once
a round closes, the next one starts over at bullet_number 1 with a
fresh budget. See src/bullets.py's module docstring.
"""
import math
import pytest

from src import state as state_module
from src import bullets
from src.strategy_tools import simulate_bullet_math


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    """Point state persistence at a per-test temp file and force the local
    JSON backend (never Supabase or BingX), no matter the ambient environment."""
    temp_state = tmp_path / "portfolio_state.json"
    monkeypatch.setattr(state_module, "STATE_PATH", str(temp_state))
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_KEY", raising=False)
    monkeypatch.delenv("BINGX_API_KEY", raising=False)
    monkeypatch.delenv("BINGX_API_SECRET", raising=False)
    monkeypatch.delenv("BINGX_AUTO_TRADE_ENABLED", raising=False)
    yield


def _insert_bullet(collateral_usd, entry_price, leverage=5.0, target_position_gain_pct=15.0,
                    opened_at=None, status="open", round_number=None, bullet_number=None):
    """Insert a bullet directly via state.py, bypassing open_bullet()'s
    guardrails (one-NEW-bullet-per-day, round cap). Used to set up
    multi-bullet test scenarios deterministically, regardless of what
    real calendar day the suite happens to run on.

    round_number/bullet_number auto-compute via the SAME logic
    open_bullet() uses (bullets._next_bullet_position) unless given
    explicitly -- most tests don't need to think about round bookkeeping
    at all; only tests specifically about round transitions override them.
    """
    math_result = simulate_bullet_math(collateral_usd, entry_price, leverage, target_position_gain_pct)
    all_bullets = state_module.get_bullets()
    active = bullets._find_active_bullets(all_bullets)
    auto_round, auto_number = bullets._next_bullet_position(all_bullets, active)
    round_number = auto_round if round_number is None else round_number
    bullet_number = auto_number if bullet_number is None else bullet_number
    fields = {
        "status": status,
        "round_number": round_number,
        "collateral_usd": collateral_usd,
        "entry_price": entry_price,
        "leverage": leverage,
        "target_position_gain_pct": target_position_gain_pct,
        "position_size_usd": math_result["position_size_usd"],
        "target_price": math_result["target_price"],
        "approx_liquidation_price": math_result["approx_liquidation_price"],
        "opened_at": opened_at or bullets._now_iso(),
        "closed_at": None,
        "closing_price": None,
        "outcome": None,
        "realized_pnl_usd": None,
        "notes": None,
    }
    return state_module.insert_bullet(fields, bullet_number)


YESTERDAY = "2020-01-01T00:00:00+00:00"  # any date guaranteed != "today"


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
    assert b["round_number"] == 1
    assert b["bullet_number"] == 1
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


def test_open_second_bullet_same_day_raises():
    bullets.open_bullet(collateral_usd=500, entry_price=60000)
    with pytest.raises(RuntimeError):
        bullets.open_bullet(collateral_usd=300, entry_price=59000)


def test_open_bullet_different_day_accumulates():
    """A bullet opened on a previous day does NOT block opening today's --
    bullets accumulate instead of requiring the previous one closed."""
    _insert_bullet(500, 60000, opened_at=YESTERDAY)
    # Today's open must succeed (no RuntimeError) and both stay active,
    # same round, bullet_number incrementing.
    b2 = bullets.open_bullet(collateral_usd=300, entry_price=61000)
    assert b2["round_number"] == 1
    assert b2["bullet_number"] == 2
    active = bullets.get_active_bullets()
    assert len(active) == 2


def test_open_bullet_round_full_raises():
    """30 ACTIVE bullets in the current round blocks a 31st -- this is a
    PER-ROUND cap, not a lifetime one (see next test)."""
    for _ in range(bullets.MAX_BULLETS_PER_ROUND):
        _insert_bullet(100, 60000, opened_at=YESTERDAY, status="open")
    with pytest.raises(RuntimeError):
        bullets.open_bullet(collateral_usd=100, entry_price=60000)


def test_round_number_and_bullet_number_reset_after_round_closes():
    """The core behavior this design exists for: MAX_BULLETS_PER_ROUND is
    NOT a lifetime cap. Once a round closes, the next one gets a fresh
    30-bullet budget and bullet_number starts over at 1."""
    b1 = _insert_bullet(500, 60000, leverage=5, opened_at=YESTERDAY)
    assert b1["round_number"] == 1
    assert b1["bullet_number"] == 1

    b2 = bullets.open_bullet(collateral_usd=300, entry_price=61000, leverage=5)
    assert b2["round_number"] == 1
    assert b2["bullet_number"] == 2

    bullets.close_all_active_bullets("tp", closing_price=62000)
    assert bullets.get_active_bullets() == []

    # The one-bullet-per-day guardrail is untouched by the round reset --
    # today's slot was already used by b2.
    with pytest.raises(RuntimeError):
        bullets.open_bullet(collateral_usd=100, entry_price=62000)

    # A backdated round-2 bullet must start over at bullet_number 1, in a
    # NEW round_number -- not bullet_number 3 of round 1.
    b3 = _insert_bullet(1000, 63000, leverage=5, opened_at=YESTERDAY)
    assert b3["round_number"] == 2
    assert b3["bullet_number"] == 1


def test_open_bullet_persists_across_reload():
    bullets.open_bullet(collateral_usd=500, entry_price=60000)
    # Fresh read from disk (new load) still sees the active bullet.
    assert len(bullets.get_active_bullets()) == 1


# --- auto-sized collateral (collateral_usd=None) ---

def test_open_bullet_auto_sizes_from_bingx_balance_at_round_start(monkeypatch):
    from src import bingx_client
    monkeypatch.setattr(bingx_client, "is_enabled", lambda: True)
    monkeypatch.setattr(bingx_client, "get_balance",
                         lambda **kw: {"asset": "VST", "free": 30000, "used": 0, "total": 30000})

    b = bullets.open_bullet(collateral_usd=None, entry_price=60000)
    # 30000 / 30 = 1000
    assert math.isclose(b["collateral_usd"], 1000.0)


def test_open_bullet_reuses_round_size_for_later_bullets_same_round(monkeypatch):
    from src import bingx_client
    monkeypatch.setattr(bingx_client, "is_enabled", lambda: True)
    monkeypatch.setattr(bingx_client, "get_balance",
                         lambda **kw: {"asset": "VST", "free": 30000, "used": 0, "total": 30000})

    _insert_bullet(1000, 60000, opened_at=YESTERDAY)  # round already has a size: 1000

    # Change the mocked balance -- if this were consulted again, size would
    # differ. It must NOT be consulted mid-round: existing bullets win.
    monkeypatch.setattr(bingx_client, "get_balance",
                         lambda **kw: {"asset": "VST", "free": 90000, "used": 0, "total": 90000})

    b = bullets.open_bullet(collateral_usd=None, entry_price=61000)
    assert math.isclose(b["collateral_usd"], 1000.0)


def test_open_bullet_explicit_collateral_overrides_auto_sizing(monkeypatch):
    from src import bingx_client
    monkeypatch.setattr(bingx_client, "is_enabled", lambda: True)
    monkeypatch.setattr(bingx_client, "get_balance",
                         lambda **kw: {"asset": "VST", "free": 30000, "used": 0, "total": 30000})

    b = bullets.open_bullet(collateral_usd=250, entry_price=60000)
    assert math.isclose(b["collateral_usd"], 250.0)


def test_open_bullet_auto_sizing_raises_if_bingx_not_configured():
    with pytest.raises(RuntimeError):
        bullets.open_bullet(collateral_usd=None, entry_price=60000)


# --- check_bullets ---

def test_check_bullets_transitions_open_to_tracking():
    bullets.open_bullet(collateral_usd=500, entry_price=60000)
    assert bullets.get_active_bullets()[0]["status"] == "open"
    result = bullets.check_bullets(current_price=60000)
    assert result["bullets"][0]["status"] == "tracking"
    # Transition is persisted, not just returned.
    assert bullets.get_active_bullets()[0]["status"] == "tracking"


def test_check_bullets_single_bullet_matches_original_math():
    bullets.open_bullet(collateral_usd=500, entry_price=60000, leverage=5, target_position_gain_pct=15)
    result = bullets.check_bullets(current_price=61800)  # exactly the target
    assert result["target_reached"] is True
    b = result["bullets"][0]
    assert b["bullet_number"] == 1
    # +3% price move at x5 = +15% on the position.
    assert math.isclose(b["price_move_pct"], 3.0)
    assert math.isclose(b["position_gain_pct"], 15.0)
    assert math.isclose(b["unrealized_pnl_usd"], 75.0)
    assert math.isclose(result["combined_unrealized_pnl_usd"], 75.0)
    assert math.isclose(result["combined_position_gain_pct"], 15.0)
    assert result["near_liquidation_any"] is False


def test_check_bullets_combines_pnl_across_multiple_bullets():
    # Two bullets, same entry/leverage/target, backdated so both coexist.
    _insert_bullet(500, 60000, leverage=5, target_position_gain_pct=15, opened_at=YESTERDAY)
    bullets.open_bullet(collateral_usd=300, entry_price=60000, leverage=5, target_position_gain_pct=15)

    result = bullets.check_bullets(current_price=61800)  # +3% price move for both
    # Bullet A: 500 * 0.15 = 75. Bullet B: 300 * 0.15 = 45. Combined = 120.
    assert math.isclose(result["combined_unrealized_pnl_usd"], 120.0)
    assert math.isclose(result["combined_collateral_usd"], 800.0)
    # 120 / 800 * 100 = 15.0 -- combined gain hits the +15% target even
    # though it's computed across two positions, not one.
    assert math.isclose(result["combined_position_gain_pct"], 15.0)
    assert result["target_reached"] is True
    assert len(result["bullets"]) == 2


def test_check_bullets_combined_target_not_reached_by_a_single_strong_bullet():
    """One bullet alone hitting +15% must NOT trip the combined target if
    another active bullet is dragging the combined average down."""
    _insert_bullet(500, 60000, leverage=5, target_position_gain_pct=15, opened_at=YESTERDAY)  # will gain
    bullets.open_bullet(collateral_usd=500, entry_price=70000, leverage=5, target_position_gain_pct=15)  # will lose

    result = bullets.check_bullets(current_price=61800)
    # Bullet A: +75. Bullet B: price_move = (61800-70000)/70000*100 = -11.71%,
    # position_gain = -58.57%, pnl = 500 * -0.5857 = -292.86.
    assert result["combined_unrealized_pnl_usd"] < 0
    assert result["target_reached"] is False


def test_check_bullets_detects_near_liquidation():
    bullets.open_bullet(collateral_usd=500, entry_price=60000, leverage=5)
    # Liquidation approx at 48000; threshold = 48000 * 1.05 = 50400.
    result = bullets.check_bullets(current_price=50000)
    assert result["near_liquidation_any"] is True
    assert result["bullets"][0]["near_liquidation"] is True


def test_check_bullets_without_active_raises():
    with pytest.raises(RuntimeError):
        bullets.check_bullets(current_price=60000)


def test_check_bullets_rejects_bad_price():
    bullets.open_bullet(collateral_usd=500, entry_price=60000)
    with pytest.raises(ValueError):
        bullets.check_bullets(current_price=0)


# --- close_all_active_bullets ---

def test_close_all_active_bullets_computes_realized_pnl_and_frees_day():
    bullets.open_bullet(collateral_usd=500, entry_price=60000, leverage=5)
    closed = bullets.close_all_active_bullets("tp", closing_price=61800)
    assert len(closed) == 1
    assert closed[0]["status"] == "closed_tp"
    assert closed[0]["outcome"] == "tp"
    # +3% move at x5 = +15% => 500 * 0.15 = 75 USD.
    assert math.isclose(closed[0]["realized_pnl_usd"], 75.0)
    assert bullets.get_active_bullets() == []


def test_close_all_active_bullets_closes_everyone_together():
    _insert_bullet(500, 60000, leverage=5, opened_at=YESTERDAY)
    bullets.open_bullet(collateral_usd=300, entry_price=61000, leverage=5)

    closed = bullets.close_all_active_bullets("tp", closing_price=62000, notes="combined round done")
    assert len(closed) == 2
    assert all(b["status"] == "closed_tp" for b in closed)
    assert all(b["notes"] == "combined round done" for b in closed)
    assert bullets.get_active_bullets() == []
    # Each bullet's realized P&L is computed from ITS OWN entry price.
    a, b = closed
    assert not math.isclose(a["realized_pnl_usd"], b["realized_pnl_usd"])


def test_close_all_active_bullets_manual_negative_pnl():
    bullets.open_bullet(collateral_usd=500, entry_price=60000, leverage=5)
    closed = bullets.close_all_active_bullets("manual", closing_price=59400)  # -1% move
    assert closed[0]["status"] == "closed_manual"
    # -1% at x5 = -5% => -25 USD.
    assert math.isclose(closed[0]["realized_pnl_usd"], -25.0)


def test_close_all_active_bullets_invalid_outcome_raises():
    bullets.open_bullet(collateral_usd=500, entry_price=60000)
    with pytest.raises(ValueError):
        bullets.close_all_active_bullets("whatever", closing_price=60000)


def test_close_all_active_bullets_without_active_raises():
    with pytest.raises(RuntimeError):
        bullets.close_all_active_bullets("tp", closing_price=60000)


# --- get_cycle_summary ---

def test_cycle_summary_aggregates_across_rounds():
    # Round 1: two bullets that close TOGETHER (as they always do in
    # reality), backdated so they don't consume "today"'s slot.
    b1 = _insert_bullet(500, 60000, leverage=5, opened_at=YESTERDAY, round_number=1, bullet_number=1)
    b2 = _insert_bullet(500, 60000, leverage=5, opened_at=YESTERDAY, round_number=1, bullet_number=2)
    for b, pnl in ((b1, 75.0), (b2, 50.0)):
        state_module.update_bullet(b["id"], {
            "status": "closed_tp", "outcome": "tp", "closing_price": 61800,
            "closed_at": bullets._now_iso(), "realized_pnl_usd": pnl,
        })

    # Round 2: opened today via the real open_bullet() call -- must
    # auto-detect this is a NEW round (no active bullets exist) and start
    # over at bullet_number 1.
    b3 = bullets.open_bullet(collateral_usd=100, entry_price=58000, leverage=5)
    assert b3["round_number"] == 2
    assert b3["bullet_number"] == 1

    summary = bullets.get_cycle_summary()
    assert summary["max_bullets_per_round"] == 30
    assert summary["round_number"] == 2
    assert summary["bullets_used_this_round"] == 1
    assert summary["bullets_remaining_this_round"] == 29
    assert summary["rounds_completed"] == 1
    assert summary["tp_rounds"] == 1
    assert math.isclose(summary["total_realized_pnl_usd"], 125.0)  # 75 + 50, lifetime across rounds
    assert summary["active_bullets_count"] == 1
    assert summary["active_bullet_numbers"] == [1]


# --- auto_trade() safety gate and orchestration ---

def test_auto_trade_noop_when_flag_not_set(monkeypatch):
    monkeypatch.delenv("BINGX_AUTO_TRADE_ENABLED", raising=False)
    result = bullets.auto_trade()
    assert result["traded"] is False
    assert "BINGX_AUTO_TRADE_ENABLED" in result["reason"]


def test_auto_trade_noop_when_flag_false(monkeypatch):
    monkeypatch.setenv("BINGX_AUTO_TRADE_ENABLED", "false")
    result = bullets.auto_trade()
    assert result["traded"] is False


def test_auto_trade_noop_when_bingx_not_configured(monkeypatch):
    monkeypatch.setenv("BINGX_AUTO_TRADE_ENABLED", "true")
    from src import bingx_client
    monkeypatch.setattr(bingx_client, "is_enabled", lambda: False)
    result = bullets.auto_trade()
    assert result["traded"] is False
    assert "BingX not configured" in result["reason"]


def test_auto_trade_opens_when_flag_enabled_and_nothing_open_today(monkeypatch):
    monkeypatch.setenv("BINGX_AUTO_TRADE_ENABLED", "true")
    from src import bingx_client
    monkeypatch.setattr(bingx_client, "is_enabled", lambda: True)
    monkeypatch.setattr(bingx_client, "get_balance",
                         lambda **kw: {"asset": "VST", "free": 30000, "used": 0, "total": 30000})
    monkeypatch.setattr(bingx_client, "open_long_position",
                         lambda collateral_usd, leverage=5.0, test=False, **kw:
                         {"id": "fake-order-1", "collateral_usd": collateral_usd, "leverage": leverage, "test": test})

    result = bullets.auto_trade()
    assert result["traded"] is True
    assert result["action"] == "open"
    assert result["order"]["leverage"] == bullets.AUTO_TRADE_LEVERAGE  # forced to 5x, not whatever the account has


def test_auto_trade_closes_when_combined_target_reached(monkeypatch):
    monkeypatch.setenv("BINGX_AUTO_TRADE_ENABLED", "true")
    from src import bingx_client, market_data
    monkeypatch.setattr(bingx_client, "is_enabled", lambda: True)
    monkeypatch.setattr(bingx_client, "close_all_long_positions",
                         lambda test=False, **kw: {"id": "fake-close-1", "test": test})
    monkeypatch.setattr(market_data, "get_price", lambda symbol: {"last_price": 61800})

    # One active bullet at +15% exactly (60000 -> 61800 at x5).
    _insert_bullet(500, 60000, leverage=5, opened_at=YESTERDAY)

    result = bullets.auto_trade()
    assert result["traded"] is True
    assert result["action"] == "close"


def test_auto_trade_does_nothing_when_already_opened_today_and_target_not_reached(monkeypatch):
    monkeypatch.setenv("BINGX_AUTO_TRADE_ENABLED", "true")
    from src import bingx_client, market_data
    monkeypatch.setattr(bingx_client, "is_enabled", lambda: True)
    monkeypatch.setattr(market_data, "get_price", lambda symbol: {"last_price": 60100})  # barely moved

    bullets.open_bullet(collateral_usd=500, entry_price=60000, leverage=5)  # opened today

    result = bullets.auto_trade()
    assert result["traded"] is False
    assert result["reason"] == "nothing to do this cycle"
