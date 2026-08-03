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
from src import bingx_client
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
                    opened_at=None, status="open", round_number=None, bullet_number=None,
                    entry_fee_usd=None):
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
        "entry_fee_usd": entry_fee_usd,
        "exit_fee_usd": None,
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


# --- check_bullets: real cross-margin liquidation price ---
# Motivated by a real gap found 2026-07-31: with 3 open bullets, the
# isolated-margin approx sat around $52k while BingX's real cross-margin
# figure was $414.70, because free account balance was backing the
# position. Before this, near_liquidation_any (which feeds a live alert
# in main.py and the daily report via get_daily_alert()) was judged
# against the isolated number alone -- guaranteed to fire falsely long
# before the account was ever in real danger.

def test_check_bullets_prefers_the_real_liquidation_price_when_given():
    bullets.open_bullet(collateral_usd=500, entry_price=60000, leverage=5)
    # Isolated approx would be 48000 (near_liquidation at price<=50400),
    # but the real cross-margin price is far below -- not actually near.
    result = bullets.check_bullets(current_price=50000, real_liquidation_price=400.0)
    assert result["near_liquidation_any"] is False
    assert result["liquidation_price"] == 400.0
    b = result["bullets"][0]
    assert b["liquidation_price_is_real"] is True
    assert b["approx_liquidation_price"] == 48000.0   # isolated value still reported
    assert math.isclose(b["pct_above_liquidation"], (50000 - 400.0) / 400.0 * 100)


def test_check_bullets_real_liquidation_price_can_still_trigger_near_liquidation():
    bullets.open_bullet(collateral_usd=500, entry_price=60000, leverage=5)
    result = bullets.check_bullets(current_price=50000, real_liquidation_price=49000.0)
    assert result["near_liquidation_any"] is True
    assert result["liquidation_price"] == 49000.0
    assert result["bullets"][0]["liquidation_price_is_real"] is True


def test_check_bullets_falls_back_to_isolated_approx_without_a_real_price():
    bullets.open_bullet(collateral_usd=500, entry_price=60000, leverage=5)
    result = bullets.check_bullets(current_price=50000)   # no real_liquidation_price
    assert result["liquidation_price"] is None
    assert result["near_liquidation_any"] is True   # same as the pre-existing isolated behavior
    assert result["bullets"][0]["liquidation_price_is_real"] is False


@pytest.mark.parametrize("bad_value", [None, 0, -100.0])
def test_check_bullets_ignores_a_non_positive_real_liquidation_price(bad_value):
    bullets.open_bullet(collateral_usd=500, entry_price=60000, leverage=5)
    result = bullets.check_bullets(current_price=50000, real_liquidation_price=bad_value)
    assert result["liquidation_price"] is None
    assert result["bullets"][0]["liquidation_price_is_real"] is False


def test_check_bullets_real_liquidation_price_applies_to_every_bullet_alike():
    """Cross margin protects/threatens the whole round together -- every
    active bullet must share the same real liquidation reading, unlike
    the isolated approximation which varies per bullet's own entry."""
    _insert_bullet(500, 55000, leverage=5, target_position_gain_pct=15, opened_at=YESTERDAY)
    bullets.open_bullet(collateral_usd=500, entry_price=65000, leverage=5, target_position_gain_pct=15)

    result = bullets.check_bullets(current_price=60000, real_liquidation_price=1000.0)
    assert {b["liquidation_price_used"] for b in result["bullets"]} == {1000.0}
    assert all(b["liquidation_price_is_real"] for b in result["bullets"])
    # The isolated approximations, by contrast, differ per bullet.
    assert len({b["approx_liquidation_price"] for b in result["bullets"]}) == 2


def test_check_bullets_includes_bullet_id_for_alert_messages():
    """Regression guard: main.py's near-liquidation alert message reads
    b['id'] -- a key check_bullets() didn't return before, which would
    have raised KeyError the first time a real alert ever fired."""
    bullets.open_bullet(collateral_usd=500, entry_price=60000, leverage=5)
    result = bullets.check_bullets(current_price=50000)
    assert isinstance(result["bullets"][0]["id"], int)


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


def test_auto_trade_opens_when_rsi_signal_fired(monkeypatch):
    monkeypatch.setenv("BINGX_AUTO_TRADE_ENABLED", "true")
    from src import bingx_client, market_data
    monkeypatch.setattr(bingx_client, "is_enabled", lambda: True)
    monkeypatch.setattr(bingx_client, "get_balance",
                         lambda **kw: {"asset": "VST", "free": 30000, "used": 0, "total": 30000})
    monkeypatch.setattr(bingx_client, "open_long_position",
                         lambda collateral_usd, leverage=5.0, test=False, **kw:
                         {"id": "fake-order-1", "collateral_usd": collateral_usd, "leverage": leverage, "test": test})
    monkeypatch.setattr(market_data, "get_intraday_rsi", lambda *a, **kw: bullets.RSI_ENTRY_THRESHOLD - 1)

    result = bullets.auto_trade()
    assert result["traded"] is True
    assert result["action"] == "open"
    assert result["triggered_by"] == "rsi_oversold"
    assert result["order"]["leverage"] == bullets.AUTO_TRADE_LEVERAGE  # forced to 5x, not whatever the account has


def test_auto_trade_waits_when_rsi_not_oversold_and_before_cutoff(monkeypatch):
    monkeypatch.setenv("BINGX_AUTO_TRADE_ENABLED", "true")
    from src import bingx_client, market_data
    monkeypatch.setattr(bingx_client, "is_enabled", lambda: True)
    monkeypatch.setattr(market_data, "get_intraday_rsi", lambda *a, **kw: bullets.RSI_ENTRY_THRESHOLD + 20)
    monkeypatch.setattr(bullets, "_past_rsi_fallback_cutoff", lambda *a, **kw: False)

    result = bullets.auto_trade()
    assert result["traded"] is False
    assert f"RSI({bullets.RSI_ENTRY_PERIOD})<{bullets.RSI_ENTRY_THRESHOLD}" in result["reason"]


def test_auto_trade_opens_at_eod_fallback_even_without_rsi_signal(monkeypatch):
    monkeypatch.setenv("BINGX_AUTO_TRADE_ENABLED", "true")
    from src import bingx_client, market_data
    monkeypatch.setattr(bingx_client, "is_enabled", lambda: True)
    monkeypatch.setattr(bingx_client, "get_balance",
                         lambda **kw: {"asset": "VST", "free": 30000, "used": 0, "total": 30000})
    monkeypatch.setattr(bingx_client, "open_long_position",
                         lambda collateral_usd, leverage=5.0, test=False, **kw:
                         {"id": "fake-order-1", "collateral_usd": collateral_usd, "leverage": leverage, "test": test})
    monkeypatch.setattr(market_data, "get_intraday_rsi", lambda *a, **kw: bullets.RSI_ENTRY_THRESHOLD + 20)
    monkeypatch.setattr(bullets, "_past_rsi_fallback_cutoff", lambda *a, **kw: True)

    result = bullets.auto_trade()
    assert result["traded"] is True
    assert result["triggered_by"] == "eod_fallback"


def test_past_rsi_fallback_cutoff():
    from datetime import datetime, timezone
    before = datetime(2026, 1, 1, 23, 44, tzinfo=timezone.utc)
    at_cutoff = datetime(2026, 1, 1, 23, 45, tzinfo=timezone.utc)
    after = datetime(2026, 1, 1, 23, 59, tzinfo=timezone.utc)
    assert bullets._past_rsi_fallback_cutoff(before) is False
    assert bullets._past_rsi_fallback_cutoff(at_cutoff) is True
    assert bullets._past_rsi_fallback_cutoff(after) is True


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


# --- sync_with_bingx() ------------------------------------------------
# REGRESSION (bug found 2026-07-24): a SELL fill's order id was never
# persisted anywhere, so re-running sync_with_bingx() kept "seeing" the
# same old, already-processed sell every time -- and matched it against
# whatever bullets happened to be active AT THAT LATER SYNC, wrongly
# closing brand new bullets using the old sell's stale price/time. Fixed
# by stamping bingx_close_order_id on every bullet a sell closes, and
# including it (not just bingx_order_id) in the "already seen" set.

def _fake_trade(order_id, side, price, cost, when):
    return {"order": order_id, "side": side, "price": price, "cost": cost, "datetime": when}


def test_sync_with_bingx_opens_bullets_from_buy_fills(monkeypatch):
    from src import bingx_client
    monkeypatch.setattr(bingx_client, "is_enabled", lambda: True)
    monkeypatch.setattr(bingx_client, "get_open_positions", lambda **kw: [])
    monkeypatch.setattr(bingx_client, "get_trade_history", lambda **kw: [
        _fake_trade("o1", "buy", 60000, 500, "2026-07-20T10:00:00Z"),
    ])

    result = bullets.sync_with_bingx()
    assert len(result["opened"]) == 1
    assert bullets.get_active_bullets()[0]["bingx_order_id"] == "o1"


def test_sync_with_bingx_does_not_reclose_bullets_opened_after_an_old_sell(monkeypatch):
    from src import bingx_client
    monkeypatch.setattr(bingx_client, "is_enabled", lambda: True)
    monkeypatch.setattr(bingx_client, "get_open_positions",
                         lambda **kw: [{"leverage": 5.0}])

    # Round 1: a buy, then a sell that closes it -- both already known
    # (as if a previous sync already processed them).
    trades = [
        _fake_trade("o1", "buy", 60000, 500, "2026-07-20T10:00:00Z"),
        _fake_trade("o2", "sell", 60600, 500, "2026-07-20T11:00:00Z"),
    ]
    monkeypatch.setattr(bingx_client, "get_trade_history", lambda **kw: list(trades))
    bullets.sync_with_bingx()
    assert bullets.get_active_bullets() == []

    # Round 2: a NEW buy shows up days later. get_trade_history() still
    # returns the FULL history (o1, o2 included) -- exactly what BingX's
    # real endpoint does, and what re-triggered the bug.
    trades.append(_fake_trade("o3", "buy", 61000, 500, "2026-07-24T10:00:00Z"))
    bullets.sync_with_bingx()

    active = bullets.get_active_bullets()
    assert len(active) == 1
    assert active[0]["bingx_order_id"] == "o3"

    # The critical assertion: syncing AGAIN (e.g. the next 15-min cycle)
    # must NOT re-process the old sell (o2) and close the bullet o3 just
    # opened.
    bullets.sync_with_bingx()
    active = bullets.get_active_bullets()
    assert len(active) == 1
    assert active[0]["bingx_order_id"] == "o3"


# --- reconcile_with_bingx() ---------------------------------------------
# Defense in depth added after the sync bug above: even if sync_with_bingx()
# runs "successfully", verify what we THINK is open still matches BingX's
# real reported position.

def test_reconcile_not_checked_when_bingx_disabled(monkeypatch):
    from src import bingx_client
    monkeypatch.setattr(bingx_client, "is_enabled", lambda: False)
    result = bullets.reconcile_with_bingx()
    assert result["checked"] is False


def test_reconcile_ok_when_amounts_match(monkeypatch):
    from src import bingx_client
    monkeypatch.setattr(bingx_client, "is_enabled", lambda: True)
    # 500 collateral * 5x / 60000 entry = 0.041666... BTC
    _insert_bullet(500, 60000, leverage=5, opened_at=YESTERDAY)
    monkeypatch.setattr(bingx_client, "get_open_positions",
                         lambda **kw: [{"contracts": 500 * 5 / 60000}])

    result = bullets.reconcile_with_bingx()
    assert result["checked"] is True
    assert result["ok"] is True
    assert result["diff_btc"] < bullets.RECONCILE_TOLERANCE_BTC


def test_reconcile_flags_mismatch(monkeypatch):
    from src import bingx_client
    monkeypatch.setattr(bingx_client, "is_enabled", lambda: True)
    _insert_bullet(500, 60000, leverage=5, opened_at=YESTERDAY)
    # BingX reports nothing open at all -- exactly the bug scenario: our
    # state thinks a bullet is active but the sync history wrongly closed
    # (or never opened) the matching real position.
    monkeypatch.setattr(bingx_client, "get_open_positions", lambda **kw: [])

    result = bullets.reconcile_with_bingx()
    assert result["checked"] is True
    assert result["ok"] is False
    assert result["real_amount_btc"] == 0
    assert result["active_amount_btc"] > 0


def test_reconcile_ok_when_nothing_active_and_nothing_open(monkeypatch):
    from src import bingx_client
    monkeypatch.setattr(bingx_client, "is_enabled", lambda: True)
    monkeypatch.setattr(bingx_client, "get_open_positions", lambda **kw: [])

    result = bullets.reconcile_with_bingx()
    assert result["checked"] is True
    assert result["ok"] is True
    assert result["active_amount_btc"] == 0
    assert result["real_amount_btc"] == 0


# --- Exchange fees ------------------------------------------------------
# Real fees, read from BingX's own trade record, subtracted from P&L
# instead of pretending trading is fee-free.

def test_check_bullets_subtracts_entry_fee_from_unrealized_pnl():
    _insert_bullet(500, 60000, leverage=5, opened_at=YESTERDAY, entry_fee_usd=2.5)
    # Price unchanged -> 0 raw gain, so unrealized P&L should be exactly -2.5
    result = bullets.check_bullets(60000)
    assert result["bullets"][0]["unrealized_pnl_usd"] == -2.5


def test_check_bullets_treats_missing_entry_fee_as_zero():
    _insert_bullet(500, 60000, leverage=5, opened_at=YESTERDAY)  # entry_fee_usd=None
    result = bullets.check_bullets(60000)
    assert result["bullets"][0]["unrealized_pnl_usd"] == 0.0


def test_close_all_active_bullets_subtracts_entry_and_exit_fees():
    _insert_bullet(500, 60000, leverage=5, opened_at=YESTERDAY, entry_fee_usd=2.0)
    # Price unchanged -> 0 raw gain. exit_fee_usd_total=10 goes entirely
    # to this one bullet (it's the only one active).
    closed = bullets.close_all_active_bullets("manual", 60000, exit_fee_usd_total=10.0)
    assert closed[0]["realized_pnl_usd"] == -12.0
    assert closed[0]["exit_fee_usd"] == 10.0


def test_close_all_active_bullets_splits_exit_fee_by_collateral_weight():
    # Bullet A: 300 collateral (25% of the 1200 total) -> 25% of the exit fee.
    # Bullet B: 900 collateral (75%) -> 75% of the exit fee.
    b1 = _insert_bullet(300, 60000, leverage=5, opened_at=YESTERDAY)
    b2 = bullets.open_bullet(collateral_usd=900, entry_price=60000, leverage=5)
    closed = bullets.close_all_active_bullets("manual", 60000, exit_fee_usd_total=20.0)
    by_id = {c["id"]: c for c in closed}
    assert by_id[b1["id"]]["exit_fee_usd"] == 5.0
    assert by_id[b2["id"]]["exit_fee_usd"] == 15.0


def test_sync_with_bingx_reads_entry_fee_from_buy_trade(monkeypatch):
    from src import bingx_client
    monkeypatch.setattr(bingx_client, "is_enabled", lambda: True)
    monkeypatch.setattr(bingx_client, "get_open_positions", lambda **kw: [])
    monkeypatch.setattr(bingx_client, "get_trade_history", lambda **kw: [
        {"order": "o1", "side": "buy", "price": 60000, "cost": 500,
         "datetime": "2026-07-20T10:00:00Z", "fee": {"currency": "USDT", "cost": 0.25}},
    ])
    bullets.sync_with_bingx()
    active = bullets.get_active_bullets()
    assert active[0]["entry_fee_usd"] == 0.25


def test_sync_with_bingx_reads_exit_fee_from_sell_trade(monkeypatch):
    from src import bingx_client
    monkeypatch.setattr(bingx_client, "is_enabled", lambda: True)
    monkeypatch.setattr(bingx_client, "get_open_positions", lambda **kw: [{"leverage": 5.0}])
    monkeypatch.setattr(bingx_client, "get_trade_history", lambda **kw: [
        {"order": "o1", "side": "buy", "price": 60000, "cost": 500,
         "datetime": "2026-07-20T10:00:00Z", "fee": {"currency": "USDT", "cost": 0.25}},
        {"order": "o2", "side": "sell", "price": 60000, "cost": 500,
         "datetime": "2026-07-20T11:00:00Z", "fee": {"currency": "USDT", "cost": 0.25}},
    ])
    result = bullets.sync_with_bingx()
    closed = result["closed"][0]
    assert closed["exit_fee_usd"] == 0.25
    # 0 raw gain (same price) - 0.25 entry - 0.25 exit
    assert closed["realized_pnl_usd"] == -0.5


# --- get_daily_alert() ---------------------------------------------------
# Deterministic (no LLM) replacement for the old "portfolio manager"
# sub-agent, added after that sub-agent (on the local Ollama backend)
# fabricated a false alert from numbers it misread.

def _quiet_drawdown(monkeypatch):
    """Stub market_data.get_trailing_high_drawdown() with a "nothing to
    see here" result. get_daily_alert() calls it live over the network
    whenever any bullet is active, so every test that reaches that branch
    must patch it -- otherwise the test suite would make a real ccxt call."""
    from src import market_data
    monkeypatch.setattr(market_data, "get_trailing_high_drawdown", lambda *a, **k: {
        "trailing_high": None, "current_price": None,
        "drawdown_from_trailing_high_pct": None, "lookback_days": 90,
    })


def test_get_daily_alert_none_when_nothing_active():
    assert bullets.get_daily_alert() is None


def test_get_daily_alert_none_when_far_from_target(monkeypatch):
    from src import market_data
    _quiet_drawdown(monkeypatch)
    # Reproduces the exact real-world bug: combined gain -3.71% vs a 15%
    # target is nowhere close -- must NOT alert.
    _insert_bullet(500, 60000, leverage=5, opened_at=YESTERDAY)
    monkeypatch.setattr(market_data, "get_price", lambda symbol: {"last_price": 59777.4})
    assert bullets.get_daily_alert() is None


def test_get_daily_alert_fires_when_close_to_target(monkeypatch):
    from src import market_data
    _quiet_drawdown(monkeypatch)
    # +12% combined at 5x needs about +2.4% price move; target is 15%,
    # gap is 3 points -- right at the alert threshold.
    _insert_bullet(500, 60000, leverage=5, opened_at=YESTERDAY)
    monkeypatch.setattr(market_data, "get_price", lambda symbol: {"last_price": 61440.0})
    alert = bullets.get_daily_alert()
    assert alert is not None
    assert "objetivo" in alert


def test_get_daily_alert_fires_when_near_liquidation(monkeypatch):
    from src import market_data
    _quiet_drawdown(monkeypatch)
    _insert_bullet(500, 60000, leverage=5, opened_at=YESTERDAY)
    # approx_liquidation_price at x5 is 48000; within LIQUIDATION_PROXIMITY_PCT (5%) of it.
    monkeypatch.setattr(market_data, "get_price", lambda symbol: {"last_price": 49500.0})
    alert = bullets.get_daily_alert()
    assert alert is not None
    assert "liquidaci" in alert.lower()


def test_get_daily_alert_uses_the_real_liquidation_price_when_bingx_is_enabled(monkeypatch):
    """When BingX is configured, get_bullet_status() must fetch the real
    cross-margin liquidation price and pass it through -- otherwise a
    price that's actually nowhere near danger (isolated approx ~48000,
    real ~400) would still fire the alert."""
    from src import market_data, bingx_client
    _quiet_drawdown(monkeypatch)
    monkeypatch.setenv("BINGX_API_KEY", "k")
    monkeypatch.setenv("BINGX_API_SECRET", "s")
    monkeypatch.setattr(bingx_client, "get_liquidation_price", lambda **k: 400.0)
    _insert_bullet(500, 60000, leverage=5, opened_at=YESTERDAY)
    # Isolated approx (48000) would flag this as near-liquidation; the
    # real price (400) means it's nowhere close.
    monkeypatch.setattr(market_data, "get_price", lambda symbol: {"last_price": 49500.0})
    assert bullets.get_daily_alert() is None


def test_get_daily_alert_falls_back_when_the_real_liquidation_lookup_fails(monkeypatch):
    from src import market_data, bingx_client
    _quiet_drawdown(monkeypatch)
    monkeypatch.setenv("BINGX_API_KEY", "k")
    monkeypatch.setenv("BINGX_API_SECRET", "s")

    def boom(**kwargs):
        raise RuntimeError("exchange unreachable")

    monkeypatch.setattr(bingx_client, "get_liquidation_price", boom)
    _insert_bullet(500, 60000, leverage=5, opened_at=YESTERDAY)
    monkeypatch.setattr(market_data, "get_price", lambda symbol: {"last_price": 49500.0})
    alert = bullets.get_daily_alert()
    assert alert is not None
    assert "aislado" in alert


# --- get_daily_alert(): round-depth/drawdown context (README Roadmap item 4) ---

def test_get_daily_alert_adds_context_line_on_a_deep_correction(monkeypatch):
    from src import market_data
    _insert_bullet(500, 60000, leverage=5, opened_at=YESTERDAY)
    # Far from target and from liquidation -- would otherwise be a quiet
    # day -- but a -8% drawdown crosses the -5% context threshold.
    monkeypatch.setattr(market_data, "get_price", lambda symbol: {"last_price": 59900.0})
    monkeypatch.setattr(market_data, "get_trailing_high_drawdown", lambda *a, **k: {
        "trailing_high": 70000.0, "current_price": 64400.0,
        "drawdown_from_trailing_high_pct": -8.0, "lookback_days": 90,
    })
    alert = bullets.get_daily_alert()
    assert alert is not None
    assert "Contexto" in alert
    assert "90" in alert           # lookback days
    assert "1/30" in alert or "1/" in alert   # round depth (1 active bullet)
    assert "recomendación" in alert.lower() or "recomendaci" in alert.lower()


def test_get_daily_alert_stays_quiet_on_a_shallow_pullback(monkeypatch):
    from src import market_data
    _insert_bullet(500, 60000, leverage=5, opened_at=YESTERDAY)
    monkeypatch.setattr(market_data, "get_price", lambda symbol: {"last_price": 59900.0})
    # -2% is below the -5% bar -- must stay quiet, same as a flat market.
    monkeypatch.setattr(market_data, "get_trailing_high_drawdown", lambda *a, **k: {
        "trailing_high": 61200.0, "current_price": 60000.0,
        "drawdown_from_trailing_high_pct": -2.0, "lookback_days": 90,
    })
    assert bullets.get_daily_alert() is None


def test_get_daily_alert_survives_a_failed_drawdown_lookup(monkeypatch):
    """Best-effort: a broken market_data call must not take down the rest
    of the alert (or the report)."""
    from src import market_data
    _insert_bullet(500, 60000, leverage=5, opened_at=YESTERDAY)
    monkeypatch.setattr(market_data, "get_price", lambda symbol: {"last_price": 61440.0})

    def boom(*a, **k):
        raise RuntimeError("exchange unreachable")

    monkeypatch.setattr(market_data, "get_trailing_high_drawdown", boom)
    alert = bullets.get_daily_alert()
    assert alert is not None            # the target-proximity line still fires
    assert "objetivo" in alert
    assert "Contexto" not in alert


def test_get_daily_alert_can_report_both_conditions_together(monkeypatch):
    # Isolate get_daily_alert()'s own condition/concatenation logic from
    # the underlying P&L math (already covered by the tests above) by
    # mocking get_bullet_status directly with both conditions true.
    _quiet_drawdown(monkeypatch)
    monkeypatch.setattr(bullets, "get_bullet_status", lambda symbol="BTC/USDT": {
        "live_status": {
            "combined_position_gain_pct": 13.0,
            "target_position_gain_pct": 15.0,
            "near_liquidation_any": True,
            "bullets": [{"bullet_number": 1, "near_liquidation": True}],
        },
    })
    alert = bullets.get_daily_alert()
    assert alert is not None
    assert "objetivo" in alert
    assert "liquidaci" in alert.lower()


# ---------------------------------------------------------------------
# BingX cross-margin liquidation price (src/bingx_client.py)
# ---------------------------------------------------------------------

def test_liquidation_price_reads_the_raw_info_field(monkeypatch):
    """ccxt's unified `liquidationPrice` comes back None for cross-margin
    positions on BingX (confirmed 2026-07-31 against the live demo
    account); the real number only exists in the raw `info` payload."""
    monkeypatch.setattr(bingx_client, "get_open_positions", lambda **k: [
        {"liquidationPrice": None, "info": {"liquidationPrice": 414.7}},
    ])
    assert bingx_client.get_liquidation_price() == pytest.approx(414.7)


def test_liquidation_price_accepts_a_string_payload(monkeypatch):
    # Most BingX raw numeric fields arrive as strings; this one happened to
    # come back as a float, so don't rely on either.
    monkeypatch.setattr(bingx_client, "get_open_positions", lambda **k: [
        {"info": {"liquidationPrice": "51234.5"}},
    ])
    assert bingx_client.get_liquidation_price() == pytest.approx(51234.5)


def test_liquidation_price_is_none_when_flat(monkeypatch):
    monkeypatch.setattr(bingx_client, "get_open_positions", lambda **k: [])
    assert bingx_client.get_liquidation_price() is None


@pytest.mark.parametrize("raw", [None, "", 0, "0", "abc", -5])
def test_liquidation_price_rejects_non_prices(monkeypatch, raw):
    """BingX reports 0 for "not applicable". A long's liquidation is always
    a positive price, and a junk value must not reach the dashboard as a
    bar endpoint."""
    monkeypatch.setattr(bingx_client, "get_open_positions", lambda **k: [
        {"info": {"liquidationPrice": raw}},
    ])
    assert bingx_client.get_liquidation_price() is None


def test_liquidation_price_skips_a_junk_position_and_uses_the_next(monkeypatch):
    monkeypatch.setattr(bingx_client, "get_open_positions", lambda **k: [
        {"info": {"liquidationPrice": 0}},
        {"info": {"liquidationPrice": 414.7}},
    ])
    assert bingx_client.get_liquidation_price() == pytest.approx(414.7)


def test_record_account_tick_sends_the_liquidation_price(monkeypatch):
    """Regression guard: the column is nullable, so a silently-dropped
    value would look exactly like "no open position" in the dashboard."""
    captured = {}

    class _Table:
        def insert(self, payload):
            captured.update(payload)
            return self

        def execute(self):
            return type("R", (), {"data": [dict(captured)]})()

    monkeypatch.setattr(state_module.db, "is_enabled", lambda: True)
    monkeypatch.setattr(state_module.db, "get_client",
                        lambda: type("C", (), {"table": lambda self, name: _Table()})())

    state_module.record_account_tick(137498.41, liquidation_price=414.7)
    assert captured == {"vst_total": 137498.41, "liquidation_price": 414.7}
