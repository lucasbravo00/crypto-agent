"""
bullets.py
-----------
State machine that tracks the leveraged futures positions ("bullets")
the user opens and closes MANUALLY on BingX, one at a time, up to a
cycle of 30 bullets.

Hard design rule (do not relax): this module NEVER opens or closes a
real order on any exchange. It only records what the user confirms
already happened and computes the live/realized P&L from that. Every
trading decision and execution stays manual, human-in-the-loop.

Bullet lifecycle (state machine):

    open -> tracking -> closed_tp       (closed at the +15% target)
                     -> closed_manual   (closed by hand for any reason)

- "open"     : just recorded, not yet checked against a live price.
- "tracking" : has been checked at least once (see check_bullet).
- "closed_*" : terminal; the slot is free again for the next bullet.

Persistence lives in state.py (state["bullets"]). The position math is
NOT re-derived here: we reuse strategy_tools.simulate_bullet_math() so
there is a single source of truth for target/liquidation prices.
"""
from __future__ import annotations

from datetime import datetime, timezone

from . import state as state_module
from .strategy_tools import simulate_bullet_math

# Business rules enforced in code, not left as conventions to remember.
MAX_BULLETS = 30
ACTIVE_STATUSES = ("open", "tracking")
CLOSED_STATUSES = ("closed_tp", "closed_manual")
VALID_OUTCOMES = ("tp", "manual")

# How close (in %) the price must get to the approximate liquidation
# price before check_bullet flags it as "near liquidation".
LIQUIDATION_PROXIMITY_PCT = 5.0


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _find_active_bullet(state: dict) -> dict | None:
    """Return the (mutable) active bullet record inside ``state``, or None.

    Returns the actual dict stored in state["bullets"], so mutating it
    and then calling state_module.save_state(state) persists the change.
    """
    for bullet in state["bullets"]:
        if bullet["status"] in ACTIVE_STATUSES:
            return bullet
    return None


def open_bullet(
    collateral_usd: float,
    entry_price: float,
    leverage: float = 5.0,
    target_position_gain_pct: float = 15.0,
) -> dict:
    """Record a newly opened leveraged bullet (does NOT touch the exchange).

    The "one bullet at a time" rule is enforced here in code: if a bullet
    is already open or tracking, this raises instead of silently allowing
    a second concurrent position. The 30-bullet cycle cap is enforced the
    same way.

    Args:
        collateral_usd: Margin posted as collateral, in USD.
        entry_price: Price at which the position was entered.
        leverage: Leverage multiplier (e.g. 5 for x5).
        target_position_gain_pct: Target gain ON THE POSITION (not on
            price), e.g. 15 for +15%.

    Returns:
        The persisted bullet record.

    Raises:
        RuntimeError: If a bullet is already active, or the 30-bullet
            cycle is already full.
    """
    state = state_module.load_state()

    active = _find_active_bullet(state)
    if active is not None:
        raise RuntimeError(
            f"A bullet is already active (id={active['id']}, "
            f"status={active['status']}). Close it before opening another "
            "(one bullet at a time)."
        )

    if len(state["bullets"]) >= MAX_BULLETS:
        raise RuntimeError(
            f"Cycle is full: {MAX_BULLETS} bullets already used."
        )

    # Single source of truth for the position math (no fees/funding).
    math = simulate_bullet_math(
        collateral_usd=collateral_usd,
        entry_price=entry_price,
        leverage=leverage,
        target_position_gain_pct=target_position_gain_pct,
    )

    bullet = {
        "id": len(state["bullets"]) + 1,
        "status": "open",
        "collateral_usd": collateral_usd,
        "entry_price": entry_price,
        "leverage": leverage,
        "target_position_gain_pct": target_position_gain_pct,
        "position_size_usd": math["position_size_usd"],
        "target_price": math["target_price"],
        "approx_liquidation_price": math["approx_liquidation_price"],
        "opened_at": _now_iso(),
        "closed_at": None,
        "closing_price": None,
        "outcome": None,
        "realized_pnl_usd": None,
        "notes": None,
    }

    state["bullets"].append(bullet)
    state_module.save_state(state)
    return bullet


def get_open_bullet(**_ignored) -> dict | None:
    """Return the currently active (open/tracking) bullet, or None."""
    state = state_module.load_state()
    return _find_active_bullet(state)


def check_bullet(current_price: float) -> dict:
    """Compute the live P&L of the active bullet at ``current_price``.

    Pure with respect to the network: the caller passes the current
    price (obtained separately via market_data.get_price()), so this
    function is testable in isolation. Its only side effect is the
    documented open -> tracking transition on the first check.

    Args:
        current_price: Latest market price of the asset.

    Returns:
        A dict with the live metrics: price move %, position gain %
        (price move x leverage), unrealized P&L in USD, whether the
        target is reached, and whether the price is near liquidation.

    Raises:
        RuntimeError: If there is no active bullet.
        ValueError: If current_price is not > 0.
    """
    if current_price <= 0:
        raise ValueError("current_price must be > 0")

    state = state_module.load_state()
    bullet = _find_active_bullet(state)
    if bullet is None:
        raise RuntimeError("No active bullet to check.")

    # Side effect: first check moves the bullet from open to tracking.
    if bullet["status"] == "open":
        bullet["status"] = "tracking"
        state_module.save_state(state)

    entry_price = bullet["entry_price"]
    leverage = bullet["leverage"]
    liq_price = bullet["approx_liquidation_price"]

    price_move_pct = (current_price - entry_price) / entry_price * 100
    position_gain_pct = price_move_pct * leverage
    unrealized_pnl_usd = bullet["collateral_usd"] * position_gain_pct / 100

    target_reached = current_price >= bullet["target_price"]

    # Distance to the (approximate) liquidation price, as a % above it.
    # For a long, liquidation is below entry; "near" means the price has
    # fallen to within LIQUIDATION_PROXIMITY_PCT of that level.
    pct_above_liquidation = (current_price - liq_price) / liq_price * 100
    near_liquidation = current_price <= liq_price * (1 + LIQUIDATION_PROXIMITY_PCT / 100)

    return {
        "id": bullet["id"],
        "status": bullet["status"],
        "entry_price": entry_price,
        "current_price": current_price,
        "leverage": leverage,
        "target_price": bullet["target_price"],
        "approx_liquidation_price": liq_price,
        "price_move_pct": round(price_move_pct, 2),
        "position_gain_pct": round(position_gain_pct, 2),
        "unrealized_pnl_usd": round(unrealized_pnl_usd, 2),
        "pct_above_liquidation": round(pct_above_liquidation, 2),
        "target_reached": target_reached,
        "near_liquidation": near_liquidation,
    }


def close_bullet(outcome: str, closing_price: float, notes: str | None = None) -> dict:
    """Record the manual close of the active bullet (does NOT touch the exchange).

    Computes realized P&L from entry vs. closing price and frees the
    "one bullet at a time" slot for the next bullet.

    Args:
        outcome: Either "tp" (closed at the +15% target) or "manual"
            (closed by hand for any other reason).
        closing_price: Price at which the position was actually closed.
        notes: Optional free-text note about the close.

    Returns:
        The updated (now terminal) bullet record.

    Raises:
        ValueError: If outcome is not "tp"/"manual" or closing_price<=0.
        RuntimeError: If there is no active bullet to close.
    """
    if outcome not in VALID_OUTCOMES:
        raise ValueError(
            f"outcome must be one of {VALID_OUTCOMES}, got {outcome!r}"
        )
    if closing_price <= 0:
        raise ValueError("closing_price must be > 0")

    state = state_module.load_state()
    bullet = _find_active_bullet(state)
    if bullet is None:
        raise RuntimeError("No active bullet to close.")

    entry_price = bullet["entry_price"]
    leverage = bullet["leverage"]
    price_move_pct = (closing_price - entry_price) / entry_price * 100
    position_gain_pct = price_move_pct * leverage
    realized_pnl_usd = bullet["collateral_usd"] * position_gain_pct / 100

    bullet["status"] = "closed_tp" if outcome == "tp" else "closed_manual"
    bullet["outcome"] = outcome
    bullet["closing_price"] = closing_price
    bullet["closed_at"] = _now_iso()
    bullet["realized_pnl_usd"] = round(realized_pnl_usd, 2)
    bullet["notes"] = notes

    state_module.save_state(state)
    return bullet


def get_cycle_summary(**_ignored) -> dict:
    """Aggregate view of the 30-bullet cycle.

    Returns:
        A dict with: bullets used/remaining over MAX_BULLETS, how many
        are closed, how many were target (tp) wins, the total realized
        P&L, and whether a bullet is currently open.
    """
    state = state_module.load_state()
    bullets = state["bullets"]

    closed = [b for b in bullets if b["status"] in CLOSED_STATUSES]
    tp_wins = [b for b in closed if b["status"] == "closed_tp"]
    total_realized_pnl = sum(b["realized_pnl_usd"] or 0 for b in closed)
    active = _find_active_bullet(state)

    return {
        "max_bullets": MAX_BULLETS,
        "bullets_used": len(bullets),
        "bullets_remaining": MAX_BULLETS - len(bullets),
        "closed": len(closed),
        "tp_wins": len(tp_wins),
        "total_realized_pnl_usd": round(total_realized_pnl, 2),
        "has_open_bullet": active is not None,
        "open_bullet_id": active["id"] if active else None,
    }
