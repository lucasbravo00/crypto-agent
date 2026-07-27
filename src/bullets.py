"""
bullets.py
-----------
State machine that tracks the leveraged futures positions ("bullets")
the user opens and closes on BingX (manually, or via auto_trade() on the
Demo Trading / VST account -- see that function's docstring), organized
into ROUNDS.

STRATEGY (corrected twice already -- this is the confirmed real one):

- The user opens at most ONE NEW bullet per calendar day. Bullets
  ACCUMULATE within a round: previous ones stay open while a new one is
  added each day. Several can be "open"/"tracking" at the same time.
- The +15% target is evaluated on the COMBINED position across every
  currently active bullet (total unrealized P&L / total collateral),
  not per individual bullet.
- When the combined target is reached, ALL active bullets close TOGETHER,
  in one action -- that's the end of the round.
- MAX_BULLETS_PER_ROUND (30) is a PER-ROUND cap, not a lifetime one: once
  a round closes, the next one starts over from bullet 1 with a fresh
  30-bullet budget. The bot can run indefinitely, round after round, for
  as long as the bull market lasts.
- If a round reaches 30 active bullets without hitting the combined
  target, no MORE bullets open that round (today's daily slot is simply
  not used), but the 30 already open keep being tracked until the
  combined target is eventually reached, however long that takes.

Two different numbers per bullet -- see the note in state.py above
get_bullets(): "id" is globally unique and never resets (safe for
update_bullet()); "bullet_number" is the position WITHIN the round
(1..30, resets every round) and is display-only. "round_number" says
which round a bullet belongs to.

Bullet lifecycle (state machine, per bullet):

    open -> tracking -> closed_tp       (closed as part of a combined +15% close)
                     -> closed_manual   (closed by hand for any other reason)

- "open"     : just recorded, not yet checked against a live price.
- "tracking" : has been checked at least once (see check_bullets).
- "closed_*" : terminal.

Persistence goes through state.py's get_bullets()/insert_bullet()/
update_bullet(), which transparently route to Supabase or the local JSON
file depending on configuration -- this module doesn't know or care which.
The position math is NOT re-derived here: we reuse
strategy_tools.simulate_bullet_math() so there is a single source of
truth for each bullet's own target/liquidation prices.

Hard design rule for open_bullet()/close_all_active_bullets()/
sync_with_bingx(): these only RECORD what already happened (manually, or
read back from BingX). The only function that can place a real order is
auto_trade(), and only when explicitly opted into -- see its docstring.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

from . import state as state_module
from .strategy_tools import simulate_bullet_math

# Strategy-documented leverage for automated trading (see auto_trade()).
# Forced explicitly before every automated open, regardless of whatever
# leverage the BingX account happens to be set to manually.
AUTO_TRADE_LEVERAGE = 5.0

# Business rules enforced in code, not left as conventions to remember.
MAX_BULLETS_PER_ROUND = 30  # per round, NOT a lifetime total -- resets every round
DEFAULT_TARGET_GAIN_PCT = 15.0  # combined target for the currently-open round
ACTIVE_STATUSES = ("open", "tracking")
CLOSED_STATUSES = ("closed_tp", "closed_manual")
VALID_OUTCOMES = ("tp", "manual")

# How close (in %) the price must get to a bullet's OWN approximate
# liquidation price before it's flagged "near liquidation". Liquidation
# risk is still evaluated per bullet -- each has its own entry price and
# isolated margin, independent of how the combined round is doing.
LIQUIDATION_PROXIMITY_PCT = 5.0


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_date(iso_string: str):
    """Parse an ISO datetime string (however state.py/Supabase returned
    it) down to just its calendar date, for the one-bullet-per-day check."""
    return datetime.fromisoformat(iso_string.replace("Z", "+00:00")).date()


def _find_active_bullets(bullets: list[dict]) -> list[dict]:
    """Return every active (open or tracking) bullet in ``bullets``."""
    return [b for b in bullets if b["status"] in ACTIVE_STATUSES]


def _opened_today(bullets: list[dict]) -> bool:
    """Global across ALL rounds: the "one new bullet per calendar day"
    rule isn't reset by a round closing -- if a round closes and a new
    one starts on the same day, that day's slot is still used."""
    today = datetime.now(timezone.utc).date()
    return any(_parse_date(b["opened_at"]) == today for b in bullets if b.get("opened_at"))


def _next_bullet_position(bullets: list[dict], active: list[dict]) -> tuple[int, int]:
    """Determine (round_number, bullet_number) for a bullet being opened
    right now.

    - Mid-round (active bullets exist): same round_number as them,
      bullet_number = count of active bullets + 1.
    - Round start (no active bullets): a fresh round -- one past the
      highest round_number seen among ALL bullets ever (0 if this is the
      very first bullet), bullet_number = 1.
    """
    if active:
        return active[0].get("round_number", 1), len(active) + 1
    max_round = max((b.get("round_number") or 0 for b in bullets), default=0)
    return max_round + 1, 1


def _auto_collateral_usd(active: list[dict]) -> float:
    """Determine the bullet size for a new bullet when the caller didn't
    specify one explicitly.

    - Mid-round (bullets already active): reuse the SAME size as the
      round's existing bullets, so every bullet in a round is equal-sized.
    - Round start (no active bullets): read the REAL account balance from
      BingX (bingx_client.get_balance()) and split it evenly across the
      round's 30-bullet budget. This is what makes the sizing compound: a
      round that closes positive grows the account balance, so the next
      round's fresh division starts from a bigger number.
    """
    if active:
        return active[0]["collateral_usd"]

    from . import bingx_client  # local import: keeps this module network-free
    # except for this one documented case, same pattern as get_bullet_status.
    if not bingx_client.is_enabled():
        raise RuntimeError(
            "Cannot auto-size a bullet: BINGX_API_KEY/BINGX_API_SECRET "
            "aren't configured, and no explicit collateral_usd was given. "
            "Either set up BingX (see .env.example) or pass collateral_usd "
            "explicitly."
        )
    balance = bingx_client.get_balance()
    total = balance.get("total")
    if not total or total <= 0:
        raise RuntimeError(
            f"Cannot auto-size a bullet: BingX demo balance looks invalid ({balance})."
        )
    return total / MAX_BULLETS_PER_ROUND


def _build_bullet_fields(
    collateral_usd: float,
    entry_price: float,
    leverage: float,
    target_position_gain_pct: float,
    round_number: int,
    opened_at: str | None = None,
    bingx_order_id: str | None = None,
    entry_fee_usd: float | None = None,
) -> dict:
    """Shared field-construction logic for a freshly-opened bullet, used
    by both open_bullet() (manual, guardrail-checked) and
    sync_with_bingx() (reconciled from real BingX trades, guardrail-free
    since it's recording what already happened, not requesting a new
    action).

    entry_fee_usd: the REAL exchange fee paid on the opening fill (read
    from BingX's own trade record by sync_with_bingx() -- see its
    docstring). None for manually-recorded bullets, where there's no
    real fill to read a fee from; treated as 0 by check_bullets() /
    close_all_active_bullets()."""
    math = simulate_bullet_math(
        collateral_usd=collateral_usd,
        entry_price=entry_price,
        leverage=leverage,
        target_position_gain_pct=target_position_gain_pct,
    )
    return {
        "status": "open",
        "round_number": round_number,
        "collateral_usd": collateral_usd,
        "entry_price": entry_price,
        "leverage": leverage,
        "target_position_gain_pct": target_position_gain_pct,
        "position_size_usd": math["position_size_usd"],
        "target_price": math["target_price"],
        "approx_liquidation_price": math["approx_liquidation_price"],
        "opened_at": opened_at or _now_iso(),
        "closed_at": None,
        "closing_price": None,
        "outcome": None,
        "realized_pnl_usd": None,
        "notes": None,
        "bingx_order_id": bingx_order_id,
        "entry_fee_usd": entry_fee_usd,
        "exit_fee_usd": None,
    }


def open_bullet(
    collateral_usd: float | None,
    entry_price: float,
    leverage: float = 5.0,
    target_position_gain_pct: float = DEFAULT_TARGET_GAIN_PCT,
) -> dict:
    """Record a newly opened leveraged bullet (does NOT touch the exchange).

    Bullets accumulate within a round: this does NOT require the
    previous bullet(s) to be closed first. The guardrails enforced here
    in code are: (1) at most one NEW bullet per calendar day, and (2) at
    most MAX_BULLETS_PER_ROUND active bullets in the CURRENT round (once
    a round closes, the next one gets a fresh budget).

    Args:
        collateral_usd: Margin posted as collateral, in USD. If None,
            it's computed automatically (see _auto_collateral_usd) --
            this is what lets a future autonomous version of this bot
            open a bullet without a human typing in an amount.
        entry_price: Price at which the position was entered.
        leverage: Leverage multiplier (e.g. 5 for x5).
        target_position_gain_pct: Target gain ON THE COMBINED POSITION
            (not on price, and not just this bullet), e.g. 15 for +15%.
            Should normally be the same value across every bullet in a
            round -- it's a strategy-wide setting, not a per-bullet one.

    Returns:
        The persisted bullet record.

    Raises:
        RuntimeError: If a bullet was already opened today, this round
            is already at its 30-bullet cap, or collateral_usd was
            omitted and couldn't be auto-computed (BingX not configured
            / bad balance).
    """
    bullets = state_module.get_bullets()
    active = _find_active_bullets(bullets)

    if _opened_today(bullets):
        raise RuntimeError(
            "A bullet was already opened today. Only one NEW bullet per "
            "calendar day -- previously opened bullets can stay active "
            "and accumulate, but today's slot is used."
        )

    if len(active) >= MAX_BULLETS_PER_ROUND:
        raise RuntimeError(
            f"This round is full: {MAX_BULLETS_PER_ROUND} bullets already active. "
            "Waiting for the combined target to close the round before a new one can start."
        )

    if collateral_usd is None:
        collateral_usd = _auto_collateral_usd(active)

    round_number, bullet_number = _next_bullet_position(bullets, active)
    fields = _build_bullet_fields(collateral_usd, entry_price, leverage, target_position_gain_pct, round_number)
    return state_module.insert_bullet(fields, bullet_number)


def get_active_bullets(**_ignored) -> list[dict]:
    """Return every currently active (open/tracking) bullet."""
    return _find_active_bullets(state_module.get_bullets())


def check_bullets(current_price: float) -> dict:
    """Compute the live, COMBINED P&L across every active bullet at
    ``current_price``.

    Pure with respect to the network: the caller passes the current
    price (obtained separately via market_data.get_price()), so this
    function is testable in isolation. Its only side effect is the
    documented open -> tracking transition on the first check of each
    bullet.

    Args:
        current_price: Latest market price of the asset.

    Returns:
        A dict with:
        - "bullets": per-bullet detail (own price move %, position gain
          %, unrealized P&L, distance to its own liquidation price).
        - combined totals across all active bullets: collateral,
          unrealized P&L, and position gain % (= combined P&L / combined
          collateral).
        - "target_reached": whether the COMBINED gain hit the round's
          target (the target_position_gain_pct recorded on the first
          active bullet -- they should all share the same value).
        - "near_liquidation_any": True if ANY individual bullet is close
          to ITS OWN liquidation price (liquidation risk stays per bullet).

    Raises:
        RuntimeError: If there are no active bullets.
        ValueError: If current_price is not > 0.
    """
    if current_price <= 0:
        raise ValueError("current_price must be > 0")

    active = _find_active_bullets(state_module.get_bullets())
    if not active:
        raise RuntimeError("No active bullets to check.")

    per_bullet = []
    total_collateral = 0.0
    total_unrealized_pnl = 0.0
    any_near_liquidation = False

    for bullet in active:
        # Side effect: first check moves each bullet from open to tracking.
        if bullet["status"] == "open":
            bullet = state_module.update_bullet(bullet["id"], {"status": "tracking"})

        entry_price = bullet["entry_price"]
        leverage = bullet["leverage"]
        liq_price = bullet["approx_liquidation_price"]

        price_move_pct = (current_price - entry_price) / entry_price * 100
        position_gain_pct = price_move_pct * leverage
        entry_fee_usd = bullet.get("entry_fee_usd") or 0
        # Only the entry fee is subtracted here: it's already been paid
        # for real. The exit fee isn't known until the position actually
        # closes (see close_all_active_bullets()), so unrealized P&L
        # can't account for it yet -- it's a real cost still to come.
        unrealized_pnl_usd = bullet["collateral_usd"] * position_gain_pct / 100 - entry_fee_usd

        pct_above_liquidation = (current_price - liq_price) / liq_price * 100
        near_liquidation = current_price <= liq_price * (1 + LIQUIDATION_PROXIMITY_PCT / 100)
        any_near_liquidation = any_near_liquidation or near_liquidation

        total_collateral += bullet["collateral_usd"]
        total_unrealized_pnl += unrealized_pnl_usd

        per_bullet.append({
            "round_number": bullet.get("round_number"),
            "bullet_number": bullet["bullet_number"],
            "status": bullet["status"],
            "entry_price": entry_price,
            "leverage": leverage,
            "approx_liquidation_price": liq_price,
            "price_move_pct": round(price_move_pct, 2),
            "position_gain_pct": round(position_gain_pct, 2),
            "unrealized_pnl_usd": round(unrealized_pnl_usd, 2),
            "pct_above_liquidation": round(pct_above_liquidation, 2),
            "near_liquidation": near_liquidation,
        })

    target_pct = active[0].get("target_position_gain_pct", DEFAULT_TARGET_GAIN_PCT)
    combined_position_gain_pct = (total_unrealized_pnl / total_collateral * 100) if total_collateral else 0.0

    return {
        "round_number": active[0].get("round_number"),
        "current_price": current_price,
        "bullets": per_bullet,
        "combined_collateral_usd": round(total_collateral, 2),
        "combined_unrealized_pnl_usd": round(total_unrealized_pnl, 2),
        "combined_position_gain_pct": round(combined_position_gain_pct, 2),
        "target_position_gain_pct": target_pct,
        "target_reached": combined_position_gain_pct >= target_pct,
        "near_liquidation_any": any_near_liquidation,
    }


def close_all_active_bullets(
    outcome: str,
    closing_price: float,
    notes: str | None = None,
    closed_at: str | None = None,
    bingx_close_order_id: str | None = None,
    exit_fee_usd_total: float = 0.0,
) -> list[dict]:
    """Record the close of EVERY currently active bullet together, in one
    action (does NOT touch the exchange) -- this ends the round. Each
    bullet's realized P&L is computed from its OWN entry price vs. this
    shared closing price, minus its own entry fee (if known) and its
    share of the round's exit fee.

    Args:
        outcome: Either "tp" (the combined round hit its +15% target) or
            "manual" (closed by hand for any other reason). Applies to
            every bullet closed in this call.
        closing_price: Price at which the round was actually closed.
        notes: Optional free-text note, applied to every closed bullet.
        closed_at: ISO timestamp to record as the close time. Defaults to
            now; sync_with_bingx() passes the real BingX fill time instead.
        bingx_close_order_id: The BingX SELL fill's order id, when this
            close came from sync_with_bingx(). Stamped on every bullet
            closed here so that SAME sell fill is never reprocessed on a
            later sync -- see sync_with_bingx()'s module note on why this
            matters (a replayed old sell fill was wrongly re-closing
            bullets opened AFTER it, confirmed 2026-07-24).
        exit_fee_usd_total: The REAL exchange fee BingX charged on the
            closing fill (read from the trade record by
            sync_with_bingx()) -- ALL active bullets close via a single
            fill, so this one fee is split across them by collateral
            share (a bullet with 2x the collateral of another absorbs 2x
            the exit fee). 0 for a manual close with no real fill to
            read a fee from.

    Returns:
        The list of updated (now terminal) bullet records.

    Raises:
        ValueError: If outcome is not "tp"/"manual" or closing_price<=0.
        RuntimeError: If there are no active bullets to close.
    """
    if outcome not in VALID_OUTCOMES:
        raise ValueError(
            f"outcome must be one of {VALID_OUTCOMES}, got {outcome!r}"
        )
    if closing_price <= 0:
        raise ValueError("closing_price must be > 0")

    active = _find_active_bullets(state_module.get_bullets())
    if not active:
        raise RuntimeError("No active bullets to close.")

    closed_at = closed_at or _now_iso()
    status = "closed_tp" if outcome == "tp" else "closed_manual"
    total_collateral = sum(b["collateral_usd"] for b in active)
    closed = []
    for bullet in active:
        entry_price = bullet["entry_price"]
        leverage = bullet["leverage"]
        price_move_pct = (closing_price - entry_price) / entry_price * 100
        position_gain_pct = price_move_pct * leverage
        entry_fee_usd = bullet.get("entry_fee_usd") or 0
        exit_fee_share = (
            exit_fee_usd_total * (bullet["collateral_usd"] / total_collateral)
            if total_collateral else 0.0
        )
        realized_pnl_usd = (
            bullet["collateral_usd"] * position_gain_pct / 100
            - entry_fee_usd - exit_fee_share
        )

        closed.append(state_module.update_bullet(bullet["id"], {
            "status": status,
            "outcome": outcome,
            "closing_price": closing_price,
            "closed_at": closed_at,
            "realized_pnl_usd": round(realized_pnl_usd, 2),
            "exit_fee_usd": round(exit_fee_share, 2),
            "notes": notes,
            "bingx_close_order_id": bingx_close_order_id,
        }))

    return closed


def get_cycle_summary(**_ignored) -> dict:
    """Aggregate view of the current round plus lifetime totals across
    every round that's run so far.

    Returns:
        A dict with: this round's bullets used/remaining over
        MAX_BULLETS_PER_ROUND, how many rounds have completed, how many
        of those hit their target ("tp"), the total realized P&L across
        every round ever, and how many bullets are currently active.
    """
    bullets = state_module.get_bullets()
    active = _find_active_bullets(bullets)
    closed = [b for b in bullets if b["status"] in CLOSED_STATUSES]

    current_round = (
        active[0].get("round_number") if active
        else max((b.get("round_number") or 0 for b in bullets), default=0)
    )

    # A round's outcome is whatever its bullets share (they always close
    # together), so counting distinct round_numbers among closed bullets
    # gives the number of completed rounds -- not len(closed), which
    # would overcount rounds that used more than one bullet.
    closed_round_outcomes: dict[int, str] = {}
    for b in closed:
        closed_round_outcomes[b.get("round_number") or 0] = b["status"]
    tp_rounds = sum(1 for status in closed_round_outcomes.values() if status == "closed_tp")

    total_realized_pnl = sum(b["realized_pnl_usd"] or 0 for b in closed)

    return {
        "round_number": current_round,
        "max_bullets_per_round": MAX_BULLETS_PER_ROUND,
        "bullets_used_this_round": len(active),
        "bullets_remaining_this_round": MAX_BULLETS_PER_ROUND - len(active),
        "rounds_completed": len(closed_round_outcomes),
        "tp_rounds": tp_rounds,
        "total_realized_pnl_usd": round(total_realized_pnl, 2),
        "active_bullets_count": len(active),
        "active_bullet_numbers": [b["bullet_number"] for b in active],
    }


def get_bullet_status(symbol: str = "BTC/USDT") -> dict:
    """Combined bullet-cycle view for the daily agent report.

    Unlike every other function in this module, this one DOES touch the
    network: it fetches the current price to compute live combined P&L
    when bullets are active. It exists so the agent needs a single tool
    call to get both the round summary and (if applicable) the active
    bullets' live combined numbers, instead of chaining
    get_cycle_summary + market_data.get_price + check_bullets itself.

    Args:
        symbol: Trading pair to fetch the current price for if any
            bullet is active, e.g. "BTC/USDT".

    Returns:
        The cycle summary dict (see get_cycle_summary), plus a
        "live_status" key: None if no bullets are active, otherwise the
        live check_bullets() result at the current market price.
    """
    summary = get_cycle_summary()
    if summary["active_bullets_count"] == 0:
        return {**summary, "live_status": None}

    from . import market_data  # local import: keeps the rest of this
    # module network-free and avoids importing ccxt/requests unless a
    # bullet is actually active.
    current_price = market_data.get_price(symbol)["last_price"]
    return {**summary, "live_status": check_bullets(current_price)}


def sync_with_bingx() -> dict:
    """Reconcile your bullet records against your REAL BingX Demo Trading
    account, using TRADE history rather than positions.

    Why trades and not positions: BingX merges same-symbol, same-side
    fills into a single position row with an averaged entry price and a
    stable position id (confirmed empirically -- adding to an existing
    long left the same positionId, just a bigger size and a recalculated
    average price). Individual bullets are indistinguishable from that
    endpoint. Trades don't merge: each fill keeps its own order id,
    price, size, and timestamp, which is what lets each one become its
    own bullet here.

    - Every BUY fill not yet linked to a bullet (via bingx_order_id)
      becomes a new bullet, sized from the fill's REAL cost and the
      account's current leverage -- not the /30 formula, which only
      estimates what to trade before an order exists. Its round/bullet
      position is computed the same way open_bullet() does.
    - Any BUY fill's opened_at is the real fill time, not "now" -- sync
      may run up to ~15 minutes after the trade (see bullet-check).
    - Any SELL fill closes every bullet active at that point, together
      -- ending the round -- using the fill's real price as the closing
      price and real fill time as the close time. Outcome ("tp" vs
      "manual") is inferred from whether the combined gain at that price
      had reached the round's target.
    - Per the confirmed strategy: does NOT enforce the one-bullet-per-day
      or 30-bullets-per-round guardrails here. Those apply to whoever
      DECIDES to trade (you today, auto_trade() later); syncing only
      records what already, verifiably happened on the exchange.

    Does NOT touch the exchange -- reads BingX, writes to Supabase/local
    state only, exactly like every other function in this module.

    Returns:
        {"synced": bool, "reason": str (if not synced), "opened": [...],
        "closed": [...]}
    """
    from . import bingx_client

    if not bingx_client.is_enabled():
        return {"synced": False, "reason": "BingX not configured", "opened": [], "closed": []}

    trades = bingx_client.get_trade_history()
    all_bullets_snapshot = state_module.get_bullets()
    # BUG FIXED 2026-07-24: this used to only track bingx_order_id (BUY
    # fills). A SELL fill's order id was never persisted anywhere, so
    # get_trade_history() kept returning that same old SELL on every
    # sync run, and it kept matching against whatever bullets happened
    # to be active AT THAT LATER MOMENT -- wrongly re-closing bullets
    # opened well after the real sell happened. Tracking
    # bingx_close_order_id too makes every fill (buy AND sell)
    # permanently "seen" after its first sync.
    known_order_ids = {
        b[field] for b in all_bullets_snapshot
        for field in ("bingx_order_id", "bingx_close_order_id")
        if b.get(field)
    }

    # Leverage isn't on the trade itself -- read it from the currently
    # open position, falling back to the strategy's documented default
    # if nothing is open (e.g. syncing a SELL that closed everything).
    positions = bingx_client.get_open_positions()
    fallback_leverage = positions[0]["leverage"] if positions else 5.0

    opened, closed = [], []

    for trade in trades:
        order_id = trade.get("order")
        if not order_id or order_id in known_order_ids:
            continue
        known_order_ids.add(order_id)  # don't double-process within this run

        side = trade.get("side")
        price = trade.get("price")
        cost = trade.get("cost")
        trade_time = trade.get("datetime")
        # Futures fees on BingX are charged in the settlement currency
        # (USDT, even on the demo/VST account) -- straightforward to read
        # directly as a USD amount, unlike DCA's BTC-denominated spot
        # fees (see dca.py). Confirmed empirically 2026-07-25.
        fee_usd = (trade.get("fee") or {}).get("cost") or 0

        if side == "buy" and price and cost:
            all_bullets = state_module.get_bullets()
            active = _find_active_bullets(all_bullets)
            round_number, bullet_number = _next_bullet_position(all_bullets, active)
            collateral_usd = cost / fallback_leverage
            fields = _build_bullet_fields(
                collateral_usd, price, fallback_leverage, DEFAULT_TARGET_GAIN_PCT, round_number,
                opened_at=trade_time, bingx_order_id=order_id, entry_fee_usd=fee_usd,
            )
            opened.append(state_module.insert_bullet(fields, bullet_number))

        elif side == "sell" and price:
            active = _find_active_bullets(state_module.get_bullets())
            if not active:
                continue  # a reduce/close we have nothing active to match against
            live = check_bullets(price)
            outcome = "tp" if live["target_reached"] else "manual"
            closed.extend(close_all_active_bullets(
                outcome, price, closed_at=trade_time, bingx_close_order_id=order_id,
                exit_fee_usd_total=fee_usd,
            ))

    return {"synced": True, "opened": opened, "closed": closed}


# Max acceptable gap between what Supabase thinks is open and what BingX
# actually reports, in BTC, before reconcile_with_bingx() flags a
# mismatch. Not zero: rounding in how each bullet's implied BTC amount is
# reconstructed (collateral*leverage/entry_price) can differ from BingX's
# own stored contract size by a tiny amount even when nothing is wrong.
RECONCILE_TOLERANCE_BTC = 0.001


def reconcile_with_bingx() -> dict:
    """Defense in depth AFTER sync_with_bingx(): verify that what
    Supabase/local state believes is currently open actually matches
    BingX's real open position, instead of trusting a "sync ran without
    raising" as proof state is correct.

    Why this exists: sync_with_bingx() completing without an exception is
    NOT the same as it having done the right thing -- confirmed the hard
    way on 2026-07-24, when a bug silently re-closed bullets that were
    genuinely still open on BingX, and every sync call kept "succeeding"
    the whole time. This function catches that class of bug even if a
    similar one is introduced again: it doesn't trust sync's bookkeeping,
    it re-derives the expected position size from whatever bullets are
    currently marked active and compares it against BingX's own reported
    position, independently.

    Read-only against BingX. Does not fix anything -- callers (see
    main.py's bullet-check) are expected to notify a human when this
    reports a mismatch, per this project's human-in-the-loop design.

    Returns:
        {"checked": bool, "reason": str (if not checked), "ok": bool,
        "active_amount_btc": float, "real_amount_btc": float,
        "diff_btc": float}
    """
    from . import bingx_client

    if not bingx_client.is_enabled():
        return {"checked": False, "reason": "BingX not configured"}

    active = _find_active_bullets(state_module.get_bullets())
    active_amount_btc = sum(
        b["collateral_usd"] * b["leverage"] / b["entry_price"] for b in active
    )

    positions = bingx_client.get_open_positions()
    real_amount_btc = sum(p.get("contracts") or 0 for p in positions)

    diff_btc = abs(active_amount_btc - real_amount_btc)
    return {
        "checked": True,
        "ok": diff_btc <= RECONCILE_TOLERANCE_BTC,
        "active_amount_btc": round(active_amount_btc, 8),
        "real_amount_btc": round(real_amount_btc, 8),
        "diff_btc": round(diff_btc, 8),
    }


def auto_trade(test: bool = False) -> dict:
    """Place REAL orders on your BingX Demo Trading account to keep the
    strategy on schedule, with NO human typing a command:
    - Opens today's bullet (auto-sized, forced to AUTO_TRADE_LEVERAGE) if
      one hasn't been opened yet today and this round isn't at its cap.
    - Closes ALL active bullets together (ending the round) if the
      combined gain has reached the round's target.

    SAFETY: this is the only function in the whole codebase that can
    place an order. It refuses to do anything unless
    BINGX_AUTO_TRADE_ENABLED=true is set in the environment -- an
    explicit, off-by-default opt-in on TOP OF bingx_client.py's
    hard-coded demo-mode lock. Two independent gates have to agree
    before a single order goes out.

    Args:
        test: Forwarded to bingx_client. WARNING: confirmed empirically
            that BingX's "test" param can still execute a real demo
            order -- do not treat this as risk-free. See
            bingx_client.py's module docstring.

    Returns:
        {"traded": bool, "reason": str (if not traded), "action":
        "open"|"close" (if traded), "order": <BingX response>}
    """
    if os.environ.get("BINGX_AUTO_TRADE_ENABLED", "").strip().lower() != "true":
        return {"traded": False, "reason": "BINGX_AUTO_TRADE_ENABLED is not 'true'"}

    from . import bingx_client
    if not bingx_client.is_enabled():
        return {"traded": False, "reason": "BingX not configured"}

    bullets = state_module.get_bullets()
    active = _find_active_bullets(bullets)

    if active:
        from . import market_data
        current_price = market_data.get_price("BTC/USDT")["last_price"]
        if check_bullets(current_price)["target_reached"]:
            order = bingx_client.close_all_long_positions(test=test)
            return {"traded": True, "action": "close", "order": order}

    if not _opened_today(bullets) and len(active) < MAX_BULLETS_PER_ROUND:
        collateral_usd = _auto_collateral_usd(active)
        order = bingx_client.open_long_position(collateral_usd, leverage=AUTO_TRADE_LEVERAGE, test=test)
        return {"traded": True, "action": "open", "order": order}

    return {"traded": False, "reason": "nothing to do this cycle"}
