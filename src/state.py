"""
state.py
---------
Persistent state management for the agent: DCA purchases and leveraged
bullets (see src/bullets.py).

Dual backend, chosen at CALL TIME (not import time) by checking
db.is_enabled(): if SUPABASE_URL / SUPABASE_KEY are set, every read/write
goes to Supabase (Postgres) instead of the local JSON file. This is what
lets state be shared between your Mac and GitHub Actions -- a local run
and a CI run both see the same DCA purchases and bullet history. If
Supabase isn't configured (e.g. in tests, which explicitly unset those
env vars), everything falls back to the JSON file exactly as in Phase 1/2.

Every function below returns the same dict/list shape regardless of which
backend served it, so callers (bullets.py, main.py, the agent tools) never
need to know which one is active.
"""
from __future__ import annotations
import json
import os
from datetime import datetime, timezone
from typing import Optional

from . import db

STATE_PATH = os.path.join(os.path.dirname(__file__), "..", "state", "portfolio_state.json")


def _default_state() -> dict:
    """Return a FRESH default state on every call.

    This must be a factory, not a module-level constant: a module-level
    dict would share its nested lists ("dca_purchases", "bullets",
    "notes") across every ``.copy()`` (a shallow copy aliases them), so
    two supposedly independent "fresh state" instances would silently
    leak entries into each other. Building a new dict each call gives
    every caller its own lists.
    """
    return {
        "phase": "dca",
        "dca_purchases": [],   # list of {date, amount_usd, price, asset}
        "bullets": [],         # used in phase 2 (see src/bullets.py)
        "notes": [],
    }


def load_state() -> dict:
    """Local-JSON-only. Supabase-backed reads go through get_dca_purchases()
    / get_bullets() instead, which is why bullets.py no longer calls this
    directly."""
    if not os.path.exists(STATE_PATH):
        state = _default_state()
        save_state(state)
        return state
    with open(STATE_PATH, "r") as f:
        state = json.load(f)
    # Defensive merge: a state file written by an older schema may lack
    # keys added later (e.g. "bullets"). Backfill any missing top-level
    # key from a fresh default so downstream code can rely on them.
    for key, default_value in _default_state().items():
        state.setdefault(key, default_value)
    return state


def save_state(state: dict) -> None:
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


# --- DCA purchases -----------------------------------------------------

def get_dca_purchases() -> list[dict]:
    if db.is_enabled():
        client = db.get_client()
        rows = (
            client.table("dca_purchases")
            .select("*")
            .order("purchased_at")
            .execute()
            .data
        )
        for row in rows:
            row.setdefault("date", row.get("purchased_at"))
        return rows
    return load_state()["dca_purchases"]


def insert_dca_purchase(
    amount_usd: float,
    price: float,
    asset: str = "BTC",
    bingx_trade_id: Optional[str] = None,
    purchased_at: Optional[str] = None,
    fee_usd: Optional[float] = None,
) -> dict:
    """Record a DCA purchase. `bingx_trade_id` / `purchased_at` are set
    when this purchase was auto-imported from a real BingX spot fill (see
    src/dca.py's sync_with_bingx()) -- bingx_trade_id is the dedupe key
    that keeps repeated syncs from double-importing the same trade.
    `fee_usd` is informational (the real fee's USD-equivalent value) --
    `price` has ALREADY been adjusted to fold the fee into the effective
    cost basis (see dca.py's _fee_adjusted()), so this field is for
    display/audit only, not added again anywhere."""
    purchased_at = purchased_at or datetime.now(timezone.utc).isoformat()
    if db.is_enabled():
        client = db.get_client()
        row = {
            "purchased_at": purchased_at,
            "amount_usd": amount_usd,
            "price": price,
            "asset": asset,
            "bingx_trade_id": bingx_trade_id,
            "fee_usd": fee_usd,
        }
        result = client.table("dca_purchases").insert(row).execute().data[0]
        result.setdefault("date", result.get("purchased_at", purchased_at))
        return result
    state = load_state()
    purchase = {
        "date": purchased_at,
        "amount_usd": amount_usd,
        "price": price,
        "asset": asset,
        "bingx_trade_id": bingx_trade_id,
        "fee_usd": fee_usd,
    }
    state["dca_purchases"].append(purchase)
    save_state(state)
    return purchase


def log_dca_purchase(amount_usd: float, price: float, asset: str = "BTC") -> dict:
    """Record a DCA purchase. This is a tool with a side effect (it
    writes state), unlike the read-only tools in market_data.py."""
    return insert_dca_purchase(amount_usd, price, asset)


def get_dca_summary(**_ignored) -> dict:
    """Summary of the accumulation phase: total invested, average entry
    price and accumulated quantity. Live valuation is intentionally not
    computed here (that needs the live price, which is another tool) to
    keep each function single-responsibility."""
    purchases = get_dca_purchases()
    total_usd = sum(p["amount_usd"] for p in purchases)
    total_qty = sum(p["amount_usd"] / p["price"] for p in purchases) if purchases else 0
    avg_price = (total_usd / total_qty) if total_qty else None
    return {
        "phase": "dca",  # not yet a togglable setting; see bullets cycle status for phase 2
        "num_purchases": len(purchases),
        "total_invested_usd": round(total_usd, 2),
        "total_qty_btc": round(total_qty, 8),
        "avg_entry_price": round(avg_price, 2) if avg_price else None,
    }


# --- Bullets -------------------------------------------------------------
# Two different numbers, don't conflate them:
#   "id"            -> globally unique, NEVER reused or reset. The only
#                      safe key for update_bullet(). Supabase's own
#                      identity column; a lifetime-incrementing counter
#                      for the local JSON backend.
#   "bullet_number" -> position WITHIN the current round (1..30). RESETS
#                      to 1 every time a new round starts (see
#                      bullets.py's round design). Display-only -- two
#                      different bullets from two different rounds can
#                      legitimately share the same bullet_number, so it
#                      must never be used to look up a specific bullet.

def get_bullets() -> list[dict]:
    if db.is_enabled():
        client = db.get_client()
        return (
            client.table("bullets")
            .select("*")
            .order("id")
            .execute()
            .data
        )
    return load_state()["bullets"]


def insert_bullet(bullet: dict, bullet_number: int) -> dict:
    """Insert a new bullet at the given bullet_number (its position within
    the round -- bullets.py computes this, since only it knows whether a
    round is in progress or a new one is starting). Returns the record
    with its globally-unique "id" assigned."""
    if db.is_enabled():
        client = db.get_client()
        row = {**bullet, "bullet_number": bullet_number}
        return client.table("bullets").insert(row).execute().data[0]
    state = load_state()
    next_id = max((b["id"] for b in state["bullets"]), default=0) + 1
    record = {**bullet, "id": next_id, "bullet_number": bullet_number}
    state["bullets"].append(record)
    save_state(state)
    return record


def update_bullet(unique_id: int, fields: dict) -> dict:
    """Update the bullet identified by its globally-unique "id" (NEVER
    bullet_number -- see the note above this section) and return the
    updated record."""
    if db.is_enabled():
        client = db.get_client()
        return (
            client.table("bullets")
            .update(fields)
            .eq("id", unique_id)
            .execute()
            .data[0]
        )
    state = load_state()
    for b in state["bullets"]:
        if b["id"] == unique_id:
            b.update(fields)
            save_state(state)
            return b
    raise RuntimeError(f"Bullet {unique_id} not found")


# --- Daily snapshots (Supabase-only; no local-file equivalent) ---------

def record_snapshot(fields: dict) -> Optional[dict]:
    """Persist a daily_snapshots row for the dashboard's historical
    charts. No-op (returns None) if Supabase isn't configured -- there is
    intentionally no local-JSON equivalent for this table."""
    if not db.is_enabled():
        return None
    client = db.get_client()
    return client.table("daily_snapshots").insert(fields).execute().data[0]


def get_snapshots(limit: int = 60) -> list[dict]:
    """Read the most recent `limit` daily_snapshots rows, oldest first.
    Empty list if Supabase isn't configured -- there is intentionally no
    local-JSON equivalent for this table (see record_snapshot())."""
    if not db.is_enabled():
        return []
    client = db.get_client()
    rows = (
        client.table("daily_snapshots")
        .select("*")
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
        .data
    )
    return list(reversed(rows))


# --- Price ticks (Supabase-only; no local-file equivalent) -------------
# Higher-frequency than daily_snapshots: written every time
# `bullet-check` runs (every 15 min via launchd), giving the dashboard's
# price chart real intraday resolution instead of one point per day.

def record_price_tick(price: float, change_24h_pct: Optional[float] = None) -> Optional[dict]:
    """Persist a lightweight price_ticks row. No-op if Supabase isn't
    configured. This exists purely for the dashboard chart -- nothing in
    this codebase reads price_ticks back."""
    if not db.is_enabled():
        return None
    client = db.get_client()
    return client.table("price_ticks").insert({
        "price": price,
        "change_24h_pct": change_24h_pct,
    }).execute().data[0]


# --- Account ticks (Supabase-only; no local-file equivalent) -----------
# Same cadence/purpose as price_ticks, but for your BingX DEMO (VST)
# account's total balance -- the capital being used to test the bullets
# strategy. NOT the real spot wallet: that one is emptied out weekly by
# design (BTC gets rotated to Nexo for yield between DCA buys), so it
# isn't a meaningful "money in account" figure -- current DCA value is
# computed directly from dca_purchases instead (see dashboard/index.html).

def record_account_tick(vst_total: float,
                        liquidation_price: Optional[float] = None) -> Optional[dict]:
    """Persist a lightweight account_ticks row. No-op if Supabase isn't
    configured.

    `liquidation_price` is BingX's OWN cross-margin figure for the open
    position (None when flat). It is stored per tick rather than derived
    in the dashboard because under cross margin it moves with the account
    balance, not just with price -- see bingx_client.get_liquidation_price."""
    if not db.is_enabled():
        return None
    client = db.get_client()
    return client.table("account_ticks").insert({
        "vst_total": vst_total,
        "liquidation_price": liquidation_price,
    }).execute().data[0]
