"""
state.py
---------
Persistent state management for the agent. Deliberately simple (a JSON
file on disk) to make the core idea obvious: an agent that operates over
time needs to "remember" which phase it is in and what happened before,
just like a streaming job needs its state checkpoint.

Possible strategy states (state["phase"]):
  - "dca"     -> accumulation phase toward BTC (implemented now)
  - "bullets" -> phase of 30 leveraged futures bullets (next milestone)
"""
from __future__ import annotations
import json
import os
from datetime import datetime, timezone

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


def log_dca_purchase(amount_usd: float, price: float, asset: str = "BTC") -> dict:
    """Record a DCA purchase. This is a tool with a side effect (it
    writes state), unlike the read-only tools in market_data.py."""
    state = load_state()
    purchase = {
        "date": datetime.now(timezone.utc).isoformat(),
        "amount_usd": amount_usd,
        "price": price,
        "asset": asset,
    }
    state["dca_purchases"].append(purchase)
    save_state(state)
    return purchase


def get_dca_summary(**_ignored) -> dict:
    """Summary of the accumulation phase: total invested, average entry
    price and accumulated quantity. Live valuation is intentionally not
    computed here (that needs the live price, which is another tool) to
    keep each function single-responsibility."""
    state = load_state()
    purchases = state["dca_purchases"]
    total_usd = sum(p["amount_usd"] for p in purchases)
    total_qty = sum(p["amount_usd"] / p["price"] for p in purchases) if purchases else 0
    avg_price = (total_usd / total_qty) if total_qty else None
    return {
        "phase": state["phase"],
        "num_purchases": len(purchases),
        "total_invested_usd": round(total_usd, 2),
        "total_qty_btc": round(total_qty, 8),
        "avg_entry_price": round(avg_price, 2) if avg_price else None,
    }
