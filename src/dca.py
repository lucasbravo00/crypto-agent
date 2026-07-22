"""
dca.py
-------
DCA sync: reconciles your recorded DCA purchases (src/state.py's
dca_purchases table) against your REAL BingX spot trade history, so you
don't have to type each purchase in by hand with `main.py buy`.

Read-only against BingX (see bingx_client.py's "REAL account access"
section) -- writes only go to Supabase/local state, exactly like
bullets.sync_with_bingx().
"""
from __future__ import annotations

from . import state as state_module


def sync_with_bingx() -> dict:
    """Import every real BTC/USDT spot BUY fill from your BingX account
    that isn't already recorded as a dca_purchase.

    Dedupe key is the trade's own id (stored as bingx_trade_id) -- safe
    to call this repeatedly (e.g. every 15 min from bullet-check).
    Ignores SELL fills: this project only tracks accumulation, not
    spot sells.

    Returns:
        {"synced": bool, "reason": str (if not synced), "imported": [...]}
    """
    from . import bingx_client

    if not bingx_client.is_enabled():
        return {"synced": False, "reason": "BingX not configured", "imported": []}

    trades = bingx_client.get_real_spot_trades()
    known_trade_ids = {
        p["bingx_trade_id"] for p in state_module.get_dca_purchases() if p.get("bingx_trade_id")
    }

    imported = []
    for trade in trades:
        if trade.get("side") != "buy":
            continue
        trade_id = trade.get("id")
        if not trade_id or trade_id in known_trade_ids:
            continue
        amount_usd = trade.get("cost")
        price = trade.get("price")
        if not amount_usd or not price:
            continue
        record = state_module.insert_dca_purchase(
            amount_usd=amount_usd,
            price=price,
            asset="BTC",
            bingx_trade_id=trade_id,
            purchased_at=trade.get("datetime"),
        )
        imported.append(record)

    return {"synced": True, "imported": imported}
