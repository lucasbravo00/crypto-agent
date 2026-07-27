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


def _fee_adjusted(trade: dict) -> tuple[float, float, float]:
    """Fold a spot buy's REAL exchange fee into (amount_usd, price,
    fee_usd), so the fee actually affects your recorded cost basis.

    Confirmed empirically (2026-07-25) that this BingX account's spot
    fees are charged in BTC (the asset being bought), not USDT -- the
    fee reduces the BTC you actually receive, it does NOT cost you extra
    USD. Since this project derives your BTC quantity as
    amount_usd / price (see state.get_dca_summary()), simply adding the
    fee to amount_usd would overstate your real holdings: the correct
    fix is an EFFECTIVE price (same USD spent, divided by the NET BTC
    you actually received), which raises your true average entry price
    exactly as much as the fee cost you.

    If a fee ever comes back in the quote currency (USDT) instead --
    not seen in practice, but handled -- that genuinely is extra USD
    spent, so it's added straight into amount_usd with the raw fill
    price left alone.
    """
    raw_price = trade["price"]
    raw_cost = trade["cost"]
    fee = trade.get("fee") or {}
    fee_cost = fee.get("cost") or 0
    fee_currency = fee.get("currency")

    if not fee_cost:
        return raw_cost, raw_price, 0.0

    if fee_currency == "BTC":
        net_amount = trade["amount"] - fee_cost
        if net_amount <= 0:
            return raw_cost, raw_price, 0.0
        fee_usd = fee_cost * raw_price  # informational only, not added to amount_usd
        return raw_cost, raw_cost / net_amount, fee_usd

    # Fee in the quote currency (or anything else): real extra USD
    # spent, quantity unaffected -- add it straight to the cost.
    return raw_cost + fee_cost, raw_price, fee_cost


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
        if not trade.get("cost") or not trade.get("price"):
            continue
        amount_usd, price, fee_usd = _fee_adjusted(trade)
        record = state_module.insert_dca_purchase(
            amount_usd=amount_usd,
            price=price,
            asset="BTC",
            bingx_trade_id=trade_id,
            purchased_at=trade.get("datetime"),
            fee_usd=fee_usd,
        )
        imported.append(record)

    return {"synced": True, "imported": imported}
