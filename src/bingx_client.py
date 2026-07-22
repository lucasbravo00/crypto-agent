"""
bingx_client.py
-----------------
Access to your BingX Demo Trading (VST) account via ccxt: reads
positions/balance/trade history, and (as of the auto-trade feature) can
place REAL orders -- but ONLY against the demo account, and ONLY when
BINGX_AUTO_TRADE_ENABLED=true (see bullets.auto_trade(), the sole
caller of the order-placing functions below).

WARNING -- the `test` parameter is NOT a safe dry-run on this account.
ccxt's docs describe `params.test=True` as hitting BingX's dedicated
test-order endpoint, which "validates the request without executing
it." Confirmed EMPIRICALLY (2026-07-22, against this project's real
demo account) that this is false for BingX's Demo Trading/VST
environment: an `open_long_position(test=True)` call executed a real
$493 order. Do not trust that parameter here again without re-verifying
against a real account first. It's kept only because it's still useful
for validating request *shape* against BingX's own validator when you
fully accept it may execute -- never call it believing it's risk-free.

SAFETY -- READ THIS BEFORE CHANGING ANYTHING BELOW:
BingX uses the SAME API key/secret for demo and live trading. The only
thing that separates a demo request from a live one is which base URL it
hits (ccxt's sandboxMode flag switches this). Because the key itself
gives no guarantee, this module hard-codes sandbox/demo mode in
_get_client() with NO parameter, flag, or environment variable able to
turn it off, and verifies the setting actually took before handing out a
usable client. There is currently no code path anywhere in this project
that can place a LIVE order. If real trading is ever wanted, that must
be a deliberate, separately-requested piece of new code -- never a
config flip.
"""
from __future__ import annotations
import os

import ccxt

# BingX's unified ccxt symbol for the BTC-USDT linear perpetual swap.
# (Different from the "BTC/USDT" spot symbol used elsewhere in this
# project for read-only price/indicator data.)
SYMBOL = "BTC/USDT:USDT"

_client = None


def is_enabled() -> bool:
    return bool(os.environ.get("BINGX_API_KEY")) and bool(os.environ.get("BINGX_API_SECRET"))


def _get_client():
    """Return a cached ccxt BingX client, hard-locked to demo/sandbox mode."""
    global _client
    if _client is not None:
        return _client

    if not is_enabled():
        raise RuntimeError(
            "BINGX_API_KEY / BINGX_API_SECRET are not set. See .env.example."
        )

    client = ccxt.bingx({
        "apiKey": os.environ["BINGX_API_KEY"],
        "secret": os.environ["BINGX_API_SECRET"],
        "enableRateLimit": True,
    })
    client.set_sandbox_mode(True)  # HARD-CODED -- see module docstring. Never parameterize this.

    # Defense in depth: verify the setting actually took before handing
    # out a client any caller could use.
    if not client.options.get("sandboxMode"):
        raise RuntimeError(
            "BingX client failed to enter sandbox/demo mode -- refusing to "
            "proceed. This is a safety check; never relax it."
        )

    _client = client
    return client


def get_open_positions(**_ignored) -> list[dict]:
    """Read your OPEN positions on the BingX Demo Trading (VST) account.

    Read-only: makes no trades. Returns BingX's raw position fields
    (entry price, size, leverage, unrealized P&L, liquidation price,
    etc.) for the BTC-USDT perpetual, or an empty list if you have no
    open position there right now.
    """
    client = _get_client()
    positions = client.fetch_positions([SYMBOL])
    # fetch_positions can include zero-size/closed entries; keep only
    # positions that actually have size.
    return [p for p in positions if p.get("contracts") not in (None, 0)]


def get_trade_history(**_ignored) -> list[dict]:
    """Read your executed trades (fills) on the BingX Demo Trading account
    for the BTC-USDT perpetual, oldest first.

    Read-only: makes no trades. This exists because fetch_positions()
    MERGES same-symbol, same-side fills into one row (confirmed
    empirically: opening a position and later adding to it produces a
    single position with an averaged entry price and the same
    positionId) -- individual bullets can't be reconstructed from
    positions. Trades don't merge: each one keeps its own order id,
    price, size and timestamp, which is what src/bullets.py's
    sync_with_bingx() uses to tell bullets apart.
    """
    client = _get_client()
    trades = client.fetch_my_trades(SYMBOL)
    return sorted(trades, key=lambda t: t.get("timestamp") or 0)


def get_balance(**_ignored) -> dict:
    """Read your virtual balance on the BingX Demo Trading account.

    Two things trip this up if you don't know them: ccxt's
    fetch_balance() defaults to the SPOT wallet (nearly empty here --
    your virtual funds live in the perpetual futures/swap wallet, hence
    the explicit type param); and BingX denominates demo funds in "VST"
    (Virtual Simulated Trading currency), not "USDT" -- confirmed
    empirically against a real demo account, not assumed.
    """
    client = _get_client()
    balance = client.fetch_balance(params={"type": "swap"})
    vst = balance.get("VST", {})
    return {
        "asset": "VST",
        "free": vst.get("free"),
        "used": vst.get("used"),
        "total": vst.get("total"),
    }


# --- Order placement -----------------------------------------------------
# Everything below WRITES to your BingX Demo Trading account. The sole
# intended caller is bullets.auto_trade(), which itself refuses to run
# unless BINGX_AUTO_TRADE_ENABLED=true. Never call these from anywhere
# else without that same gate.

def set_leverage(leverage: float, **_ignored) -> dict:
    """Set the leverage for the LONG side of BTC-USDT on the Demo Trading
    account. Isolated margin, hedge mode (this account's dualSidePosition
    is True -- confirmed via fetch_position_mode()), so this only affects
    the LONG side, not any SHORT position. Called before every automated
    open so bullets always use the strategy's documented leverage,
    regardless of whatever the account happens to be set to manually.
    """
    client = _get_client()
    return client.set_leverage(int(leverage), SYMBOL, params={"side": "LONG"})


def open_long_position(collateral_usd: float, leverage: float = 5.0, test: bool = False, **_ignored) -> dict:
    """Place a REAL order on the BingX Demo Trading account: a MARKET BUY
    to open or add to the LONG position, sized from
    collateral_usd * leverage at the current market price.

    Args:
        collateral_usd: Margin to post, in USD (VST).
        leverage: Leverage to force before placing the order.
        test: Forwards to BingX's "test" endpoint param. DO NOT treat
            this as a safe dry-run -- confirmed empirically that it can
            still execute a real order on the Demo Trading account
            despite the name. See the module docstring.

    Returns:
        BingX's raw order response. ALWAYS assume this order executed.
    """
    client = _get_client()
    set_leverage(leverage)
    price = client.fetch_ticker(SYMBOL)["last"]
    notional_usd = collateral_usd * leverage
    amount = notional_usd / price
    return client.create_order(
        SYMBOL, "market", "buy", amount,
        params={"hedged": True, "test": test},
    )


def close_all_long_positions(test: bool = False, **_ignored) -> dict:
    """Place a REAL order on the BingX Demo Trading account: a MARKET
    SELL (reduceOnly) to close the ENTIRE current LONG position in one
    shot.

    Args:
        test: Forwards to BingX's "test" endpoint param. DO NOT treat
            this as a safe dry-run -- see open_long_position()'s
            docstring and the module docstring for why.

    Returns:
        BingX's raw order response (ALWAYS assume it executed), or
        {"closed": False, ...} if there was no open position to close.
    """
    client = _get_client()
    positions = get_open_positions()
    if not positions:
        return {"closed": False, "reason": "no open position to close"}
    amount = positions[0]["contracts"]
    return client.create_order(
        SYMBOL, "market", "sell", amount,
        params={"hedged": True, "reduceOnly": True, "test": test},
    )
