"""
backtest.py
------------
Simulates the bullet strategy against real historical daily candles.

SCOPE, deliberately: this strategy is a BULL-market tool only. Run over
a full multi-year history it would inevitably show losses, because it
would be simulating a long-only leveraged strategy through bear markets
it was never meant to be running in -- deciding *when* a bull market is
underway is the trader's judgment call, made outside this code (the
same reason the live agent never emits buy/sell signals). So the date
range is an explicit, required input: you pick the bull period, this
measures how the mechanics would have performed inside it.

Data source is Binance rather than BingX (which this project uses
everywhere else): BingX's public API only returns ~1000 daily candles
(back to late 2023), not enough to reach the 2020-2021 cycle. Binance's
goes back to 2017. Same asset, effectively the same price series.

Documented simplifications (read before trusting a number):
- Liquidation is modeled as round equity (full round balance +
  combined unrealized P&L) reaching zero. Real exchanges liquidate
  somewhat EARLIER, holding back maintenance margin, so results here
  are optimistic about surviving deep drawdowns.
- Intraday ordering within a day is unknown from a daily candle. If a
  single day's range would have triggered BOTH liquidation (at its low)
  and the take-profit (at its high), liquidation is assumed to have
  happened first -- the conservative reading.
- Funding rates are not modeled. Fees are (see TAKER_FEE_RATE).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import ccxt

from .bullets import AUTO_TRADE_LEVERAGE, DEFAULT_TARGET_GAIN_PCT, MAX_BULLETS_PER_ROUND

# BingX's real taker fee, confirmed empirically from this project's own
# executed orders (fee 11.747448 on 23494.89692 notional = exactly
# 0.05%). Charged on entry and again on exit.
TAKER_FEE_RATE = 0.0005


def fetch_daily_candles(start_date: str, end_date: str, symbol: str = "BTC/USDT") -> list:
    """Daily OHLCV candles in [start_date, end_date], inclusive, from
    Binance's public API (no key needed). Paginated: Binance caps each
    request at 1000 candles."""
    exchange = ccxt.binance({"enableRateLimit": True})
    since = exchange.parse8601(f"{start_date}T00:00:00Z")
    end_ts = exchange.parse8601(f"{end_date}T23:59:59Z")

    candles: list = []
    while since < end_ts:
        batch = exchange.fetch_ohlcv(symbol, timeframe="1d", since=since, limit=1000)
        if not batch:
            break
        candles.extend(c for c in batch if c[0] <= end_ts)
        if len(batch) < 1000:
            break
        since = batch[-1][0] + 24 * 60 * 60 * 1000
    return candles


def _combined_pnl(bullets: list[dict], price: float, leverage: float) -> float:
    """Combined unrealized P&L across every open bullet at `price`."""
    return sum(
        b["collateral_usd"] * leverage * (price - b["entry_price"]) / b["entry_price"]
        for b in bullets
    )


def _target_price(bullets: list[dict], leverage: float, target_gain_pct: float) -> float:
    """Exact price at which the COMBINED position reaches its target gain.

    Solved in closed form rather than checked against the candle's high,
    so a take-profit is booked at the target itself -- booking it at the
    day's high would overstate profit by however far price overshot.
    """
    total_collateral = sum(b["collateral_usd"] for b in bullets)
    weighted = sum(b["collateral_usd"] / b["entry_price"] for b in bullets)
    return (total_collateral * (target_gain_pct / 100 + leverage)) / (leverage * weighted)


def _liquidation_price(bullets: list[dict], balance: float, leverage: float) -> float:
    """Exact price at which round equity (balance + combined P&L) hits
    zero. Under CROSS margin the whole round balance backs every open
    bullet together -- which is the point of the 30-bullet budget: the
    part not yet deployed is real headroom, not idle cash (see the
    README's cross-margin note)."""
    total_collateral = sum(b["collateral_usd"] for b in bullets)
    weighted = sum(b["collateral_usd"] / b["entry_price"] for b in bullets)
    return (leverage * total_collateral - balance) / (leverage * weighted)


def run_backtest(
    start_date: str,
    end_date: str,
    initial_balance_usd: float = 10000.0,
    symbol: str = "BTC/USDT",
    leverage: float = AUTO_TRADE_LEVERAGE,
    target_gain_pct: float = DEFAULT_TARGET_GAIN_PCT,
    max_bullets_per_round: int = MAX_BULLETS_PER_ROUND,
    candles: list | None = None,
) -> dict:
    """Simulate the round-based bullet strategy day by day.

    Mirrors the live rules in src/bullets.py: at most one NEW bullet per
    calendar day (global, so a round closing does not free up that day's
    slot), bullets accumulate within a round up to max_bullets_per_round,
    the target is evaluated on the COMBINED position, and hitting it
    closes every bullet together and ends the round. Bullet size is the
    round's starting balance / max_bullets_per_round, recomputed at each
    round start -- so profitable rounds compound into bigger bullets.

    Args:
        start_date / end_date: "YYYY-MM-DD", inclusive. YOU choose these
            (a bull-market stretch) -- see the module docstring.
        candles: Optional pre-fetched OHLCV, mainly for tests. Fetched
            from Binance when omitted.

    Returns:
        Summary dict: final balance, return %, per-round detail, counts
        of take-profit vs liquidated vs still-open rounds, max drawdown.
    """
    if candles is None:
        candles = fetch_daily_candles(start_date, end_date, symbol)
    if not candles:
        raise ValueError(f"No candles returned for {symbol} in {start_date}..{end_date}")

    balance = initial_balance_usd
    rounds: list[dict] = []
    equity_curve: list[float] = []
    peak_equity = initial_balance_usd
    max_drawdown_pct = 0.0

    open_bullets: list[dict] = []
    round_number = 1
    round_start_balance = balance
    bullet_size = balance / max_bullets_per_round
    round_opened_at: str | None = None

    def _close_round(exit_price: float, exit_date: str, outcome: str) -> None:
        nonlocal balance, open_bullets, round_number, round_start_balance, bullet_size, round_opened_at
        pnl = _combined_pnl(open_bullets, exit_price, leverage)
        exit_notional = sum(b["collateral_usd"] for b in open_bullets) * leverage
        exit_fee = exit_notional * TAKER_FEE_RATE
        balance += pnl - exit_fee
        if outcome == "liquidated":
            balance = 0.0  # equity wiped out; nothing survives the round
        rounds.append({
            "round_number": round_number,
            "outcome": outcome,
            "opened_at": round_opened_at,
            "closed_at": exit_date,
            "bullets_used": len(open_bullets),
            "start_balance_usd": round(round_start_balance, 2),
            "end_balance_usd": round(balance, 2),
            "pnl_usd": round(balance - round_start_balance, 2),
            "exit_price": round(exit_price, 2),
        })
        open_bullets = []
        round_number += 1
        round_start_balance = balance
        bullet_size = balance / max_bullets_per_round
        round_opened_at = None

    last_open_date: str | None = None
    for ts, open_price, high, low, close, _volume in candles:
        date = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).date().isoformat()

        # --- Open the day's bullet (at the daily open: the live bot runs
        # its bullet-check right after midnight UTC) ---
        can_open = (
            balance > 0
            and date != last_open_date
            and len(open_bullets) < max_bullets_per_round
        )
        if can_open:
            entry_fee = bullet_size * leverage * TAKER_FEE_RATE
            if bullet_size > 0 and balance - entry_fee > 0:
                balance -= entry_fee
                open_bullets.append({"entry_price": open_price, "collateral_usd": bullet_size})
                last_open_date = date
                if round_opened_at is None:
                    round_opened_at = date

        if not open_bullets:
            equity_curve.append(balance)
            continue

        # --- Resolve the day: liquidation (checked at the low) takes
        # precedence over take-profit (checked at the high) when a single
        # day's range would have hit both. See module docstring. ---
        liq_price = _liquidation_price(open_bullets, balance, leverage)
        tp_price = _target_price(open_bullets, leverage, target_gain_pct)

        if low <= liq_price:
            _close_round(liq_price, date, "liquidated")
        elif high >= tp_price:
            _close_round(tp_price, date, "take_profit")

        equity = balance + (_combined_pnl(open_bullets, close, leverage) if open_bullets else 0.0)
        equity_curve.append(equity)
        peak_equity = max(peak_equity, equity)
        if peak_equity > 0:
            max_drawdown_pct = max(max_drawdown_pct, (peak_equity - equity) / peak_equity * 100)

    # A round still open when the window ends is reported, not closed:
    # marking it a win or a loss would invent an exit the data doesn't have.
    final_equity = balance
    if open_bullets:
        last_close = candles[-1][4]
        unrealized = _combined_pnl(open_bullets, last_close, leverage)
        final_equity = balance + unrealized
        rounds.append({
            "round_number": round_number,
            "outcome": "still_open",
            "opened_at": round_opened_at,
            "closed_at": None,
            "bullets_used": len(open_bullets),
            "start_balance_usd": round(round_start_balance, 2),
            "end_balance_usd": round(final_equity, 2),
            "pnl_usd": round(final_equity - round_start_balance, 2),
            "exit_price": None,
        })

    closed_rounds = [r for r in rounds if r["outcome"] != "still_open"]
    return {
        "symbol": symbol,
        "start_date": start_date,
        "end_date": end_date,
        "days_simulated": len(candles),
        "leverage": leverage,
        "target_gain_pct": target_gain_pct,
        "max_bullets_per_round": max_bullets_per_round,
        "initial_balance_usd": round(initial_balance_usd, 2),
        "final_balance_usd": round(final_equity, 2),
        "total_return_pct": round((final_equity / initial_balance_usd - 1) * 100, 2),
        "rounds_total": len(rounds),
        "rounds_take_profit": sum(1 for r in closed_rounds if r["outcome"] == "take_profit"),
        "rounds_liquidated": sum(1 for r in closed_rounds if r["outcome"] == "liquidated"),
        "round_still_open": any(r["outcome"] == "still_open" for r in rounds),
        "max_drawdown_pct": round(max_drawdown_pct, 2),
        "rounds": rounds,
    }
