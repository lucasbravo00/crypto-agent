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

COIN-MARGINED (INVERSE) CONTRACTS. Collateral and P&L are in BTC, not
USDT -- this is how the strategy is actually traded, and it changes the
math materially versus a USDT-margined position:

    PnL_btc = collateral * leverage * (1 - entry_price / price)

Two consequences worth internalizing:
  1. BTC gains are asymptotically capped at `collateral * leverage`, no
     matter how far price runs. Upside is not linear.
  2. Liquidation arrives EARLIER than the linear intuition, because the
     collateral itself loses USD value as price falls. At 5x a lone
     bullet liquidates at -16.67%, not -20%.

Data source is Binance rather than BingX (which this project uses
everywhere else): BingX's public API only returns ~1000 daily candles
(back to late 2023), not enough to reach the 2020-2021 cycle. Binance's
goes back to 2017. Same asset, effectively the same price series.

Documented simplifications (read before trusting a number):
- Liquidation is modeled as round equity (full round BTC balance +
  combined unrealized P&L) reaching zero. Real exchanges liquidate
  somewhat EARLIER, holding back maintenance margin, so results here
  are still optimistic about surviving deep drawdowns.
- Intraday ordering within a day is unknown from a daily candle. If a
  single day's range would have triggered BOTH liquidation (at its low)
  and the take-profit (at its high), liquidation is assumed to have
  happened first -- the conservative reading.
- Funding rates are not modeled. Fees are (see TAKER_FEE_RATE).
"""
from __future__ import annotations

from datetime import datetime, timezone

import ccxt

from .bullets import AUTO_TRADE_LEVERAGE, DEFAULT_TARGET_GAIN_PCT, MAX_BULLETS_PER_ROUND

# BingX's real taker fee, confirmed empirically from this project's own
# executed orders (fee 11.747448 on 23494.89692 notional = exactly
# 0.05%). Charged on entry and again on exit.
TAKER_FEE_RATE = 0.0005

# Cycle-top de-risking, modeled on how the strategy's originator trades
# it: assume the run ALWAYS ends in a liquidation, and shrink exposure as
# the market heats up so that liquidation eats a smaller share of the
# cycle's gains. Signal is the Mayer Multiple (price / 200d SMA), which
# historically ran >2.4 near cycle tops. Full exposure at or below
# MAYER_FULL_SIZE, scaling linearly to zero at MAYER_STOP.
MAYER_FULL_SIZE = 1.5
MAYER_STOP = 2.4

# CRITICAL distinction, and the whole reason `derisk_mode` exists.
# Under CROSS margin a liquidation wipes the entire trading balance, not
# just the collateral currently deployed. So shrinking bullet SIZE alone
# ("bullet_size") cannot protect accumulated profit -- the untouched
# balance is still sitting in the account backing the position, and dies
# with it. The only thing that actually protects gains is moving BTC OUT
# of the trading account ("withdraw"), where a liquidation can't reach
# it. Confirmed empirically: see the README's backtest section.
DERISK_MODES = ("off", "bullet_size", "withdraw")

SMA_WARMUP_DAYS = 200  # the Mayer Multiple's denominator


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


def _rolling_sma(values: list[float], period: int) -> list[float | None]:
    result: list[float | None] = []
    running = 0.0
    for i, v in enumerate(values):
        running += v
        if i >= period:
            running -= values[i - period]
        result.append(running / period if i >= period - 1 else None)
    return result


def _size_factor(mayer: float | None) -> float:
    """Fraction of the normal bullet size to commit, given how hot the
    market looks. 1.0 = full size, 0.0 = stop opening new bullets.
    Unknown Mayer (not enough warmup history) is treated as full size."""
    if mayer is None:
        return 1.0
    if mayer <= MAYER_FULL_SIZE:
        return 1.0
    if mayer >= MAYER_STOP:
        return 0.0
    return (MAYER_STOP - mayer) / (MAYER_STOP - MAYER_FULL_SIZE)


# --- Inverse (coin-margined) contract math ------------------------------
# Every bullet: {"entry_price": E, "collateral_btc": c}. Leverage L.
#   S = sum(c_i)          total BTC collateral deployed
#   W = sum(c_i * E_i)    collateral weighted by entry price

def _totals(bullets: list[dict]) -> tuple[float, float]:
    s = sum(b["collateral_btc"] for b in bullets)
    w = sum(b["collateral_btc"] * b["entry_price"] for b in bullets)
    return s, w


def _combined_pnl_btc(bullets: list[dict], price: float, leverage: float) -> float:
    """Combined unrealized P&L, in BTC, across every open bullet."""
    s, w = _totals(bullets)
    return leverage * s - leverage * w / price


def _target_price_btc(bullets: list[dict], leverage: float, target_gain_pct: float) -> float:
    """Price at which combined P&L reaches target_gain_pct MORE BTC than
    the BTC collateral committed."""
    s, w = _totals(bullets)
    return (leverage * w) / (s * (leverage - target_gain_pct / 100))


def _target_price_usd(bullets: list[dict], leverage: float, target_gain_pct: float) -> float:
    """Price at which the USD VALUE of the position is up target_gain_pct
    versus the USD value committed. Triggers earlier than the BTC target:
    part of the gain comes from the collateral itself appreciating, not
    from the trade."""
    s, w = _totals(bullets)
    return (w * (1 + target_gain_pct / 100 + leverage)) / (s * (1 + leverage))


def _liquidation_price(bullets: list[dict], balance_btc: float, leverage: float) -> float:
    """Price at which round equity (BTC balance + combined P&L) hits
    zero. Under CROSS margin the whole round balance backs every open
    bullet together -- which is the point of the 30-bullet budget: the
    part not yet deployed is real headroom, not idle cash."""
    s, w = _totals(bullets)
    return (leverage * w) / (balance_btc + leverage * s)


def run_backtest(
    start_date: str,
    end_date: str,
    initial_balance_btc: float = 1.0,
    symbol: str = "BTC/USDT",
    leverage: float = AUTO_TRADE_LEVERAGE,
    target_gain_pct: float = DEFAULT_TARGET_GAIN_PCT,
    max_bullets_per_round: int = MAX_BULLETS_PER_ROUND,
    target_mode: str = "btc",
    derisk_mode: str = "withdraw",
    candles: list | None = None,
    warmup_candles: list | None = None,
) -> dict:
    """Simulate the round-based bullet strategy day by day, coin-margined.

    Mirrors the live rules in src/bullets.py: at most one NEW bullet per
    calendar day (global, so a round closing does not free up that day's
    slot), bullets accumulate within a round up to max_bullets_per_round,
    the target is evaluated on the COMBINED position, and hitting it
    closes every bullet together and ends the round. Bullet size is the
    round's starting BTC balance / max_bullets_per_round, recomputed at
    each round start -- so profitable rounds compound.

    Args:
        start_date / end_date: "YYYY-MM-DD", inclusive. YOU choose these
            (a bull-market stretch) -- see the module docstring.
        initial_balance_btc: Starting collateral, in BTC.
        target_mode: "btc" (close when the position has gained
            target_gain_pct MORE BTC) or "usd" (close when its USD value
            is up target_gain_pct). Which one is actually better is an
            empirical question -- run both and compare.
        derisk_mode: How to de-risk as the Mayer Multiple approaches
            cycle-top territory. "off" = never; "bullet_size" = open
            smaller bullets but leave the whole balance in the account;
            "withdraw" = move BTC out of the trading account entirely, so
            a liquidation cannot reach it. See DERISK_MODES -- under
            cross margin only "withdraw" actually protects profit.
        candles / warmup_candles: Optional pre-fetched OHLCV, mainly for
            tests. warmup_candles are the SMA_WARMUP_DAYS candles BEFORE
            start_date, needed to have a valid Mayer Multiple on day one.

    Returns:
        Summary dict: final BTC balance, BTC return %, per-round detail,
        counts of take-profit vs liquidated rounds, max drawdown.
    """
    if target_mode not in ("btc", "usd"):
        raise ValueError(f"target_mode must be 'btc' or 'usd', got {target_mode!r}")
    if derisk_mode not in DERISK_MODES:
        raise ValueError(f"derisk_mode must be one of {DERISK_MODES}, got {derisk_mode!r}")

    if candles is None:
        # Fetch extra history before start_date so the 200d SMA behind the
        # Mayer Multiple is already valid on the first simulated day.
        warmup_start = (
            datetime.fromisoformat(start_date).toordinal() - SMA_WARMUP_DAYS - 20
        )
        warmup_start_date = datetime.fromordinal(warmup_start).date().isoformat()
        all_candles = fetch_daily_candles(warmup_start_date, end_date, symbol)
        start_ts = ccxt.binance().parse8601(f"{start_date}T00:00:00Z")
        warmup_candles = [c for c in all_candles if c[0] < start_ts]
        candles = [c for c in all_candles if c[0] >= start_ts]
    warmup_candles = warmup_candles or []

    if not candles:
        raise ValueError(f"No candles returned for {symbol} in {start_date}..{end_date}")

    # Mayer Multiple per simulated day, from a 200d SMA that spans the
    # warmup window too.
    all_closes = [c[4] for c in warmup_candles] + [c[4] for c in candles]
    sma200 = _rolling_sma(all_closes, SMA_WARMUP_DAYS)
    offset = len(warmup_candles)
    mayer_by_day = [
        (all_closes[offset + i] / sma200[offset + i]) if sma200[offset + i] else None
        for i in range(len(candles))
    ]

    pick_target = _target_price_btc if target_mode == "btc" else _target_price_usd

    balance = initial_balance_btc
    banked_btc = 0.0  # withdrawn from the trading account; liquidation-proof
    rounds: list[dict] = []
    peak_equity = initial_balance_btc
    max_drawdown_pct = 0.0

    open_bullets: list[dict] = []
    round_number = 1
    round_start_balance = balance
    round_opened_at: str | None = None

    def _close_round(exit_price: float, exit_date: str, outcome: str) -> None:
        nonlocal balance, open_bullets, round_number, round_start_balance, round_opened_at
        pnl_btc = _combined_pnl_btc(open_bullets, exit_price, leverage)
        s, _ = _totals(open_bullets)
        # Exit fee is charged on notional (USD), paid in BTC at exit price.
        exit_fee_btc = (s * leverage * exit_price) * TAKER_FEE_RATE / exit_price
        balance += pnl_btc - exit_fee_btc
        if outcome == "liquidated":
            balance = 0.0  # equity wiped out; nothing survives the round
        rounds.append({
            "round_number": round_number,
            "outcome": outcome,
            "opened_at": round_opened_at,
            "closed_at": exit_date,
            "bullets_used": len(open_bullets),
            "start_balance_btc": round(round_start_balance, 8),
            "end_balance_btc": round(balance, 8),
            "pnl_btc": round(balance - round_start_balance, 8),
            "exit_price": round(exit_price, 2),
        })
        open_bullets = []
        round_number += 1
        round_start_balance = balance
        round_opened_at = None

    last_open_date: str | None = None
    for i, (ts, open_price, high, low, close, _volume) in enumerate(candles):
        date = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).date().isoformat()
        mayer = mayer_by_day[i]

        # --- De-risk before doing anything else, if the cycle looks hot ---
        factor = _size_factor(mayer) if derisk_mode != "off" else 1.0
        if derisk_mode == "withdraw" and not open_bullets and factor < 1.0:
            # Only safe to move funds out between rounds (nothing open to
            # under-collateralize). Exposure is a fraction of TOTAL wealth
            # (trading + already-banked), not of the trading balance alone
            # -- otherwise this would skim (1-factor) off the account on
            # every single round and compound to zero regardless of the
            # market. Never claws banked BTC back in: one-way ratchet.
            total_wealth = balance + banked_btc
            target_exposure = total_wealth * factor
            if target_exposure < balance:
                banked_btc += balance - target_exposure
                balance = target_exposure
                round_start_balance = balance

        # --- Open the day's bullet (at the daily open: the live bot runs
        # its bullet-check right after midnight UTC) ---
        size_factor = factor if derisk_mode == "bullet_size" else 1.0
        bullet_size = (round_start_balance / max_bullets_per_round) * size_factor
        can_open = (
            balance > 0
            and bullet_size > 0
            and date != last_open_date
            and len(open_bullets) < max_bullets_per_round
        )
        if can_open:
            entry_fee_btc = bullet_size * leverage * TAKER_FEE_RATE
            if balance - entry_fee_btc > 0:
                balance -= entry_fee_btc
                open_bullets.append({"entry_price": open_price, "collateral_btc": bullet_size})
                last_open_date = date
                if round_opened_at is None:
                    round_opened_at = date

        if not open_bullets:
            continue

        # --- Resolve the day: liquidation (checked at the low) takes
        # precedence over take-profit (checked at the high) when a single
        # day's range would have hit both. See module docstring. ---
        liq_price = _liquidation_price(open_bullets, balance, leverage)
        tp_price = pick_target(open_bullets, leverage, target_gain_pct)

        if low <= liq_price:
            _close_round(liq_price, date, "liquidated")
        elif high >= tp_price:
            _close_round(tp_price, date, "take_profit")

        equity = balance + (
            _combined_pnl_btc(open_bullets, close, leverage) if open_bullets else 0.0
        )
        peak_equity = max(peak_equity, equity)
        if peak_equity > 0:
            max_drawdown_pct = max(max_drawdown_pct, (peak_equity - equity) / peak_equity * 100)

    # A round still open when the window ends is reported, not closed:
    # marking it a win or a loss would invent an exit the data doesn't have.
    final_balance = balance
    if open_bullets:
        last_close = candles[-1][4]
        final_balance = balance + _combined_pnl_btc(open_bullets, last_close, leverage)
        rounds.append({
            "round_number": round_number,
            "outcome": "still_open",
            "opened_at": round_opened_at,
            "closed_at": None,
            "bullets_used": len(open_bullets),
            "start_balance_btc": round(round_start_balance, 8),
            "end_balance_btc": round(final_balance, 8),
            "pnl_btc": round(final_balance - round_start_balance, 8),
            "exit_price": None,
        })

    closed_rounds = [r for r in rounds if r["outcome"] != "still_open"]
    end_price = candles[-1][4]
    total_btc = final_balance + banked_btc
    return {
        "symbol": symbol,
        "start_date": start_date,
        "end_date": end_date,
        "days_simulated": len(candles),
        "leverage": leverage,
        "target_gain_pct": target_gain_pct,
        "target_mode": target_mode,
        "derisk_mode": derisk_mode,
        "max_bullets_per_round": max_bullets_per_round,
        "initial_balance_btc": round(initial_balance_btc, 8),
        "trading_balance_btc": round(final_balance, 8),
        "banked_btc": round(banked_btc, 8),
        "final_balance_btc": round(total_btc, 8),
        "btc_return_pct": round((total_btc / initial_balance_btc - 1) * 100, 2),
        "final_value_usd": round(total_btc * end_price, 2),
        "end_price": round(end_price, 2),
        "rounds_total": len(rounds),
        "rounds_take_profit": sum(1 for r in closed_rounds if r["outcome"] == "take_profit"),
        "rounds_liquidated": sum(1 for r in closed_rounds if r["outcome"] == "liquidated"),
        "round_still_open": any(r["outcome"] == "still_open" for r in rounds),
        "max_drawdown_pct": round(max_drawdown_pct, 2),
        "rounds": rounds,
    }
