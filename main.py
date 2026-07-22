"""
main.py
--------
Entry point. Available commands:

    python main.py report        -> run the full agent and deliver the report (email/telegram/console)
    python main.py buy 50        -> record a 50 USD DCA purchase at the CURRENT BTC price (auto-fetched)
    python main.py buy 50 60000  -> record a 50 USD DCA purchase at a price you specify
    python main.py dca           -> summary of your DCA purchases (no LLM, instant)
    python main.py metrics       -> raw market and cycle metrics (no LLM)
    python main.py bullet 500          -> math of a 500 USD x5 bullet at the current price
    python main.py bullet 500 60000 3  -> same with entry price 60000 and x3 leverage

    --- Phase 2: tracking manual leveraged bullets on BingX ---
    Bullets ACCUMULATE: at most one NEW bullet per calendar day, but
    previous ones stay open. The +15% target is evaluated on the
    COMBINED position across every active bullet, and when it's hit,
    ALL active bullets close together in one action, ending the round --
    the next one starts over from bullet 1 with a fresh 30-bullet budget
    (this is a PER-ROUND cap, not a lifetime one; the bot can run
    indefinitely). Bullet size auto-computes as (BingX demo balance / 30)
    at the start of each round, so a profitable round compounds into a
    bigger bullet size next round -- no amount to type in by hand.
    python main.py bullet-open               -> auto-size today's new bullet (balance/30) at the CURRENT price
    python main.py bullet-open 500           -> override with an explicit 500 USD instead
    python main.py bullet-open 500 60000 5 15 -> same, explicit entry / leverage / target %
    python main.py bullet-status             -> live COMBINED P&L of all active bullets (no notification)
    python main.py bullet-check              -> same, and notify if combined target hit / any bullet near liquidation
    python main.py bullet-close tp           -> close ALL active bullets at the CURRENT price (outcome tp/manual)
    python main.py bullet-close manual 59000 "stopped out" -> close all at a given price with a note
    python main.py bullet-history             -> current round's progress (out of 30 bullets)
                                                plus lifetime totals across every round

NOTE: bullet-* commands NEVER touch the exchange. They only record what
you already did manually on BingX and compute the P&L from that.

    --- BingX Demo Trading (VST) integration ---
    python main.py bingx-positions       -> your REAL open positions on the demo account
    python main.py bingx-balance         -> your VST (virtual USDT) balance
    python main.py bingx-sync            -> reconcile bullets against your REAL BingX trade
                                             history (also runs automatically inside bullet-check)
    python main.py bingx-auto-trade      -> places a REAL (demo) order if one is due today
                                             (also runs automatically inside bullet-check, but
                                             ONLY if BINGX_AUTO_TRADE_ENABLED=true in .env)
    python main.py bingx-auto-trade --test -> same, forwarding BingX's "test" order param --
                                             ⚠️ CONFIRMED this can still execute a real demo
                                             order despite the name (2026-07-22). Do NOT treat
                                             --test as risk-free; it exists only for request-
                                             shape debugging, never as a safe dry-run.

NOTE: bingx-positions/balance/sync NEVER place or cancel an order.
bingx-auto-trade CAN place a real demo order -- see src/bingx_client.py
and src/bullets.py's auto_trade() for the safety design: hard-coded demo
mode (no path to your live account) PLUS an explicit, off-by-default
BINGX_AUTO_TRADE_ENABLED env var gate. Both must agree before any order
is placed.

The agent's "brain" is chosen with LLM_BACKEND in .env:
    LLM_BACKEND=claude   -> Anthropic API (default)
    LLM_BACKEND=ollama   -> local model via Ollama (free)

The output channel is chosen with NOTIFY_CHANNEL in .env:
    console (default) | email | telegram | all
"""
import json
import os
import sys
from dotenv import load_dotenv

load_dotenv()

from src import state, notify, db  # noqa: E402


def _get_agent_module():
    backend = os.environ.get("LLM_BACKEND", "claude").lower()
    if backend == "ollama":
        from src import agent_ollama as agent_module
    elif backend == "claude":
        from src import agent as agent_module
    else:
        raise ValueError(f"Unknown LLM_BACKEND: {backend!r} (use 'claude' or 'ollama')")
    print(f"[backend: {backend}]")
    return agent_module


def cmd_report():
    symbol = os.environ.get("SYMBOL", "BTC/USDT")
    agent_module = _get_agent_module()
    text = agent_module.run_daily_report(symbol=symbol)
    notify.notify(text, subject_prefix=f"Report {symbol}")
    _record_snapshot(symbol, text)


def _record_snapshot(symbol: str, report_text: str) -> None:
    """Best-effort: push a structured snapshot to Supabase (if configured)
    for the future dashboard's historical charts. Re-fetches the same
    metrics the agent's tools already gathered (cheap, public, read-only
    APIs) because the agent's free-text report isn't structured data.
    Never raises -- a snapshot failure must not break report delivery."""
    if not db.is_enabled():
        return
    try:
        from src import market_data, bullets
        price = market_data.get_price(symbol)
        indicators = market_data.get_indicators(symbol)
        cycle = market_data.get_cycle_metrics(symbol)
        fg = market_data.get_fear_greed_index()
        dom = market_data.get_btc_dominance()
        dca = state.get_dca_summary()
        bullet_cycle = bullets.get_cycle_summary()
        state.record_snapshot({
            "price": price["last_price"],
            "change_24h_pct": price["change_24h_pct"],
            "sma50": indicators["sma50"],
            "sma200": indicators["sma200"],
            "rsi14": indicators["rsi_14"],
            "sma200w": cycle["sma_200w"],
            "mayer_multiple": cycle["mayer_multiple"],
            "drawdown_from_high_pct": cycle["drawdown_from_high_pct"],
            "weekly_rsi14": cycle["weekly_rsi_14"],
            "fear_greed_value": fg["value"],
            "fear_greed_classification": fg["classification"],
            "btc_dominance_pct": dom["btc_dominance_pct"],
            "total_invested_usd": dca["total_invested_usd"],
            "total_qty_btc": dca["total_qty_btc"],
            "avg_entry_price": dca["avg_entry_price"],
            # daily_snapshots keeps the original column names (bullets_used,
            # bullets_remaining, tp_wins) even though bullets.py's own
            # vocabulary is now round-based (bullets_used_this_round,
            # etc.) -- avoids a DB migration + dashboard rewrite for what's
            # still the same shape of info, just scoped to the current round.
            "bullets_used": bullet_cycle["bullets_used_this_round"],
            "bullets_remaining": bullet_cycle["bullets_remaining_this_round"],
            "tp_wins": bullet_cycle["tp_rounds"],
            "total_realized_pnl_usd": bullet_cycle["total_realized_pnl_usd"],
            "report_text": report_text,
        })
        print("Snapshot saved to Supabase ✅")
    except Exception as exc:
        print(f"⚠️ Could not save snapshot to Supabase: {exc}")


def cmd_buy(args: list[str]):
    if not args:
        print("Usage: python main.py buy AMOUNT_USD [PRICE]")
        sys.exit(1)
    amount_usd = float(args[0])
    if len(args) >= 2:
        price = float(args[1])
    else:
        from src import market_data
        symbol = os.environ.get("SYMBOL", "BTC/USDT")
        print(f"Fetching current {symbol} price from BingX...")
        price = market_data.get_price(symbol)["last_price"]
        print(f"Current price: {price}")
    purchase = state.log_dca_purchase(amount_usd, price)
    print("Purchase recorded:", purchase)
    print("Summary:", state.get_dca_summary())


def cmd_dca():
    print(json.dumps(state.get_dca_summary(), indent=2, ensure_ascii=False))


def cmd_metrics():
    from src import market_data
    symbol = os.environ.get("SYMBOL", "BTC/USDT")
    print(json.dumps({
        "price": market_data.get_price(symbol),
        "indicators_daily": market_data.get_indicators(symbol),
        "cycle_metrics": market_data.get_cycle_metrics(symbol),
        "fear_greed": market_data.get_fear_greed_index(),
        "btc_dominance": market_data.get_btc_dominance(),
    }, indent=2, ensure_ascii=False))


def cmd_bullet(args: list[str]):
    if not args:
        print("Usage: python main.py bullet COLLATERAL_USD [ENTRY_PRICE] [LEVERAGE]")
        sys.exit(1)
    from src import strategy_tools
    collateral = float(args[0])
    if len(args) >= 2:
        entry = float(args[1])
    else:
        from src import market_data
        symbol = os.environ.get("SYMBOL", "BTC/USDT")
        entry = market_data.get_price(symbol)["last_price"]
        print(f"Using current {symbol} price: {entry}")
    leverage = float(args[2]) if len(args) >= 3 else 5.0
    result = strategy_tools.simulate_bullet_math(collateral, entry, leverage=leverage)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def _current_price() -> float:
    from src import market_data
    symbol = os.environ.get("SYMBOL", "BTC/USDT")
    price = market_data.get_price(symbol)["last_price"]
    print(f"Current {symbol} price: {price}")
    return price


def cmd_bullet_open(args: list[str]):
    from src import bullets
    # COLLATERAL_USD is optional: omit it to auto-size the bullet (BingX
    # balance / 30 at round start, or the round's existing size otherwise).
    collateral = float(args[0]) if len(args) >= 1 else None
    entry = float(args[1]) if len(args) >= 2 else _current_price()
    leverage = float(args[2]) if len(args) >= 3 else 5.0
    target_pct = float(args[3]) if len(args) >= 4 else bullets.DEFAULT_TARGET_GAIN_PCT
    bullet = bullets.open_bullet(collateral, entry, leverage=leverage, target_position_gain_pct=target_pct)
    print("Bullet opened (recorded, NOT sent to any exchange):")
    print(json.dumps(bullet, indent=2, ensure_ascii=False))


def cmd_bullet_status():
    from src import bullets
    if not bullets.get_active_bullets():
        print("No active bullets. Use 'bullet-open' to record one.")
        return
    result = bullets.check_bullets(_current_price())
    print(json.dumps(result, indent=2, ensure_ascii=False))


def cmd_bullet_check():
    from src import bullets, market_data
    symbol = os.environ.get("SYMBOL", "BTC/USDT")
    price_data = market_data.get_price(symbol)
    price = price_data["last_price"]
    print(f"Current {symbol} price: {price}")

    # Runs every 15 min via launchd regardless of bullet state -- this is
    # the dashboard's price history "heartbeat" (see state.record_price_tick).
    if db.is_enabled():
        try:
            state.record_price_tick(price, price_data.get("change_24h_pct"))
        except Exception as exc:
            print(f"⚠️ Could not record price tick: {exc}")

    from src import bingx_client
    if bingx_client.is_enabled():
        # auto_trade() no-ops internally unless BINGX_AUTO_TRADE_ENABLED=true
        # -- always safe to call. Runs BEFORE sync so an order placed just
        # now is picked up as a bullet in this same cycle.
        try:
            trade_result = bullets.auto_trade()
            if trade_result["traded"]:
                print(f"BingX auto-trade: {trade_result['action']} — {trade_result['order']}")
        except Exception as exc:
            print(f"⚠️ Could not run BingX auto-trade: {exc}")

        try:
            sync_result = bullets.sync_with_bingx()
            if sync_result["opened"] or sync_result["closed"]:
                print(f"BingX sync: {len(sync_result['opened'])} bullet(s) opened, "
                      f"{len(sync_result['closed'])} bullet(s) closed from real trades.")
        except Exception as exc:
            print(f"⚠️ Could not sync with BingX: {exc}")

    if not bullets.get_active_bullets():
        print("No active bullets. Use 'bullet-open' to record one.")
        return
    result = bullets.check_bullets(price)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["target_reached"] or result["near_liquidation_any"]:
        alerts = []
        if result["target_reached"]:
            alerts.append(
                f"🎯 Combined target reached: {result['combined_position_gain_pct']}% "
                f">= target {result['target_position_gain_pct']}% across "
                f"{len(result['bullets'])} active bullet(s)."
            )
        if result["near_liquidation_any"]:
            near = [b for b in result["bullets"] if b["near_liquidation"]]
            alerts.append(
                "⚠️ Near liquidation: bullet(s) " +
                ", ".join(f"#{b['id']} ({b['pct_above_liquidation']}% above approx. liquidation)" for b in near)
            )
        alerts.append(
            f"Combined unrealized P&L: {result['combined_unrealized_pnl_usd']} USD "
            f"({result['combined_position_gain_pct']}%). Decide and execute manually on BingX."
        )
        notify.notify("\n".join(alerts), subject_prefix="Bullet alert")


def cmd_bullet_close(args: list[str]):
    if not args:
        print("Usage: python main.py bullet-close OUTCOME [CLOSING_PRICE] [NOTES]  (OUTCOME = tp | manual)")
        print("Closes ALL currently active bullets together.")
        sys.exit(1)
    from src import bullets
    outcome = args[0]
    closing_price = float(args[1]) if len(args) >= 2 else _current_price()
    notes = args[2] if len(args) >= 3 else None
    closed = bullets.close_all_active_bullets(outcome, closing_price, notes=notes)
    print(f"Closed {len(closed)} bullet(s) (recorded):")
    print(json.dumps(closed, indent=2, ensure_ascii=False))
    print("Cycle summary:", json.dumps(bullets.get_cycle_summary(), indent=2, ensure_ascii=False))


def cmd_bullet_history():
    from src import bullets
    print(json.dumps(bullets.get_cycle_summary(), indent=2, ensure_ascii=False))


def cmd_bingx_positions():
    from src import bingx_client
    positions = bingx_client.get_open_positions()
    if not positions:
        print("No open positions on your BingX Demo Trading account.")
        return
    print(json.dumps(positions, indent=2, ensure_ascii=False, default=str))


def cmd_bingx_balance():
    from src import bingx_client
    print(json.dumps(bingx_client.get_balance(), indent=2, ensure_ascii=False))


def cmd_bingx_sync():
    from src import bullets
    result = bullets.sync_with_bingx()
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


def cmd_bingx_auto_trade(args: list[str]):
    from src import bullets
    test = "--test" in args
    if test:
        print("⚠️  --test forwards BingX's 'test' order param, which we confirmed can STILL "
              "execute a real demo order (see src/bingx_client.py). Treat this run as real.\n")
    result = bullets.auto_trade(test=test)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd, rest = sys.argv[1], sys.argv[2:]
    if cmd == "report":
        cmd_report()
    elif cmd == "buy":
        cmd_buy(rest)
    elif cmd == "dca":
        cmd_dca()
    elif cmd == "metrics":
        cmd_metrics()
    elif cmd == "bullet":
        cmd_bullet(rest)
    elif cmd == "bullet-open":
        cmd_bullet_open(rest)
    elif cmd == "bullet-status":
        cmd_bullet_status()
    elif cmd == "bullet-check":
        cmd_bullet_check()
    elif cmd == "bullet-close":
        cmd_bullet_close(rest)
    elif cmd == "bullet-history":
        cmd_bullet_history()
    elif cmd == "bingx-positions":
        cmd_bingx_positions()
    elif cmd == "bingx-balance":
        cmd_bingx_balance()
    elif cmd == "bingx-sync":
        cmd_bingx_sync()
    elif cmd == "bingx-auto-trade":
        cmd_bingx_auto_trade(rest)
    else:
        print(__doc__)
