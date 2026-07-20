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
    python main.py bullet-open 500           -> record an open x5 bullet at the CURRENT price
    python main.py bullet-open 500 60000 5 15 -> same, explicit entry / leverage / target %
    python main.py bullet-status             -> live P&L of the open bullet (no notification)
    python main.py bullet-check              -> same, and notify if target hit / near liquidation
    python main.py bullet-close tp           -> close the bullet at the CURRENT price (outcome tp/manual)
    python main.py bullet-close manual 59000 "stopped out" -> close at a given price with a note
    python main.py bullet-history            -> summary of the 30-bullet cycle

NOTE: bullet-* commands NEVER touch the exchange. They only record what
you already did manually on BingX and compute the P&L from that.

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
            "bullets_used": bullet_cycle["bullets_used"],
            "bullets_remaining": bullet_cycle["bullets_remaining"],
            "tp_wins": bullet_cycle["tp_wins"],
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
    if not args:
        print("Usage: python main.py bullet-open COLLATERAL_USD [ENTRY_PRICE] [LEVERAGE] [TARGET_PCT]")
        sys.exit(1)
    from src import bullets
    collateral = float(args[0])
    entry = float(args[1]) if len(args) >= 2 else _current_price()
    leverage = float(args[2]) if len(args) >= 3 else 5.0
    target_pct = float(args[3]) if len(args) >= 4 else 15.0
    bullet = bullets.open_bullet(collateral, entry, leverage=leverage, target_position_gain_pct=target_pct)
    print("Bullet opened (recorded, NOT sent to any exchange):")
    print(json.dumps(bullet, indent=2, ensure_ascii=False))


def cmd_bullet_status():
    from src import bullets
    if bullets.get_open_bullet() is None:
        print("No open bullet. Use 'bullet-open' to record one.")
        return
    result = bullets.check_bullet(_current_price())
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

    if bullets.get_open_bullet() is None:
        print("No open bullet. Use 'bullet-open' to record one.")
        return
    result = bullets.check_bullet(price)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["target_reached"] or result["near_liquidation"]:
        alerts = []
        if result["target_reached"]:
            alerts.append(f"🎯 Target reached: price {result['current_price']} >= target {result['target_price']}.")
        if result["near_liquidation"]:
            alerts.append(
                f"⚠️ Near liquidation: price {result['current_price']} is "
                f"{result['pct_above_liquidation']}% above approx. liquidation "
                f"{result['approx_liquidation_price']}."
            )
        alerts.append(
            f"Unrealized P&L: {result['unrealized_pnl_usd']} USD "
            f"(position {result['position_gain_pct']}%). Decide and execute manually on BingX."
        )
        notify.notify("\n".join(alerts), subject_prefix="Bullet alert")


def cmd_bullet_close(args: list[str]):
    if not args:
        print("Usage: python main.py bullet-close OUTCOME [CLOSING_PRICE] [NOTES]  (OUTCOME = tp | manual)")
        sys.exit(1)
    from src import bullets
    outcome = args[0]
    closing_price = float(args[1]) if len(args) >= 2 else _current_price()
    notes = args[2] if len(args) >= 3 else None
    bullet = bullets.close_bullet(outcome, closing_price, notes=notes)
    print("Bullet closed (recorded):")
    print(json.dumps(bullet, indent=2, ensure_ascii=False))
    print("Cycle summary:", json.dumps(bullets.get_cycle_summary(), indent=2, ensure_ascii=False))


def cmd_bullet_history():
    from src import bullets
    print(json.dumps(bullets.get_cycle_summary(), indent=2, ensure_ascii=False))


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
    else:
        print(__doc__)
