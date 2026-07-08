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

from src import state, notify  # noqa: E402


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
    else:
        print(__doc__)
