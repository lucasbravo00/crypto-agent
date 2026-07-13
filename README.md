# Crypto Strategy Agent

An LLM tool-calling agent that builds an objective daily report on BTC
market context, your DCA accumulation progress, and (once you're in that
phase) your manual leveraged "bullet" cycle on BingX — delivered to your
mailbox (e.g. Outlook), Telegram, or the console.

The agent never trades and never gives buy/sell signals: it gathers
data, summarizes it, and leaves every decision to the human (deliberate
human-in-the-loop design). It supports the user's real strategy, which
runs entirely outside the code:

1. **DCA phase** (current): accumulate BTC via manual spot purchases
   during a bear market, recorded with `python main.py buy`.
2. **Bullet phase** (once the user decides — manually — that a bull
   market has started): up to 30 sequential x5 leveraged futures
   positions on BingX, one at a time, each targeting +15% gain on the
   position. Opened and closed by hand on the exchange; this project
   only *records* what happened and computes the math/P&L from it.

## Architecture

```
main.py                  -> CLI entry point; picks LLM backend and output channel
src/
  market_data.py          -> read-only tools: price, daily indicators,
                             cycle metrics (200w SMA, Mayer Multiple, drawdown,
                             weekly RSI), fear & greed, BTC dominance
                             (BingX/CoinGecko), all with retry logic
  strategy_tools.py       -> pure math of a leveraged bullet position (no network)
  state.py                -> stateful tool: records DCA purchases + bullets (persistent JSON)
  bullets.py              -> bullet state machine: open -> tracking -> closed_tp/closed_manual;
                             enforces "one bullet at a time" and the 30-bullet cycle cap in code
  notify.py               -> output dispatcher: console / email / telegram
  email_notifier.py       -> SMTP delivery to your mailbox (e.g. Outlook)
  telegram_notifier.py    -> Telegram delivery (optional)
  agent.py                -> tool-calling loop on the Claude API
  agent_ollama.py         -> the SAME loop on a local model via Ollama
tests/
  test_logic.py           -> unit tests for market_data/strategy_tools (no network, no LLM)
  test_bullets.py         -> unit tests for the bullet state machine (isolated temp state)
state/                    -> persistent state (auto-created)
logs/                     -> JSONL trace of every agent decision (auto-created)
```

## Design decisions

- **Pure, isolated tools**: `pytest tests/` runs with no network and no LLM.
- **Swappable LLM backend** (Claude / Ollama) without touching the tools —
  the tools define *what the agent can do*; the backend defines *who
  decides when to do it*.
- **Swappable output channel** (email / telegram / console) without
  touching the agent.
- **Hard iteration cap** against infinite tool-calling loops, plus tool
  error handling that never crashes the run. The Ollama backend uses a
  higher cap than Claude's: local models are inconsistent about batching
  multiple tool calls per turn (sometimes several at once, sometimes one
  at a time), so it needs more budget to cover the same required tools.
- **Code-level, not prompt-level, guardrails**: `REQUIRED_TOOLS` blocks a
  final answer that skipped a mandatory tool, and `bullets.py` raises if
  you try to open a second bullet while one is active, or exceed the
  30-bullet cycle. Prompt wording alone ("you MUST call every tool")
  was tested and found insufficient, even on Claude.
- **Retries with backoff** on all network calls (public APIs fail
  sporadically).
- **Structured JSONL logging** of every tool call, its input and its
  result, so any report can be audited after the fact.
- **Human-in-the-loop by design**: the agent informs; it never executes
  trades or emits signals.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env    # then fill it in
```

- **Claude backend**: get an API key at https://console.anthropic.com and
  set `ANTHROPIC_API_KEY`.
- **Ollama backend**: install https://ollama.com, run
  `ollama pull llama3.1`, and set `LLM_BACKEND=ollama`.
- **Email delivery**: Outlook/Microsoft removed basic-auth SMTP sending
  (fully rejected since April 2026), so the report is sent *from* another
  SMTP provider (Gmail app password, or Brevo's free tier) *to* your
  mailbox. See `.env.example`. Or keep `NOTIFY_CHANNEL=console` for zero
  setup.

## Usage

```bash
# --- DCA phase ---
python main.py buy 50            # record a 50 USD DCA purchase at the current price
python main.py buy 50 60000      # same, with a manual price
python main.py dca               # DCA summary (no LLM)
python main.py metrics           # all raw market/cycle metrics (no LLM)
python main.py bullet 500        # math of a 500 USD x5 bullet at the current price (no state)

# --- Bullet phase: tracking manual leveraged positions on BingX ---
# These commands NEVER touch the exchange. They only record what you
# already did manually on BingX and compute the P&L from that.
python main.py bullet-open 500           # record an open x5 bullet at the CURRENT price
python main.py bullet-open 500 60000 5 15 # same, explicit entry / leverage / target %
python main.py bullet-status             # live P&L of the open bullet (no notification)
python main.py bullet-check              # same, and notify if target hit / near liquidation
python main.py bullet-close tp           # close the bullet at the CURRENT price (outcome tp/manual)
python main.py bullet-history            # summary of the 30-bullet cycle

# --- Agent + tests ---
python main.py report            # run the full agent and deliver the report
python -m pytest tests/ -v       # unit tests for the pure logic
```

## Roadmap

1. **Scheduling**: cron/launchd for the daily report and for a
   higher-frequency `bullet-check` (liquidation risk needs faster
   reaction than once a day).
2. **Multi-agent**: a market-analyst agent + a bullet-manager agent,
   coordinated.
3. **Memory**: compare against previous reports to detect regime changes.

## Disclaimer

This project is for educational and informational purposes. Nothing it
produces is financial advice. Leveraged trading carries a substantial
risk of loss: at x5 leverage an adverse price move of roughly 20% wipes
out the position's collateral (before fees and maintenance margin, which
make it happen sooner).
