# Crypto Strategy Agent — Phase 1 (DCA + Market Monitor)

An LLM tool-calling agent that builds an objective daily report on BTC
market context and your DCA progress, and delivers it to your mailbox
(e.g. Outlook), Telegram, or the console. Foundation for Phase 2
(leveraged "bullet" position tracking).

The agent never trades and never gives buy/sell signals: it gathers
data, summarizes it, and leaves every decision to the human
(deliberate human-in-the-loop design).

## Architecture

```
main.py                  -> CLI entry point; picks LLM backend and output channel
src/
  market_data.py          -> read-only tools: price, daily indicators,
                             cycle metrics (200w SMA, Mayer Multiple, drawdown,
                             weekly RSI), fear & greed, BTC dominance
                             (BingX/CoinGecko), all with retry logic
  strategy_tools.py       -> pure math of a leveraged bullet position
  state.py                -> stateful tool: records DCA purchases (persistent JSON)
  notify.py               -> output dispatcher: console / email / telegram
  email_notifier.py       -> SMTP delivery to your mailbox (e.g. Outlook)
  telegram_notifier.py    -> Telegram delivery (optional)
  agent.py                -> tool-calling loop on the Claude API
  agent_ollama.py         -> the SAME loop on a local model via Ollama
tests/test_logic.py       -> unit tests for the pure logic (no network, no LLM)
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
  error handling that never crashes the run.
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
python main.py buy 50            # record a 50 USD DCA purchase at the current price
python main.py buy 50 60000      # same, with a manual price
python main.py dca               # DCA summary (no LLM)
python main.py metrics           # all raw market/cycle metrics (no LLM)
python main.py bullet 500        # math of a 500 USD x5 bullet at the current price
python main.py report            # run the full agent and deliver the report
python -m pytest tests/ -v       # unit tests for the pure logic
```

## Roadmap

1. **Phase 2 — bullet management**: a per-bullet state machine
   (`pending -> open -> tracking -> closed`), live P&L vs the target,
   alerts when approaching it.
2. **Scheduling**: cron/scheduler for the daily report.
3. **Multi-agent**: a market-analyst agent + a bullet-manager agent,
   coordinated.
4. **Memory**: compare against previous reports to detect regime changes.

## Disclaimer

This project is for educational and informational purposes. Nothing it
produces is financial advice. Leveraged trading carries a substantial
risk of loss: at x5 leverage an adverse price move of roughly 20% wipes
out the position's collateral (before fees and maintenance margin, which
make it happen sooner).
