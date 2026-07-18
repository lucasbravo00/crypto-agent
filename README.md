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
  state.py                -> stateful tool: DCA purchases + bullets, dual backend (see below)
  bullets.py              -> bullet state machine: open -> tracking -> closed_tp/closed_manual;
                             enforces "one bullet at a time" and the 30-bullet cycle cap in code
                             (also enforced at the DB level when using Supabase)
  db.py                   -> thin Supabase client wrapper (optional; see Setup)
  notify.py               -> output dispatcher: console / email / telegram
  email_notifier.py       -> SMTP delivery to your mailbox (e.g. Outlook)
  telegram_notifier.py    -> Telegram delivery (optional)
  agent.py                -> tool-calling loop on the Claude API
  agent_ollama.py         -> the SAME loop on a local model via Ollama
tests/
  test_logic.py           -> unit tests for market_data/strategy_tools (no network, no LLM)
  test_bullets.py         -> unit tests for the bullet state machine (isolated temp state,
                             Supabase force-disabled regardless of the ambient environment)
state/                    -> local JSON state, used only when Supabase isn't configured
logs/                     -> JSONL trace of every agent decision (auto-created)
supabase/schema.sql       -> run once in the Supabase SQL Editor to create the tables
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
- **Dual storage backend, chosen at call time**: if `SUPABASE_URL` /
  `SUPABASE_KEY` are set, DCA purchases and bullets live in Postgres
  (Supabase) instead of the local JSON file — this is what lets state be
  shared between your Mac and GitHub Actions. If unset, everything falls
  back to the JSON file exactly as in the original single-machine design.
  Tests force-disable Supabase regardless of the ambient environment, so
  they never touch the real database.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate   # isolated env for this project
pip install -r requirements.txt
cp .env.example .env    # then fill it in
```

A virtual environment matters here specifically because `supabase-py`
needs a `websockets` version that can conflict with other unrelated
Python projects on the same machine if installed globally.

- **Claude backend**: get an API key at https://console.anthropic.com and
  set `ANTHROPIC_API_KEY`.
- **Ollama backend**: install https://ollama.com, run
  `ollama pull llama3.1`, and set `LLM_BACKEND=ollama`.
- **Email delivery**: Outlook/Microsoft removed basic-auth SMTP sending
  (fully rejected since April 2026), so the report is sent *from* another
  SMTP provider (Gmail app password, or Brevo's free tier) *to* your
  mailbox. See `.env.example`. Or keep `NOTIFY_CHANNEL=console` for zero
  setup.
- **Supabase (optional)**: create a free project at https://supabase.com,
  run `supabase/schema.sql` once in its SQL Editor, then set
  `SUPABASE_URL` and `SUPABASE_KEY` (the `service_role` key) in `.env`.
  Skip this entirely to keep using the local JSON file — nothing else
  changes.

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

## Automation (macOS launchd)

Two LaunchAgents keep this running without manual invocation. They live
in `~/Library/LaunchAgents/` (system config, not part of this repo) and
write their run logs to `logs/launchd_*.out.log` / `.err.log`.

| Job | Schedule | Runs |
|---|---|---|
| `com.cryptoagent.dailyreport` | daily at 08:00 | `python main.py report` (full LLM report, delivered via `NOTIFY_CHANNEL`) |
| `com.cryptoagent.bulletcheck` | every 15 minutes | `python main.py bullet-check` (no LLM; only notifies on target/liquidation alerts) |

```bash
launchctl list | grep cryptoagent                                 # confirm both are loaded
launchctl kickstart -k gui/$(id -u)/com.cryptoagent.dailyreport   # force a run right now
launchctl bootout gui/$(id -u)/com.cryptoagent.bulletcheck        # disable a job
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.cryptoagent.bulletcheck.plist  # re-enable it
```

**Caveat**: with `LLM_BACKEND=ollama`, the Ollama app must be running in
the background for `dailyreport` to work — if the Mac is off (asleep is
fine; launchd catches up missed runs on wake) or Ollama isn't running at
08:00, check `logs/launchd_dailyreport.err.log` for the failure.

## Web dashboard

`dashboard/index.html` is a single, build-step-free web page (Supabase JS +
Chart.js from CDNs) that reads the Supabase data and shows status cards, a
yesterday-vs-today comparison, time-series charts (price, Fear & Greed,
Mayer Multiple, distance to the 200-week SMA), the DCA and bullet tables,
and the latest generated report. It only reads and renders — it never
places or suggests trades.

Security: the page uses the **anon** key (safe to expose — it's gated by
Row Level Security). Run `supabase/security.sql` once to enable RLS +
read-only policies, then in the Supabase dashboard disable public sign-ups
and create your single Auth user. Logged out, the anon key can read
nothing; only your account sees the data. Never put the `service_role`
key in the frontend.

Preview locally:

```bash
python3 -m http.server 4173 --directory dashboard   # then open http://localhost:4173
```

Deploy (free): host the `dashboard/` folder as a static site on Vercel or
Netlify (no build command; output/root directory = `dashboard`).

## Roadmap

1. **Multi-agent**: a market-analyst agent + a bullet-manager agent,
   coordinated.
2. **Memory**: compare against previous reports to detect regime changes.

## Disclaimer

This project is for educational and informational purposes. Nothing it
produces is financial advice. Leveraged trading carries a substantial
risk of loss: at x5 leverage an adverse price move of roughly 20% wipes
out the position's collateral (before fees and maintenance margin, which
make it happen sooner).
