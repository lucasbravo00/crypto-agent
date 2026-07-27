# Crypto Strategy Agent

An LLM tool-calling agent that builds an objective daily report on BTC
market context, your real DCA accumulation, and your leveraged "bullet"
strategy — delivered to your mailbox (e.g. Outlook), Telegram, or the
console.

The agent never trades and never gives buy/sell signals for real money:
it gathers data, summarizes it, and leaves every decision to the human
(deliberate human-in-the-loop design). The one deliberate, narrowly
scoped exception is `bullets.auto_trade()`, which can place orders
**only** on BingX's Demo Trading (VST/virtual funds) account, and only
if you explicitly opt in — see "BingX integration" below. The real
strategy this project supports, BTC only, runs as two parallel legs:

1. **DCA (real money, ongoing)**: weekly BTC spot purchases on BingX.
   Auto-imported from your real trade history (`dca-sync`) — nothing to
   type by hand. Between buys, the BTC is rotated out to Nexo for yield,
   so BingX's spot wallet itself isn't a meaningful "money in account"
   figure; current DCA value is computed from the purchase history
   directly, wherever the BTC physically sits.
2. **Bullets (currently testing on BingX Demo/VST)**: x5 leveraged
   futures positions, at most one NEW one per calendar day, accumulating
   within a **round**. The +15% target is evaluated on the **combined**
   position across every currently active bullet, not per bullet; when
   it's hit, ALL active bullets close together, ending the round. Each
   round has its own fresh 30-bullet budget (`MAX_BULLETS_PER_ROUND`) —
   this resets every round, it is not a lifetime cap. Bullet size
   auto-computes as `(BingX balance / 30)` at the start of each round, so
   a profitable round compounds into bigger bullets next round. Real
   money starts here in roughly 3 months; until then this runs entirely
   on demo/virtual funds to prove the strategy and the code out.

## Architecture

```
main.py                  -> CLI entry point; picks LLM backend and output channel
src/
  market_data.py          -> read-only tools: price, daily indicators,
                             cycle metrics (200w SMA, Mayer Multiple, drawdown,
                             weekly RSI), fear & greed, BTC dominance
                             (BingX/CoinGecko), all with retry logic
  strategy_tools.py       -> pure math of a leveraged bullet position (no network)
  state.py                -> stateful tool: DCA purchases + bullets + account/price
                             ticks, dual backend (see below)
  bullets.py              -> bullet state machine: open -> tracking -> closed_tp/closed_manual,
                             organized into ROUNDS (see above). Two separate ids per
                             bullet: "id" (globally unique, safe update key) vs
                             "bullet_number" (resets every round, display-only).
                             sync_with_bingx() reconciles bullets against REAL BingX
                             trade history; reconcile_with_bingx() is a read-only,
                             independent second check that what we THINK is active
                             still matches BingX's real position (see Design decisions).
                             auto_trade() is the only function that can place a real
                             (demo-only) order.
  dca.py                  -> sync_with_bingx() imports real spot BTC buy fills from
                             your REAL BingX account into dca_purchases, deduped by
                             trade id -- no manual entry needed once it's set up.
  bingx_client.py         -> BingX access via ccxt, split into two clearly separated
                             halves: (1) DEMO/sandbox trading -- hard-coded, no code
                             path to your live account, can place orders (used only by
                             bullets.auto_trade()); (2) REAL account, READ-ONLY -- used
                             only to import real DCA trade history/balance, never
                             places an order.
  db.py                    -> thin Supabase client wrapper (optional; see Setup)
  notify.py                -> output dispatcher: console / email / telegram
  email_notifier.py        -> SMTP delivery to your mailbox (e.g. Outlook)
  telegram_notifier.py     -> Telegram delivery (optional)
  agent.py                 -> two coordinated sub-agents (Market Analyst,
                             Portfolio Manager) on the Claude API
  agent_ollama.py           -> the SAME two-sub-agent design on a local
                             model via Ollama
tests/
  test_logic.py            -> unit tests for market_data/strategy_tools (no network, no LLM)
  test_bullets.py          -> unit tests for the bullet state machine, including sync
                             and reconciliation (isolated temp state, Supabase/BingX
                             force-disabled regardless of the ambient environment)
  test_dca.py               -> unit tests for DCA sync against (mocked) BingX
state/                     -> local JSON state, used only when Supabase isn't configured
logs/                      -> JSONL trace of every agent decision (auto-created)
dashboard/index.html       -> single-file web dashboard (see "Web dashboard" below)
supabase/schema.sql        -> run once in the Supabase SQL Editor to create the tables
supabase/migration_*.sql   -> incremental migrations, run in order if your project
                             predates a given feature (each file explains why it exists)
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
  you try to open a second bullet on the same calendar day, or exceed
  the round's 30-bullet cap. Prompt wording alone ("you MUST call every
  tool") was tested and found insufficient, even on Claude.
- **Rounds, not a lifetime cap**: the strategy allows bullets to
  accumulate (one new one/day) up to 30 within a round; when the
  combined position hits +15%, ALL active bullets close together and a
  **new** round starts from bullet 1 with a fresh 30-bullet budget. This
  required separating each bullet's globally-unique `id` (the only safe
  key for updates) from its `bullet_number` (position within the current
  round — resets every round, so two different bullets from two
  different rounds can legitimately share one). Getting this wrong once,
  before the split existed, would have let `update_bullet()` silently
  corrupt a closed bullet from an earlier round that happened to share a
  `bullet_number` with a currently-active one.
- **CROSS margin, not isolated**: the 30-bullet-per-round budget is meant
  to be a reserve, rarely if ever actually reached — with ISOLATED
  margin that reserve was pointless, since each bullet's own collateral
  is its own liquidation wall regardless of how much unused budget
  exists (a ~20% adverse move at 5x wipes a bullet out on its own).
  Switched to CROSS (2026-07-25, after round 2 lost $1,037 to exactly
  this): now the whole account's collateral backs every open bullet
  together, so unused budget is real headroom — a drawdown deeper than
  20% can be absorbed by adding bullets at better prices instead of
  being liquidated bullet-by-bullet. `bingx_client.open_long_position()`
  forces cross mode before a round's first bullet (the only moment
  BingX allows changing it — it refuses mid-round, while flat with
  contracts open). Known follow-up, not yet done:
  `strategy_tools.simulate_bullet_math()`'s `approx_liquidation_price`
  still assumes isolated math per bullet; under cross margin the real
  liquidation point is a function of the whole round's combined
  position vs. total account equity, not any single bullet in
  isolation — that display is now illustrative, not literal.
- **Sync from trade history, not positions**: BingX MERGES same-symbol,
  same-side positions into one row with an averaged entry price (opening
  a position and later adding to it keeps the same `positionId`).
  Confirmed empirically against a real demo account before trusting it.
  Individual bullets are indistinguishable from `fetch_positions()`, so
  `sync_with_bingx()` reads `fetch_my_trades()` instead — each fill keeps
  its own order id, price, size and timestamp, which is what lets it
  become its own bullet, linked via a unique `bingx_order_id` column so
  syncing twice never double-creates one.
- **Reconciliation is a SEPARATE check from sync, on purpose**: a real
  bug (2026-07-24) showed that `sync_with_bingx()` completing without an
  exception is not proof it did the right thing — a SELL fill's order id
  was never persisted, so every later sync re-processed that same old
  fill and silently re-closed bullets that had opened well after it,
  using its stale price/timestamp. Fixed by persisting
  `bingx_close_order_id` too, but also by adding
  `reconcile_with_bingx()`: an independent, read-only check (run every
  `bullet-check`) that re-derives the BTC amount implied by whatever
  bullets are currently marked active and compares it against BingX's
  own reported open position, notifying a human on any mismatch instead
  of assuming sync's bookkeeping is correct. Lesson generalized: don't
  trust a write path's own success signal to also mean the resulting
  state is correct — verify independently, especially before real money
  is involved.
- **BingX integration, split into two independent halves**: BingX uses
  the SAME API key for demo (VST/virtual funds) and live trading — the
  only thing separating them is which base URL a request hits, so the
  key itself gives no guarantee. `bingx_client.py` hard-codes sandbox
  mode for everything related to bullets, with no parameter or env var
  able to turn it off — the ONLY function anywhere that can place an
  order is `bullets.auto_trade()`, gated by an explicit, off-by-default
  `BINGX_AUTO_TRADE_ENABLED` env var on top of that hard-coded demo lock
  (two independent gates have to agree). Separately, a small **real,
  read-only** half (`get_real_spot_trades`, `get_real_spot_balance`)
  exists only to import DCA purchase history — by convention no function
  built on that real client may ever call `create_order`.
- **`test` order param is NOT a safe dry-run**: confirmed empirically
  (2026-07-22) that BingX's own "validates without executing" endpoint
  parameter can still execute a real demo order. Documented prominently
  in `bingx_client.py`; never assumed to be risk-free again.
- **Auto-sized bullets, compounding by design**: if `collateral_usd` is
  omitted, `open_bullet()` computes it from your real BingX balance
  divided by 30 at the start of each round (reusing the same size for
  every bullet within that round). A round that closes positive grows
  the account balance, so the next round's division starts from a bigger
  number — this is what lets `auto_trade()` size a bullet with no human
  typing an amount.
- **Multi-agent report**: `run_daily_report()` coordinates two independent
  sub-agents — a Market Analyst (price/indicators/cycle/sentiment tools
  only) and a Portfolio Manager (DCA/bullet tools only), each with its
  own prompt and `REQUIRED_TOOLS` set — then concatenates their text. No
  synthesis LLM call: cheaper, faster, one fewer failure point, at the
  cost of two sections instead of one fused narrative. Caught in testing:
  giving the market analyst fewer, more focused tools didn't stop the
  local Ollama model from once fabricating an indicator we never gave it
  (MACD) and once mislabeling BTC dominance as BNB's — fixed with an
  explicit "only report what a tool actually returned" rule in the prompt.
- **Retries with backoff** on all network calls (public APIs fail
  sporadically).
- **Structured JSONL logging** of every tool call, its input and its
  result, so any report can be audited after the fact.
- **Human-in-the-loop by design**: the agent informs; the ONLY code path
  that can execute anything against real infrastructure is the
  demo-only, opt-in `auto_trade()` — everything touching real money
  (DCA, the report, the dashboard) is purely read-only or requires a
  human to type a command.
- **Dual storage backend, chosen at call time**: if `SUPABASE_URL` /
  `SUPABASE_KEY` are set, all state (DCA purchases, bullets, price/account
  ticks) lives in Postgres (Supabase) instead of the local JSON file —
  this is what lets state be shared between your Mac and GitHub Actions.
  If unset, everything falls back to the JSON file. Tests force-disable
  Supabase and BingX regardless of the ambient environment, so they never
  touch real data.

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
- **Report language**: set `REPORT_LANGUAGE` (ISO code, e.g. `en`, `es`).
- **Email delivery**: Outlook/Microsoft removed basic-auth SMTP sending
  (fully rejected since April 2026), so the report is sent *from* another
  SMTP provider (Gmail app password, or Brevo's free tier) *to* your
  mailbox. See `.env.example`. Or keep `NOTIFY_CHANNEL=console` for zero
  setup.
- **Supabase (optional)**: create a free project at https://supabase.com,
  run `supabase/schema.sql` once in its SQL Editor, then set
  `SUPABASE_URL` and `SUPABASE_KEY` (the `service_role` key) in `.env`.
  Skip this entirely to keep using the local JSON file — nothing else
  changes. If your project predates a feature, run the relevant
  `supabase/migration_*.sql` file(s) in order (each explains what it's
  for and why).
- **BingX (optional, one API key covers both halves)**: generate an API
  key at BingX -> User Center -> API Management, then set
  `BINGX_API_KEY` / `BINGX_API_SECRET` in `.env`. This single key is used
  for both the demo bullets integration (`bingx-*` commands,
  `bullet-check`'s sync/reconcile/auto-trade) and the real, read-only DCA
  sync (`dca-sync`). Bullet auto-trading additionally requires
  `BINGX_AUTO_TRADE_ENABLED=true` (off by default).

## Usage

```bash
# --- DCA (real money) ---
python main.py dca-sync          # import real BTC/USDT spot buys from BingX (read-only)
python main.py buy 50            # manually record a 50 USD DCA purchase instead
python main.py dca               # DCA summary (no LLM)
python main.py metrics           # all raw market/cycle metrics (no LLM)
python main.py bullet 500        # math of a 500 USD x5 bullet at the current price (no state)

# --- Bullets: tracking + auto-trading leveraged positions on BingX Demo ---
# Bullets ACCUMULATE within a round (at most one new one per day). The
# +15% target and close apply to the COMBINED currently-active set, and
# closing ends the round -- the next one starts over with a fresh
# 30-bullet budget.
python main.py bullet-open               # auto-size today's new bullet (BingX balance / 30) at the CURRENT price
python main.py bullet-open 500           # override with an explicit 500 USD instead
python main.py bullet-open 500 60000 5 15 # same, explicit entry / leverage / target %
python main.py bullet-status             # live COMBINED P&L of all active bullets (no notification)
python main.py bullet-check              # runs the FULL 15-min cycle: price tick, auto-trade,
                                          # bullet sync, reconciliation, DCA sync, account tick,
                                          # alerts if combined target hit / near liquidation
python main.py bullet-close tp           # close ALL active bullets at the CURRENT price (outcome tp/manual)
python main.py bullet-history            # this round's progress (out of 30) + lifetime totals

# --- BingX Demo Trading (VST) ---
python main.py bingx-positions           # your REAL open positions (merged by BingX, see Design decisions)
python main.py bingx-balance             # your VST (virtual funds) balance
python main.py bingx-sync                # reconcile bullets against REAL trade history (also runs inside bullet-check)
python main.py bingx-reconcile           # independent check: does active bullet state match BingX's real position?
python main.py bingx-auto-trade          # places a REAL (demo) order if one is due today
                                          # (also runs inside bullet-check, but ONLY if
                                          # BINGX_AUTO_TRADE_ENABLED=true)

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
| `com.cryptoagent.bulletcheck` | every 15 minutes | `python main.py bullet-check` (no LLM; price tick, auto-trade, bullet sync + reconciliation, DCA sync, account tick — only notifies on alerts/mismatches) |

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
Chart.js from CDNs) that only reads and renders — it never places or
suggests trades. It shows:

- Status cards: an estimated total ("real DCA value + demo BingX
  balance", clearly labeled since one leg is virtual money), live BTC
  price, DCA value/gain (computed from `dca_purchases` directly — not
  from wherever the BTC physically sits, since it rotates to Nexo for
  yield between buys), the demo VST balance, and the current round's
  bullet usage/P&L.
- A yesterday-vs-today comparison.
- **Market-cycle charts, independent of how long this agent has been
  running**: full BTC price history since 2017 (log scale, via Binance's
  public klines endpoint — CoinGecko's free tier caps history at 365
  days, not enough for a 200-week moving average), Mayer Multiple and
  200-week SMA distance computed from that same series, and the Fear &
  Greed Index since 2018 (via alternative.me). Plus a "very recent"
  section: 15-min-resolution price and BTC dominance (the latter only as
  long as this agent has been running — no free historical source for
  dominance was found).
- The DCA and bullets tables (bullets show round/bullet number and
  whether each was opened manually or synced from a real BingX order),
  and the latest generated report text.

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

## Backtesting

`python main.py backtest START_DATE END_DATE [INITIAL_BALANCE]` replays
the round-based bullet mechanics against real historical daily candles
(from Binance — BingX's public API only reaches back to late 2023, not
far enough for the 2020-2021 cycle). It touches no account and no state.

The date range is a required, deliberate input: this strategy is
long-only and was never meant to run through a bear market. Judging when
a bull market is underway is the trader's call, not the code's — the
same reason the live agent never emits buy/sell signals.

**What the first runs showed (2026-07-26).** Across both bull cycles the
mechanics compound impressively and then lose everything to a single
event:

| Range | Rounds won | Result |
|---|---|---|
| 2020-10-01 → 2021-11-08 | 66 straight | $10k → $28.4k, then **liquidated** (2021-06-08, the May-2021 crash) |
| 2023-01-01 → 2025-07-24 | 88 straight | $10k → $191k, then **liquidated** (2025-02-28) |
| 2023 only | 41 straight | $10k → $32k (+220%), round still open at year end |
| 2024 only | 43 straight | $10k → $41.6k (+316%), round still open at year end |

The failure mode is the same both times, and it is NOT the bear market
the strategy is designed to sit out: once all 30 bullets are deployed,
the round's total collateral equals the account balance, so the whole
account is effectively at 5x — and a ~20% drop below the round's
*weighted average entry* liquidates everything. Verified against the raw
candles rather than trusted from the summary: the 2025 round opened
2025-01-22 at ~$106k, averaged ~$99.5k across 30 bullets, and BTC's
2025-02-28 low of $78,258 is almost exactly 20% below that. Both wipeouts
happened *inside* bull markets, during ordinary corrections.

Note also the max drawdown even in the winning years (-71% in 2024, -76%
in 2023): that is unrealized drawdown *within* rounds, i.e. what the
account looks like mid-round before a target is reached.

Documented simplifications (see `src/backtest.py`): liquidation is
modeled as round equity hitting zero, which is *optimistic* — real
exchanges liquidate earlier by holding maintenance margin. When a single
day's range spans both the liquidation price and the target, liquidation
is assumed to have happened first. Funding rates are not modeled; the
real 0.05% taker fee is.

## Roadmap

1. **Live-verify the auto-close path**: `auto_trade()`'s branch that
   closes a round when the combined +15% target is actually reached in a
   real, automatic `bullet-check` cycle has only been unit-tested and
   manually invoked once (an emergency cleanup) — worth watching for
   naturally or testing deliberately.
2. **Memory**: compare against previous reports (now stored in
   `daily_snapshots`) to detect regime changes.
3. **Telegram bidirectional bot**: respond to commands from the chat
   (e.g. `/bullet-open`) talking directly to Supabase, same pattern as
   the dashboard.
4. **Backtesting**: simulate the bullet cycle against historical BingX
   data.

Done: BingX demo auto-trading, DCA auto-sync from real BingX trades,
round-based bullet accounting, reconciliation hardening, auto-trade
notifications, the web dashboard (including market-cycle charts).

## Disclaimer

This project is for educational and informational purposes. Nothing it
produces is financial advice. Leveraged trading carries a substantial
risk of loss: at x5 leverage an adverse price move of roughly 20% wipes
out the position's collateral (before fees and maintenance margin, which
make it happen sooner).
