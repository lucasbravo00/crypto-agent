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
   a profitable round compounds into bigger bullets next round. WHEN
   within the day a bullet opens is timed by live 15-minute RSI (waits
   for oversold, falls back to end-of-day) — see "Intraday entry timing".
   Real money starts here in roughly 3 months; until then this runs
   entirely on demo/virtual funds to prove the strategy and the code out.

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
  memory.py               -> temporal context for the report: today's key market
                             indicators vs 1/7/30 days ago, with trend labels computed
                             here rather than left for the LLM to infer.
  backtest.py             -> offline simulation of the bullet strategy against real
                             historical candles, coin-margined (see "Backtesting").
                             Also compares intraday ENTRY TIMING (fixed daily-open vs
                             15m RSI-oversold vs an idealized day-low ceiling). Touches
                             no account and no state.
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
  test_memory.py            -> unit tests for the memory/trend comparisons
  test_backtest.py          -> unit tests for the backtest, incl. hand-derived inverse
                             contract math (synthetic candles, no network)
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
- **RSI entry timing (optional)**: `RSI_ENTRY_THRESHOLD` (default 25)
  controls how oversold live 15m RSI must get before `auto_trade()` opens
  the day's bullet — see `bullets.RSI_ENTRY_*` and "Intraday entry
  timing" below.

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

`python main.py backtest START_DATE END_DATE [BTC] [--detail]` replays the
round-based bullet mechanics against real historical daily candles (from
Binance — BingX's public API only reaches back to late 2023, not far
enough for the 2020-2021 cycle). It touches no account and no state.

**Coin-margined.** Collateral and P&L are in BTC, not USDT, because that
is how the strategy is actually traded. This is not a cosmetic
difference: `PnL_btc = collateral * leverage * (1 - entry/price)`, so BTC
gains are asymptotically capped at `collateral * leverage`, and
liquidation arrives *earlier* than linear intuition suggests — at 5x a
fully-deployed round liquidates at **-16.67%**, not -20%, because the
collateral itself loses USD value on the way down.

The date range is a required, deliberate input: this strategy is
long-only and was never meant to run through a bear market. Judging when
a bull market is underway is the trader's call, not the code's — the
same reason the live agent never emits buy/sell signals.

Each run compares six variants: target measured in **BTC** vs **USD**,
each with de-risking **off**, by **bullet_size** (smaller bullets as the
Mayer Multiple heats up), or by **withdraw** (moving BTC out of the
trading account entirely).

### What the runs showed (2026-07-26), starting from 1.0 BTC

| Cycle | de-risk `off` | de-risk `bullet_size` | de-risk `withdraw` |
|---|---|---|---|
| 2020-10 → 2021-11 | **-100%** | **-100%** | **+45.5%** (1.455 BTC) |
| 2023-01 → 2025-07 | **-100%** | **-100%** | **+69.4%** (1.694 BTC) |

Three findings, each verified against the raw candles rather than
trusted from a summary:

**1. Without de-risking the strategy always ends at zero.** Both cycles
ran dozens of consecutive winning rounds (63 and 58) and then gave it all
back to a single liquidation. Critically, neither wipeout happened at a
cycle top or in the bear market this strategy is designed to sit out —
both were *ordinary corrections inside a bull run*: 2021-05-19 (the May
crash) and 2025-02-28. Once all 30 bullets are deployed the round's
collateral equals the account balance, so the whole account is
effectively at 5x and a ~17% drop below the round's weighted average
entry ends it.

**2. Shrinking bullet size does not protect anything.** This was the
originator's stated mechanism, and the backtest says it fails: under
CROSS margin a liquidation takes the entire trading balance, not just the
collateral currently deployed. Smaller bullets leave the same balance
sitting in the account backing them, and it dies with the position —
`bullet_size` scored identically to `off` (-100%) in both cycles. The
stated *goal* ("so the liquidation won't consume all the gains so far")
is real and correct; only the mechanism doesn't achieve it under cross
margin.

**3. What does work is withdrawing BTC out of the account.** Same Mayer
Multiple signal, but instead of trading smaller, move the funds where a
liquidation cannot reach them (exposure targeted as a fraction of total
wealth, one-way ratchet, only between rounds). Both cycles then finish
positive *in BTC terms* — including 2023-2025, which still took its
liquidation but had already banked most of the gains by then.

**4. The BTC-denominated target beat the USD one** in both cycles (+69.4%
vs +40.2%, and +45.5% vs +35.5%). The USD target triggers earlier because
part of a USD gain comes from the collateral appreciating rather than
from the trade, so it books smaller wins and gives up more upside.

Caveat worth keeping in mind: the Mayer Multiple never exceeded **1.85**
in the 2023-2025 cycle — it never came close to the 2.4 that marked
previous tops, and the February 2025 liquidation happened with Mayer at
~1.03, i.e. a *cold* market. The de-risking signal is far from a top
detector; it worked here mostly by trimming exposure steadily, not by
calling the top.

### Reactive de-risking (`drawdown`) and why top-detection was abandoned

Measured against BTC's *real* cycle tops, every price-derived top
detector tried here failed:

| Real top | Price | Mayer | Mayer percentile | Weekly RSI |
|---|---|---|---|---|
| 2021-11-10 | $64,882 | 1.42 | 74% | 71 |
| 2025-10-06 | $124,659 | **1.18** | **49%** | **61** |

At the actual 2025 top the Mayer Multiple sat at the 49th percentile of
its own trailing two-year distribution — the median. Pi Cycle Top fired
once (2021-04-12, near a *local* top) and never again, including not at
either real top. A self-calibrating percentile rank fires ~320-355 days
early, at $18.6k and $94.3k respectively — you would de-risk through the
entire best part of the run. These indicators degraded as BTC matured
and its volatility compressed; thresholds calibrated on 2013-2017 no
longer get reached.

So `drawdown` mode does not try to predict anything: it simply stops
opening new bullets while price sits more than X% below its trailing
N-day high, and resumes when it recovers. Under cross margin this really
does cut liquidation risk, because the liquidation price rises with
every bullet added — freezing exposure freezes it too.

**Parameter sweep, both cycles, starting from 1.0 BTC:**

| Config | 2020-21 | 2023-25 | survived both |
|---|---|---|---|
| 30d / -5% | 2.103 | 6.638 | **yes** |
| 90d / -5% | 1.837 | 4.185 | **yes** |
| 180d / -5% | 1.605 | 3.363 | **yes** |
| 60d / -10% | 0.000 | 13.098 | no |
| 90d / -10% | 0.000 | 11.179 | no |
| 60d / -20% | 4.367 | 0.000 | no |
| *(10 other combos)* | | | no |

**Read this table carefully, because the headline numbers are a trap.**
The best single result (13.098 BTC, 60d/-10%) is a total wipeout in the
other cycle. So is 60d/-20%, the 2020-21 winner. Only 3 of 16
combinations survived both cycles, and all three are the *tightest*
brake (-5%) — the threshold matters far more than the lookback. Picking
a config by peak return is curve-fitting to one crash.

Two caveats that keep this honest: there are exactly **two** liquidation
events in the whole sample (n=2), which is nowhere near enough to
separate skill from luck; and whether a brake saves a given round comes
down to whether it happened to be engaged on the specific day of a
specific crash. Treat "-5% is robust" as a weak prior, not a result.

A lookahead-bias bug was found and fixed while building this (the
trailing high originally included the current day's high, which is not
knowable at the daily open when the decision is made). The table above
is post-fix.

### Intraday entry timing (`backtest-timing`)

Until 2026-07-28 the live bot bought its one daily bullet at whatever
price BTC happened to be right after midnight UTC — an arbitrary moment,
not a chosen one. `python main.py backtest-timing` was built to ask
whether waiting for a better intraday price would help, holding
everything else fixed (target=btc, de-risk=off) so only entry timing
varies. `rsi_oversold` came out ahead (see the results below) and is now
what `auto_trade()` actually does live — this section is both the
backtest tool's docs and the record of why that choice was made.

- **`fixed`** — the OLD live behavior, kept as the baseline everything
  else is measured against: buy at the daily open.
- **`rsi_oversold<N`** — the CURRENT live behavior: wait for 15-minute
  RSI to drop below N that day, buy at that candle's close; if it never
  fires, buy at the day's last close anyway (the one-bullet-per-day
  cadence is never skipped). REALISTIC: only uses information available
  at the moment of the decision, same as the next one.
- **`bollinger_lower<S>`** — same idea, different math: wait for price to
  close below the lower Bollinger Band (20-period SMA minus S standard
  deviations), a volatility-based signal instead of RSI's momentum-based
  one. Same end-of-day fallback.
- **`day_low`** — buys at the day's actual lowest price. NOT a real
  strategy — only knowable in hindsight — included purely as an
  idealized ceiling: the most any same-day timing approach could
  possibly have captured.

**The two cycles disagree, which is itself the finding:**

| Cycle | fixed | RSI\<20 | RSI\<25 | RSI\<30 | RSI\<35 | Boll 1.5sd | Boll 2.0sd | Boll 2.5sd | day_low (ceiling) |
|---|---|---|---|---|---|---|---|---|---|
| 2020-10 → 2021-11 | **-100%** (liq) | **+583%** | +580% | +578% | -100% (liq) | -100% (liq) | -100% (liq) | -100% (liq) | +600% |
| 2023-01 → 2023-12 | +219% | +198% | +198% | +164% | +219% | +198% | +197% | +197% | +436% |

In 2020-2021, RSI timing at the tighter thresholds (20/25/30) avoided
the exact liquidation that `fixed` suffered — worth understanding *why*,
not just that it did: only buying on intraday dips pulls the round's
*weighted average entry price* down, which pulls the liquidation price
down with it, buying more room before the same crash reaches it. RSI<35
barely filters anything (it triggers on almost any red candle), stayed
close to `fixed`, and got liquidated too — consistent with that
mechanism, not contradicting it.

**The Bollinger result is the more interesting one: it did NOT avoid the
same liquidation, at any of the three widths tried.** Two "oversold"
signals, same crash, opposite outcomes. Volatility itself was rising
sharply through that decline (the band was widening as fast as price was
falling), so a band-based trigger stayed permissive when RSI — a bounded
momentum oscillator, indifferent to how wide the recent range has become
— did not. This is exactly why both were worth testing rather than
picking one on reputation: "an oversold indicator" is not a
interchangeable category, the specific mechanism determines whether the
lower-average-entry effect actually shows up.

In 2023, entry timing made close to no difference either way — every
realistic variant landed within a few points of `fixed`. No crash to
dodge that year, so all timing did was occasionally delay entries into a
market that mostly went up anyway — a cost with little corresponding
benefit, and no signal came out ahead of simply buying at the open.

**Read this the same way as the drawdown table above: two cycles is not
enough to call this settled.** The 2020-2021 result is a genuine,
mechanistically-explained effect, not noise — but it's one liquidation
event, avoided by one technique (RSI) and not another (Bollinger), in
one crash. Whether either helps next time depends on whether there's a
crash to avoid at all, and on which flavor of "oversold" happens to
match how that particular crash unfolds.

### Which candle size for the RSI? (`backtest-rsi-timeframe`)

A follow-up question once RSI was chosen: what timeframe should it be
computed on? `python main.py backtest-rsi-timeframe` fetches 5-minute
candles once and builds every other size by aggregating them (Binance
has no native 10m/45m interval), holding the threshold fixed at 25 so
only candle size varies:

| Timeframe | 2020-2021 | 2023 |
|---|---|---|
| 5m | **-100% (liquidated)** | +195.9% |
| 10m | +579.1% | +182.7% |
| 15m | +578.8% | +197.4% |
| 20m | +583.4% | +218.9% |
| 30m | +584.9% | +220.4% |
| 45m | +585.9% | +198.6% |
| 60m | +586.4% | +221.0% |

**5m is the one to avoid, clearly — not a matter of degree.** RSI on
candles that short is too noisy to filter anything: it crosses below 25
on ordinary micro-fluctuations, so it ends up firing almost as readily
as `fixed` does, and it got liquidated in the exact same crash `fixed`
did. Everything from 10m to 60m landed in a flat, tightly-clustered
plateau (±8 points of BTC return in each cycle) — no candle size in that
range stood out enough to prefer on this evidence. Live entry timing
uses **1h**, the top of that plateau, picked as a reasonable default
rather than a measured winner.

Documented simplifications (see `src/backtest.py`): liquidation is
modeled as round equity hitting zero, which is *optimistic* — real
exchanges liquidate earlier by holding maintenance margin. When a single
day's range spans both the liquidation price and the target, liquidation
is assumed to have happened first. Funding rates are not modeled; the
real 0.05% taker fee is.

## Division of labour: what the bot decides, and what it must not

An explicit design boundary, decided 2026-07-27, refined 2026-07-28:

**The bot owns the mechanical part.** *Whether* to open a bullet (one per
calendar day, sized at `balance / 30`, at 5x, in cross margin) and
*when within the day* to time that entry, plus closing the round when
the combined target is reached. Both are well-defined rules, not
judgment calls, so both are safe to automate.

**The human owns the strategy's arc.** *How much* to expose as the cycle
matures — when to scale down, when to stop opening altogether, when to
pull profit out, when to walk away. Those depend on reading momentum and
market context, a judgment call, and stay manual — informed by the daily
report and the dashboard, decided by the user.

Concretely:

- **Entry timing is LIVE** (since 2026-07-28): `bullets.auto_trade()`
  waits for live RSI(14) on `RSI_ENTRY_TIMEFRAME` candles (1h) to drop
  below `RSI_ENTRY_THRESHOLD` (env var, default 25) before opening the
  day's bullet, falling back to opening anyway at 23:45 UTC if the
  signal never fires that day — see `market_data.get_intraday_rsi()` and
  `bullets.RSI_ENTRY_*`. Backtested first against both bull cycles
  (`backtest-timing`, then `backtest-rsi-timeframe` for the candle size
  itself — 5m turned out badly broken, too noisy to filter anything;
  everything 10m-60m landed in a flat, statistically-indistinguishable
  plateau, and 1h was picked as the top of that safe range) before going
  live: see "Intraday entry timing"
  below for the comparison, including the honest caveat about how thin
  that evidence actually is (one real liquidation event survived).
- **Position sizing / de-risking stays BACKTESTING-ONLY, on purpose.**
  `src/backtest.py`'s `derisk_mode` (Mayer-based sizing, withdrawals,
  drawdown brake) decides *how much* exposure to carry as a cycle
  matures — that is the judgment call that stays with the human, not
  the bot. Verified rather than assumed: `MAYER_FULL_SIZE`, `MAYER_STOP`,
  `DRAWDOWN_*`, `_size_factor()` and `derisk_mode` appear nowhere in
  `bullets.py`, `bingx_client.py`, or the `bullet-check` flow.
  `backtest.py` imports neither `state` nor `db` and never calls
  `create_order`, so it cannot write state or touch an exchange even by
  accident.

If automated de-risking is ever wanted, it must be a deliberate,
separately-requested feature — the same way entry timing became one, not
something that quietly grows out of the backtester.

## Roadmap

1. **Watch how live RSI entry timing actually performs**: it's only been
   live since 2026-07-28, backed by real but thin backtest evidence (one
   real liquidation event survived, out of a sample of two bull cycles —
   see "Intraday entry timing"). Worth tracking on the demo account
   before trusting it, and revisiting `RSI_ENTRY_THRESHOLD` if live
   behavior diverges from what the backtest suggested.
2. **Live-verify the auto-close path**: `auto_trade()`'s branch that
   closes a round when the combined +15% target is actually reached in a
   real, automatic `bullet-check` cycle has only been unit-tested and
   manually invoked once (an emergency cleanup) — worth watching for
   naturally or testing deliberately.
3. **Cross-margin liquidation display**: `strategy_tools.simulate_bullet_math()`
   still reports a per-bullet `approx_liquidation_price` using isolated,
   USDT-margined math. Under cross margin with BTC collateral the real
   liquidation point is a function of the whole round vs. account equity
   (the backtest already models this correctly; the live display does not).
4. **Surface backtest context in the daily report**: the human makes the
   de-risking calls, so the report is where that judgment gets informed —
   e.g. showing how deep the current round is into its 30-bullet budget
   and how far price sits below its recent highs. Data for a decision,
   never the decision itself.
5. **Telegram bidirectional bot**: respond to commands from the chat
   (e.g. `/bullet-open`) talking directly to Supabase, same pattern as
   the dashboard.

Done: BingX demo auto-trading, DCA auto-sync from real BingX trades,
round-based bullet accounting, reconciliation hardening, auto-trade
notifications, real exchange fees in P&L, the web dashboard (including
market-cycle charts), report memory (1/7/30-day trend context),
Predictive Ranges, the coin-margined backtester, and live RSI-timed
bullet entries.

## Disclaimer

This project is for educational and informational purposes. Nothing it
produces is financial advice. Leveraged trading carries a substantial
risk of loss: at x5 leverage an adverse price move of roughly 20% wipes
out the position's collateral (before fees and maintenance margin, which
make it happen sooner).
