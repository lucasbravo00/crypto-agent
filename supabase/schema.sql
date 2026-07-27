-- crypto-agent state schema
-- Run this once in the Supabase SQL Editor (Dashboard -> SQL Editor -> New query).
--
-- Security note: these tables are created WITHOUT Row Level Security (RLS).
-- That's intentional for now: the only client that talks to Supabase is
-- this project's own backend code (your Mac + GitHub Actions), always using
-- the "service_role" key, which bypasses RLS anyway. If a public dashboard
-- is ever built on top of this data using the "anon" key, RLS policies
-- MUST be added before that happens -- otherwise anyone with the anon key
-- could read/write this data.

create table if not exists dca_purchases (
    id bigint generated always as identity primary key,
    purchased_at timestamptz not null default now(),
    amount_usd numeric not null check (amount_usd > 0),
    price numeric not null check (price > 0),
    asset text not null default 'BTC',
    -- BingX spot trade id, set when this purchase was auto-imported by
    -- src/dca.py's sync_with_bingx() instead of typed in by hand via
    -- `main.py buy`. NULL for manually-recorded purchases. Unique so
    -- syncing twice never double-imports the same real trade.
    bingx_trade_id text unique,
    -- Informational only: the real fee's USD-equivalent value. `price`
    -- above has ALREADY been adjusted to fold the fee into the cost
    -- basis (see dca.py's _fee_adjusted()) -- this column is for
    -- display/audit, never added again anywhere.
    fee_usd numeric
);

create table if not exists bullets (
    id bigint generated always as identity primary key,
    bullet_number int not null,              -- position WITHIN its round (1..30, resets each round)
    round_number int,                        -- which accumulation round this bullet belongs to
    status text not null check (status in ('open', 'tracking', 'closed_tp', 'closed_manual')),
    collateral_usd numeric not null check (collateral_usd > 0),
    entry_price numeric not null check (entry_price > 0),
    leverage numeric not null default 5,
    target_position_gain_pct numeric not null default 15,
    position_size_usd numeric not null,
    target_price numeric not null,
    approx_liquidation_price numeric not null,
    opened_at timestamptz not null default now(),
    closed_at timestamptz,
    closing_price numeric,
    outcome text check (outcome in ('tp', 'manual')),
    realized_pnl_usd numeric,
    notes text,
    -- BingX order ID that created this bullet, when opened via the
    -- trade-history sync (src/bullets.py's sync_with_bingx()) instead of
    -- manual bullet-open. NULL for manually-recorded bullets. Unique so
    -- syncing twice never creates a duplicate bullet for the same order.
    bingx_order_id text unique,
    -- BingX order ID of the SELL fill that closed this bullet, when
    -- closed via sync_with_bingx(). NOT unique (every bullet closed in
    -- the same round-ending sell shares this value) -- but critical:
    -- without persisting it, that same historical sell fill gets
    -- replayed on every later sync and wrongly re-closes whatever
    -- bullets happen to be active at that later point (fixed 2026-07-24).
    bingx_close_order_id text,
    -- REAL exchange fees, read from BingX's own trade record by
    -- sync_with_bingx() and subtracted from P&L. entry_fee_usd is this
    -- bullet's own opening fee; exit_fee_usd is its share of the whole
    -- round's single closing fee, split by collateral weight (see
    -- close_all_active_bullets()). Both NULL for manually-recorded
    -- bullets -- there's no real fill to read a fee from.
    entry_fee_usd numeric,
    exit_fee_usd numeric
);

-- NOTE: there is deliberately NO uniqueness constraint limiting how many
-- bullets can be active at once. The real strategy allows one NEW bullet
-- per calendar day to accumulate on top of previously-opened ones; that
-- guardrail (one open per day, not one active at a time) lives in code,
-- in src/bullets.py's open_bullet(), not in the schema.

create table if not exists daily_snapshots (
    id bigint generated always as identity primary key,
    created_at timestamptz not null default now(),

    -- market data at the time of the report
    price numeric,
    change_24h_pct numeric,
    sma50 numeric,
    sma200 numeric,
    rsi14 numeric,
    sma200w numeric,
    mayer_multiple numeric,
    drawdown_from_high_pct numeric,
    weekly_rsi14 numeric,
    fear_greed_value int,
    fear_greed_classification text,
    btc_dominance_pct numeric,

    -- DCA summary at the time of the report
    total_invested_usd numeric,
    total_qty_btc numeric,
    avg_entry_price numeric,

    -- bullet cycle summary at the time of the report
    bullets_used int,
    bullets_remaining int,
    tp_wins int,
    total_realized_pnl_usd numeric,

    -- the full LLM-generated report text, for a real historical record
    report_text text
);

-- Lightweight, high-frequency price history for the dashboard's price
-- chart. Populated by `python main.py bullet-check` (runs every 15 min
-- via launchd on your Mac) -- no separate scheduler needed. Independent
-- of daily_snapshots, which only gets one row/day from the full report.
create table if not exists price_ticks (
    id bigint generated always as identity primary key,
    created_at timestamptz not null default now(),
    price numeric not null,
    change_24h_pct numeric
);

-- Same cadence/purpose as price_ticks, but for your BingX DEMO (VST)
-- account's total balance -- the capital being used to test the bullets
-- strategy. (NOT the real spot wallet: that one is emptied out weekly by
-- design, since BTC gets rotated to Nexo for yield between DCA buys --
-- current DCA value is computed directly from dca_purchases instead.)
create table if not exists account_ticks (
    id bigint generated always as identity primary key,
    created_at timestamptz not null default now(),
    vst_total numeric not null
);
