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
    asset text not null default 'BTC'
);

create table if not exists bullets (
    id bigint generated always as identity primary key,
    bullet_number int not null,              -- position in the 30-bullet cycle (1..30)
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
    notes text
);

-- DB-level enforcement of "one bullet at a time": a partial unique index on
-- a constant expression means at most one row can have an active status.
-- Mirrors the application-level check in src/bullets.py -- belt and suspenders.
create unique index if not exists bullets_one_active_idx
    on bullets ((true))
    where status in ('open', 'tracking');

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
