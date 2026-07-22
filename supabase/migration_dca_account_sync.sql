-- Migration: DCA auto-sync from real BingX spot trades + account balance
-- tracking. Run ONCE in the Supabase SQL Editor if your project already
-- existed before this change (schema.sql already has this for fresh installs).

-- Dedupe key for auto-imported DCA purchases (src/dca.py's sync_with_bingx()).
alter table dca_purchases add column if not exists bingx_trade_id text unique;

-- New table: same cadence/purpose as price_ticks, but for your REAL BingX
-- spot wallet balance (BTC held + free/uninvested USDT).
create table if not exists account_ticks (
    id bigint generated always as identity primary key,
    created_at timestamptz not null default now(),
    btc_total numeric not null,
    usdt_free numeric not null
);

alter table account_ticks enable row level security;

create policy "authenticated can read account_ticks"
    on account_ticks for select to authenticated using (true);
