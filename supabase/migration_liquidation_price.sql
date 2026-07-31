-- Migration: account_ticks also stores BingX's OWN liquidation price for
-- the open BTC-USDT position, so the dashboard can anchor the negative
-- end of the round-progress bar to the real liquidation instead of an
-- arbitrary -15%.
--
-- Why it has to come from the exchange and cannot be computed here:
-- the strategy runs in CROSS margin, where the liquidation price depends
-- on the WHOLE account balance, not on the position's own collateral.
-- bullets.approx_liquidation_price still models the isolated case and is
-- NOT usable for this (see the Roadmap item in README.md). Measured
-- 2026-07-31 with 3 open bullets: the isolated approximation sat around
-- $50k while BingX's real cross-margin figure was $414.70, because
-- ~109k VST of free balance was backing the position.
--
-- Nullable on purpose: with no open position there is no liquidation
-- price, and the dashboard falls back to its previous behaviour.
--
-- Run this once in the Supabase SQL Editor if your project predates it.

alter table account_ticks add column if not exists liquidation_price numeric;

comment on column account_ticks.liquidation_price is
    'BingX-reported cross-margin liquidation price for the open BTC-USDT position; null when flat.';

-- Reload PostgREST's schema cache. WITHOUT THIS the ALTER TABLE above
-- appears to have worked -- `select *` returns the new column, because
-- that query goes straight to Postgres -- while every INSERT that
-- mentions it fails with:
--
--   PGRST204: Could not find the 'liquidation_price' column of
--             'account_ticks' in the schema cache
--
-- because PostgREST validates INSERT payload keys against its own cached
-- schema. The failure is silent from the app's side: main.py catches it,
-- prints a warning to the launchd log and moves on, so account_ticks just
-- quietly stops being written. This bit us on 2026-07-31, and the same
-- error is already in the log for `vst_total` and for the table itself
-- from earlier migrations -- so it is a recurring trap, not a one-off.
-- Every future migration that adds a column should end with this line.
notify pgrst, 'reload schema';
