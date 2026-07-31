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
