-- Migration: track REAL exchange fees, subtracted from bullet P&L and
-- folded into the DCA cost basis, instead of pretending trading is
-- fee-free.
--
-- Bullets: entry_fee_usd is a bullet's own opening fee; exit_fee_usd is
-- its share of the round's single closing fee, split by collateral
-- weight (see src/bullets.py's close_all_active_bullets()).
--
-- DCA: fee_usd is informational only -- `price` on dca_purchases is
-- already adjusted to fold the real (BTC-denominated) fee into the
-- effective cost basis (see src/dca.py's _fee_adjusted()).

alter table dca_purchases add column if not exists fee_usd numeric;
alter table bullets add column if not exists entry_fee_usd numeric;
alter table bullets add column if not exists exit_fee_usd numeric;
