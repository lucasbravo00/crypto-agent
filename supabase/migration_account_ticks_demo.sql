-- Migration: account_ticks now tracks the BingX DEMO (VST) account's
-- total balance, not the real spot wallet.
--
-- Why: the real spot wallet is emptied out on purpose every week (the
-- user rotates BTC to Nexo for yield in between DCA buys), so
-- btc_total/usdt_free from the real account were structurally close to
-- zero and misleading as a "money in account" figure. The real-money
-- side of the dashboard only needs the DCA purchase history (already
-- covered by dca_purchases + the bingx_trade_id sync) -- current BTC
-- value is computed from that directly, wherever the BTC physically is.
-- The demo VST balance is what's actually useful to track live: it's the
-- capital being used to test the bullets strategy right now.

alter table account_ticks drop column if exists btc_total;
alter table account_ticks drop column if exists usdt_free;
alter table account_ticks add column if not exists vst_total numeric not null default 0;
alter table account_ticks alter column vst_total drop default;
