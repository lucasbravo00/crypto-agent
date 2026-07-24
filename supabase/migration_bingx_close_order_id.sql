-- Migration: fixes a real bug where sync_with_bingx() could wrongly
-- re-close bullets that were opened AFTER a round had already ended.
--
-- Root cause: the SELL fill that closes a round was never persisted
-- anywhere, so get_trade_history() kept returning that same old sell on
-- every 15-min sync, and the code matched it against whatever bullets
-- happened to be active AT THAT LATER TIME -- immediately closing brand
-- new bullets using the OLD sell's stale price/timestamp. Confirmed
-- 2026-07-24: bullets opened on 07-23 and 07-24 were stamped as closed
-- at 07-22T00:15:34, the timestamp of an unrelated, already-processed
-- sell from two days earlier.
--
-- Fix: persist the closing sell's order id on every bullet it closes, so
-- it's never reprocessed (see src/bullets.py's sync_with_bingx()).

alter table bullets add column if not exists bingx_close_order_id text;
