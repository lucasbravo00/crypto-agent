-- Incremental migration: removes the "one active bullet at a time"
-- database constraint. The real strategy allows bullets to accumulate
-- (one new one per calendar day, closed together as a group once the
-- combined position hits +15%), so that constraint was based on an
-- incorrect earlier understanding of the strategy. Run in the SQL Editor.
--
-- Safe to run even if the index doesn't exist (IF EXISTS).

drop index if exists bullets_one_active_idx;
