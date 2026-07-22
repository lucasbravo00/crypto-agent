-- Incremental migration: bullet_number now resets to 1 at the start of
-- each round (a round = bullets that accumulate together until the
-- combined +15% target closes them all at once). round_number
-- disambiguates "bullet 3 of round 1" from "bullet 3 of round 2".
-- Already included in schema.sql for anyone setting up fresh -- this
-- file is only for an existing project that already ran that once.

alter table bullets add column if not exists round_number int;

-- Backfill: every bullet that already exists in this project belongs to
-- round 1 (the only round that has run so far).
update bullets set round_number = 1 where round_number is null;
