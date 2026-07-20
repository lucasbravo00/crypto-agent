-- Incremental migration: adds price_ticks (already included in schema.sql
-- and security.sql for anyone setting up fresh -- this file is only for
-- an existing project that already ran those once). Run in the SQL Editor.

create table if not exists price_ticks (
    id bigint generated always as identity primary key,
    created_at timestamptz not null default now(),
    price numeric not null,
    change_24h_pct numeric
);

alter table price_ticks enable row level security;

create policy "authenticated can read price_ticks"
    on price_ticks for select to authenticated using (true);
