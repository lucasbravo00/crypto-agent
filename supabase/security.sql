-- crypto-agent dashboard security
-- Run this ONCE in the Supabase SQL Editor, AFTER schema.sql, and BEFORE
-- exposing the anon key in the web dashboard.
--
-- What this does:
--   1. Enables Row Level Security (RLS) on all three tables. With RLS on and
--      no permissive policy, the "anon" key (the one embedded in the public
--      dashboard) can read NOTHING -- a logged-out visitor sees an empty page.
--   2. Adds SELECT-only policies for the "authenticated" role. Once you log in
--      through Supabase Auth, the dashboard can READ the data, but still can't
--      insert/update/delete from the browser (no write policies exist).
--
-- Your Python backend (Mac + GitHub Actions) is unaffected: it uses the
-- "service_role" key, which bypasses RLS entirely. Reads and writes there
-- keep working exactly as before.

alter table dca_purchases   enable row level security;
alter table bullets         enable row level security;
alter table daily_snapshots enable row level security;
alter table price_ticks     enable row level security;

-- Read-only access for logged-in users. "using (true)" means any authenticated
-- session may read every row -- which is fine here because sign-ups are
-- disabled and only your single account exists (see the dashboard setup steps).
create policy "authenticated can read dca_purchases"
    on dca_purchases for select to authenticated using (true);

create policy "authenticated can read bullets"
    on bullets for select to authenticated using (true);

create policy "authenticated can read daily_snapshots"
    on daily_snapshots for select to authenticated using (true);

create policy "authenticated can read price_ticks"
    on price_ticks for select to authenticated using (true);
