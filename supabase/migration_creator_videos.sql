-- Migration: track YouTube videos from the creators you follow, so the
-- daily report can mention a NEW crypto video once and never again.
--
-- Why a table and not just "look at the RSS each run": the report runs
-- daily but a video stays in the channel's RSS feed for ~15 entries, so
-- without a record of what was already summarized the same video would
-- be re-reported every day until it fell out of the feed.
--
-- `is_crypto` is stored (not just filtered on) on purpose: a video that
-- was judged non-crypto must ALSO be remembered, otherwise every run
-- would re-download its transcript and re-ask the LLM about it.
--
-- `summary` is the short synthesized take that goes into the report --
-- never the raw transcript. Transcripts are thousands of words, belong
-- to their creators, and are not stored here.
--
-- Run this once in the Supabase SQL Editor if your project predates it.

create table if not exists creator_videos (
    id bigint generated always as identity primary key,
    created_at timestamptz not null default now(),
    channel_id text not null,
    channel_name text,
    video_id text not null unique,   -- YouTube's own id; the dedup key
    title text,
    published_at timestamptz,
    is_crypto boolean not null default false,
    summary text,                     -- null when is_crypto is false
    reported_at timestamptz           -- set once it has been put in a report
);

create index if not exists creator_videos_published_idx
    on creator_videos (published_at desc);

alter table creator_videos enable row level security;

create policy "authenticated can read creator_videos"
    on creator_videos for select to authenticated using (true);

-- Reload PostgREST's schema cache. WITHOUT THIS the table above appears
-- to exist -- `select *` works, because that goes straight to Postgres --
-- while every INSERT fails with PGRST205/PGRST204 ("Could not find the
-- table/column ... in the schema cache"), because PostgREST validates
-- writes against its own cached schema. The failure is silent from the
-- app's side (creators.py catches it and skips the digest), so the
-- feature would just quietly never work. This already bit this project
-- twice -- see migration_liquidation_price.sql.
notify pgrst, 'reload schema';
