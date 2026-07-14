"""
db.py
------
Thin Supabase client wrapper. Every other module reaches Supabase through
this file, so there is a single place that knows how (or whether) the
connection is configured.

Optional dependency: if SUPABASE_URL / SUPABASE_KEY aren't both set in the
environment, this module reports itself as disabled and state.py falls
back to the local JSON file exactly as before. Checked at call time (not
import time) so tests can freely enable/disable it via monkeypatch/env.
"""
from __future__ import annotations
import os

_client = None


def is_enabled() -> bool:
    return bool(os.environ.get("SUPABASE_URL")) and bool(os.environ.get("SUPABASE_KEY"))


def get_client():
    """Return a cached Supabase client, or None if not configured."""
    global _client
    if not is_enabled():
        return None
    if _client is None:
        from supabase import create_client  # local import: no hard dependency when unused
        _client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
    return _client
