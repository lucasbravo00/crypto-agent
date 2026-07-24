"""
memory.py
----------
Gives the agent temporal context: how today's key market indicators
compare against 1, 7, and 30 days ago, instead of every report reading
like an isolated snapshot. Deltas and trend labels are computed here in
Python, not left for the LLM to infer -- consistent with how this
project avoids letting a model guess at things a tool could tell it
directly (see the MACD/BNB fabrication lesson in agent.py's prompts).

Market-only scope, deliberately: DCA/bullet history isn't compared here.
"""
from __future__ import annotations
from datetime import datetime, timedelta, timezone
from typing import Optional

from . import market_data
from . import state as state_module

WINDOWS_DAYS = (1, 7, 30)

# How far a metric has to move before it's labeled a trend instead of
# "estable" -- one threshold per metric since each moves on a different
# scale. Chosen with the user, 2026-07-24.
METRICS = [
    {"key": "price", "snapshot_col": "price", "threshold": 3.0, "pct": True},
    {"key": "rsi14", "snapshot_col": "rsi14", "threshold": 5.0, "pct": False},
    {"key": "fear_greed", "snapshot_col": "fear_greed_value", "threshold": 10.0, "pct": False},
    {"key": "mayer_multiple", "snapshot_col": "mayer_multiple", "threshold": 0.1, "pct": False},
    {"key": "btc_dominance_pct", "snapshot_col": "btc_dominance_pct", "threshold": 1.0, "pct": False},
]


def _parse_date(iso_string: str):
    return datetime.fromisoformat(iso_string.replace("Z", "+00:00")).date()


def _closest_snapshot(snapshots: list[dict], target_date, tolerance_days: int = 1) -> Optional[dict]:
    """The snapshot whose created_at is closest to target_date, within
    `tolerance_days`. Report runs don't land at exactly the same moment
    every day, and some days can be missed entirely (Mac asleep, Ollama
    down, etc.), so an exact-date match would too often come up empty."""
    best, best_diff = None, None
    for snap in snapshots:
        if not snap.get("created_at"):
            continue
        diff = abs((_parse_date(snap["created_at"]) - target_date).days)
        if diff <= tolerance_days and (best_diff is None or diff < best_diff):
            best, best_diff = snap, diff
    return best


def _trend_label(change: Optional[float], threshold: float) -> str:
    if change is None:
        return "sin_dato"
    if change > threshold:
        return "subiendo"
    if change < -threshold:
        return "bajando"
    return "estable"


def _compare_metric(metric: dict, current: Optional[float], past_snapshot: Optional[dict]) -> dict:
    then = past_snapshot.get(metric["snapshot_col"]) if past_snapshot else None
    if current is None or then is None:
        return {"value_then": None, "delta": None, "delta_pct": None, "trend": "sin_dato"}

    delta = current - then
    delta_pct = (delta / then * 100) if then else None
    change_for_label = delta_pct if metric["pct"] else delta
    return {
        "value_then": round(then, 4),
        "delta": round(delta, 4),
        "delta_pct": round(delta_pct, 2) if delta_pct is not None else None,
        "trend": _trend_label(change_for_label, metric["threshold"]),
    }


def get_market_memory(symbol: str = "BTC/USDT", **_ignored) -> dict:
    """Compare today's price/RSI/Fear&Greed/Mayer Multiple/BTC dominance
    against 1, 7, and 30 days ago. Every window returns pre-computed
    deltas and a trend label ("subiendo"/"bajando"/"estable") per metric
    -- not raw history for the model to interpret freely. A window comes
    back as "sin_dato" wherever there isn't a snapshot close enough yet
    (e.g. this agent hasn't been running for 30 days)."""
    current = {
        "price": market_data.get_price(symbol)["last_price"],
        "rsi14": market_data.get_indicators(symbol)["rsi_14"],
        "fear_greed": market_data.get_fear_greed_index()["value"],
        "mayer_multiple": market_data.get_cycle_metrics(symbol)["mayer_multiple"],
        "btc_dominance_pct": market_data.get_btc_dominance()["btc_dominance_pct"],
    }

    snapshots = state_module.get_snapshots(limit=60)
    today = datetime.now(timezone.utc).date()

    windows = {}
    for days_back in WINDOWS_DAYS:
        past = _closest_snapshot(snapshots, today - timedelta(days=days_back))
        windows[f"{days_back}d"] = {
            metric["key"]: _compare_metric(metric, current[metric["key"]], past)
            for metric in METRICS
        }

    return {"symbol": symbol, "current": current, "windows": windows}
