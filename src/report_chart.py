"""
report_chart.py
----------------
Renders the daily report's market context as a PNG, so the email and the
Telegram message carry a picture of what the text is describing instead
of only the words.

Design decisions worth knowing:

1. matplotlib with the "Agg" backend, chosen over a hosted chart service
   (QuickChart and friends). The report already runs in GitHub Actions
   with no display, and Agg is the headless renderer -- it must be
   selected BEFORE pyplot is imported, which is why the two lines below
   are in that order and not alphabetised by an autoformatter.

2. Nothing here talks to a third party. The image is built from the same
   candles market_data already fetches for the report, so adding the
   chart adds no new network dependency and no new place your data can
   leak to.

3. The RSI series here deliberately mirrors market_data._rsi's SIMPLE
   average over the last N deltas, NOT the more common Wilder smoothing.
   They would produce visibly different numbers, and a chart whose last
   RSI point disagreed with the RSI quoted in the text would undermine
   the whole point of attaching it.

4. Chart generation is best-effort by contract: build_report_chart()
   returns None instead of raising. A broken chart must never stop the
   report from being delivered -- the text is the product, the image is
   a garnish.
"""
from __future__ import annotations

import io
import os
from datetime import datetime, timezone
from typing import Optional

import matplotlib
matplotlib.use("Agg")   # headless: must precede the pyplot import
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Same palette as the dashboard, so the report and the web view read as
# one product rather than two.
BG        = "#0B0E13"
PANEL     = "#11151C"
GRID      = "#1C222C"
TEXT      = "#E8ECF2"
DIM       = "#97A1B0"
FAINT     = "#5C6673"
ORANGE    = "#F7931A"
GREEN     = "#2EBD85"
RED       = "#F6465D"
BLUE      = "#4C8DFF"
PURPLE    = "#A78BFA"

# Days actually drawn. The fetch asks for far more (see CANDLES_FETCHED)
# because an SMA200 needs 200 candles of history BEFORE the first plotted
# point -- otherwise the long average would only appear part-way across
# the chart.
DAYS_SHOWN = 180
CANDLES_FETCHED = 400


def _sma_series(values: list[float], period: int) -> list[Optional[float]]:
    out: list[Optional[float]] = [None] * len(values)
    total = 0.0
    for i, v in enumerate(values):
        total += v
        if i >= period:
            total -= values[i - period]
        if i >= period - 1:
            out[i] = total / period
    return out


def _rsi_series(closes: list[float], period: int = 14) -> list[Optional[float]]:
    """Rolling RSI using a SIMPLE mean of the last `period` gains/losses,
    matching market_data._rsi exactly (see this module's docstring, point
    3). Cost is O(n*period), which at 400 candles is nothing."""
    out: list[Optional[float]] = [None] * len(closes)
    for i in range(period, len(closes)):
        gains = losses = 0.0
        for j in range(i - period + 1, i + 1):
            delta = closes[j] - closes[j - 1]
            gains += max(delta, 0.0)
            losses += max(-delta, 0.0)
        out[i] = 100.0 if losses == 0 else 100 - (100 / (1 + (gains / period) / (losses / period)))
    return out


def _fmt_usd(v: float) -> str:
    return "$" + f"{v:,.0f}"


def _style_axes(ax) -> None:
    ax.set_facecolor(PANEL)
    ax.grid(True, color=GRID, linewidth=0.7, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=FAINT, labelsize=8, length=0)


def build_report_chart(symbol: str = "BTC/USDT") -> Optional[bytes]:
    """Return a PNG of the market context, or None if anything at all goes
    wrong. Never raises: callers treat the image as optional."""
    if os.environ.get("REPORT_CHART_ENABLED", "true").lower() not in ("1", "true", "yes"):
        return None
    try:
        return _build(symbol)
    except Exception as exc:
        # Deliberately swallowed: see point 4 in the module docstring.
        print(f"⚠️ Could not build the report chart: {exc}")
        return None


def _build(symbol: str) -> bytes:
    from . import market_data

    candles = market_data.get_ohlcv(symbol, timeframe="1d", limit=CANDLES_FETCHED)
    closes = [c[4] for c in candles]
    dates = [datetime.fromtimestamp(c[0] / 1000, tz=timezone.utc) for c in candles]

    sma50 = _sma_series(closes, 50)
    sma200 = _sma_series(closes, 200)
    rsi = _rsi_series(closes, 14)

    # Only the tail is drawn; the leading candles existed purely to give
    # the moving averages a running start.
    n = min(DAYS_SHOWN, len(closes))
    d, c = dates[-n:], closes[-n:]
    s50, s200, r14 = sma50[-n:], sma200[-n:], rsi[-n:]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 6.0), dpi=100,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.12},
        sharex=True,
    )
    fig.patch.set_facecolor(BG)

    # ---- price panel ----
    _style_axes(ax1)
    ax1.plot(d, c, color=ORANGE, linewidth=1.9, label="BTC", zorder=5)
    ax1.fill_between(d, c, min(c), color=ORANGE, alpha=0.07, zorder=1)
    if any(v is not None for v in s50):
        ax1.plot(d, s50, color=BLUE, linewidth=1.2, label="SMA 50", zorder=4)
    if any(v is not None for v in s200):
        ax1.plot(d, s200, color=PURPLE, linewidth=1.2, label="SMA 200", zorder=4)

    last_price = c[-1]
    ax1.scatter([d[-1]], [last_price], color=ORANGE, s=34, zorder=6,
                edgecolors=BG, linewidths=1.6)
    ax1.annotate(
        _fmt_usd(last_price), xy=(d[-1], last_price), xytext=(6, 0),
        textcoords="offset points", va="center", ha="left",
        color=BG, fontsize=9, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.32", facecolor=ORANGE, edgecolor="none"),
    )

    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: _fmt_usd(v)))
    ax1.tick_params(axis="y", labelcolor=DIM)
    # Opaque panel-coloured box: the legend sits over the SMA200 line at
    # the top-left on any chart where price fell across the window.
    leg = ax1.legend(loc="upper left", fontsize=8.5, ncol=3,
                     handlelength=1.4, columnspacing=1.4,
                     frameon=True, facecolor=PANEL, edgecolor="none", framealpha=0.92)
    for txt in leg.get_texts():
        txt.set_color(DIM)
    # Headroom on the right so the price label never collides with the edge.
    ax1.set_xlim(d[0], d[-1] + (d[-1] - d[-2]) * 9)

    # ---- RSI panel ----
    _style_axes(ax2)
    ax2.plot(d, r14, color=TEXT, linewidth=1.3, zorder=5)
    ax2.axhline(70, color=RED, linewidth=0.9, linestyle="--", alpha=0.55, zorder=2)
    ax2.axhline(30, color=GREEN, linewidth=0.9, linestyle="--", alpha=0.55, zorder=2)
    ax2.axhspan(70, 100, color=RED, alpha=0.06, zorder=1)
    ax2.axhspan(0, 30, color=GREEN, alpha=0.06, zorder=1)
    ax2.set_ylim(0, 100)
    ax2.set_yticks([30, 50, 70])
    ax2.tick_params(axis="y", labelcolor=DIM)
    ax2.text(0.008, 0.93, "RSI 14", transform=ax2.transAxes,
             color=DIM, fontsize=8.5, va="top",
             bbox=dict(boxstyle="round,pad=0.22", facecolor=PANEL, edgecolor="none"))

    ax2.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=8))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax2.tick_params(axis="x", labelcolor=FAINT)

    _add_header(fig, symbol, last_price, closes)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=BG, bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)
    return buf.getvalue()


def _flow_text(fig, x: float, y: float, parts: list[tuple]) -> None:
    """Lay text fragments left-to-right, each one starting where the
    previous ended. Hardcoded x offsets were the obvious first attempt and
    they break the moment a number changes width -- a five-figure BTC
    price collided with the symbol. Agg gives us a renderer up front, so
    the widths can just be measured."""
    renderer = fig.canvas.get_renderer()
    for text, color, size, weight in parts:
        if not text:
            continue
        artist = fig.text(x, y, text, color=color, fontsize=size,
                          fontweight=weight, va="bottom", ha="left")
        width = artist.get_window_extent(renderer=renderer).width / fig.bbox.width
        x += width + 0.018   # gap between fragments, in figure coords


def _add_header(fig, symbol: str, last_price: float, closes: list[float]) -> None:
    """Title plus a one-line strip of the cycle numbers the report text
    tends to reference. Each lookup is individually optional: a rate-
    limited Fear & Greed call should cost us that one chip, not the whole
    image."""
    from . import market_data

    change_pct = ((last_price / closes[-2]) - 1) * 100 if len(closes) >= 2 else None
    chg_color = GREEN if (change_pct or 0) >= 0 else RED
    chg_text = f"{change_pct:+.2f}% 24h" if change_pct is not None else ""

    _flow_text(fig, 0.008, 1.012, [
        (symbol,                 TEXT,      13.5, "bold"),
        (_fmt_usd(last_price),   TEXT,      12.5, "normal"),
        (chg_text,               chg_color, 10.5, "bold"),
    ])

    chips: list[tuple[str, str]] = []
    try:
        cyc = market_data.get_cycle_metrics(symbol)
        if cyc.get("mayer_multiple") is not None:
            chips.append(("Mayer", f"{cyc['mayer_multiple']:.2f}"))
        if cyc.get("pct_distance_to_sma200w") is not None:
            chips.append(("vs SMA200w", f"{cyc['pct_distance_to_sma200w']:+.1f}%"))
    except Exception:
        pass
    try:
        fng = market_data.get_fear_greed_index()
        chips.append(("Fear & Greed", f"{fng['value']} · {fng['classification']}"))
    except Exception:
        pass

    if chips:
        line = "     ".join(f"{k}  {v}" for k, v in chips)
        fig.text(0.008, 0.972, line, color=DIM, fontsize=9, va="bottom", ha="left")

    fig.text(0.992, 1.012,
             datetime.now(timezone.utc).strftime("%d %b %Y · %H:%M UTC"),
             color=FAINT, fontsize=8.5, va="bottom", ha="right")
