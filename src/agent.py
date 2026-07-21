"""
agent.py
---------
Multi-agent version of the daily report: two independent sub-agents, each
with its own system prompt, its own tool subset, and its own
REQUIRED_TOOLS coverage guardrail, coordinated by run_daily_report():

    1. Market Analyst  -> price, indicators, cycle metrics, sentiment.
                           Knows nothing about the user's own portfolio.
    2. Portfolio Manager -> the user's DCA purchases and bullet cycle.
                           Knows nothing about broader market context.

Each sub-agent runs the same core tool-calling loop pattern:

    1. Send the model the goal + the available tools.
    2. The model decides (it, not us) which tool(s) to call and with
       which parameters.
    3. We execute those tools in our Python code and return the result.
    4. The model may request more tools, or produce the final answer.
    5. Repeat until the model returns final text (or until an iteration
       cap is hit — the key production guardrail against infinite loops).

run_daily_report() runs both loops in sequence and concatenates their
text output into one report. No extra "synthesis" LLM call: it's a
deliberate simplicity tradeoff (cheaper, faster, one fewer place for
things to go wrong) over a single fused narrative.

Everything each sub-agent decides is recorded in logs/agent_log.jsonl
(tagged with which agent logged it) so decisions can be audited after
the fact.
"""
from __future__ import annotations
import json
import os
from datetime import datetime, timezone

import anthropic

from . import bullets, market_data, state, strategy_tools

MODEL = "claude-sonnet-4-6"
MAX_ITERATIONS = 10  # hard anti-infinite-loop cap, per sub-agent
LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "logs", "agent_log.jsonl")

# --- Tools each sub-agent can invoke ----------------------------------------
# The "name" and "description" are what the model reads to decide when to
# use them: description quality matters as much as the code itself.

_TOOL_GET_PRICE = {
    "name": "get_price",
    "description": "Current price and 24h change for a symbol on BingX.",
    "input_schema": {
        "type": "object",
        "properties": {"symbol": {"type": "string", "default": "BTC/USDT"}},
    },
}
_TOOL_GET_INDICATORS = {
    "name": "get_indicators",
    "description": (
        "Objective technical indicators (SMA50, SMA200, RSI14, distance "
        "to SMA200) computed from real BingX candles. Predicts nothing; "
        "returns verifiable numbers only."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "symbol": {"type": "string", "default": "BTC/USDT"},
            "timeframe": {"type": "string", "default": "1d"},
        },
    },
}
_TOOL_GET_CYCLE_METRICS = {
    "name": "get_cycle_metrics",
    "description": (
        "Objective market-cycle position metrics: 200-week SMA and "
        "distance to it, Mayer Multiple (price/200d SMA), drawdown from "
        "the available-history high, and weekly RSI. Useful context for "
        "an accumulation strategy in a bear market. Data, not predictions."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"symbol": {"type": "string", "default": "BTC/USDT"}},
    },
}
_TOOL_SIMULATE_BULLET_MATH = {
    "name": "simulate_bullet_math",
    "description": (
        "Pure calculation (no market data) of the math of a long "
        "futures position: price move required for the target, target "
        "price, USD profit, and approximate liquidation distance. Use "
        "when the user asks about their leveraged bullet strategy."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "collateral_usd": {"type": "number"},
            "entry_price": {"type": "number"},
            "leverage": {"type": "number", "default": 5},
            "target_position_gain_pct": {"type": "number", "default": 15},
        },
        "required": ["collateral_usd", "entry_price"],
    },
}
_TOOL_GET_CURRENT_DATE = {
    "name": "get_current_date",
    "description": "Today's real date. Use this if the report needs to reference 'today' — never guess or infer a date.",
    "input_schema": {"type": "object", "properties": {}},
}
_TOOL_GET_FEAR_GREED = {
    "name": "get_fear_greed_index",
    "description": "Crypto market Fear & Greed index (0-100).",
    "input_schema": {"type": "object", "properties": {}},
}
_TOOL_GET_BTC_DOMINANCE = {
    "name": "get_btc_dominance",
    "description": "BTC dominance percentage over total crypto market cap.",
    "input_schema": {"type": "object", "properties": {}},
}
_TOOL_GET_DCA_SUMMARY = {
    "name": "get_dca_summary",
    "description": "Summary of the user's recorded DCA purchases: total invested, accumulated quantity and average entry price.",
    "input_schema": {"type": "object", "properties": {}},
}
_TOOL_GET_BULLET_STATUS = {
    "name": "get_bullet_status",
    "description": (
        "Status of the user's 30-bullet leveraged-futures cycle: how "
        "many bullets used/remaining, closed count, target (tp) wins, "
        "total realized P&L, and if a bullet is currently open its "
        "live P&L, distance to target, and distance to approximate "
        "liquidation at the current market price. These bullets are "
        "opened and closed MANUALLY by the user on BingX; this tool "
        "only reports recorded state, it never opens/closes anything."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"symbol": {"type": "string", "default": "BTC/USDT"}},
    },
}

MARKET_TOOLS = [
    _TOOL_GET_CURRENT_DATE,
    _TOOL_GET_PRICE,
    _TOOL_GET_INDICATORS,
    _TOOL_GET_CYCLE_METRICS,
    _TOOL_GET_FEAR_GREED,
    _TOOL_GET_BTC_DOMINANCE,
]
MARKET_REQUIRED_TOOLS = {t["name"] for t in MARKET_TOOLS}

PORTFOLIO_TOOLS = [
    _TOOL_GET_CURRENT_DATE,
    _TOOL_GET_DCA_SUMMARY,
    _TOOL_GET_BULLET_STATUS,
    _TOOL_SIMULATE_BULLET_MATH,
]
PORTFOLIO_REQUIRED_TOOLS = {t["name"] for t in PORTFOLIO_TOOLS} - {"simulate_bullet_math"}

# One shared implementation map: each sub-agent only ever sees the subset
# of names in its own `tools` list, so offering the full map here is safe.
TOOL_IMPL = {
    "get_price": lambda **kw: market_data.get_price(**kw),
    "get_indicators": lambda **kw: market_data.get_indicators(**kw),
    "get_cycle_metrics": lambda **kw: market_data.get_cycle_metrics(**kw),
    "simulate_bullet_math": lambda **kw: strategy_tools.simulate_bullet_math(**kw),
    "get_current_date": lambda **kw: market_data.get_current_date(),
    "get_fear_greed_index": lambda **kw: market_data.get_fear_greed_index(),
    "get_btc_dominance": lambda **kw: market_data.get_btc_dominance(),
    "get_dca_summary": lambda **kw: state.get_dca_summary(),
    "get_bullet_status": lambda **kw: bullets.get_bullet_status(**kw),
}


def _market_analyst_prompt() -> str:
    language = os.environ.get("REPORT_LANGUAGE", "en")
    return f"""You are the MARKET ANALYST sub-agent of a crypto strategy
assistant. Your job is ONLY the market-context section of a daily
report — nothing about the user's own portfolio or bullet positions,
that is a separate sub-agent's job.

1. You MUST call every one of these tools before writing your answer:
   get_current_date, get_price, get_indicators, get_cycle_metrics,
   get_fear_greed_index, get_btc_dominance. Do not skip any.
2. NEVER state or imply a date/year from memory. If you reference
   "today", use only what get_current_date returned.
3. ONLY report numbers and indicators that a tool call actually
   returned. Do NOT mention, infer, or fabricate any other metric (e.g.
   MACD, Bollinger Bands, an RSI on a timeframe you weren't given) that
   wasn't returned by one of the tools above. get_btc_dominance returns
   BITCOIN's dominance specifically — never attribute that number to any
   other coin (e.g. BNB, ETH).
4. Summarize the data clearly. NEVER give buy/sell signals or assert
   whether the market's bottom or top has arrived — that call belongs
   to the user alone. If a metric sits in a historically extreme zone
   you may note that as a historical fact, clarifying it's no guarantee
   about the future.
5. Close with 1-2 neutral lines about which market data is worth
   watching over the next days (no trading instructions).

Be concise: max ~120 words, plain text, no markdown headers (this text
is concatenated with another sub-agent's section afterward).
Write in this language (ISO code): {language}."""


def _portfolio_manager_prompt() -> str:
    language = os.environ.get("REPORT_LANGUAGE", "en")
    return f"""You are the PORTFOLIO MANAGER sub-agent of a crypto
strategy assistant, for a user who:
- Is in a DCA accumulation phase toward BTC during a bear market, and
  may separately be running a manual leveraged-futures "bullet" cycle
  on BingX (up to 30 sequential x5 positions, one at a time, each
  targeting +15% on the position). You never open, close, or suggest
  opening/closing any position — every trade is manual, on the
  exchange, decided by the user alone.

Your job is ONLY the portfolio-status section of a daily report —
covering the user's own DCA purchases and bullet cycle, not general
market context, that is a separate sub-agent's job.

1. You MUST call every one of these tools before writing your answer:
   get_current_date, get_dca_summary, get_bullet_status. Report plainly
   even if there are zero purchases or bullets yet — state that fact
   rather than omitting the section.
2. NEVER state or imply a date/year from memory.
3. If get_bullet_status shows an open bullet, report its live P&L,
   distance to the target price, and distance to the approximate
   liquidation price as plain facts. NEVER tell the user to close,
   hold, add margin, or otherwise act on that position.
4. NEVER give buy/sell signals or advice about DCA pace or sizing.

Be concise: max ~120 words, plain text, no markdown headers (this text
is concatenated with another sub-agent's section afterward).
Write in this language (ISO code): {language}."""


def _log(event: dict) -> None:
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    event["timestamp"] = datetime.now(timezone.utc).isoformat()
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def _run_subagent(
    role: str,
    tools: list[dict],
    required_tools: set[str],
    system_prompt: str,
    user_message: str,
) -> str:
    """Run one sub-agent's tool-calling loop to completion and return its
    final text. Same guardrail pattern as the original single-agent
    design (code-level REQUIRED_TOOLS enforcement, hard iteration cap),
    just parameterized so both sub-agents share the mechanics."""
    client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY from the environment
    messages = [{"role": "user", "content": user_message}]
    called_tools: set[str] = set()

    for iteration in range(MAX_ITERATIONS):
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=system_prompt,
            tools=tools,
            messages=messages,
        )

        _log({
            "agent": role,
            "iteration": iteration,
            "stop_reason": response.stop_reason,
            "blocks": [b.type for b in response.content],
        })

        if response.stop_reason != "tool_use":
            missing = required_tools - called_tools
            if missing and iteration < MAX_ITERATIONS - 1:
                # Code-level enforcement: don't accept a final answer that
                # skipped required tools, regardless of what the prompt asked.
                _log({"agent": role, "iteration": iteration, "coverage_gap": sorted(missing)})
                messages.append({"role": "assistant", "content": response.content})
                messages.append({
                    "role": "user",
                    "content": (
                        "Before your final answer you still need to call these "
                        f"tools, which you haven't used yet: {', '.join(sorted(missing))}. "
                        "Call them now."
                    ),
                })
                continue
            final_text = "".join(
                b.text for b in response.content if b.type == "text"
            )
            return final_text

        # One or more tool calls to resolve.
        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            called_tools.add(block.name)
            fn = TOOL_IMPL.get(block.name)
            try:
                result = fn(**(block.input or {})) if fn else {"error": "unknown tool"}
                is_error = False
            except Exception as exc:  # never let a tool exception crash the agent
                result = {"error": str(exc)}
                is_error = True

            _log({"agent": role, "iteration": iteration, "tool": block.name, "input": block.input, "result": result})

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": json.dumps(result, ensure_ascii=False),
                "is_error": is_error,
            })

        messages.append({"role": "user", "content": tool_results})

    return f"⚠️ The {role} sub-agent hit the iteration cap without finishing (check logs)."


def run_daily_report(symbol: str = "BTC/USDT") -> str:
    """Coordinate the two sub-agents and concatenate their sections."""
    market_text = _run_subagent(
        "market_analyst", MARKET_TOOLS, MARKET_REQUIRED_TOOLS,
        _market_analyst_prompt(),
        f"Build the market-context section of the daily report for {symbol}.",
    )
    portfolio_text = _run_subagent(
        "portfolio_manager", PORTFOLIO_TOOLS, PORTFOLIO_REQUIRED_TOOLS,
        _portfolio_manager_prompt(),
        "Build the DCA-and-bullet-cycle section of the daily report.",
    )
    return f"{market_text}\n\n{portfolio_text}"
