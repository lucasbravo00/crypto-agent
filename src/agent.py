"""
agent.py
---------
This is the core pattern of every LLM tool-calling agent:

    1. Send the model the goal + the available tools.
    2. The model decides (it, not us) which tool(s) to call and with
       which parameters.
    3. We execute those tools in our Python code and return the result.
    4. The model may request more tools, or produce the final answer.
    5. Repeat until the model returns final text (or until an iteration
       cap is hit — the key production guardrail against infinite loops).

Everything the agent decides is recorded in logs/agent_log.jsonl so its
decisions can be audited afterwards.
"""
from __future__ import annotations
import json
import os
from datetime import datetime, timezone

import anthropic

from . import bullets, market_data, state, strategy_tools

MODEL = "claude-sonnet-4-6"
MAX_ITERATIONS = 10  # hard anti-infinite-loop cap
LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "logs", "agent_log.jsonl")

# --- Tools the agent can invoke --------------------------------------------
# The "name" and "description" are what the model reads to decide when to
# use them: description quality matters as much as the code itself.
TOOLS = [
    {
        "name": "get_price",
        "description": "Current price and 24h change for a symbol on BingX.",
        "input_schema": {
            "type": "object",
            "properties": {"symbol": {"type": "string", "default": "BTC/USDT"}},
        },
    },
    {
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
    },
    {
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
    },
    {
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
    },
    {
        "name": "get_current_date",
        "description": "Today's real date. Use this if the report needs to reference 'today' — never guess or infer a date.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_fear_greed_index",
        "description": "Crypto market Fear & Greed index (0-100).",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_btc_dominance",
        "description": "BTC dominance percentage over total crypto market cap.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_dca_summary",
        "description": "Summary of the user's recorded DCA purchases: total invested, accumulated quantity and average entry price.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
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
    },
]

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

REQUIRED_TOOLS = set(TOOL_IMPL.keys()) - {"simulate_bullet_math"}

def _system_prompt() -> str:
    language = os.environ.get("REPORT_LANGUAGE", "en")
    return f"""You are a crypto strategy assistant agent for a user who:
- Is in a DCA accumulation phase toward BTC during a bear market, and
  may separately be running a manual leveraged-futures "bullet" cycle
  on BingX (up to 30 sequential x5 positions, one at a time, each
  targeting +15% on the position). You never open, close, or suggest
  opening/closing any position — every trade is manual, on the
  exchange, decided by the user alone.
- Believes the market is approaching a bottom, but you must NOT make
  that call for them.

Your job in this run is to build a brief, objective daily report:
1. You MUST call every one of these tools before writing your answer:
   get_current_date, get_price, get_indicators, get_cycle_metrics,
   get_fear_greed_index, get_btc_dominance, get_dca_summary,
   get_bullet_status. Do not skip any of them, even if some seem less
   relevant (e.g. get_bullet_status may report no bullets used yet —
   state that plainly rather than omitting the section).
2. NEVER state or imply a date/year from memory. If the report needs a
   date, use only what get_current_date returned.
3. Summarize the data clearly, without giving buy/sell signals or
   asserting whether the bottom has arrived. You provide context; the
   decision belongs to the user. If a metric sits in a historically
   extreme zone you may note that as a historical fact, clarifying it is
   no guarantee about the future.
4. If get_bullet_status shows an open bullet, report its live P&L,
   distance to the target price, and distance to the approximate
   liquidation price as plain facts. NEVER tell the user to close, hold,
   add margin, or otherwise act on that position — you only report the
   numbers; the manual decision on BingX is entirely theirs.
5. Close the report with 1-2 neutral lines about which data would be
   worth watching over the next days (no trading instructions).

Be concise: the final report must not exceed ~180 words, in plain text
suitable for an email or short message.
Write the final report in this language (ISO code): {language}."""


def _log(event: dict) -> None:
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    event["timestamp"] = datetime.now(timezone.utc).isoformat()
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def run_daily_report(symbol: str = "BTC/USDT") -> str:
    client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY from the environment
    messages = [
        {
            "role": "user",
            "content": f"Build the daily strategy report for {symbol}.",
        }
    ]
    
    called_tools: set[str] = set()

    for iteration in range(MAX_ITERATIONS):
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=_system_prompt(),
            tools=TOOLS,
            messages=messages,
        )

        _log({
            "iteration": iteration,
            "stop_reason": response.stop_reason,
            "blocks": [b.type for b in response.content],
        })

        if response.stop_reason != "tool_use":
            missing = REQUIRED_TOOLS - called_tools
            if missing and iteration < MAX_ITERATIONS - 1:
                # Code-level enforcement: don't accept a final answer that
                # skipped required tools, regardless of what the prompt asked.
                _log({"iteration": iteration, "coverage_gap": sorted(missing)})
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

            _log({"iteration": iteration, "tool": block.name, "input": block.input, "result": result})

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": json.dumps(result, ensure_ascii=False),
                "is_error": is_error,
            })

        messages.append({"role": "user", "content": tool_results})

    return "⚠️ The agent hit the iteration cap without finishing (check logs)."
