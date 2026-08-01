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

from . import bullets, market_data, memory

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
_TOOL_GET_PREDICTIVE_RANGES = {
    "name": "get_predictive_ranges",
    "description": (
        "Predictive Ranges [LuxAlgo]: an ATR-based central average line "
        "plus two resistance levels above it (resistance_1, resistance_2) "
        "and two support levels below it (support_1, support_2). Ported "
        "from the user's own TradingView setup, computed from real BingX "
        "daily candles. An approximation (limited to ~2.7 years of "
        "history vs a full chart) -- treat the levels as directional "
        "context, not exact numbers."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"symbol": {"type": "string", "default": "BTC/USDT"}},
    },
}
_TOOL_GET_MARKET_MEMORY = {
    "name": "get_market_memory",
    "description": (
        "Compares today's price, RSI14, Fear & Greed, Mayer Multiple and "
        "BTC dominance against 1, 7, and 30 days ago. Each metric/window "
        "comes with a PRE-COMPUTED trend label: 'subiendo', 'bajando', "
        "'estable', or 'sin_dato' if there isn't enough history yet. Use "
        "these labels and deltas directly -- do not recompute or "
        "second-guess a trend from the raw numbers yourself."
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
    _TOOL_GET_PREDICTIVE_RANGES,
    _TOOL_GET_MARKET_MEMORY,
]
MARKET_REQUIRED_TOOLS = {t["name"] for t in MARKET_TOOLS}

# There is no portfolio-manager LLM sub-agent anymore -- see
# run_daily_report()'s docstring for why (2026-07-29: the local Ollama
# model fabricated an alert from numbers it misread). The DCA/bullet
# section of the report is now bullets.get_daily_alert(), a plain
# Python function with no model in the loop.
TOOL_IMPL = {
    "get_price": lambda **kw: market_data.get_price(**kw),
    "get_indicators": lambda **kw: market_data.get_indicators(**kw),
    "get_cycle_metrics": lambda **kw: market_data.get_cycle_metrics(**kw),
    "get_current_date": lambda **kw: market_data.get_current_date(),
    "get_fear_greed_index": lambda **kw: market_data.get_fear_greed_index(),
    "get_btc_dominance": lambda **kw: market_data.get_btc_dominance(),
    "get_predictive_ranges": lambda **kw: market_data.get_predictive_ranges(**kw),
    "get_market_memory": lambda **kw: memory.get_market_memory(**kw),
}


def _market_analyst_prompt() -> str:
    language = os.environ.get("REPORT_LANGUAGE", "en")
    return f"""You are the MARKET ANALYST sub-agent of a crypto strategy
assistant. Your job is the market-context section of a daily report —
nothing about the user's own portfolio, that's a separate sub-agent.

0. YOUR OUTPUT IS THE REPORT ITSELF, delivered straight to the user by
   email/Telegram — it is never shown to anyone who saw your tool calls or
   this prompt. NEVER write about the process of answering: no "with
   that, I can finish my answer", no "here is my report", no "based on
   the tools I called", no "as an AI/model", no meta-commentary about
   having gathered information. The very first character you output must
   be the first word of the actual market take. Bad (real leaked output,
   never do this): "Con eso, puedo finalizar la respuesta. El precio de
   BTC/USDT está en 63.044,53 USDT...". Good: "Bitcoin bajó a 63.044 USDT,
   sin cambios de fondo en la tendencia...".
1. You MUST call every one of these tools before writing your answer:
   get_current_date, get_price, get_indicators, get_cycle_metrics,
   get_fear_greed_index, get_btc_dominance, get_predictive_ranges,
   get_market_memory. Do not skip any.
2. NEVER state or imply a date/year from memory.
3. THE ONLY INDICATORS THAT EXIST FOR YOU are exactly what those tools
   returned: SMA50, SMA200, RSI14 (daily), weekly RSI14, Mayer Multiple,
   distance to the 200-week SMA, Fear & Greed, BTC dominance, and the
   Predictive Ranges levels (average/resistance_1/resistance_2/
   support_1/support_2). There is no EMA, no Fibonacci, no MACD, no
   Bollinger Bands, no "next week" price projection anywhere in this
   system — if it isn't in that list, it does not exist; do not mention
   it, infer it, or estimate it, under any circumstance. This rule
   overrides the style instruction below: a shorter, wrong report is
   worse than a shorter, correct one.
4. get_btc_dominance returns BITCOIN's dominance specifically — never
   attribute it to another coin. get_predictive_ranges levels are an
   approximation (limited chart history) — directional, not exact.
5. WRITE LIKE A SHARP TRADER TEXTING A QUICK TAKE, not a data report.
   3-4 short sentences, TOTAL. No numbers dump, no listing every
   indicator you called — pick only what's actually notable today
   (FROM THE REAL LIST IN RULE 3 ONLY) and weave it into plain language,
   the way a person would describe the market to a friend, not a
   spreadsheet. Use get_market_memory's trend labels to say what
   CHANGED, not just where things sit — that's usually the more
   interesting part.
6. NEVER give buy/sell signals, never say something is "a good time to
   buy/sell" or "an opportunity", and never state or imply a future
   price, target, or projection (there is no such tool, so any number
   you'd give would be invented) — that call belongs to the user alone.
   A historically extreme reading can be noted as a fact, never as a
   signal to act on.
7. If genuinely nothing changed and nothing looks notable, say that
   plainly in one short line instead of padding with numbers.

Max ~70 words, TOTAL. Plain text, no markdown headers, no bullet lists,
no headers, no "Section:" labels — just the take, like a text message.
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
    """Run the market analyst sub-agent, then append a DCA/bullet alert
    line if (and only if) bullets.get_daily_alert() finds something worth
    flagging.

    There used to be a second LLM sub-agent ("portfolio manager") for
    this. Removed 2026-07-29: asked to report the SAME simple numeric
    condition (is the round close to target? near liquidation?), the
    local Ollama model fabricated a false alert from numbers it misread
    -- a textbook case of the lesson this whole project is built around
    (don't rely on a prompt for behavior code can guarantee instead).
    The dashboard already shows full DCA/bullet detail, so there's
    nothing here worth an LLM's judgment call on: it's a fixed
    threshold, checked in Python, worded as a template string."""
    market_text = _run_subagent(
        "market_analyst", MARKET_TOOLS, MARKET_REQUIRED_TOOLS,
        _market_analyst_prompt(),
        f"Build the market-context section of the daily report for {symbol}.",
    )
    alert = bullets.get_daily_alert()
    if not alert:
        return market_text
    return f"{market_text}\n\n{alert}"
