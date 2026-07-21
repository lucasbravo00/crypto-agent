"""
agent_ollama.py
-----------------
Same multi-agent design as agent.py (two independent sub-agents —
Market Analyst and Portfolio Manager — coordinated by run_daily_report()),
but the "brain" is an open-weight model running LOCALLY on your machine
through Ollama, instead of the Claude API.

Key differences vs agent.py (interview-worthy details):

1. Ollama runs locally at http://localhost:11434 — you need the Ollama
   app installed and running, and the model already pulled
   (`ollama pull llama3.1`). Without that, this will not work.

2. No need to hand-write tool schemas in JSON: the Ollama library reads
   the Python functions directly (type hints + docstrings) and builds
   the schema itself. More convenient, but also less explicit — worth
   knowing both styles.

3. Only some models truly support tool calling (llama3.1, llama3.2,
   qwen2.5, mistral-nemo, among others). A model without it will simply
   never return tool_calls, and the agent would wait for something that
   never happens — which is why the iteration cap and the "no tool
   calls" handling matter just as much here.

4. Decision quality ("which tool, which parameters?") is generally less
   reliable than Claude's, especially on small (7-8B) models. It is
   common for them to call a tool with a malformed parameter or skip a
   tool they should use. The log (logs/agent_ollama_log.jsonl, tagged
   per sub-agent) lets you compare this objectively against the Claude
   backend.

Splitting into two smaller sub-agents (6 required tools for the Market
Analyst, 3 for the Portfolio Manager, vs. 8 in the original single-agent
design) also gives local models less to juggle in one pass, which helps
with the batching inconsistency described in MAX_ITERATIONS below.
"""
from __future__ import annotations
import json
import os
from datetime import datetime, timezone

import ollama

from . import bullets, market_data, state, strategy_tools

# Local model to use. Must be pulled beforehand:
#   ollama pull llama3.1
MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1")
# Higher than agent.py's cap: local models are inconsistent about batching
# multiple tool calls into one turn (sometimes several at once, sometimes
# one at a time). This is a per-sub-agent budget; each sub-agent now has
# fewer required tools than the original single-agent design (6 and 3
# instead of 8), so 16 remains a safe ceiling rather than a tight one.
MAX_ITERATIONS = 16
LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "logs", "agent_ollama_log.jsonl")

# We pass the real Python functions directly: Ollama builds the schema
# on its own by reading each function's type hints + docstring.
MARKET_TOOL_FUNCTIONS = [
    market_data.get_current_date,
    market_data.get_price,
    market_data.get_indicators,
    market_data.get_cycle_metrics,
    market_data.get_fear_greed_index,
    market_data.get_btc_dominance,
]
MARKET_REQUIRED_TOOLS = {fn.__name__ for fn in MARKET_TOOL_FUNCTIONS}

PORTFOLIO_TOOL_FUNCTIONS = [
    market_data.get_current_date,
    state.get_dca_summary,
    bullets.get_bullet_status,
    strategy_tools.simulate_bullet_math,
]
PORTFOLIO_REQUIRED_TOOLS = {fn.__name__ for fn in PORTFOLIO_TOOL_FUNCTIONS} - {"simulate_bullet_math"}

# One shared implementation map (get_current_date appears in both function
# lists; the dict comprehension naturally dedupes it to the same function).
TOOL_IMPL = {fn.__name__: fn for fn in MARKET_TOOL_FUNCTIONS + PORTFOLIO_TOOL_FUNCTIONS}


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
   to the user alone.
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
  on BingX. You never open, close, or suggest opening/closing any
  position — every trade is manual, on the exchange, decided by the
  user alone.

Your job is ONLY the portfolio-status section of a daily report —
covering the user's own DCA purchases and bullet cycle, not general
market context, that is a separate sub-agent's job.

1. Use get_current_date, get_dca_summary, and get_bullet_status. Report
   plainly even if there are zero purchases or bullets yet.
2. If get_bullet_status shows an open bullet, report its live P&L,
   distance to target, and distance to approximate liquidation as plain
   facts. NEVER tell the user to close, hold, or add margin.
3. NEVER give buy/sell signals or advice about DCA pace or sizing.

Be concise: max ~120 words, plain text, no markdown headers (this text
is concatenated with another sub-agent's section afterward).
Write in this language (ISO code): {language}."""


def _log(event: dict) -> None:
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    event["timestamp"] = datetime.now(timezone.utc).isoformat()
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")


def _run_subagent(
    role: str,
    tool_functions: list,
    required_tools: set[str],
    system_prompt: str,
    user_message: str,
) -> str:
    """Run one sub-agent's tool-calling loop to completion and return its
    final text. Same guardrail pattern as the original single-agent
    design (code-level REQUIRED_TOOLS enforcement, hard iteration cap),
    just parameterized so both sub-agents share the mechanics."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    called_tools: set[str] = set()

    for iteration in range(MAX_ITERATIONS):
        response = ollama.chat(model=MODEL, messages=messages, tools=tool_functions)
        msg = response.message

        _log({
            "agent": role,
            "iteration": iteration,
            "has_tool_calls": bool(msg.tool_calls),
            "content_preview": (msg.content or "")[:200],
        })

        if not msg.tool_calls:
            missing = required_tools - called_tools
            if missing and iteration < MAX_ITERATIONS - 1:
                _log({"agent": role, "iteration": iteration, "coverage_gap": sorted(missing)})
                messages.append(msg)
                messages.append({
                    "role": "user",
                    "content": (
                        "Before your final answer you still need to call these "
                        f"tools, which you haven't used yet: {', '.join(sorted(missing))}. "
                        "Call them now."
                    ),
                })
                continue
            return msg.content or f"⚠️ The {role} sub-agent returned no text (check logs)."

        messages.append(msg)
        for call in msg.tool_calls:
            called_tools.add(call.function.name)
            fn = TOOL_IMPL.get(call.function.name)
            try:
                result = fn(**call.function.arguments) if fn else {"error": "unknown tool"}
            except Exception as exc:
                result = {"error": str(exc)}

            _log({"agent": role, "iteration": iteration, "tool": call.function.name, "input": call.function.arguments, "result": result})

            messages.append({
                "role": "tool",
                "tool_name": call.function.name,
                "content": json.dumps(result, ensure_ascii=False, default=str),
            })

    return f"⚠️ The {role} sub-agent hit the iteration cap without finishing (check logs)."


def run_daily_report(symbol: str = "BTC/USDT") -> str:
    """Coordinate the two sub-agents and concatenate their sections."""
    market_text = _run_subagent(
        "market_analyst", MARKET_TOOL_FUNCTIONS, MARKET_REQUIRED_TOOLS,
        _market_analyst_prompt(),
        f"Build the market-context section of the daily report for {symbol}.",
    )
    portfolio_text = _run_subagent(
        "portfolio_manager", PORTFOLIO_TOOL_FUNCTIONS, PORTFOLIO_REQUIRED_TOOLS,
        _portfolio_manager_prompt(),
        "Build the DCA-and-bullet-cycle section of the daily report.",
    )
    return f"{market_text}\n\n{portfolio_text}"
