"""
agent_ollama.py
-----------------
Same idea as agent.py (tool-calling loop with an iteration cap and
logging), but the "brain" is an open-weight model running LOCALLY on
your machine through Ollama, instead of the Claude API.

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
   tool they should use. The log (logs/agent_ollama_log.jsonl) lets you
   compare this objectively against the Claude backend.
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
MAX_ITERATIONS = 10
LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "logs", "agent_ollama_log.jsonl")

# We pass the real Python functions directly: Ollama builds the schema
# on its own by reading each function's type hints + docstring.
TOOL_FUNCTIONS = [
    market_data.get_current_date,
    market_data.get_price,
    market_data.get_indicators,
    market_data.get_cycle_metrics,
    market_data.get_fear_greed_index,
    market_data.get_btc_dominance,
    state.get_dca_summary,
    bullets.get_bullet_status,
    strategy_tools.simulate_bullet_math,
]
TOOL_IMPL = {fn.__name__: fn for fn in TOOL_FUNCTIONS}
REQUIRED_TOOLS = {fn.__name__ for fn in TOOL_FUNCTIONS} - {"simulate_bullet_math"}

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
1. Use the available tools to gather real market data and their DCA
   status. Always include the cycle metrics (get_cycle_metrics) and the
   bullet cycle status (get_bullet_status) — report it plainly even if
   no bullets have been used yet.
2. Summarize the data clearly, without giving buy/sell signals or
   asserting whether the bottom has arrived.
3. If get_bullet_status shows an open bullet, report its live P&L,
   distance to target, and distance to approximate liquidation as plain
   facts. NEVER tell the user to close, hold, or add margin — you only
   report numbers; the manual decision on BingX is entirely theirs.
4. Close the report with 1-2 neutral lines about which data would be
   worth watching over the next days.

Be concise: the final report must not exceed ~180 words, in plain text
suitable for an email or short message.
Write the final report in this language (ISO code): {language}."""


def _log(event: dict) -> None:
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    event["timestamp"] = datetime.now(timezone.utc).isoformat()
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")


def run_daily_report(symbol: str = "BTC/USDT") -> str:
    messages = [
        {"role": "system", "content": _system_prompt()},
        {"role": "user", "content": f"Build the daily strategy report for {symbol}."},
    ]
    called_tools: set[str] = set()

    for iteration in range(MAX_ITERATIONS):
        response = ollama.chat(model=MODEL, messages=messages, tools=TOOL_FUNCTIONS)
        msg = response.message

        _log({
            "iteration": iteration,
            "has_tool_calls": bool(msg.tool_calls),
            "content_preview": (msg.content or "")[:200],
        })

        if not msg.tool_calls:
            missing = REQUIRED_TOOLS - called_tools
            if missing and iteration < MAX_ITERATIONS - 1:
                _log({"iteration": iteration, "coverage_gap": sorted(missing)})
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
            return msg.content or "⚠️ The model returned no text (check logs)."

        messages.append(msg)
        for call in msg.tool_calls:
            called_tools.add(call.function.name)
            fn = TOOL_IMPL.get(call.function.name)
            try:
                result = fn(**call.function.arguments) if fn else {"error": "unknown tool"}
            except Exception as exc:
                result = {"error": str(exc)}

            _log({"iteration": iteration, "tool": call.function.name, "input": call.function.arguments, "result": result})

            messages.append({
                "role": "tool",
                "tool_name": call.function.name,
                "content": json.dumps(result, ensure_ascii=False, default=str),
            })

    return "⚠️ The agent hit the iteration cap without finishing (check logs)."
