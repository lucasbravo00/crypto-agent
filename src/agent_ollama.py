"""
agent_ollama.py
-----------------
Same design as agent.py -- a Market Analyst LLM sub-agent for the
market-context section of the report, plus a deterministic (no LLM)
DCA/bullet alert from bullets.get_daily_alert() -- but the "brain" is an
open-weight model running LOCALLY on your machine through Ollama,
instead of the Claude API.

There used to be a second LLM sub-agent ("Portfolio Manager") for the
DCA/bullet section. Removed 2026-07-29: asked to flag a simple numeric
condition (round close to target? near liquidation?), this local model
fabricated a false alert from numbers it misread. Moved to plain Python
instead -- see run_daily_report().

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
   common for them to call a tool with a malformed parameter, skip a
   tool they should use, or -- confirmed directly, see above -- invent
   content that sounds plausible but isn't grounded in any tool result.
   The log (logs/agent_ollama_log.jsonl) lets you audit this after the
   fact.
"""
from __future__ import annotations
import json
import os
import re
from datetime import datetime, timezone

import ollama

from . import bullets, creators, market_data, memory

# Trigger phrases for a leaked "meta-commentary" preamble -- the model
# narrating its own answering process instead of just answering.
# Real example that motivated this (2026-08-01, REPORT_LANGUAGE=es):
#   "Con eso, puedo finalizar la respuesta. El precio de BTC/USDT..."
# Covers only en/es (the two languages actually used by REPORT_LANGUAGE
# so far) -- a report in another language would not be caught here.
_PREAMBLE_TRIGGERS = re.compile(
    r"\b("
    r"con (eso|esto|esta información|esa información)|"
    r"aquí (te )?(tienes|dejo|va|está)|"
    r"basá?ndome en|basado en (lo|los|las)|"
    r"puedo (ahora )?(finalizar|responder|concluir|dar)|"
    r"he (recopilado|reunido|obtenido)|"
    r"esta es (mi|la) respuesta|mi respuesta (es|sería)|"
    r"como (un )?modelo de lenguaje|como (una )?ia\b|"
    r"with that|here is|here's|i can now (finish|answer|conclude)|"
    r"based on the tools|as an ai|i have gathered|below is my"
    r")\b",
    re.IGNORECASE,
)


def _strip_leaked_preamble(text: str) -> str:
    """Drop a leading sentence in which the model narrates its own
    answering process instead of answering (see _PREAMBLE_TRIGGERS'
    docstring for the real example that prompted this).

    Deliberately narrow: only strips the FIRST sentence, and only when it
    both matches a trigger phrase AND contains no digit. The digit check
    is the safety rail -- a real market sentence almost always carries a
    price or a percentage, while an empty preamble like "Con eso, puedo
    finalizar la respuesta." never does. Without it, a legitimate opening
    sentence that happens to start with "Con el..." could be stripped by
    mistake. This is a narrow patch for a specific observed failure, not
    a general profanity/quality filter -- most malformed output is still
    left to the prompt (see _market_analyst_prompt's rule 0)."""
    if not text:
        return text
    # ":" and newline count as sentence boundaries too, not just ". "/"! " --
    # "Aquí tienes el informe: Bitcoin..." has no period before the real
    # content, and without this the "first sentence" swallows the whole
    # text, leaves nothing after it, and the empty-report guard below
    # silently gives up and returns the untouched original.
    match = re.match(r"\s*(.+?[.!:])(\s+|$)", text, re.DOTALL)
    if not match:
        return text
    first_sentence = match.group(1)
    if _PREAMBLE_TRIGGERS.search(first_sentence) and not re.search(r"\d", first_sentence):
        rest = text[match.end():].lstrip()
        return rest or text   # never return an empty report over a suspicious one
    return text

# Local model to use. Must be pulled beforehand:
#   ollama pull llama3.1
MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1")
# Local models are inconsistent about batching multiple tool calls into
# one turn (sometimes several at once, sometimes one at a time), so this
# stays generous even though there's only one sub-agent's 8 required
# tools to cover now.
MAX_ITERATIONS = 16
LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "logs", "agent_ollama_log.jsonl")

def get_predictive_ranges(symbol: str = "BTC/USDT") -> dict:
    """Predictive Ranges [LuxAlgo]: an ATR-based central average line
    plus two resistance levels above it and two support levels below it.
    Ported from the user's own TradingView setup, computed from real
    BingX daily candles. An approximation (limited chart history) --
    treat the levels as directional context, not exact numbers.

    A thin wrapper, deliberately hiding market_data.get_predictive_ranges's
    `length`/`mult` parameters: Ollama auto-generates a tool's schema
    from ALL of a Python function's parameters, so passing that function
    directly let the model invent its own `mult` (confirmed 2026-07-29:
    it tried 2 and 0.5 on different calls, nothing like the backtested
    default of 6.0), silently discarding the calibration the whole
    feature was built and verified around. Only `symbol` is exposed here
    on purpose -- length/mult always come from market_data's own
    defaults, never from the model."""
    return market_data.get_predictive_ranges(symbol)


# We pass the real Python functions directly: Ollama builds the schema
# on its own by reading each function's type hints + docstring.
MARKET_TOOL_FUNCTIONS = [
    market_data.get_current_date,
    market_data.get_price,
    market_data.get_indicators,
    market_data.get_cycle_metrics,
    market_data.get_fear_greed_index,
    market_data.get_btc_dominance,
    get_predictive_ranges,
    memory.get_market_memory,
]
MARKET_REQUIRED_TOOLS = {fn.__name__ for fn in MARKET_TOOL_FUNCTIONS}

# There is no portfolio-manager LLM sub-agent anymore -- see
# run_daily_report()'s docstring for why (2026-07-29: this local model
# fabricated an alert from numbers it misread). The DCA/bullet section of
# the report is now bullets.get_daily_alert(), a plain Python function
# with no model in the loop.
TOOL_IMPL = {fn.__name__: fn for fn in MARKET_TOOL_FUNCTIONS}


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
        f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")


def summarize(system_prompt: str, user_text: str, max_tokens: int = 300) -> str:
    """One-shot completion with NO tools and no agent loop -- the local
    counterpart of agent.summarize(), same contract (see its docstring).

    `max_tokens` maps to Ollama's num_predict. The leaked-preamble
    stripper is applied here too: the same local model that narrated its
    own answering process into the daily report will happily do it to a
    video summary as well.
    """
    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        options={"num_predict": max_tokens},
    )
    return _strip_leaked_preamble((response.message.content or "").strip())


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
    """Run the market analyst sub-agent, then append a DCA/bullet alert
    line if (and only if) bullets.get_daily_alert() finds something worth
    flagging.

    There used to be a second LLM sub-agent ("portfolio manager") for
    this. Removed 2026-07-29: asked to report the SAME simple numeric
    condition (is the round close to target? near liquidation?), this
    local Ollama model fabricated a false alert from numbers it misread
    -- a textbook case of the lesson this whole project is built around
    (don't rely on a prompt for behavior code can guarantee instead).
    The dashboard already shows full DCA/bullet detail, so there's
    nothing here worth an LLM's judgment call on: it's a fixed
    threshold, checked in Python, worded as a template string."""
    market_text = _run_subagent(
        "market_analyst", MARKET_TOOL_FUNCTIONS, MARKET_REQUIRED_TOOLS,
        _market_analyst_prompt(),
        f"Build the market-context section of the daily report for {symbol}.",
    )
    # Safety net for a real, observed failure (2026-08-01): after several
    # rounds of "you still need to call these tools" nudging, this local
    # model sometimes narrates its own answering process as the opening
    # sentence ("Con eso, puedo finalizar la respuesta.") instead of just
    # answering. Rule 0 of the prompt above asks it not to; this catches
    # what slips through anyway (see _strip_leaked_preamble's docstring).
    market_text = _strip_leaked_preamble(market_text)
    # Both extras are optional and independently best-effort: each
    # returns None rather than raising, so a quiet day (or a broken
    # YouTube feed) simply leaves that section out of the report.
    sections = [market_text, bullets.get_daily_alert(), creators.get_creator_digest()]
    return "\n\n".join(s for s in sections if s)
