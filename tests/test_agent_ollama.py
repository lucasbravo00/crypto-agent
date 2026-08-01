"""
Tests for src/agent_ollama.py's _strip_leaked_preamble -- a code-level
safety net for a real, observed failure (2026-08-01): after several rounds
of "you still need to call these tools" nudging, the local model
sometimes narrates its own answering process as the opening sentence
instead of just answering the report. No network, no Ollama.
"""
from src.agent_ollama import _strip_leaked_preamble


def test_strips_the_exact_leaked_preamble_that_motivated_this():
    text = (
        "Con eso, puedo finalizar la respuesta. El precio de BTC/USDT "
        "está en 63.044,53 USDT. El indicador SMA50 está en 63.406,05."
    )
    result = _strip_leaked_preamble(text)
    assert result == "El precio de BTC/USDT está en 63.044,53 USDT. El indicador SMA50 está en 63.406,05."
    assert "Con eso" not in result


def test_strips_an_english_leaked_preamble():
    text = "With that, I can finish my answer. Bitcoin is up 2% today, trading at $64,000."
    result = _strip_leaked_preamble(text)
    assert result == "Bitcoin is up 2% today, trading at $64,000."


def test_strips_a_here_is_my_report_style_preamble():
    text = "Aquí tienes el informe: Bitcoin bajó levemente hoy, sin cambios de tendencia."
    result = _strip_leaked_preamble(text)
    assert result == "Bitcoin bajó levemente hoy, sin cambios de tendencia."


def test_leaves_clean_reports_untouched():
    text = "Bitcoin bajó a 63.044 USDT, sin cambios de fondo en la tendencia de mediano plazo."
    assert _strip_leaked_preamble(text) == text


def test_does_not_strip_a_legitimate_sentence_that_shares_a_trigger_word():
    """The safety rail: a real market sentence carries a number. A
    legitimate opening like "Con el RSI en 44..." must survive even
    though it starts with the same word ("con") as a trigger phrase --
    the digit is what tells them apart."""
    text = "Con el RSI en 44.47, el mercado se ve estable en el corto plazo."
    assert _strip_leaked_preamble(text) == text


def test_only_strips_the_first_sentence_even_with_multiple_periods():
    text = "Con eso, puedo finalizar la respuesta. Bitcoin subió 1%. El RSI está en 50."
    result = _strip_leaked_preamble(text)
    assert result == "Bitcoin subió 1%. El RSI está en 50."


def test_never_returns_an_empty_report():
    """If the ENTIRE text is a preamble with nothing after it, stripping
    would leave nothing to send -- keep the suspicious text instead of
    delivering a blank report."""
    text = "Con eso, puedo finalizar la respuesta."
    assert _strip_leaked_preamble(text) == text


def test_handles_empty_and_none_input():
    assert _strip_leaked_preamble("") == ""
    assert _strip_leaked_preamble(None) is None


def test_strips_a_based_on_the_tools_preamble():
    text = "Based on the tools I called. Bitcoin trades at $64,000, up 1% today."
    result = _strip_leaked_preamble(text)
    assert result == "Bitcoin trades at $64,000, up 1% today."


def test_does_not_strip_a_comma_joined_preamble_with_no_sentence_break():
    """Documented limitation, not a bug: without a '.', '!', ':' or
    newline separating the meta-remark from the real content, the two
    are one sentence and the digit-safety-rail can't tell them apart
    without risking a legitimate lead-in clause. The prompt's rule 0 is
    the first line of defense for this shape; this is only the net."""
    text = "Based on the tools I called, Bitcoin trades at $64,000, up 1% today."
    assert _strip_leaked_preamble(text) == text
