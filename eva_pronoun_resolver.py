"""
eva_pronoun_resolver.py — LLM-driven pronoun-followup detection +
antecedent extraction.

P6 (2026-05-08): replaces the regex-based pronoun-followup pipeline
in eva_verifier_logic.build_required_memory_params. The original
pipeline was two heuristics in series:

    _is_pronoun_followup(query)
        — regex over query shape ("check it" / "really? do that" / ...)
        — answers ONLY "is this a follow-up?", not "to what?"

    _extract_topical_nouns_from_recent_turns(turns)
        — regex over assistant text (quoted phrases, capitalised
          bigrams, lowercase singletons)
        — provides candidate antecedents

The two heuristics composed serially, so each layer's misses
compounded. P5 → P5.1 already required two regex revisions to catch
common skepticism prefixes ("really?", "wait,") — and the next
unseen variant ("hold on,", "sorry,", "um...") would require yet
another revision.

This module collapses both questions into a single DeepSeek call.
The LLM reads the user query plus recent_turns and returns:

    {
        "needs_resolution": <bool>,
        "antecedents":      <list[str], 1..3 phrases ranked>,
        "confidence":       <float 0..1>,
    }

Public surface:
    PronounResolution     — dataclass holding the resolver verdict
    resolve_pronoun(...)  — main entry point used by verifier
    PROMPT_PRONOUN_RESOLVER — the system prompt sent to DeepSeek

Design contracts (locked in docs/P6_pronoun_resolver_refactor_v3.md):

  - State is caller-owned (JudgeState shared with eva_intent_judge).
    The caller resets state.pronoun_call_count at turn boundaries.

  - Budget pool is INDEPENDENT from LLM_JUDGE_MAX_CALLS_PER_TURN.
    PRE PROBE / Plan B exhausting their pool does NOT starve the
    resolver, and vice versa. See JudgeState.pronoun_call_count.

  - Three execution stages, in order:
      1. Cheap gate — short-circuit if query is empty / too long /
         lacks pronoun trigger tokens. Returns source="skip" without
         touching the LLM.
      2. LLM main path — DeepSeek call. Returns source="llm" on
         success, falls through to Stage 3 on failure.
      3. Regex fallback — delegates to eva_verifier_logic._is_pronoun_*
         helpers. Returns source="regex". This stage is REMOVED in
         P6.4; until then it preserves pre-P6 behaviour when the
         LLM is unavailable.

  - confidence < PRONOUN_RESOLVER_MIN_CONFIDENCE is treated as
    needs_resolution=False. The LLM is calibrated such that
    confidence < 0.6 means "I'm guessing" — better to miss a pronoun
    than to inject a wrong antecedent into the verifier query.

  - antecedents are BARE noun phrases without articles ("music box"
    not "the music box"). Matches the downstream stop-word filter
    in build_required_memory_params, where leading "the"/"a" would
    be stripped to empty anyway.
"""

from dataclasses import dataclass, field
from typing import List, Optional


__all__ = [
    "PronounResolution",
    "resolve_pronoun",
    "PROMPT_PRONOUN_RESOLVER",
]


# ============================================================
# Cheap gate constants
# ============================================================
# Tokens whose appearance is necessary (but not sufficient) for a
# pronoun follow-up. If NONE of these appear in a short query, we
# skip the LLM call. The list is intentionally narrow — adding
# nouns like "thing" / "stuff" creates false positives for ordinary
# questions (e.g. "what's the best thing to eat").
_PRONOUN_TRIGGERS = frozenset({
    "it", "that", "this", "them", "those", "these",
    "him", "her", "his", "hers",
    "again", "too",            # "do it again", "try that too"
})

# Short interjections that often PRECEDE a pronoun follow-up but
# don't themselves count as triggers — used for word-cap normalisation
# (we don't want a 4-word interjection-prefix to push us over the
# word cap when the resolvable part is only 2 words).
_LEADING_INTERJECTIONS = frozenset({
    "really", "wait", "hmm", "huh", "hold", "on",
    "sorry", "um", "uh", "oh", "ah", "okay", "ok", "well",
    "hey", "yeah", "yep",
})


# ============================================================
# Verdict dataclass
# ============================================================
@dataclass
class PronounResolution:
    """Verdict returned by resolve_pronoun().

    Fields:
        needs_resolution:
            True iff the user query refers back to something in the
            conversation history and cannot be searched on its own.
            Callers treat False as "use the cleaned query unchanged".

        antecedents:
            Ranked noun phrases (most likely first), 1..3 entries.
            Empty list when needs_resolution=False.
            Always BARE noun phrases — no articles, no possessives
            other than the original possessor ("Master's birthday"
            keeps the apostrophe-s).

        confidence:
            LLM self-reported certainty in [0.0, 1.0]. The caller
            applies PRONOUN_RESOLVER_MIN_CONFIDENCE to decide whether
            to honour needs_resolution=True.

        source:
            One of {"skip", "llm", "regex", "off"}.
            - "skip": cheap gate rejected the query.
            - "llm":  Stage 2 succeeded.
            - "regex": Stage 3 fallback fired (LLM unavailable /
              parse error / over budget). Removed in P6.4.
            - "off":  resolver disabled by config.

        reasoning:
            Optional debug string. Populated only when
            PRONOUN_RESOLVER_DEBUG=True (in either LLM or regex
            stages). Never used to drive control flow.
    """
    needs_resolution: bool = False
    antecedents: List[str] = field(default_factory=list)
    confidence: float = 1.0
    source: str = "skip"
    reasoning: str = ""


# ============================================================
# System prompt — DeepSeek main path
# ============================================================
# P6.3 latency tuning (2026-05-08): prompt was ~390 tokens with 4
# examples. Compressed to ~170 tokens, 2 examples — DeepSeek server
# generation time scales with prompt + output tokens, so trimming
# the system prompt is the biggest free latency win available
# without architecture changes. Quality must be re-validated via
# small shadow probe (target: P50 drop 30-50%, no agreement-rate
# regression).
PROMPT_PRONOUN_RESOLVER = (
    "Resolve pronoun antecedents for a chatbot turn. "
    "Output STRICT JSON only:\n"
    '{"needs_resolution": <bool>, '
    '"antecedents": ["<bare noun phrase>", ...], '
    '"confidence": <0.0-1.0>}\n\n'
    "Rules:\n"
    "- needs_resolution=true ONLY when the message can't be understood "
    "without history (contains it/that/them/this/those, or short "
    "follow-ups like \"check it\", \"do it again\", \"really?\").\n"
    "- antecedents: bare noun phrases, NO articles "
    "(\"music box\" not \"the music box\"). Keep possessives "
    "(\"Master's birthday\").\n"
    "- 1-3 ranked candidates, most likely first.\n"
    "- needs_resolution=false → antecedents=[].\n"
    "- confidence < 0.6 means guessing.\n\n"
    "Examples:\n"
    "user: \"check it\"\n"
    "history: [..., assistant: \"I have my music box.\"]\n"
    "→ {\"needs_resolution\": true, \"antecedents\": [\"music box\"], "
    "\"confidence\": 0.95}\n\n"
    "user: \"what's the weather?\"\n"
    "history: [...]\n"
    "→ {\"needs_resolution\": false, \"antecedents\": [], "
    "\"confidence\": 1.0}\n\n"
    "Input JSON: query + recent_turns (newest last). "
    "Output JSON only, no prose."
)


# ============================================================
# Stage 1 — cheap gate
# ============================================================
def _cheap_gate(query: str, max_words: int) -> bool:
    """Return True iff the query is plausibly a pronoun follow-up
    worth the LLM call.

    Rules:
      - Empty / whitespace-only query: reject.
      - More than max_words tokens: reject. Real follow-ups are
        short ("check it", "really? do that"); a long sentence
        almost always contains its own anchor and doesn't need
        antecedent resolution.
      - No pronoun trigger token appears: reject. We only resolve
        antecedents when there's literally a pronoun or short-form
        cue to resolve.

    Tokenisation is intentionally cheap: split on whitespace,
    lowercase, strip ASCII punctuation. We don't import re for
    this — the function runs on every user turn and the cost
    matters.
    """
    if not isinstance(query, str):
        return False
    qs = query.strip()
    if not qs:
        return False
    # Cheap tokenise: whitespace split, lowercase, strip punctuation.
    raw_tokens = qs.lower().split()
    tokens = []
    for t in raw_tokens:
        # Strip leading/trailing ASCII punctuation. Keep apostrophes
        # inside words ("master's", "what's").
        t = t.strip(".,!?;:\"'()[]{}<>")
        if t:
            tokens.append(t)
    if not tokens:
        return False
    if len(tokens) > max_words:
        return False
    # Need at least one trigger token in the query.
    if not any(t in _PRONOUN_TRIGGERS for t in tokens):
        return False
    return True


# ============================================================
# Stage 2 — LLM main path
# ============================================================
def _call_llm(query: str, recent_turns) -> Optional[dict]:
    """Issue the DeepSeek call. Returns parsed dict or None.

    None means "LLM unavailable / parse failure". Caller falls
    through to Stage 3.
    """
    from eva_config import PRONOUN_RESOLVER_DEBUG
    from eva_tools_runtime import call_deepseek_judge
    import json as _json

    payload_dict = {"query": query}
    if recent_turns:
        compact_turns = []
        for t in recent_turns[-3:]:
            u = (t.get("user") or "").strip() if isinstance(t, dict) else ""
            a = (t.get("assistant") or "").strip() if isinstance(t, dict) else ""
            if not (u or a):
                continue
            compact_turns.append({
                "user": u[:300],
                "assistant": a[:400],
            })
        if compact_turns:
            payload_dict["recent_turns"] = compact_turns

    payload = _json.dumps(payload_dict, ensure_ascii=False)
    try:
        result = call_deepseek_judge(
            PROMPT_PRONOUN_RESOLVER, payload,
            debug=PRONOUN_RESOLVER_DEBUG,
        )
    except Exception as e:
        if PRONOUN_RESOLVER_DEBUG:
            print(f"        | [PRONOUN] LLM call raised: {e!r}")
        return None
    if not isinstance(result, dict):
        return None
    # call_deepseek_judge returns {"ok": None, "error": ...} on its
    # own error path. Treat as failure.
    if "error" in result and "needs_resolution" not in result:
        if PRONOUN_RESOLVER_DEBUG:
            print(f"        | [PRONOUN] LLM error: "
                  f"{result.get('error')!r}")
        return None
    return result


def _parse_llm_verdict(raw: dict) -> Optional[PronounResolution]:
    """Parse the DeepSeek response into a PronounResolution.

    Returns None on any schema violation — the caller will then fall
    through to Stage 3. We are strict here: a malformed verdict is
    LESS useful than the regex fallback's heuristic.
    """
    if not isinstance(raw, dict):
        return None
    needs = raw.get("needs_resolution")
    ants = raw.get("antecedents")
    conf = raw.get("confidence")
    if not isinstance(needs, bool):
        return None
    if not isinstance(ants, list):
        return None
    if not isinstance(conf, (int, float)):
        return None
    # Coerce antecedents to strings, trim, drop empties, cap at 3.
    cleaned_ants: List[str] = []
    for a in ants:
        if not isinstance(a, str):
            continue
        s = a.strip()
        # Defensive: strip leading articles even though prompt
        # forbids them — real LLM output occasionally slips.
        for art in ("the ", "a ", "an "):
            if s.lower().startswith(art):
                s = s[len(art):].strip()
                break
        if s and s not in cleaned_ants:
            cleaned_ants.append(s)
        if len(cleaned_ants) >= 3:
            break

    return PronounResolution(
        needs_resolution=needs,
        antecedents=cleaned_ants,
        confidence=max(0.0, min(1.0, float(conf))),
        source="llm",
        reasoning="",
    )


# ============================================================
# Stage 3 — regex fallback (removed in P6.4)
# ============================================================
def _regex_fallback(query: str, recent_turns) -> PronounResolution:
    """Delegate to eva_verifier_logic._is_pronoun_followup +
    _extract_topical_nouns_from_recent_turns. Wraps the result in a
    PronounResolution.

    This stage exists only during P6.0–P6.3. P6.4 deletes the regex
    helpers entirely; this function then returns
    PronounResolution(needs_resolution=False, source="skip").
    """
    # Lazy import — eva_verifier_logic itself will eventually call
    # back into resolve_pronoun, so we MUST NOT import at module top.
    #
    # Two failure modes look identical to a naive `from X import Y` —
    # disambiguate explicitly:
    #   (a) module fails to load (transient: missing dep, syntax error
    #       in eva_verifier_logic). This is a REAL environment problem
    #       we don't want to hide; degrade gracefully but mark the
    #       reasoning so it surfaces in trace.
    #   (b) module loads but the regex helpers are gone (P6.4 final
    #       state, intentional deletion).
    try:
        import eva_verifier_logic as _vl
    except Exception as exc:
        # (a) — log via reasoning. Don't re-raise: the resolver's
        # contract is to never raise for routine failure.
        return PronounResolution(
            needs_resolution=False,
            antecedents=[],
            confidence=1.0,
            source="skip",
            reasoning=f"regex fallback unavailable: "
                      f"eva_verifier_logic import failed ({exc!r})",
        )
    _is_pronoun_followup = getattr(_vl, "_is_pronoun_followup", None)
    _extract_topical_nouns_from_recent_turns = getattr(
        _vl, "_extract_topical_nouns_from_recent_turns", None,
    )
    if (_is_pronoun_followup is None
            or _extract_topical_nouns_from_recent_turns is None):
        # (b) — P6.4 state.
        return PronounResolution(
            needs_resolution=False,
            antecedents=[],
            confidence=1.0,
            source="skip",
            reasoning="regex helpers removed (P6.4)",
        )

    matched = bool(_is_pronoun_followup(query))
    if not matched:
        return PronounResolution(
            needs_resolution=False,
            antecedents=[],
            confidence=1.0,
            source="regex",
            reasoning="regex: no pronoun-followup pattern matched",
        )
    # P6.1 contract: keep max_terms=6 to match the legacy
    # build_required_memory_params behaviour bit-for-bit. The new
    # 1..3 contract from § 五 of the v3 plan applies to the LLM
    # main path; the regex fallback intentionally preserves the
    # pre-P6 keyword set so MODE="regex_only" ships dark with no
    # observable downstream change. (P6.4 deletes this fallback
    # entirely, at which point the cap becomes moot.)
    try:
        terms = _extract_topical_nouns_from_recent_turns(
            recent_turns or [], max_terms=6,
        )
    except Exception:
        terms = []
    # Strip stop-word artefacts the upstream helper might leak through
    # (it already filters, but a defensive pass costs nothing).
    cleaned: List[str] = []
    for t in terms:
        if not isinstance(t, str):
            continue
        s = t.strip()
        if s and s not in cleaned:
            cleaned.append(s)
    return PronounResolution(
        needs_resolution=True,
        antecedents=cleaned,
        # Regex confidence is fixed — it has no probability estimate.
        # Pick a value above the default min_confidence so callers
        # honour the verdict.
        confidence=0.75,
        source="regex",
        reasoning="regex: pattern matched, antecedents from "
                  "_extract_topical_nouns_from_recent_turns",
    )


# ============================================================
# Public entry point
# ============================================================
def resolve_pronoun(
    query: str,
    recent_turns,
    *,
    state,
) -> PronounResolution:
    """Decide whether `query` needs antecedent resolution and, if so,
    extract the antecedent(s).

    Args:
        query: cleaned user text. Caller is responsible for
            applying clean_user_text() upstream — this function
            does NOT re-clean.
        recent_turns: list of dicts with keys "user" / "assistant".
            Newest last. May be empty or None.
        state: JudgeState (eva_intent_judge.JudgeState) carrying the
            per-turn cache and pronoun_call_count budget. Required
            keyword-only argument so callers can't accidentally
            forget to share state across resolver / judge calls.

    Returns:
        PronounResolution. Always returns a verdict; never raises
        for routine failure (LLM down, parse error, etc.).
    """
    from eva_config import (
        ENABLE_PRONOUN_RESOLVER,
        PRONOUN_RESOLVER_MODE,
        PRONOUN_RESOLVER_MIN_CONFIDENCE,
        PRONOUN_RESOLVER_DEBUG,
        PRONOUN_RESOLVER_MAX_WORDS,
        PRONOUN_RESOLVER_MAX_CALLS_PER_TURN,
    )

    # ----- Disabled fast path -----
    if (not ENABLE_PRONOUN_RESOLVER) or PRONOUN_RESOLVER_MODE == "off":
        return PronounResolution(
            needs_resolution=False,
            antecedents=[],
            confidence=1.0,
            source="off",
            reasoning="resolver disabled by config",
        )

    if not isinstance(query, str) or not query.strip():
        return PronounResolution(source="skip")

    # ----- Cache lookup -----
    # Key includes a fingerprint of recent_turns so the same query
    # under different histories doesn't collide.
    turns_fp = ""
    if recent_turns:
        try:
            joined = "|".join(
                ((t.get("assistant") or "")[:64].strip().lower())
                for t in recent_turns[-3:]
                if isinstance(t, dict)
            )
            turns_fp = joined[:200]
        except Exception:
            turns_fp = ""
    cache_key = ("PRONOUN", f"{query.strip().lower()}|{turns_fp}")
    cached = state.cache.get(cache_key) if state is not None else None
    if isinstance(cached, PronounResolution):
        return cached

    # ----- Stage 1: cheap gate -----
    if not _cheap_gate(query, PRONOUN_RESOLVER_MAX_WORDS):
        verdict = PronounResolution(
            needs_resolution=False,
            antecedents=[],
            confidence=1.0,
            source="skip",
            reasoning="cheap gate: empty / too long / no trigger token",
        )
        if state is not None:
            state.cache[cache_key] = verdict
        if PRONOUN_RESOLVER_DEBUG:
            short_q = query[:60] + ("…" if len(query) > 60 else "")
            print(f"        | [PRONOUN] q={short_q!r} source=skip "
                  f"(cheap gate)")
        return verdict

    # ----- P6.2 shadow mode -----
    # Orthogonal to MODE. When SHADOW=True and MODE="llm_first", run
    # BOTH paths, log a comparison trace, but adopt the regex verdict
    # (preserves P6.1 decision behaviour). Lets operators sample the
    # divergence rate before committing to P6.3.
    #
    # Skip caching in this branch: we'd otherwise need to cache both
    # verdicts together, and the per-turn cost is bounded (≤2 LLM
    # calls per turn × small fraction of turns past the cheap gate).
    from eva_config import PRONOUN_RESOLVER_SHADOW
    if PRONOUN_RESOLVER_MODE == "llm_first" and PRONOUN_RESOLVER_SHADOW:
        import time as _time
        regex_v = _regex_fallback(query, recent_turns)
        llm_v: Optional[PronounResolution] = None
        latency_ms: Optional[float] = None
        # Honour the budget — shadow LLM calls are real LLM calls.
        if state.pronoun_call_count < PRONOUN_RESOLVER_MAX_CALLS_PER_TURN:
            state.pronoun_call_count += 1
            t0 = _time.perf_counter()
            raw = _call_llm(query, recent_turns)
            latency_ms = (_time.perf_counter() - t0) * 1000.0
            if raw is not None:
                parsed = _parse_llm_verdict(raw)
                if parsed is not None:
                    llm_v = parsed
        _shadow_trace(query, regex_v, llm_v, latency_ms=latency_ms)
        # Adopt regex verdict (P6.1 behaviour). Apply min_confidence
        # for consistency with the regular flow — no-op for regex
        # (fixed 0.75 conf > default 0.6) but keeps the contract
        # uniform.
        if regex_v.confidence < PRONOUN_RESOLVER_MIN_CONFIDENCE:
            regex_v = PronounResolution(
                needs_resolution=False,
                antecedents=[],
                confidence=regex_v.confidence,
                source=regex_v.source,
                reasoning=regex_v.reasoning + " (below min_confidence)",
            )
        if PRONOUN_RESOLVER_DEBUG:
            _trace(query, regex_v)
        return regex_v

    # ----- regex_only mode short-circuit -----
    # P6.1 starts in this mode: wiring lands but behaviour is
    # bit-identical to pre-P6 (regex still drives every decision).
    if PRONOUN_RESOLVER_MODE == "regex_only":
        verdict = _regex_fallback(query, recent_turns)
        # Honour min_confidence even on regex (regex is fixed-conf,
        # so this is a no-op in practice — but keeps the contract
        # uniform across modes).
        if verdict.confidence < PRONOUN_RESOLVER_MIN_CONFIDENCE:
            verdict = PronounResolution(
                needs_resolution=False,
                antecedents=[],
                confidence=verdict.confidence,
                source=verdict.source,
                reasoning=verdict.reasoning + " (below min_confidence)",
            )
        if state is not None:
            state.cache[cache_key] = verdict
        if PRONOUN_RESOLVER_DEBUG:
            _trace(query, verdict)
        return verdict

    # ----- Stage 2: LLM main path -----
    # Budget check. Note: independent counter, NOT call_count.
    if state is None or state.pronoun_call_count >= PRONOUN_RESOLVER_MAX_CALLS_PER_TURN:
        if PRONOUN_RESOLVER_DEBUG:
            cnt = (state.pronoun_call_count if state is not None else "no-state")
            print(f"        | [PRONOUN] budget exhausted "
                  f"({cnt}/{PRONOUN_RESOLVER_MAX_CALLS_PER_TURN}); "
                  f"falling through to regex")
        verdict = _regex_fallback(query, recent_turns)
        if state is not None:
            state.cache[cache_key] = verdict
        if PRONOUN_RESOLVER_DEBUG:
            _trace(query, verdict)
        return verdict

    # Increment BEFORE call so a raised exception can't loop us.
    state.pronoun_call_count += 1
    # UX placeholder so operators see the ~3s DeepSeek call as
    # "in flight" rather than dead air. Lazy import keeps the
    # resolver's module-load surface unchanged for shadow-mode tests.
    from eva_intent_judge import announce_pending_llm
    short_q = query[:60] + ("…" if len(query) > 60 else "")
    announce_pending_llm(f"resolve_pronoun: q={short_q!r}")
    raw = _call_llm(query, recent_turns)
    parsed: Optional[PronounResolution] = None
    if raw is not None:
        parsed = _parse_llm_verdict(raw)

    if parsed is None:
        # LLM unavailable or schema-malformed → Stage 3.
        verdict = _regex_fallback(query, recent_turns)
    else:
        # Apply min_confidence: low-confidence LLM verdicts are
        # demoted to needs_resolution=False, keeping the antecedents
        # for debug visibility but not driving downstream behaviour.
        if parsed.confidence < PRONOUN_RESOLVER_MIN_CONFIDENCE:
            verdict = PronounResolution(
                needs_resolution=False,
                antecedents=parsed.antecedents,
                confidence=parsed.confidence,
                source="llm",
                reasoning="below min_confidence — not honoured",
            )
        else:
            verdict = parsed

    state.cache[cache_key] = verdict
    if PRONOUN_RESOLVER_DEBUG:
        _trace(query, verdict)
    return verdict


# ============================================================
# Trace helpers
# ============================================================
def _trace(query: str, verdict: PronounResolution) -> None:
    """Print a one-line trace summary. Only called when
    PRONOUN_RESOLVER_DEBUG=True; cheap to inline.
    """
    short_q = query[:60] + ("…" if len(query) > 60 else "")
    ants = verdict.antecedents
    print(
        f"        | [PRONOUN] q={short_q!r} source={verdict.source} "
        f"needs={verdict.needs_resolution} "
        f"antecedents={ants} conf={verdict.confidence:.2f}"
    )


def _normalise_for_overlap(s: str) -> str:
    """Light normalisation for Jaccard comparison. Case-fold + trim;
    no lemma / stemming yet (would require spaCy). The shadow trace
    can flag false-mismatches caused by plurals or capitalisation;
    if those dominate the diff log we'll add stemming, but pulling
    in a heavy NLP dep speculatively isn't justified.
    """
    return (s or "").strip().lower()


def _jaccard(a, b) -> float:
    """Jaccard similarity of two string sequences (case-fold normalised).

    Returns 1.0 when both lists are empty (vacuously equal — both
    paths agree there's no antecedent).
    """
    sa = {_normalise_for_overlap(x) for x in (a or []) if isinstance(x, str)}
    sa.discard("")
    sb = {_normalise_for_overlap(x) for x in (b or []) if isinstance(x, str)}
    sb.discard("")
    if not sa and not sb:
        return 1.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 1.0


def _shadow_trace(
    query: str,
    regex_v: PronounResolution,
    llm_v: Optional[PronounResolution],
    *,
    latency_ms: Optional[float] = None,
) -> None:
    """Print the [PRONOUN-SHADOW] comparison line for P6.2.

    Format (matches v3 plan § 七, plus latency_ms for the P95 metric):
        | [PRONOUN-SHADOW] q='...' regex_needs=T llm_needs=T
        |                  regex_terms=[...] llm_ants=[...]
        |                  agree=T overlap=0.50 llm_conf=0.92 latency_ms=512.3

    When llm_v is None (LLM unavailable / parse failed / over budget),
    prints llm_needs=None and overlap=N/A so operators can quantify
    LLM availability separately from agreement. latency_ms is still
    populated when the LLM was *called* but parse failed — that's the
    "DeepSeek timed out / returned malformed JSON" case which still
    counts toward the latency distribution. latency_ms=None means the
    call was skipped (over budget).

    Always prints regardless of PRONOUN_RESOLVER_DEBUG — the whole
    point of shadow mode is to gather audit data; gating it on a
    debug flag would defeat the purpose.
    """
    short_q = query[:60] + ("…" if len(query) > 60 else "")
    lat_str = (f"{latency_ms:.1f}" if latency_ms is not None else "N/A")
    if llm_v is None:
        print(
            f"        | [PRONOUN-SHADOW] q={short_q!r} "
            f"regex_needs={regex_v.needs_resolution} llm_needs=None"
        )
        print(
            f"        |                  regex_terms={regex_v.antecedents} "
            f"llm_ants=None"
        )
        print(
            f"        |                  agree=N/A overlap=N/A "
            f"latency_ms={lat_str} (LLM unavailable)"
        )
        return
    overlap = _jaccard(regex_v.antecedents, llm_v.antecedents)
    agree = (
        regex_v.needs_resolution == llm_v.needs_resolution
        and overlap >= 0.5
    )
    print(
        f"        | [PRONOUN-SHADOW] q={short_q!r} "
        f"regex_needs={regex_v.needs_resolution} "
        f"llm_needs={llm_v.needs_resolution}"
    )
    print(
        f"        |                  regex_terms={regex_v.antecedents} "
        f"llm_ants={llm_v.antecedents}"
    )
    print(
        f"        |                  agree={agree} "
        f"overlap={overlap:.2f} llm_conf={llm_v.confidence:.2f} "
        f"latency_ms={lat_str}"
    )
