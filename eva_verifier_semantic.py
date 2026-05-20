"""
eva_verifier_semantic.py — LLM-based, history-aware semantic verifier (P1).

Replaces the regex-based semantic checks in eva_verifier_logic.py
(currently DEMOTED to soft-pass via ENABLE_SEMANTIC_HARD_FAIL=False).
Those regex checks suffered three structural problems:

  1. They tried to do PRONOUN / PERSPECTIVE / REFERENT reasoning with
     local pattern matches, which by design cannot read the chain of
     prior turns where the referent was established.
  2. The verifier's input was (answer, latest_user_text) — no history
     at all. The Turn-3 birthday loop (user asks "your birthday" in
     turn 3 of a thread already grounded on Eva's birthday in turns 1
     and 2) was un-resolvable from that input.
  3. Hard-failing on a regex false positive triggered regenerate, the
     same regex flagged the new answer too, and the dispatcher fell
     through to a canned "I got the pronouns tangled" apology — Eva
     would self-correct an answer that was actually right.

The semantic verifier ships in three modes (config:
SEMANTIC_VERIFIER_MODE):

  shadow  - default. Runs alongside the regex checks; logs verdicts
            into agent.turn_evidence.semantic_verdicts but does NOT
            influence dispatch. This is data-collection mode.
  warn    - verdict 'fail' becomes a soft warning logged to stdout
            (still does not change ok/not-ok).
  hard    - verdict 'fail' produces VerifyResult(ok=False, hard=True)
            with a non-canned reason; the regex semantic checks are
            considered superseded.

Failure mode (DeepSeek down / 401 / timeout / parse error): the
verifier returns SemanticVerdict(severity='pass', ...) with a noted
error. NEVER raises into the verifier dispatcher. Fail-open over
fail-closed — same posture as the rest of P0/P1.

Public surface:
    SemanticVerdict   - dataclass returned to the dispatcher
    SemanticVerifier  - stateful per-agent instance (cache + budget)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ============================================================
# Verdict dataclass
# ============================================================
@dataclass
class SemanticVerdict:
    """Structured judgement from the LLM auditor."""
    severity: str                     # 'pass' | 'warn' | 'fail'
    issues: List[Dict[str, Any]] = field(default_factory=list)
    raw_response: Optional[str] = None
    error: Optional[str] = None       # set when LLM call failed; severity='pass'
    cached: bool = False
    skipped_reason: Optional[str] = None  # 'disabled' | 'budget' | 'no_key'

    @property
    def ok(self) -> bool:
        """Convenience: did the verifier accept the answer?"""
        return self.severity != "fail"

    @property
    def is_failure(self) -> bool:
        return self.severity == "fail"

    def to_log(self) -> dict:
        return {
            "severity": self.severity,
            "issues": self.issues,
            "error": self.error,
            "cached": self.cached,
            "skipped_reason": self.skipped_reason,
        }


# ============================================================
# Prompts
# ============================================================
_SYSTEM_INSTRUCTION = """You are a strict correctness auditor for a conversational
AI assistant called Eva. Eva talks to one user (Master / Rosm). You will
receive a JSON payload describing the recent conversation, the user's
current message, Eva's just-produced answer, and a small `evidence_summary`
of facts the system already verified for this turn. Your job is to decide
whether Eva's answer has any of the bugs below — and ONLY these bugs.

# How to reason (do this in your head, then output JSON)

STEP 1 — Build a pronoun referent table for Eva's answer.
For every first/second-person reference in the answer ("I", "me", "my",
"mine", "you", "your", "yours"), name the entity it points to (Eva or
Rosm). Use the user's current message and the conversation history to
fix referents. Remember: Eva is the speaker of the answer, the user
(Rosm) is the addressee — so Eva's "I/my/mine" = Eva, Eva's "you/your/
yours" = Rosm. The user's "my" in the prior turn refers to Rosm; the
user's "your" refers to Eva.

STEP 2 — List every concrete factual claim Eva makes (dates, names,
numbers, ownership statements like "X is mine / X is yours / X belongs
to Y"). For each, note whether it can be checked against `evidence_summary`
or against another claim in the same answer.

STEP 3 — Apply the four bug checks below. Be conservative: only flag
when the bug is unambiguous given the data you have.

# Bug catalog

1. pronoun_referent_mismatch
   The answer attributes something to the wrong person via pronoun.
   Concrete pattern: Eva says "that's your X, not mine" or "your X is Y"
   when in fact X is Eva's (or vice versa), based on the user's question
   and the evidence_summary. The classic failure is Eva flipping a
   self-referential possession to the user, e.g. user asks about Eva's
   birthday, Eva replies "your birthday is July 7" attributing Eva's
   own date to the user.
   IMPORTANT: do not confuse `evidence_summary.memory_subject`. If the
   memory record's subject is "Rosm", that fact belongs to the user; if
   it's "Eva", it belongs to Eva. Read the subject explicitly.

2. internal_self_contradiction
   The answer makes two factual claims that cannot both be true. The
   common shape is one sentence saying "X is yours, not mine" and a
   later sentence saying "my X is the same value" — those contradict
   each other regardless of tone. Tsundere phrasing, hedging, or
   playful denial does NOT excuse a literal contradiction. If sentence
   A asserts ownership-by-Rosm of value V and sentence B asserts
   ownership-by-Eva of the same value V, that is a contradiction unless
   the answer explicitly notes the coincidence ("they're the same date").

3. perspective_slip
   Eva narrates about herself in the wrong grammatical person — e.g.
   referring to "Eva" in the third person inside her own answer when
   first person was clearly expected, or speaking AS the user.

4. fact_conflict_with_evidence
   The answer states a concrete fact (date, name, count) that
   contradicts a value in `evidence_summary`. Example: evidence has
   `date_days_until: 61` but the answer says "about 30 days". Only
   flag when the contradiction is direct and unambiguous; rough
   restatements ("about two months" for 61 days) are NOT contradictions.

# Style guard — DO NOT flag any of these

  - Tone, register, formality, tsundere or persona flavour.
  - Hedging, deflection, "I don't have a specific memory about that"
    style answers when memory was weak.
  - Missing information the user did not ask about.
  - Stylistic choices about WHEN to use first vs second person,
    provided the referent is correct.
  - Anything requiring outside-world knowledge you cannot verify
    from the payload.

# Severity rules

  - "fail": at least one bug found with confidence >= 0.80 AND the bug
    would visibly mislead the user. internal_self_contradiction and
    pronoun_referent_mismatch on factual claims are the textbook fail
    cases.
  - "warn": a plausible bug at lower confidence (0.55-0.79), or a
    contradiction softened by phrasing such that you're unsure if a
    user would be misled.
  - "pass": no bug, or all candidates below 0.55 confidence.

# Output format — STRICT JSON, no commentary, no markdown fences

  {"verdict": "pass" | "warn" | "fail",
   "issues": [
     {"type": "pronoun_referent_mismatch" | "internal_self_contradiction"
              | "perspective_slip" | "fact_conflict_with_evidence",
      "evidence": "<short quote from the answer + one-line reasoning>",
      "confidence": 0.0-1.0}
   ]}

If there are no issues, output {"verdict": "pass", "issues": []}.
Never invent a fifth bug type. Never include any field outside the
schema above."""


def _build_user_payload(answer: str,
                        latest_user_text: str,
                        history: List[Dict[str, str]],
                        evidence_summary: Optional[Dict[str, Any]]) -> str:
    """Compose the JSON payload sent as the user message."""
    payload = {
        "speaker": "Rosm",
        "assistant_name": "Eva",
        "conversation": [
            {"user": (h.get("user") or "").strip()[:600],
             "assistant": (h.get("assistant") or "").strip()[:600]}
            for h in history
        ],
        "current_user_text": (latest_user_text or "").strip()[:800],
        "current_answer": (answer or "").strip()[:1500],
        "evidence_summary": evidence_summary or {},
    }
    return json.dumps(payload, ensure_ascii=False)


# ============================================================
# Verifier
# ============================================================
class SemanticVerifier:
    """Per-agent semantic verifier. One instance lives on ChatAgent
    (agent.semantic_verifier) for the agent's lifetime; per-turn
    state (call counter, cache) is reset in reset_for_new_turn()."""

    def __init__(self):
        self._cache: Dict[str, SemanticVerdict] = {}
        self._calls_this_turn: int = 0

    # ------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------
    def reset_for_new_turn(self) -> None:
        self._cache.clear()
        self._calls_this_turn = 0

    # ------------------------------------------------------------
    # Public entrypoint
    # ------------------------------------------------------------
    def verify(self,
               answer: str,
               latest_user_text: str,
               history: List[Dict[str, str]],
               evidence_summary: Optional[Dict[str, Any]] = None) -> SemanticVerdict:
        """Return a verdict for `answer` given the surrounding context.

        Never raises. On any internal error returns a 'pass' verdict
        with `.error` set so the dispatcher can keep moving. The
        caller decides whether to act on the verdict (shadow / warn /
        hard) based on SEMANTIC_VERIFIER_MODE."""
        # Lazy imports — verifier_logic imports this module at top
        # level, but config and the deepseek client may not be ready
        # at import-time during boot.
        from eva_config import (
            ENABLE_SEMANTIC_VERIFIER,
            SEMANTIC_VERIFIER_MAX_CALLS_PER_TURN,
            SEMANTIC_VERIFIER_CACHE_ENABLED,
            SEMANTIC_VERIFIER_DEBUG,
            DEEPSEEK_API_KEY,
        )

        if not ENABLE_SEMANTIC_VERIFIER:
            return SemanticVerdict(severity="pass", skipped_reason="disabled")
        if not DEEPSEEK_API_KEY:
            return SemanticVerdict(severity="pass", skipped_reason="no_key")

        # Cache lookup (per-turn, cleared in reset_for_new_turn)
        cache_key = self._cache_key(answer, latest_user_text, history, evidence_summary)
        if SEMANTIC_VERIFIER_CACHE_ENABLED and cache_key in self._cache:
            cached = self._cache[cache_key]
            return SemanticVerdict(
                severity=cached.severity,
                issues=list(cached.issues),
                raw_response=cached.raw_response,
                error=cached.error,
                cached=True,
                skipped_reason=cached.skipped_reason,
            )

        # Per-turn budget
        if self._calls_this_turn >= int(SEMANTIC_VERIFIER_MAX_CALLS_PER_TURN):
            if SEMANTIC_VERIFIER_DEBUG:
                print(f"        | [DEBUG] semantic verifier: turn budget exhausted "
                      f"({self._calls_this_turn} calls)")
            return SemanticVerdict(severity="pass", skipped_reason="budget")

        verdict = self._call_judge(
            answer=answer,
            latest_user_text=latest_user_text,
            history=history,
            evidence_summary=evidence_summary,
            debug=bool(SEMANTIC_VERIFIER_DEBUG),
        )
        self._calls_this_turn += 1
        if SEMANTIC_VERIFIER_CACHE_ENABLED:
            self._cache[cache_key] = verdict
        return verdict

    # ------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------
    def _cache_key(self,
                   answer: str,
                   latest_user_text: str,
                   history: List[Dict[str, str]],
                   evidence_summary: Optional[Dict[str, Any]]) -> str:
        h = hashlib.sha1()
        h.update((answer or "").encode("utf-8", "replace"))
        h.update(b"\x1e")
        h.update((latest_user_text or "").encode("utf-8", "replace"))
        h.update(b"\x1e")
        for turn in history or []:
            h.update((turn.get("user") or "").encode("utf-8", "replace"))
            h.update(b"\x1f")
            h.update((turn.get("assistant") or "").encode("utf-8", "replace"))
            h.update(b"\x1f")
        h.update(b"\x1e")
        h.update(json.dumps(evidence_summary or {}, sort_keys=True,
                            ensure_ascii=False).encode("utf-8", "replace"))
        return h.hexdigest()

    def _call_judge(self,
                    answer: str,
                    latest_user_text: str,
                    history: List[Dict[str, str]],
                    evidence_summary: Optional[Dict[str, Any]],
                    debug: bool) -> SemanticVerdict:
        try:
            from eva_tools_runtime import call_deepseek_judge
        except Exception as e:
            return SemanticVerdict(severity="pass",
                                   error=f"import_failed:{e!r}")

        payload = _build_user_payload(
            answer=answer,
            latest_user_text=latest_user_text,
            history=history,
            evidence_summary=evidence_summary,
        )
        try:
            from eva_config import SEMANTIC_VERIFIER_TIMEOUT_SECONDS
            timeout = float(SEMANTIC_VERIFIER_TIMEOUT_SECONDS)
        except Exception:
            timeout = None  # call_deepseek_judge falls back to LLM_JUDGE_TIMEOUT_SECONDS

        try:
            result = call_deepseek_judge(
                _SYSTEM_INSTRUCTION,
                payload,
                debug=debug,
                timeout=timeout,
            )
        except Exception as e:
            return SemanticVerdict(severity="pass",
                                   error=f"judge_call_failed:{e!r}")

        # call_deepseek_judge contract:
        #   - on success: returns the parsed DeepSeek JSON body
        #     directly (e.g. {"verdict": "...", "issues": [...]}).
        #     The body has NO "ok" or "error" key unless the model
        #     itself produced one.
        #   - on failure: returns {"ok": None, "error": "<why>"}.
        # The original P1 code checked `result.get("ok") is None`,
        # which is True for *both* outcomes since success bodies
        # also have no "ok" key — that wrongly classified every
        # well-formed verdict as 'judge_unavailable'. The correct
        # discriminator is the presence of an explicit error.
        if not isinstance(result, dict):
            return SemanticVerdict(severity="pass",
                                   error=f"unexpected_judge_result_type:{type(result).__name__}")
        if "error" in result and result.get("ok") is None:
            return SemanticVerdict(severity="pass",
                                   error=f"judge_unavailable:{result.get('error')}")

        # Parse the judge body into our verdict shape. The local helper
        # accepts both the {"verdict": ..., "issues": [...]} contract
        # we asked for AND a degenerate {"ok": true/false} shape that
        # call_deepseek_judge sometimes flattens to.
        verdict, issues, raw = _coerce_judge_payload(result)
        return SemanticVerdict(
            severity=verdict,
            issues=issues,
            raw_response=raw,
            error=None,
        )


# ============================================================
# Judge payload coercion
# ============================================================
_VALID_SEVERITIES = {"pass", "warn", "fail"}


def _coerce_judge_payload(result: dict):
    """Normalise the DeepSeek body into (severity, issues, raw_text)."""
    raw_text = json.dumps(result, ensure_ascii=False)[:1500]

    # Preferred shape: result already has "verdict" / "issues".
    verdict = result.get("verdict")
    if isinstance(verdict, str) and verdict.strip().lower() in _VALID_SEVERITIES:
        sev = verdict.strip().lower()
        issues = result.get("issues") or []
        if not isinstance(issues, list):
            issues = []
        return sev, issues, raw_text

    # Sometimes the body is the parsed JSON object nested under "data".
    inner = result.get("data") if isinstance(result.get("data"), dict) else None
    if inner:
        v2 = inner.get("verdict")
        if isinstance(v2, str) and v2.strip().lower() in _VALID_SEVERITIES:
            sev = v2.strip().lower()
            issues = inner.get("issues") or []
            if not isinstance(issues, list):
                issues = []
            return sev, issues, raw_text

    # Degenerate shape: {"ok": True/False}. Treat ok=False as warn (we
    # don't know which check failed, refuse to escalate to fail).
    ok = result.get("ok")
    if ok is True:
        return "pass", [], raw_text
    if ok is False:
        return "warn", [{"type": "judge_flat_ok_false",
                         "evidence": "judge returned ok=False without verdict body",
                         "confidence": 0.5}], raw_text

    # Anything else: treat as pass with an unknown-shape note. Logged
    # but never escalated, so a malformed judge response cannot break
    # the conversation.
    return "pass", [], raw_text
