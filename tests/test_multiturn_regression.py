"""
test_multiturn_regression.py — Multi-turn integration regression suite.

Why this file is separate from test_p2_regression.py
-----------------------------------------------------
test_p2_regression.py is the 21-case single-turn regression baseline.
That suite has been green for many sessions and we don't want to disturb
its schema. Multi-turn behaviors (verifier loop fallback, deferred
compound queries, anaphoric continuation) need a different schema —
each case is a sequence of (user, expected) pairs, not a single query —
and a different runner — history is preserved across turns within a
case rather than reset.

So multi-turn lives here. Long-term these two files may merge into one
unified regression runner; for now they're independent so we can iterate
on the multi-turn schema without risking the single-turn baseline.

Schema
------
Each case is:

    {
        "id": "case_id",
        "user_name": "Rosm",        # optional, defaults to "Rosm"
        "dialogue": [
            {
                "user": "first user message",
                "expected": {       # per-turn expected (optional)
                    "must_contain": [...],
                    "must_contain_any": [...],
                    "must_not_contain": [...],
                    "must_not_equal_canned": True,   # see §D
                    "must_match_log_pattern": "...", # log substring check
                    "must_not_match_log_pattern": "...",
                },
            },
            {"user": "...", "expected": {...}},
            ...
        ],
        "case_level": {             # case-level expected (optional)
            "must_match_log_pattern_anywhere": "...",
        },
    }

Per-turn `expected` field semantics (all optional, all soft-pass on absent):

  - must_contain:           list[str], all must be present in answer (unnegated)
  - must_contain_any:       list[str], at least one must be present
  - must_not_contain:       list[str], none may be present
  - must_not_equal_canned:  bool, fail if answer == any of the known canned
                            fallback strings — used to verify D actually saved
                            a self-validating answer rather than letting the
                            verifier-loop discard it
  - must_match_log_pattern:    substring of agent stdout for this turn
  - must_not_match_log_pattern: substring that must NOT appear in this turn's log

Case-level `case_level`:
  - must_match_log_pattern_anywhere: substring that must appear in the
    *combined* log across all turns. Use when a behavior is required.
  - if_log_contains_then: dict {trigger: required}. For each entry,
    IF trigger appears in combined log THEN required must also appear.
    If trigger is absent, assertion is vacuously true. Use for
    "if X happens, Y must happen" — e.g. "if verifier loop fires, D
    must self-validate". Useful when triggering condition X can't be
    reliably forced by case design.

Usage
-----
    cd /home/claude/eva_split   (or wherever Eva lives)
    from test_multiturn_regression import run
    run(agent)                          # all cases
    run(agent, only="birthday_countdown") # one case

In Colab, after building the agent:
    import test_multiturn_regression as mtr
    mtr.run(agent)
"""

import contextlib
import io
import os
import re
import sys
from typing import Any, Dict, List, Optional

# R-6 (2026-05-13): when launched as `python tests/test_multiturn_regression.py`
# from project root, sys.path[0] is `tests/`, not the project root — module
# imports like `from eva_history import ...` would fail. Insert project root.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ============================================================
# Canned fallback message catalog
# ============================================================
# Hard-coded list of the exact strings safe_fallback_for_hard_verifier_failure
# can return. Updated when eva_verifier_logic.py adds new canned messages.
# This is the lookup table for the `must_not_equal_canned` per-turn assertion.
#
# Source: eva_verifier_logic.safe_fallback_for_hard_verifier_failure (the
# branches without phase-2 validation), as of 2026-05-06 + TODO 5 D.
KNOWN_CANNED_STRINGS = {
    "I need to verify that with WebSearch first, Master. I won’t pretend stale guesses are fresh news.",
    "I need to check memory first, Master. I won’t pretend I verified it when I didn’t.",
    "I need the current date before giving a day count, Master. No fake arithmetic from me.",
    "I tried to call a tool in the wrong place, Master. Let me execute it properly instead of showing you the command.",
    "I need to verify that before answering, Master.",
}


# ============================================================
# Negation cue (mirrors test_p2_regression.py)
# ============================================================
_NEGATION_CUE_RE = re.compile(
    r"\b(no|not|never|n't|cannot|can't|won't|don't|doesn't|didn't|isn't|"
    r"aren't|haven't|hasn't|wouldn't|couldn't|shouldn't|none|neither|nor)\b"
)


def _contains_unnegated(needle, text):
    if not needle or not text:
        return False
    needle_lc = needle.lower()
    text_lc = text.lower()
    for m in re.finditer(re.escape(needle_lc), text_lc):
        prefix = text_lc[max(0, m.start() - 30): m.start()]
        if _NEGATION_CUE_RE.search(prefix):
            continue
        return True
    return False


# ============================================================
# Test cases
# ============================================================
#
# Split rationale (post 2026-05-06 multi-turn run + TODO 5 D analysis)
# ---------------------------------------------------------------------
# The original birthday_countdown case from TODO.md was framed as a
# multi-turn FAILURE mode requiring D to recover. Real-model run on
# the SFT-trained Eva showed that the model now completes the compound
# query in a single turn (turn 1), and turn 3 ("do it") never enters
# the verifier hard-fail path at all. The case-level
# "must_match_log_pattern_anywhere" was failing not because D was broken
# but because D had no reason to fire.
#
# Two cases now:
#
#   1. birthday_compound_single_turn
#      Locks in the *progress*: model is expected to complete the
#      compound query within turn 2 (or earlier), and follow-up
#      anaphoric "do it" is expected to give a coherent reply without
#      hitting the canned fallback. NO requirement that D fires.
#      Regression alarm: if the model regresses to deferred completion,
#      this case starts failing.
#
#   2. birthday_deferred_forced
#      User input is hand-split to discourage single-turn completion:
#      turn 1 explicitly asks for date-only ("don't calculate"), turn 2
#      then asks for the day count. This dialogue shape COULD push the
#      model into the deferred path, which COULD push the verifier into
#      a loop, which COULD trigger D. None of those are guaranteed —
#      verifier only checks `missing_date_calculation_evidence` when
#      the answer carries day-count arithmetic shape, and the gate
#      function `_question_needs_time_arithmetic` short-circuits on
#      anaphoric inputs that don't carry day/week/until keywords.
#      Both conditions need to align for D to fire, which is rare in
#      practice with the current architecture.
#
#      So this case uses a **conditional** case-level assertion via the
#      new `if_log_contains_then` schema:
#         IF "VERIFIER LOOP DETECTED" appears in the log
#         THEN "SELF-VALIDATED, KEEPING PHASE-2" must also appear.
#      If the loop never fires, the conditional is vacuous and the
#      case passes on per-turn assertions alone. This means:
#        - D firing → case validates D worked correctly.
#        - D not firing → case still validates the deferred dialogue
#          path produces a non-canned answer.
#        - D firing but BROKEN → case fails (catches D regressions).
#
# ============================================================

BIRTHDAY_COMPOUND_SINGLE_TURN_CASE = {
    "id": "birthday_compound_single_turn",
    "user_name": "Rosm",
    "dialogue": [
        {
            "user": "so who are you?",
            "expected": {
                "must_contain_any": ["Eva"],
            },
        },
        {
            "user": "so when is your birthday and how many days until it?",
            "expected": {
                # Both halves of the compound query should be answered
                # in this single turn. Date half:
                "must_contain_any": ["July 7", "July 7th", "Jul 7", "7月7"],
                # Day-count half: must mention days. We don't pin a
                # specific number because the actual count depends on
                # today's date (the test runs on whatever calendar day
                # Colab is run; system clock decides).
                "must_contain": ["day"],
                "must_not_equal_canned": True,
            },
        },
        {
            "user": "do it",
            "expected": {
                # Anaphoric continuation. Either the model re-asserts
                # the day count (most likely) or asks a clarifying
                # question — both fine, just don't return canned.
                "must_not_equal_canned": True,
            },
        },
    ],
    # No case-level pattern: D is not expected to fire on this path.
    # If the model regresses to needing D here, that's a separate
    # signal worth seeing, but it doesn't count as a case failure.
}


BIRTHDAY_DEFERRED_FORCED_CASE = {
    "id": "birthday_deferred_forced",
    "user_name": "Rosm",
    "dialogue": [
        {
            "user": "what is your birthday? Just give the date — do not calculate days yet.",
            "expected": {
                "must_contain_any": ["July 7", "July 7th", "Jul 7", "7月7"],
                # If the model ignores "do not calculate" and pre-emptively
                # gives a day count, that's not a failure for THIS case
                # (it's basically the single-turn case again). We don't
                # forbid 'days' here.
            },
        },
        {
            "user": "now calculate the days until then.",
            "expected": {
                # Day count must appear, in any of the supported
                # languages, and must not be the canned message.
                "must_contain_any": ["days", "day", "天"],
                "must_not_equal_canned": True,
            },
        },
    ],
    "case_level": {
        # Conditional D verification. If the verifier-loop branch is
        # reached during this case (any turn's log), D must self-validate.
        # If the loop is never reached, this is a no-op.
        "if_log_contains_then": {
            "VERIFIER LOOP DETECTED": "SELF-VALIDATED, KEEPING PHASE-2",
        },
    },
}


MULTITURN_CASES = [
    BIRTHDAY_COMPOUND_SINGLE_TURN_CASE,
    BIRTHDAY_DEFERRED_FORCED_CASE,
]


# ============================================================
# Tee for capturing per-turn stdout
# ============================================================
class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            try:
                st.write(s)
            except Exception:
                pass

    def flush(self):
        for st in self.streams:
            try:
                st.flush()
            except Exception:
                pass


# ============================================================
# Per-turn scoring
# ============================================================
def _score_turn(turn_idx, turn_spec, answer, log):
    """Apply per-turn assertions. Returns (passed, list_of_failures)."""
    expected = turn_spec.get("expected", {}) or {}
    failures = []

    answer = (answer or "").strip()

    for needle in expected.get("must_contain", []) or []:
        if not _contains_unnegated(needle, answer):
            failures.append(f"must_contain {needle!r} missing (unnegated) from answer")

    any_terms = expected.get("must_contain_any", []) or []
    if any_terms:
        if not any(_contains_unnegated(t, answer) for t in any_terms):
            failures.append(
                f"must_contain_any: none of {any_terms!r} appeared in answer"
            )

    for forbidden in expected.get("must_not_contain", []) or []:
        if _contains_unnegated(forbidden, answer):
            failures.append(
                f"must_not_contain {forbidden!r} appeared (unnegated) in answer"
            )

    if expected.get("must_not_equal_canned", False):
        if answer in KNOWN_CANNED_STRINGS:
            failures.append(
                f"answer == canned fallback string ({answer!r}); "
                f"D self-validation did not save this turn"
            )

    pattern = expected.get("must_match_log_pattern")
    if pattern and pattern not in log:
        failures.append(f"log missing pattern {pattern!r} this turn")

    neg_pattern = expected.get("must_not_match_log_pattern")
    if neg_pattern and neg_pattern in log:
        failures.append(f"log contains forbidden pattern {neg_pattern!r} this turn")

    return (len(failures) == 0), failures


# ============================================================
# Case-level scoring
# ============================================================
def _score_case(case, turn_results):
    """Apply case-level assertions across all turns combined."""
    case_level = case.get("case_level", {}) or {}
    failures = []

    combined_log = "\n".join(t["log"] for t in turn_results)

    pattern = case_level.get("must_match_log_pattern_anywhere")
    if pattern and pattern not in combined_log:
        failures.append(f"case-level: log never matched {pattern!r}")

    # Conditional assertion: if_log_contains_then is a dict
    # {trigger_pattern: required_pattern}. For each entry, IF the
    # trigger appears in combined_log, THEN the required pattern must
    # also appear. If trigger is absent, the assertion is vacuously
    # satisfied. Used for D verification on cases where D's trigger
    # condition can't be reliably forced by case design — see
    # BIRTHDAY_DEFERRED_FORCED_CASE.
    conditional = case_level.get("if_log_contains_then") or {}
    for trigger, required in conditional.items():
        if trigger in combined_log:
            if required not in combined_log:
                failures.append(
                    f"case-level: log contains {trigger!r} but is missing "
                    f"required follow-up {required!r}"
                )

    return (len(failures) == 0), failures


# ============================================================
# Per-case runner — loops over dialogue turns sharing history
# ============================================================
def _run_one_multiturn(agent, case):
    """Run a multi-turn case and return a dict with per-turn + case-level results.

    The agent's history_manager is reset ONCE at the start of the case
    (so we get a fresh session) and then NOT reset between dialogue
    turns within the case (so the agent sees the conversation
    accumulating, just like real chat).
    """
    case_id = case["id"]
    user_name = case.get("user_name", "Rosm")

    # Reset history once for the case.
    hm = getattr(agent, "history_manager", None)
    if hm is not None:
        hm.history = []
        hm.current_turn = None
        if hasattr(hm, "compressed_kv"):
            hm.compressed_kv = []
        if hasattr(hm, "image_registry"):
            hm.image_registry = {}
        if hasattr(hm, "image_order"):
            hm.image_order = []
    # Clear P2 turn-cache + last_memory state.
    # NOTE: last_memory / dialog_focus are normally sticky across turns
    # within a session — that's correct production behavior. We clear
    # them here only to start each case from a clean slate, not as part
    # of a per-turn reset. The agent will repopulate them naturally as
    # the dialogue progresses.
    for attr in ("active_memory_turn_key", "active_memory_context"):
        if hasattr(agent, attr):
            setattr(agent, attr, "")
    # R-6: last_memory / dialog_focus 是 dataclass，调用 reset()
    last_mem = getattr(agent, "last_memory", None)
    if last_mem is not None and hasattr(last_mem, "reset"):
        last_mem.reset()
    focus = getattr(agent, "dialog_focus", None)
    if focus is not None and hasattr(focus, "reset"):
        focus.reset()

    turn_results = []
    case_passed = True
    case_failures = []

    for turn_idx, turn_spec in enumerate(case["dialogue"]):
        user_text = turn_spec["user"]

        buf = io.StringIO()
        tee = _Tee(buf)
        answer = ""
        err = None

        with contextlib.redirect_stdout(tee):
            try:
                if hasattr(agent, "run"):
                    answer = agent.run(user_text, user_name=user_name)
                else:
                    err = "agent has no .run() method"
            except Exception as e:
                import traceback
                err = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"

        log = buf.getvalue()

        turn_passed, turn_failures = _score_turn(turn_idx, turn_spec, answer, log)

        # An unhandled exception in agent.run is always a turn failure,
        # even if the assertions happen to be empty.
        if err:
            turn_passed = False
            turn_failures.insert(0, f"agent.run raised: {err.splitlines()[0]}")

        turn_results.append({
            "turn_idx": turn_idx,
            "user": user_text,
            "answer": (answer or "").strip(),
            "error": err,
            "log": log,
            "passed": turn_passed,
            "failures": turn_failures,
        })

        if not turn_passed:
            case_passed = False

        # Early termination on hard error: if agent raised, later turns
        # depend on session state that's now corrupt.
        if err:
            break

    # Case-level checks (only if all turns ran).
    if all(t["error"] is None for t in turn_results):
        case_level_passed, case_level_failures = _score_case(case, turn_results)
        if not case_level_passed:
            case_passed = False
            case_failures.extend(case_level_failures)

    return {
        "id": case_id,
        "passed": case_passed,
        "case_failures": case_failures,
        "turn_results": turn_results,
    }


# ============================================================
# Pretty-printer
# ============================================================
def _print_case_result(result):
    case_id = result["id"]
    passed = result["passed"]
    tag = "✓ PASS" if passed else "✗ FAIL"
    print(f"\n[{tag}] {case_id}")

    for t in result["turn_results"]:
        turn_tag = "✓" if t["passed"] else "✗"
        ans_short = t["answer"][:100] + ("…" if len(t["answer"]) > 100 else "")
        print(f"  turn {t['turn_idx']}  [{turn_tag}]  user={t['user'][:60]!r}")
        print(f"         answer={ans_short!r}")
        if t["error"]:
            print(f"         ERROR: {t['error'].splitlines()[0]}")
        for f in t["failures"]:
            print(f"         × {f}")

    for f in result["case_failures"]:
        print(f"  case-level × {f}")


# ============================================================
# Public entry
# ============================================================
def run(agent, only=None, save_to=None):
    """Run the multi-turn regression suite.

    Args:
        agent: a built ChatAgent instance.
        only: optional case id to run a single case (e.g. "birthday_countdown").
        save_to: optional path to dump the per-turn JSON for offline analysis.

    Returns:
        list of case-result dicts.
    """
    if only is not None:
        cases = [c for c in MULTITURN_CASES if c["id"] == only]
        if not cases:
            print(f"[multiturn-regression] no case matches {only!r}; "
                  f"available: {[c['id'] for c in MULTITURN_CASES]}")
            return []
    else:
        cases = MULTITURN_CASES

    print(f"\n{'='*72}")
    print(f"multi-turn regression suite — {len(cases)} case(s)")
    print(f"{'='*72}")

    results = []
    for case in cases:
        r = _run_one_multiturn(agent, case)
        results.append(r)
        _print_case_result(r)

    n_pass = sum(1 for r in results if r["passed"])
    print(f"\n{'='*72}")
    print(f"  RESULT: {n_pass}/{len(results)} cases passed")
    print(f"{'='*72}\n")

    if save_to:
        import json
        # Strip log strings to keep the JSON readable; keep failure summaries.
        light = []
        for r in results:
            light.append({
                "id": r["id"],
                "passed": r["passed"],
                "case_failures": r["case_failures"],
                "turns": [
                    {
                        "turn_idx": t["turn_idx"],
                        "user": t["user"],
                        "answer": t["answer"],
                        "passed": t["passed"],
                        "failures": t["failures"],
                        "error": t["error"],
                    }
                    for t in r["turn_results"]
                ],
            })
        with open(save_to, "w", encoding="utf-8") as f:
            json.dump(light, f, ensure_ascii=False, indent=2)
        print(f"  saved JSON to {save_to}")

    return results


# ============================================================
# Self-test (runs without a real agent — uses StubAgent)
# ============================================================
class _StubHistoryManager:
    def __init__(self):
        self.history = []
        self.current_turn = None
        self.compressed_kv = []
        self.image_registry = {}
        self.image_order = []
        self.user_name = "Rosm"


class _StubAgent:
    """Hardcoded-script agent for framework self-test.

    Returns a pre-configured sequence of (answer, log) pairs, in order,
    one per call to .run(). Lets us verify that _run_one_multiturn loops
    correctly and that scoring catches both pass and fail cases without
    needing a real model.
    """
    def __init__(self, scripted_turns):
        # Each entry: (answer_str, log_to_emit_to_stdout)
        self.scripted_turns = scripted_turns
        self._call_count = 0
        self.history_manager = _StubHistoryManager()
        # R-6: stub 用 dataclass，与生产 ChatAgent 行为对齐
        from eva_history import LastMemoryState, DialogFocus
        self.last_memory = LastMemoryState()
        self.dialog_focus = DialogFocus()
        self.active_memory_turn_key = ""
        self.active_memory_context = ""

    def run(self, user_text, user_name="Rosm", image_path=None):
        if self._call_count >= len(self.scripted_turns):
            raise RuntimeError(
                f"StubAgent: ran out of scripted turns at call {self._call_count}"
            )
        answer, log = self.scripted_turns[self._call_count]
        self._call_count += 1
        # Emit the log to stdout so _Tee captures it as if the real agent
        # had printed it.
        if log:
            print(log)
        return answer


def _self_test():
    """Verify framework with a stubbed-out agent BEFORE shipping to Colab.

    Five sub-tests covering the new split + conditional logic:

      1. compound-single-turn HAPPY    — model answers compound query
                                          in turn 2, follow-up "do it"
                                          gives non-canned reply →
                                          case PASS, no D needed.
      2. compound-single-turn REGRESS  — turn 2 returns canned message
                                          (model regressed to forced
                                          fallback) → case FAIL.
      3. deferred-forced D-fires       — verifier loop fires AND D
                                          self-validates → case PASS,
                                          conditional satisfied.
      4. deferred-forced D-broken      — verifier loop fires but D
                                          fails to self-validate →
                                          case FAIL on conditional.
      5. deferred-forced D-skipped     — verifier loop never fires →
                                          conditional vacuous, case
                                          PASSES on per-turn alone.
    """
    print(f"\n{'='*72}")
    print("FRAMEWORK SELF-TEST (stub agent — not real model)")
    print(f"{'='*72}")

    # ---- Sub-test 1: compound-single-turn HAPPY ----
    print("\n--- self-test 1: compound_single_turn happy path (expect PASS) ---")
    happy_agent = _StubAgent([
        ("Well, well, Master. I'm Eva Louisa, your AI maid.", ""),
        ("My birthday is July 7th, Master. There are exactly 61 days until then~", ""),
        ("Fine, fine! It's July 7th, sixty-one days from now. Don't forget!", ""),
    ])
    r1 = _run_one_multiturn(happy_agent, BIRTHDAY_COMPOUND_SINGLE_TURN_CASE)
    _print_case_result(r1)
    assert r1["passed"], f"self-test 1 expected PASS, got {r1['case_failures']}"
    print("  ✓ self-test 1 OK\n")

    # ---- Sub-test 2: compound-single-turn REGRESS to canned ----
    print("--- self-test 2: compound_single_turn turn 2 canned (expect FAIL) ---")
    regress_agent = _StubAgent([
        ("I'm Eva, Master.", ""),
        # Turn 2 returns the canned message — model regressed.
        ("I need the current date before giving a day count, Master. No fake arithmetic from me.", ""),
        ("Fine, July 7th — 61 days from now.", ""),
    ])
    r2 = _run_one_multiturn(regress_agent, BIRTHDAY_COMPOUND_SINGLE_TURN_CASE)
    _print_case_result(r2)
    assert not r2["passed"], "self-test 2 expected FAIL, got PASS"
    turn1_failures = r2["turn_results"][1]["failures"]
    assert any("canned" in f for f in turn1_failures), \
        f"self-test 2 should flag turn 1 canned, got {turn1_failures}"
    print("  ✓ self-test 2 OK (failed for the RIGHT reason)\n")

    # ---- Sub-test 3: deferred-forced, D fires correctly ----
    print("--- self-test 3: deferred_forced D fires correctly (expect PASS) ---")
    d_fires_agent = _StubAgent([
        ("My birthday is July 7th, Master.", ""),
        ("There are 61 days until July 7th, Master.",
         "        | --- VERIFIER LOOP DETECTED (repeated reason=missing_date_calculation_evidence) -> FALLBACK ---\n"
         "        | self-validate parsed: month=7 day=7 days_claimed=61 days_expected=61 lang=en -> MATCH\n"
         "        | --- VERIFIER LOOP DETECTED -> SELF-VALIDATED, KEEPING PHASE-2 ---"),
    ])
    r3 = _run_one_multiturn(d_fires_agent, BIRTHDAY_DEFERRED_FORCED_CASE)
    _print_case_result(r3)
    assert r3["passed"], f"self-test 3 expected PASS, got {r3['case_failures']}"
    print("  ✓ self-test 3 OK\n")

    # ---- Sub-test 4: deferred-forced, verifier loop fires but D broken ----
    print("--- self-test 4: deferred_forced verifier loop fires but D broken (expect FAIL) ---")
    d_broken_agent = _StubAgent([
        ("My birthday is July 7th.", ""),
        # Verifier loop fires (log mentions it), D *should* self-validate
        # but doesn't (no SELF-VALIDATED line in log). Model still emits
        # a non-canned answer somehow — but the conditional assertion
        # catches the missing SELF-VALIDATED.
        ("There are 61 days until July 7th, Master.",
         "        | --- VERIFIER LOOP DETECTED (repeated reason=missing_date_calculation_evidence) -> FALLBACK ---"),
    ])
    r4 = _run_one_multiturn(d_broken_agent, BIRTHDAY_DEFERRED_FORCED_CASE)
    _print_case_result(r4)
    assert not r4["passed"], "self-test 4 expected FAIL"
    assert any("SELF-VALIDATED" in f for f in r4["case_failures"]), \
        f"self-test 4 should flag missing SELF-VALIDATED follow-up, got {r4['case_failures']}"
    print("  ✓ self-test 4 OK (conditional assertion fired correctly)\n")

    # ---- Sub-test 5: deferred-forced, verifier loop never fires ----
    # This is the "current real-model behavior" path: deferred dialogue
    # but model gracefully handles it without entering verifier loop.
    # D never needs to fire; case passes on per-turn assertions alone.
    print("--- self-test 5: deferred_forced no verifier loop (expect PASS, conditional vacuous) ---")
    no_loop_agent = _StubAgent([
        ("My birthday is July 7th.", ""),
        ("61 days until July 7th, Master.",
         "        | (no verifier issue this turn)"),
    ])
    r5 = _run_one_multiturn(no_loop_agent, BIRTHDAY_DEFERRED_FORCED_CASE)
    _print_case_result(r5)
    assert r5["passed"], f"self-test 5 expected PASS, got {r5['case_failures']}"
    print("  ✓ self-test 5 OK (conditional vacuous, case passed on per-turn)\n")

    print(f"{'='*72}")
    print("FRAMEWORK SELF-TEST: ALL 5 SUB-TESTS PASS")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    _self_test()
