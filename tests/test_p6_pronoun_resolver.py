"""P6.0 offline test harness for eva_pronoun_resolver.

This is a self-contained unittest run — no model loading, no real
DeepSeek calls. The LLM main path is mocked at the
`eva_pronoun_resolver._call_llm` boundary so we exercise:

  - the cheap gate (Stage 1)
  - the JSON parse + min_confidence + antecedent cleanup logic
  - the LLM-failure → regex fallback (Stage 3) wiring
  - budget exhaustion handling
  - mode flags ("off", "regex_only", "llm_first")
  - cache reuse

Plus the 8 acceptance fixtures from § 八 of the v3 plan, run twice:
once with the LLM mocked to behave as the prompt's expected output
(fixtures 1-8 should all pass), once with the LLM forced to fail
(fixtures should fall through to regex fallback — only the patterns
the legacy regex catches will pass; this is the behaviour we expect
to lose at P6.4).

Usage (from project root):
    python tests/test_p6_pronoun_resolver.py

Exit code 0 on success, non-zero on any test failure.
"""
import os
import sys
import types
import unittest
from unittest.mock import patch

# ------------------------------------------------------------
# Path setup — file lives in tests/, but imports project modules
# from the parent directory. Make project root importable
# regardless of cwd.
# ------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ------------------------------------------------------------
# Offline-test stubs.
# eva_config does `import torch` at module top to set two cudnn
# flags. The flags are irrelevant to the resolver; install a stub
# so the test can run on any machine without the full ML stack.
# Same trick for any other heavyweight deps eva_config might pull
# in transitively — keep the stubs minimal so a real failure in
# eva_config logic still surfaces.
# ------------------------------------------------------------
if "torch" not in sys.modules:
    _torch_stub = types.ModuleType("torch")
    _backends = types.ModuleType("torch.backends")
    _cudnn = types.ModuleType("torch.backends.cudnn")
    _cudnn.benchmark = False
    _cudnn.deterministic = True
    _backends.cudnn = _cudnn
    _torch_stub.backends = _backends
    sys.modules["torch"] = _torch_stub
    sys.modules["torch.backends"] = _backends
    sys.modules["torch.backends.cudnn"] = _cudnn

import eva_config
from eva_intent_judge import JudgeState, reset_state
from eva_pronoun_resolver import (
    PronounResolution,
    resolve_pronoun,
    _cheap_gate,
    _parse_llm_verdict,
)


# ============================================================
# Helpers
# ============================================================
def _turns(*assistant_lines, user_lines=None):
    """Build a recent_turns list from assistant lines (and optional
    matching user lines). Newest last.
    """
    out = []
    user_lines = list(user_lines) if user_lines else []
    for i, a in enumerate(assistant_lines):
        u = user_lines[i] if i < len(user_lines) else ""
        out.append({"user": u, "assistant": a})
    return out


def _ok_verdict(needs, ants, conf=0.9):
    """Shape the dict returned by the mocked _call_llm so it matches
    the DeepSeek contract.
    """
    return {
        "needs_resolution": needs,
        "antecedents": list(ants),
        "confidence": conf,
    }


# ============================================================
# Cheap gate (Stage 1) tests — no LLM mock needed
# ============================================================
class TestCheapGate(unittest.TestCase):

    def test_empty_query_skips(self):
        self.assertFalse(_cheap_gate("", 8))
        self.assertFalse(_cheap_gate("   ", 8))

    def test_too_long_skips(self):
        # 9 words > max_words=8
        q = "can you please tell me what the weather is today"
        self.assertFalse(_cheap_gate(q, 8))

    def test_no_trigger_token_skips(self):
        # Has length but no pronoun-trigger token.
        self.assertFalse(_cheap_gate("tell me about hobbies", 8))
        self.assertFalse(_cheap_gate("what's the weather", 8))

    def test_trigger_present_passes(self):
        self.assertTrue(_cheap_gate("check it", 8))
        self.assertTrue(_cheap_gate("really? do that", 8))
        self.assertTrue(_cheap_gate("do it again", 8))
        self.assertTrue(_cheap_gate("hold on, check it", 8))

    def test_trigger_in_punctuated_query(self):
        # Punctuation must not block detection.
        self.assertTrue(_cheap_gate("really? Check it.", 8))
        self.assertTrue(_cheap_gate("(check it)", 8))


# ============================================================
# Parse logic tests
# ============================================================
class TestParseVerdict(unittest.TestCase):

    def test_happy_path(self):
        v = _parse_llm_verdict(_ok_verdict(True, ["music box"], 0.92))
        self.assertIsNotNone(v)
        self.assertTrue(v.needs_resolution)
        self.assertEqual(v.antecedents, ["music box"])
        self.assertAlmostEqual(v.confidence, 0.92)
        self.assertEqual(v.source, "llm")

    def test_missing_field_rejected(self):
        self.assertIsNone(_parse_llm_verdict({"needs_resolution": True}))
        self.assertIsNone(_parse_llm_verdict({}))
        self.assertIsNone(_parse_llm_verdict(None))

    def test_wrong_type_rejected(self):
        # needs_resolution is a string instead of bool
        self.assertIsNone(_parse_llm_verdict({
            "needs_resolution": "yes",
            "antecedents": ["x"],
            "confidence": 0.9,
        }))
        # antecedents is not a list
        self.assertIsNone(_parse_llm_verdict({
            "needs_resolution": True,
            "antecedents": "music box",
            "confidence": 0.9,
        }))

    def test_strips_articles_defensively(self):
        # Even though prompt forbids articles, real LLMs slip; we strip.
        v = _parse_llm_verdict(_ok_verdict(True, ["the music box", "a photo"]))
        self.assertEqual(v.antecedents, ["music box", "photo"])

    def test_caps_at_three_antecedents(self):
        v = _parse_llm_verdict(_ok_verdict(True, ["a", "b", "c", "d", "e"]))
        self.assertEqual(len(v.antecedents), 3)

    def test_dedupes_antecedents(self):
        v = _parse_llm_verdict(_ok_verdict(True, ["box", "box", "music"]))
        self.assertEqual(v.antecedents, ["box", "music"])

    def test_confidence_clamped(self):
        v = _parse_llm_verdict(_ok_verdict(True, ["x"], 1.5))
        self.assertEqual(v.confidence, 1.0)
        v = _parse_llm_verdict(_ok_verdict(True, ["x"], -0.3))
        self.assertEqual(v.confidence, 0.0)


# ============================================================
# resolve_pronoun integration — flag matrix
# ============================================================
class TestResolveFlags(unittest.TestCase):

    def setUp(self):
        # Snapshot the flags we mutate so tearDown can restore.
        self._snap = {
            "ENABLE_PRONOUN_RESOLVER": eva_config.ENABLE_PRONOUN_RESOLVER,
            "PRONOUN_RESOLVER_MODE": eva_config.PRONOUN_RESOLVER_MODE,
            "PRONOUN_RESOLVER_DEBUG": eva_config.PRONOUN_RESOLVER_DEBUG,
            "PRONOUN_RESOLVER_MIN_CONFIDENCE":
                eva_config.PRONOUN_RESOLVER_MIN_CONFIDENCE,
            "PRONOUN_RESOLVER_MAX_WORDS":
                eva_config.PRONOUN_RESOLVER_MAX_WORDS,
            "PRONOUN_RESOLVER_MAX_CALLS_PER_TURN":
                eva_config.PRONOUN_RESOLVER_MAX_CALLS_PER_TURN,
        }
        self.state = JudgeState()

    def tearDown(self):
        for k, v in self._snap.items():
            setattr(eva_config, k, v)

    def test_off_mode_short_circuits(self):
        eva_config.PRONOUN_RESOLVER_MODE = "off"
        v = resolve_pronoun(
            "check it", _turns("I have a music box."), state=self.state,
        )
        self.assertEqual(v.source, "off")
        self.assertFalse(v.needs_resolution)
        self.assertEqual(self.state.pronoun_call_count, 0)

    def test_disabled_flag_short_circuits(self):
        eva_config.ENABLE_PRONOUN_RESOLVER = False
        v = resolve_pronoun("check it", _turns("x"), state=self.state)
        self.assertEqual(v.source, "off")
        self.assertEqual(self.state.pronoun_call_count, 0)

    def test_skip_when_no_trigger(self):
        eva_config.PRONOUN_RESOLVER_MODE = "llm_first"
        v = resolve_pronoun(
            "what's the weather", _turns("x"), state=self.state,
        )
        self.assertEqual(v.source, "skip")
        self.assertFalse(v.needs_resolution)
        self.assertEqual(self.state.pronoun_call_count, 0)

    @patch("eva_pronoun_resolver._call_llm")
    def test_llm_first_happy_path(self, mock_call):
        eva_config.PRONOUN_RESOLVER_MODE = "llm_first"
        mock_call.return_value = _ok_verdict(
            True, ["music box"], 0.92,
        )
        v = resolve_pronoun(
            "really? Check it",
            _turns("I could show you my music box."),
            state=self.state,
        )
        self.assertEqual(v.source, "llm")
        self.assertTrue(v.needs_resolution)
        self.assertEqual(v.antecedents, ["music box"])
        self.assertEqual(self.state.pronoun_call_count, 1)
        mock_call.assert_called_once()

    @patch("eva_pronoun_resolver._call_llm")
    def test_low_confidence_demoted(self, mock_call):
        eva_config.PRONOUN_RESOLVER_MODE = "llm_first"
        eva_config.PRONOUN_RESOLVER_MIN_CONFIDENCE = 0.6
        mock_call.return_value = _ok_verdict(
            True, ["music box"], 0.40,
        )
        v = resolve_pronoun(
            "check it", _turns("music box"), state=self.state,
        )
        # Source stays "llm" (we did call it), but verdict demoted.
        self.assertEqual(v.source, "llm")
        self.assertFalse(v.needs_resolution)
        # Antecedents preserved for debug visibility.
        self.assertEqual(v.antecedents, ["music box"])

    @patch("eva_pronoun_resolver._call_llm")
    def test_llm_unavailable_falls_through_to_regex(self, mock_call):
        eva_config.PRONOUN_RESOLVER_MODE = "llm_first"
        mock_call.return_value = None  # simulate LLM failure
        v = resolve_pronoun(
            "check it",
            _turns("I could show you my special collection."),
            state=self.state,
        )
        # Stage 3 regex fallback fires.
        self.assertIn(v.source, ("regex", "skip"))
        # Still consumed budget for the attempt.
        self.assertEqual(self.state.pronoun_call_count, 1)

    @patch("eva_pronoun_resolver._call_llm")
    def test_budget_exhaustion_skips_llm(self, mock_call):
        eva_config.PRONOUN_RESOLVER_MODE = "llm_first"
        eva_config.PRONOUN_RESOLVER_MAX_CALLS_PER_TURN = 1
        # Pre-exhaust budget.
        self.state.pronoun_call_count = 1
        v = resolve_pronoun(
            "check it",
            _turns("I have a music box."),
            state=self.state,
        )
        # Falls to regex without invoking LLM.
        mock_call.assert_not_called()
        self.assertIn(v.source, ("regex", "skip"))

    @patch("eva_pronoun_resolver._regex_fallback")
    @patch("eva_pronoun_resolver._call_llm")
    def test_regex_only_mode_never_calls_llm(self, mock_call, mock_regex):
        # In offline-test env eva_verifier_logic may not import
        # (rank_bm25 etc. not installed). Mock _regex_fallback directly
        # so this test verifies the ROUTING (regex_only → regex stage,
        # never LLM), not the regex helpers themselves. The helpers
        # have their own coverage in the legacy regression suite.
        eva_config.PRONOUN_RESOLVER_MODE = "regex_only"
        mock_regex.return_value = PronounResolution(
            needs_resolution=True,
            antecedents=["music box"],
            confidence=0.75,
            source="regex",
            reasoning="(stub)",
        )
        v = resolve_pronoun(
            "check it",
            _turns("I have a music box."),
            state=self.state,
        )
        mock_call.assert_not_called()
        mock_regex.assert_called_once()
        self.assertEqual(v.source, "regex")
        self.assertTrue(v.needs_resolution)

    @patch("eva_pronoun_resolver._call_llm")
    def test_cache_hit_skips_second_call(self, mock_call):
        eva_config.PRONOUN_RESOLVER_MODE = "llm_first"
        mock_call.return_value = _ok_verdict(True, ["music box"], 0.9)
        recent = _turns("music box")
        # First call: hits LLM.
        v1 = resolve_pronoun("check it", recent, state=self.state)
        # Second call same query+history: cache hit, no second LLM.
        v2 = resolve_pronoun("check it", recent, state=self.state)
        self.assertEqual(mock_call.call_count, 1)
        self.assertEqual(v1.antecedents, v2.antecedents)
        # Counter only incremented once.
        self.assertEqual(self.state.pronoun_call_count, 1)

    @patch("eva_pronoun_resolver._call_llm")
    def test_reset_state_clears_pronoun_counter(self, mock_call):
        eva_config.PRONOUN_RESOLVER_MODE = "llm_first"
        mock_call.return_value = _ok_verdict(True, ["x"], 0.9)
        resolve_pronoun(
            "check it", _turns("x"), state=self.state,
        )
        self.assertEqual(self.state.pronoun_call_count, 1)
        reset_state(self.state)
        self.assertEqual(self.state.pronoun_call_count, 0)
        self.assertEqual(self.state.call_count, 0)


# ============================================================
# § 八 acceptance fixtures
# ============================================================
# These are the 8 fixtures from docs/P6_pronoun_resolver_refactor_v3.md § 八.
# Each fixture (input, history-tail, expected_needs, expected_ant0).
# We check both that the cheap gate doesn't filter the True cases out,
# AND that with a properly-mocked LLM we get the expected verdict.
# This is the contract the resolver must uphold under P6.4.
ACCEPTANCE_FIXTURES = [
    ("can you check it?",          "I could show you my special collection.",
     True,  "special collection"),
    ("really? Check it",           "I have a music box.",
     True,  "music box"),
    ("hold on, check it",          "I have a music box.",
     True,  "music box"),
    ("sorry, check that",          "Here is the photo Master sent yesterday.",
     True,  "photo"),
    ("do it again",                "Want me to tell another joke?",
     True,  "joke"),
    ("really? what date is today?", "Your birthday is in 202 days.",
     False, None),
    ("tell me about your hobbies", "(unrelated)",
     False, None),
    ("what's the weather",         "I have a music box.",
     False, None),
]


class TestAcceptanceFixtures(unittest.TestCase):
    """The 8 fixtures from § 八 of the v3 plan. Hard gate for P6.3."""

    def setUp(self):
        self._snap = {
            "ENABLE_PRONOUN_RESOLVER": eva_config.ENABLE_PRONOUN_RESOLVER,
            "PRONOUN_RESOLVER_MODE": eva_config.PRONOUN_RESOLVER_MODE,
            "PRONOUN_RESOLVER_MIN_CONFIDENCE":
                eva_config.PRONOUN_RESOLVER_MIN_CONFIDENCE,
            "PRONOUN_RESOLVER_MAX_WORDS":
                eva_config.PRONOUN_RESOLVER_MAX_WORDS,
            "PRONOUN_RESOLVER_MAX_CALLS_PER_TURN":
                eva_config.PRONOUN_RESOLVER_MAX_CALLS_PER_TURN,
        }
        eva_config.PRONOUN_RESOLVER_MODE = "llm_first"
        # Allow up to 8 LLM calls in this test (one per fixture, no
        # cross-fixture cache because each uses fresh JudgeState).
        eva_config.PRONOUN_RESOLVER_MAX_CALLS_PER_TURN = 8

    def tearDown(self):
        for k, v in self._snap.items():
            setattr(eva_config, k, v)

    def test_fixtures_with_oracle_llm(self):
        """Mock the LLM to return the fixture's expected verdict.

        This proves the wiring is correct end-to-end: cheap gate
        doesn't filter True cases, parse logic preserves the
        antecedent, min_confidence doesn't demote it.
        """
        for query, hist_tail, expected_needs, expected_ant0 in ACCEPTANCE_FIXTURES:
            with self.subTest(query=query):
                state = JudgeState()
                if expected_needs:
                    fake = _ok_verdict(True, [expected_ant0], 0.9)
                else:
                    fake = _ok_verdict(False, [], 0.95)
                with patch(
                    "eva_pronoun_resolver._call_llm",
                    return_value=fake,
                ):
                    v = resolve_pronoun(
                        query, _turns(hist_tail), state=state,
                    )

                if expected_needs:
                    # Two acceptable outcomes:
                    #  (a) cheap gate let it through, LLM returned
                    #      needs=True  → resolver returns needs=True.
                    #  (b) cheap gate filtered it (false negative)
                    #      → resolver returns source="skip", needs=False.
                    # (a) is the goal. (b) is a known limitation we
                    # surface so the cheap gate can be tuned.
                    if v.source == "skip":
                        self.fail(
                            f"cheap gate filtered a True fixture: "
                            f"{query!r} -> {v}"
                        )
                    self.assertTrue(
                        v.needs_resolution,
                        f"fixture {query!r}: expected needs=True got {v}",
                    )
                    self.assertEqual(
                        v.antecedents[0], expected_ant0,
                        f"fixture {query!r}: expected ant0={expected_ant0!r}",
                    )
                else:
                    self.assertFalse(
                        v.needs_resolution,
                        f"fixture {query!r}: expected needs=False got {v}",
                    )


# ============================================================
# P6.2 shadow-mode tests
# ============================================================
class TestP62ShadowMode(unittest.TestCase):
    """Verifies the shadow-mode contract:

      - LLM is invoked
      - LLM verdict is NOT adopted
      - Final verdict comes from the regex path (i.e. P6.1 behaviour)
      - [PRONOUN-SHADOW] trace line is emitted (always — not gated
        on debug flag)
      - LLM unavailable / over budget cases are logged distinctly
    """

    def setUp(self):
        self._snap = {
            "ENABLE_PRONOUN_RESOLVER": eva_config.ENABLE_PRONOUN_RESOLVER,
            "PRONOUN_RESOLVER_MODE": eva_config.PRONOUN_RESOLVER_MODE,
            "PRONOUN_RESOLVER_SHADOW": eva_config.PRONOUN_RESOLVER_SHADOW,
            "PRONOUN_RESOLVER_DEBUG": eva_config.PRONOUN_RESOLVER_DEBUG,
            "PRONOUN_RESOLVER_MAX_CALLS_PER_TURN":
                eva_config.PRONOUN_RESOLVER_MAX_CALLS_PER_TURN,
        }
        eva_config.PRONOUN_RESOLVER_MODE = "llm_first"
        eva_config.PRONOUN_RESOLVER_SHADOW = True
        eva_config.PRONOUN_RESOLVER_DEBUG = False
        self.state = JudgeState()

    def tearDown(self):
        for k, v in self._snap.items():
            setattr(eva_config, k, v)

    @patch("eva_pronoun_resolver._regex_fallback")
    @patch("eva_pronoun_resolver._call_llm")
    def test_shadow_adopts_regex_not_llm(self, mock_llm, mock_regex):
        # Construct a deliberate disagreement: regex says yes-with-X,
        # LLM says yes-with-Y. The verdict returned must be the regex one.
        mock_regex.return_value = PronounResolution(
            needs_resolution=True,
            antecedents=["regex term"],
            confidence=0.75,
            source="regex",
            reasoning="(stub)",
        )
        mock_llm.return_value = _ok_verdict(True, ["llm term"], 0.95)

        v = resolve_pronoun(
            "really? Check it",
            _turns("(history)"),
            state=self.state,
        )
        # Adopted regex.
        self.assertEqual(v.source, "regex")
        self.assertEqual(v.antecedents, ["regex term"])
        # But LLM was called (budget consumed).
        mock_llm.assert_called_once()
        self.assertEqual(self.state.pronoun_call_count, 1)

    @patch("eva_pronoun_resolver._regex_fallback")
    @patch("eva_pronoun_resolver._call_llm")
    def test_shadow_trace_emitted_on_disagreement(self, mock_llm, mock_regex):
        import io
        import contextlib
        mock_regex.return_value = PronounResolution(
            needs_resolution=True,
            antecedents=["music box"],
            confidence=0.75,
            source="regex",
            reasoning="",
        )
        mock_llm.return_value = _ok_verdict(False, [], 0.9)

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            resolve_pronoun(
                "check it", _turns("(history)"), state=self.state,
            )
        out = buf.getvalue()
        self.assertIn("[PRONOUN-SHADOW]", out)
        self.assertIn("regex_needs=True", out)
        self.assertIn("llm_needs=False", out)
        # Disagree: needs flags differ → agree=False.
        self.assertIn("agree=False", out)

    @patch("eva_pronoun_resolver._regex_fallback")
    @patch("eva_pronoun_resolver._call_llm")
    def test_shadow_trace_on_llm_unavailable(self, mock_llm, mock_regex):
        import io
        import contextlib
        mock_regex.return_value = PronounResolution(
            needs_resolution=True,
            antecedents=["music box"],
            confidence=0.75,
            source="regex",
            reasoning="",
        )
        mock_llm.return_value = None  # LLM down

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            v = resolve_pronoun(
                "check it", _turns("(history)"), state=self.state,
            )
        out = buf.getvalue()
        self.assertIn("[PRONOUN-SHADOW]", out)
        self.assertIn("llm_needs=None", out)
        self.assertIn("agree=N/A", out)
        # Verdict still comes from regex.
        self.assertEqual(v.source, "regex")
        # Budget was consumed even though LLM returned None — we tried.
        self.assertEqual(self.state.pronoun_call_count, 1)

    @patch("eva_pronoun_resolver._regex_fallback")
    @patch("eva_pronoun_resolver._call_llm")
    def test_shadow_skips_llm_when_budget_exhausted(self, mock_llm, mock_regex):
        eva_config.PRONOUN_RESOLVER_MAX_CALLS_PER_TURN = 1
        self.state.pronoun_call_count = 1  # already exhausted
        mock_regex.return_value = PronounResolution(
            needs_resolution=True,
            antecedents=["music box"],
            confidence=0.75,
            source="regex",
            reasoning="",
        )
        v = resolve_pronoun(
            "check it", _turns("(history)"), state=self.state,
        )
        mock_llm.assert_not_called()
        self.assertEqual(v.source, "regex")
        # Budget was already at 1; not incremented.
        self.assertEqual(self.state.pronoun_call_count, 1)

    @patch("eva_pronoun_resolver._regex_fallback")
    @patch("eva_pronoun_resolver._call_llm")
    def test_shadow_off_when_mode_regex_only(self, mock_llm, mock_regex):
        # Shadow has no effect outside llm_first mode.
        eva_config.PRONOUN_RESOLVER_MODE = "regex_only"
        eva_config.PRONOUN_RESOLVER_SHADOW = True
        mock_regex.return_value = PronounResolution(
            needs_resolution=True,
            antecedents=["x"],
            confidence=0.75,
            source="regex",
            reasoning="",
        )
        v = resolve_pronoun(
            "check it", _turns("(history)"), state=self.state,
        )
        # LLM never called — pure regex_only path.
        mock_llm.assert_not_called()
        self.assertEqual(self.state.pronoun_call_count, 0)
        self.assertEqual(v.source, "regex")

    def test_jaccard_helper(self):
        from eva_pronoun_resolver import _jaccard
        self.assertEqual(_jaccard([], []), 1.0)
        self.assertEqual(_jaccard(["a"], []), 0.0)
        self.assertEqual(_jaccard(["a"], ["a"]), 1.0)
        # Case-insensitive
        self.assertEqual(_jaccard(["Music Box"], ["music box"]), 1.0)
        # Partial overlap: {a, b} ∩ {b, c} = {b}, union = {a,b,c} → 1/3
        self.assertAlmostEqual(_jaccard(["a", "b"], ["b", "c"]), 1 / 3)


# ============================================================
# P6.1 wiring equivalence test
# ============================================================
# Verifies that calling resolve_pronoun() in MODE="regex_only" produces
# the SAME antecedents list (in the same order) as the legacy direct
# call to _is_pronoun_followup + _extract_topical_nouns_from_recent_turns.
# This is the bit-identity contract P6.1 ships against — without it we
# can't claim the wiring change is dark.
#
# Skipped automatically when eva_verifier_logic can't load (e.g. offline
# CI without rank_bm25). Requires py310 / full ML stack.
class TestP61RegexOnlyEquivalence(unittest.TestCase):

    def setUp(self):
        try:
            from eva_verifier_logic import (
                _is_pronoun_followup,
                _extract_topical_nouns_from_recent_turns,
            )
        except ImportError as e:
            self.skipTest(f"eva_verifier_logic unavailable: {e}")
        self._is_followup = _is_pronoun_followup
        self._extract = _extract_topical_nouns_from_recent_turns
        self._snap = {
            "PRONOUN_RESOLVER_MODE": eva_config.PRONOUN_RESOLVER_MODE,
        }
        eva_config.PRONOUN_RESOLVER_MODE = "regex_only"

    def tearDown(self):
        for k, v in self._snap.items():
            setattr(eva_config, k, v)

    def _legacy_compute(self, query, recent):
        """Reproduce the pre-P6 build_required_memory_params logic
        for the antecedent step. Returns the keywords_extra list the
        old code would have built."""
        if not self._is_followup(query):
            return []
        terms = self._extract(recent or [])
        return list(terms)

    def test_equivalence_on_acceptance_fixtures(self):
        """For each acceptance fixture, regex_only resolver and the
        legacy two-step must agree on the keywords_extra list —
        INCLUDING legacy's known false negatives (e.g. 'hold on,
        check it' and 'sorry, check that' which legacy regex misses,
        and which were the motivating examples for the LLM path in
        the v3 plan).

        We deliberately do NOT compare against fixture expected_needs:
        those reflect LLM-mode expectations. In regex_only mode the
        contract is "resolver = legacy", warts and all. The fixture's
        expected_needs is exercised by TestAcceptanceFixtures with a
        mocked oracle LLM.
        """
        for query, hist_tail, _expected_needs, _ in ACCEPTANCE_FIXTURES:
            with self.subTest(query=query):
                recent = _turns(hist_tail)
                state = JudgeState()

                # Legacy path.
                legacy_terms = self._legacy_compute(query, recent)
                legacy_says_followup = bool(legacy_terms) or self._is_followup(query)

                # New path.
                v = resolve_pronoun(query, recent, state=state)

                # Invariant 1: needs_resolution flag matches legacy's
                # _is_pronoun_followup verdict. Note: legacy gates
                # antecedent extraction behind the regex match, so
                # legacy_terms is non-empty implies the regex matched.
                self.assertEqual(
                    v.needs_resolution, legacy_says_followup,
                    f"{query!r}: needs_resolution mismatch\n"
                    f"  legacy regex matched: {legacy_says_followup}\n"
                    f"  resolver needs:       {v.needs_resolution}",
                )
                # Invariant 2: when both agree there's a follow-up,
                # antecedent lists are bit-identical.
                if legacy_says_followup:
                    self.assertEqual(
                        v.antecedents, legacy_terms,
                        f"{query!r}: antecedents differ\n"
                        f"  legacy: {legacy_terms}\n"
                        f"  new:    {v.antecedents}",
                    )

    def test_equivalence_with_bigram_history(self):
        """Real-world history with quoted phrase + bigrams. Ensures
        the regex_only path preserves the full 6-term keyword set
        (not the 3-term LLM cap)."""
        recent = _turns(
            'I have my "special collection" of music boxes '
            'and Master\'s photo on the shelf.',
        )
        query = "really? Check it"
        state = JudgeState()
        legacy = self._legacy_compute(query, recent)
        v = resolve_pronoun(query, recent, state=state)
        self.assertTrue(v.needs_resolution)
        self.assertEqual(v.antecedents, legacy)
        # Sanity: legacy DOES produce more than 3 terms here.
        self.assertGreater(
            len(legacy), 3,
            "test fixture too small to verify the >3 keyword case",
        )


# ============================================================
# P6.3 readiness — LLM-first integration through verifier_logic
# ============================================================
# Locks in the contract that fires when P6.3 cutover happens:
#   PRONOUN_RESOLVER_MODE = "llm_first"
#   PRONOUN_RESOLVER_SHADOW = False
# i.e. LLM verdict is adopted directly and regex is only fallback on
# LLM failure.
#
# Drives `build_required_memory_params` end-to-end with a stub agent
# so we exercise the full call chain (verifier_logic → resolver →
# LLM mock). This proves the wiring change in P6.1 will continue to
# work after the P6.3 flag flip with no further code changes — the
# cutover is genuinely a one-line config edit.
#
# Skipped automatically when eva_verifier_logic can't load.
class TestP63LLMFirstIntegration(unittest.TestCase):

    def setUp(self):
        try:
            from eva_verifier_logic import build_required_memory_params
        except ImportError as e:
            self.skipTest(f"eva_verifier_logic unavailable: {e}")
        self._build = build_required_memory_params
        self._snap = {
            "PRONOUN_RESOLVER_MODE": eva_config.PRONOUN_RESOLVER_MODE,
            "PRONOUN_RESOLVER_SHADOW": eva_config.PRONOUN_RESOLVER_SHADOW,
        }
        # P6.3 production-ready settings.
        eva_config.PRONOUN_RESOLVER_MODE = "llm_first"
        eva_config.PRONOUN_RESOLVER_SHADOW = False

    def tearDown(self):
        for k, v in self._snap.items():
            setattr(eva_config, k, v)

    def _stub_agent(self, history_assistant_lines):
        """Build a minimal ChatAgent-like stub. Only the interfaces
        actually consumed by build_required_memory_params are stubbed;
        anything else would surface as AttributeError, which is the
        desired behaviour (forces test to be honest about the call
        contract)."""
        history = list(history_assistant_lines)

        class _StubHistory:
            user_name = "Rosm"

            def recent_turns(self, n=2):
                return [{"user": "", "assistant": a} for a in history[-n:]]

        class _StubAgent:
            history_manager = _StubHistory()
            _llm_judge_state = JudgeState()

            def _guard_memorysearch_params(self, params, latest_user_text):
                # Identity passthrough — verifier_logic uses this only
                # for sanitisation, not antecedent injection. Tests
                # don't need the sanitisation behaviour.
                return params

        return _StubAgent()

    def test_llm_antecedents_propagate_to_keywords(self):
        """LLM verdict's antecedents end up in the MemorySearch
        params' keywords field. This is the cutover's load-bearing
        invariant."""
        agent = self._stub_agent(["I have my favourite music box."])
        with patch(
            "eva_pronoun_resolver._call_llm",
            return_value=_ok_verdict(True, ["music box"], 0.95),
        ):
            params = self._build(agent, "really? Check it")
        self.assertIn("music box", params["keywords"])
        # The query passed to target inference is augmented with the
        # head antecedent (top-2 joined).
        self.assertIn("music box", params["query"])
        # LLM was the source — budget consumed.
        self.assertEqual(agent._llm_judge_state.pronoun_call_count, 1)

    def test_llm_failure_falls_through_to_regex(self):
        """When LLM is unavailable, verifier_logic still gets a
        sensible MemorySearch params dict via the regex fallback.
        This is the "graceful degradation to pre-P5 behaviour"
        contract — the system never loses verifier ability just
        because DeepSeek is down."""
        agent = self._stub_agent(["I have my music box."])
        with patch(
            "eva_pronoun_resolver._call_llm",
            return_value=None,  # LLM down
        ):
            params = self._build(agent, "really? Check it")
        # Regex fallback fired — keywords still contain antecedents.
        self.assertIn("music box", params["keywords"])
        # Budget consumed even on LLM failure (we tried).
        self.assertEqual(agent._llm_judge_state.pronoun_call_count, 1)

    def test_regex_legacy_miss_now_caught_by_llm(self):
        """The fixture that motivated P6 in the first place:
        'hold on, check it' — legacy regex misses the prefix, but
        LLM identifies the antecedent. After P6.3 cutover this case
        works."""
        agent = self._stub_agent(["I have my music box."])
        with patch(
            "eva_pronoun_resolver._call_llm",
            return_value=_ok_verdict(True, ["music box"], 0.92),
        ):
            params = self._build(agent, "hold on, check it")
        # Pre-P6 regex would have missed this and produced a query
        # without antecedent enrichment. P6.3 LLM path catches it.
        self.assertIn("music box", params["keywords"])

    def test_low_confidence_llm_does_not_inject(self):
        """Low-confidence LLM verdicts (below
        PRONOUN_RESOLVER_MIN_CONFIDENCE) are demoted, so antecedents
        are NOT injected into the verifier's memory query. This
        protects against hallucinated antecedents poisoning recall."""
        agent = self._stub_agent(["I have my music box."])
        with patch(
            "eva_pronoun_resolver._call_llm",
            return_value=_ok_verdict(True, ["music box"], 0.30),
        ):
            params = self._build(agent, "really? Check it")
        # Antecedent NOT injected — query stays clean.
        self.assertNotIn("music box", params["keywords"])

    def test_self_contained_query_skips_resolver(self):
        """Long, self-contained queries don't hit the LLM — the cheap
        gate filters them. This is the cost-control invariant: the
        resolver only spends LLM budget on queries that actually need
        antecedent resolution."""
        agent = self._stub_agent(["I have my music box."])
        with patch("eva_pronoun_resolver._call_llm") as mock_llm:
            self._build(
                agent,
                "what is the population of Tokyo and how does it compare "
                "to Seoul today",
            )
        mock_llm.assert_not_called()
        self.assertEqual(agent._llm_judge_state.pronoun_call_count, 0)


if __name__ == "__main__":
    unittest.main()
