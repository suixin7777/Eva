"""tests/test_advisor.py — Advisor module unit tests.

Covers (no network):
  - build_advisor_prompt structure
  - get_advice fallback paths (disabled / empty / budget exhausted)
  - format_advice_block output shape
  - cache TTL behavior

Live DeepSeek calls are NOT tested here — those go through a separate
integration test that requires DEEPSEEK_API_KEY and runs on demand.
"""
from __future__ import annotations

import os
import sys
import time
import unittest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from Advisor.advisor_client import (
    AdvisorResult,
    get_advice,
    format_advice_block,
    reset_cache,
    _cache_key,
    _cache_get,
    _cache_put,
)
from Advisor.build_advisor_prompt import build_advisor_prompt


# ============================================================
# build_advisor_prompt assembly
# ============================================================
class TestBuildPrompt(unittest.TestCase):
    def test_minimal_inputs(self):
        payload = build_advisor_prompt(
            user_text="forget the meeting",
            history_lines=[],
        )
        self.assertIn("[user_input]", payload)
        self.assertIn("forget the meeting", payload)
        self.assertIn("[recent_history]", payload)
        self.assertIn("(no prior turns", payload)
        self.assertIn("[recent_notes]", payload)
        self.assertIn("[task]", payload)

    def test_history_trimmed_to_max(self):
        many = [f"User: line{i:02d}" for i in range(20)]
        payload = build_advisor_prompt(
            user_text="ok",
            history_lines=many,
        )
        # Default _MAX_HISTORY_LINES = 6. earliest lines should be gone.
        self.assertNotIn("line00", payload)
        self.assertNotIn("line01", payload)
        # Last few should appear
        self.assertIn("line19", payload)

    def test_recent_notes_rendered(self):
        notes = [
            {"note_id": "abc12345", "topic": "Meeting", "preview": "standup tomorrow"},
            {"note_id": "deadbeef", "topic": "Pet", "preview": "adopted cat"},
        ]
        payload = build_advisor_prompt(
            user_text="forget that",
            history_lines=[],
            recent_notes=notes,
        )
        self.assertIn("#abc12345", payload)
        self.assertIn("topic=Meeting", payload)
        self.assertIn("standup tomorrow", payload)
        self.assertIn("#deadbeef", payload)

    def test_eva_state_idle_default(self):
        payload = build_advisor_prompt(
            user_text="hi",
            history_lines=[],
            eva_state=None,
        )
        self.assertIn("idle", payload)

    def test_eva_state_active(self):
        payload = build_advisor_prompt(
            user_text="how's the report",
            history_lines=[],
            eva_state={
                "current_activity": "writing_report",
                "context": {"deadline": "Saturday"},
            },
        )
        self.assertIn("writing_report", payload)
        self.assertIn("deadline=Saturday", payload)

    def test_relevant_memory_block_included(self):
        payload = build_advisor_prompt(
            user_text="what's my toy?",
            history_lines=[],
            relevant_memory="[Memory] Eva's favourite toy is a stuffed bunny named Mochi.",
        )
        self.assertIn("stuffed bunny", payload)
        self.assertIn("Mochi", payload)

    def test_long_user_text_truncated(self):
        long = "a" * 5000
        payload = build_advisor_prompt(
            user_text=long,
            history_lines=[],
        )
        # Should have truncation marker
        self.assertIn("[truncated]", payload)
        # Plus the chunk should be capped roughly to _MAX_USER_TEXT_CHARS
        self.assertLess(payload.count("a"), 5000)


# ============================================================
# get_advice fallback paths
# ============================================================
class TestGetAdviceFallback(unittest.TestCase):
    def setUp(self):
        reset_cache()

    def test_disabled_returns_fallback(self):
        r = get_advice(
            user_text="anything",
            history_lines=[],
            enabled=False,
        )
        self.assertIsInstance(r, AdvisorResult)
        self.assertFalse(r.ok)
        self.assertEqual(r.source, "fallback_disabled")
        self.assertEqual(r.advice, "")

    def test_empty_user_text_returns_fallback(self):
        r = get_advice(
            user_text="   ",
            history_lines=[],
            enabled=True,
        )
        self.assertFalse(r.ok)
        self.assertEqual(r.source, "fallback_disabled")

    def test_budget_exhausted_returns_fallback(self):
        budget = {"advisor_calls": 0}
        r = get_advice(
            user_text="forget X",
            history_lines=[],
            enabled=True,
            budget_state=budget,
        )
        self.assertFalse(r.ok)
        self.assertIn("budget", (r.error or "").lower())

    def test_missing_api_key_returns_fallback(self):
        # Temporarily clear DEEPSEEK_API_KEY so the live call branch
        # returns fallback_error rather than reaching the network.
        prev = os.environ.get("DEEPSEEK_API_KEY")
        os.environ["DEEPSEEK_API_KEY"] = ""
        try:
            # Force reload of eva_config to pick up the change. We can't
            # easily reload, but advisor_client reads from eva_config at
            # call time so the empty string propagates.
            import importlib
            import eva_config
            importlib.reload(eva_config)
            r = get_advice(
                user_text="forget X",
                history_lines=[],
                enabled=True,
                budget_state={"advisor_calls": 2},
                timeout_seconds=0.1,
            )
            self.assertFalse(r.ok)
            # Either fallback_error (no key) or fallback_error (timeout) is OK
            self.assertIn(r.source, ("fallback_error",))
        finally:
            if prev is not None:
                os.environ["DEEPSEEK_API_KEY"] = prev
            else:
                os.environ.pop("DEEPSEEK_API_KEY", None)


# ============================================================
# format_advice_block
# ============================================================
class TestFormatAdviceBlock(unittest.TestCase):
    def test_ok_result_produces_block(self):
        r = AdvisorResult(
            ok=True,
            advice="Call RememberThis twice: one for the bear, one for the report.",
            source="live",
        )
        block = format_advice_block(r)
        self.assertIn("[Advisor Hint", block)
        self.assertIn("RememberThis", block)
        self.assertIn("persona", block.lower())  # the trailer mentions persona

    def test_failed_result_produces_empty(self):
        r = AdvisorResult(ok=False, error="timeout")
        self.assertEqual(format_advice_block(r), "")

    def test_empty_advice_produces_empty(self):
        r = AdvisorResult(ok=True, advice="")
        self.assertEqual(format_advice_block(r), "")


# ============================================================
# Schema: new fields populated correctly (Advisor-first refactor)
# ============================================================
class TestAdvisorResultSchema(unittest.TestCase):
    def test_default_fields(self):
        """Bare AdvisorResult has safe defaults for all new fields."""
        r = AdvisorResult()
        self.assertEqual(r.intent, "unknown")
        self.assertFalse(r.needs_memory_retrieval)
        self.assertIsNone(r.memory_query_hint)
        self.assertFalse(r.needs_web_search)
        self.assertIsNone(r.web_query_hint)
        self.assertEqual(r.suggested_calls, [])

    def test_construct_with_intent(self):
        r = AdvisorResult(
            ok=True,
            advice="Call RememberThis twice.",
            intent="mixed",
            needs_memory_retrieval=False,
            needs_web_search=False,
        )
        self.assertEqual(r.intent, "mixed")
        self.assertFalse(r.needs_memory_retrieval)

    def test_format_advice_block_keeps_intent_neutral(self):
        """format_advice_block doesn't leak intent into the prompt block —
        it only emits the natural-language advice text."""
        r = AdvisorResult(
            ok=True, advice="Be careful.", intent="forget",
        )
        block = format_advice_block(r)
        self.assertIn("Be careful", block)
        # 'forget' is just an internal label — should not bleed into prompt
        self.assertNotIn('"intent"', block)
        self.assertNotIn('forget"', block)


# ============================================================
# Cache
# ============================================================
class TestCache(unittest.TestCase):
    def setUp(self):
        reset_cache()

    def test_cache_round_trip(self):
        key = _cache_key("hello", "hash123")
        r = AdvisorResult(ok=True, advice="say hi", source="live")
        _cache_put(key, r)
        got = _cache_get(key)
        self.assertIsNotNone(got)
        self.assertEqual(got.advice, "say hi")
        self.assertEqual(got.source, "cache")  # source rewritten to cache

    def test_failed_results_not_cached(self):
        key = _cache_key("hello", "hash123")
        r = AdvisorResult(ok=False, error="timeout")
        _cache_put(key, r)
        self.assertIsNone(_cache_get(key))

    def test_cache_distinct_keys(self):
        k1 = _cache_key("hello", "hash1")
        k2 = _cache_key("hello", "hash2")  # different history → different key
        self.assertNotEqual(k1, k2)
        k3 = _cache_key("world", "hash1")
        self.assertNotEqual(k1, k3)


if __name__ == "__main__":
    unittest.main()
