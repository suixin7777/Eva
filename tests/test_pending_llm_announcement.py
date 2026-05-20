"""Offline tests for the [PENDING DeepSeek] UX placeholder
(announce_pending_llm) added 2026-05-08.

Backstop: the verifier-repair path stacks two synchronous DeepSeek
calls (~3s each — pronoun resolver + synthesize_tool_thought) between
`--- ANSWER VERIFIER FAILED ---` and the rewrite block. Without a
marker the operator stares at dead air. announce_pending_llm prints
a one-line `[PENDING DeepSeek] <label>` placeholder so the latency
cause is visible.

These tests pin:
  1. The helper's output shape.
  2. resolve_pronoun fires the placeholder ONLY when it actually
     calls the LLM (not on cache hit / budget exhaustion / cheap-gate
     skip).

Usage (from project root):
    python tests/test_pending_llm_announcement.py
"""
import io
import os
import sys
import types
import contextlib
import unittest
from unittest.mock import patch

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

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
from eva_intent_judge import JudgeState, announce_pending_llm
import eva_pronoun_resolver
from eva_pronoun_resolver import resolve_pronoun


_TURNS = [
    {"user": "remember the music box?",
     "assistant": "Of course — the wooden one Master gave me."},
]


@contextlib.contextmanager
def _capture_stdout():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield buf


class TestAnnouncePendingLLM(unittest.TestCase):

    def test_basic_format(self):
        with _capture_stdout() as buf:
            announce_pending_llm("resolve_pronoun: q='check it'")
        out = buf.getvalue()
        self.assertIn("[PENDING DeepSeek]", out)
        self.assertIn("resolve_pronoun", out)
        self.assertTrue(out.startswith("        | "),
                        "must use the trace indent prefix")
        self.assertTrue(out.endswith("\n"))

    def test_long_label_truncated(self):
        long_label = "x" * 500
        with _capture_stdout() as buf:
            announce_pending_llm(long_label)
        out = buf.getvalue()
        self.assertIn("…", out, "long labels should ellipsis")
        # Body should not contain the full 500-char run.
        self.assertNotIn("x" * 200, out)


class TestResolvePronounAnnouncement(unittest.TestCase):
    """Verify resolve_pronoun fires the placeholder iff it actually
    calls the LLM. Skipping paths (cheap-gate, cache hit, budget
    exhaustion) must NOT emit the placeholder — they have no latency
    to absorb."""

    def setUp(self):
        self._snap = (
            getattr(eva_config, "PRONOUN_RESOLVER_MODE", None),
            getattr(eva_config, "PRONOUN_RESOLVER_SHADOW", None),
            getattr(eva_config, "ENABLE_PRONOUN_RESOLVER", None),
        )
        eva_config.PRONOUN_RESOLVER_MODE = "llm_first"
        eva_config.PRONOUN_RESOLVER_SHADOW = False
        eva_config.ENABLE_PRONOUN_RESOLVER = True

    def tearDown(self):
        (eva_config.PRONOUN_RESOLVER_MODE,
         eva_config.PRONOUN_RESOLVER_SHADOW,
         eva_config.ENABLE_PRONOUN_RESOLVER) = self._snap

    def test_placeholder_fires_on_llm_call(self):
        state = JudgeState()
        with _capture_stdout() as buf, patch(
            "eva_pronoun_resolver._call_llm",
            return_value='{"needs_resolution": true, '
                         '"antecedents": ["music box"], '
                         '"confidence": 0.9}',
        ):
            resolve_pronoun("really? Check it", _TURNS, state=state)
        out = buf.getvalue()
        self.assertIn("[PENDING DeepSeek]", out)
        self.assertIn("resolve_pronoun", out)

    def test_no_placeholder_on_cheap_gate_skip(self):
        """Long sentences fail the cheap gate -> no LLM call -> no placeholder."""
        long_q = "could you possibly tell me about your hobbies and interests"
        state = JudgeState()
        with _capture_stdout() as buf, patch(
            "eva_pronoun_resolver._call_llm",
            return_value='{"needs_resolution": false, "antecedents": [], "confidence": 1.0}',
        ) as mock_llm:
            resolve_pronoun(long_q, _TURNS, state=state)
        self.assertEqual(mock_llm.call_count, 0,
                         "cheap-gate skip must not call LLM")
        self.assertNotIn("[PENDING DeepSeek]", buf.getvalue())

    def test_no_placeholder_on_cache_hit(self):
        """Second call with same query+turns must skip LLM and placeholder."""
        state = JudgeState()
        # Prime the cache with a first call.
        with patch(
            "eva_pronoun_resolver._call_llm",
            return_value='{"needs_resolution": true, '
                         '"antecedents": ["music box"], '
                         '"confidence": 0.9}',
        ):
            resolve_pronoun("really? Check it", _TURNS, state=state)
        # Second call: should hit cache.
        with _capture_stdout() as buf, patch(
            "eva_pronoun_resolver._call_llm",
            return_value='{"needs_resolution": true, "antecedents": [], "confidence": 0.9}',
        ) as mock_llm:
            resolve_pronoun("really? Check it", _TURNS, state=state)
        self.assertEqual(mock_llm.call_count, 0,
                         "cache hit must not call LLM")
        self.assertNotIn("[PENDING DeepSeek]", buf.getvalue())

    def test_no_placeholder_on_budget_exhaustion(self):
        """Budget exhausted -> falls through to regex fallback, no LLM."""
        state = JudgeState()
        state.pronoun_call_count = 999  # well over cap
        with _capture_stdout() as buf, patch(
            "eva_pronoun_resolver._call_llm",
            return_value='{"needs_resolution": true, "antecedents": [], "confidence": 0.9}',
        ) as mock_llm:
            resolve_pronoun("really? Check it", _TURNS, state=state)
        self.assertEqual(mock_llm.call_count, 0,
                         "budget exhaustion must not call LLM")
        self.assertNotIn("[PENDING DeepSeek]", buf.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=2)
