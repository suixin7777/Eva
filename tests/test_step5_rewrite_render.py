"""Offline tests for the STEP-5 trace-rewrite render helper
(_render_step5_rewrite_block) added 2026-05-08.

Backstop: when the verifier rejects a phase-1 answer, the controller
rewrites the most recent assistant step from answer-shape into tool-
call shape. Prior THOUGHT/ANSWER text is already on the operator's
screen — it can't be retroactively struck through. The render helper
prints a loud, unambiguous boundary so the operator sees that the
prior phase-1 output is now superseded, plus prefixes the synthesised
replacement thought + tool_code with [REWRITTEN].

Two styles are supported via TRACE_REWRITE_STYLE config:
  - "ansi"  — bold-yellow header + dim-strike supersede notice
  - "ascii" — plain === bars and bracketed labels

These tests pin both renderings.

Usage (from project root):
    python tests/test_step5_rewrite_render.py
"""
import os
import sys
import types
import unittest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# torch stub — eva_config imports torch for two cudnn flags only.
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
from eva_verifier_logic import _render_step5_rewrite_block


class TestStep5RewriteRender(unittest.TestCase):

    def setUp(self):
        # Each test runs against a defined style; restore default in tearDown
        # so test ordering doesn't matter.
        self._saved_style = getattr(eva_config, "TRACE_REWRITE_STYLE", "ansi")

    def tearDown(self):
        eva_config.TRACE_REWRITE_STYLE = self._saved_style

    # --- structural invariants (style-agnostic) -----------------------

    def test_ascii_block_has_required_elements(self):
        eva_config.TRACE_REWRITE_STYLE = "ascii"
        lines = _render_step5_rewrite_block(
            "I should consult memory first.",
            'MemorySearch(query="music box")',
        )
        joined = "\n".join(lines)
        self.assertIn("STEP-5 TRACE REWRITE", joined)
        self.assertIn("SUPERSEDED", joined)
        self.assertIn("[REWRITTEN]", joined)
        self.assertIn("thought:", joined)
        self.assertIn("tool_code:", joined)
        self.assertIn("MemorySearch", joined)

    def test_ansi_block_has_required_elements(self):
        eva_config.TRACE_REWRITE_STYLE = "ansi"
        lines = _render_step5_rewrite_block(
            "I should consult memory first.",
            'MemorySearch(query="music box")',
        )
        joined = "\n".join(lines)
        self.assertIn("STEP-5 TRACE REWRITE", joined)
        self.assertIn("SUPERSEDED", joined)
        self.assertIn("[REWRITTEN]", joined)

    def test_ascii_no_escape_codes(self):
        """ASCII style must produce zero \\033 escape sequences."""
        eva_config.TRACE_REWRITE_STYLE = "ascii"
        lines = _render_step5_rewrite_block("t", "X()")
        joined = "\n".join(lines)
        self.assertNotIn("\033", joined,
                         "ascii style leaked an ANSI escape code")

    def test_ansi_includes_escape_codes(self):
        """ANSI style must include \\033 sequences for header + supersede."""
        eva_config.TRACE_REWRITE_STYLE = "ansi"
        lines = _render_step5_rewrite_block("t", "X()")
        joined = "\n".join(lines)
        self.assertIn("\033[", joined)
        # Reset codes appear paired with each formatted region.
        self.assertGreaterEqual(joined.count("\033[0m"), 4)

    def test_indent_prefix_applied_to_every_line(self):
        eva_config.TRACE_REWRITE_STYLE = "ascii"
        custom_indent = ">>>>"
        lines = _render_step5_rewrite_block("t", "X()", indent=custom_indent)
        # Every emitted line begins with the indent, including the empty
        # spacer line at the top.
        for ln in lines:
            self.assertTrue(
                ln.startswith(custom_indent),
                f"line did not begin with indent: {ln!r}",
            )

    # --- thought truncation -------------------------------------------

    def test_long_thought_is_truncated_with_ellipsis(self):
        eva_config.TRACE_REWRITE_STYLE = "ascii"
        long_thought = "x" * 200
        lines = _render_step5_rewrite_block(long_thought, "X()")
        joined = "\n".join(lines)
        # Truncation happens at 90 chars + a single ellipsis char.
        self.assertIn("…", joined)
        # Full 200-char string must NOT appear in the output.
        self.assertNotIn("x" * 100, joined)

    def test_short_thought_is_not_truncated(self):
        eva_config.TRACE_REWRITE_STYLE = "ascii"
        short = "ok"
        lines = _render_step5_rewrite_block(short, "X()")
        joined = "\n".join(lines)
        self.assertNotIn("…", joined)
        self.assertIn("'ok'", joined)


if __name__ == "__main__":
    unittest.main(verbosity=2)
