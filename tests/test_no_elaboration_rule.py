"""Offline tests for the [NO ELABORATION RULE] anti-fabrication backstop
in eva_verifier_logic (added 2026-05-08).

Backdrop: eva_memory_legacy emits the prompt-side `[NO ELABORATION RULE]`
warning when memory retrieval is low-confidence. The phase-2 model is
told to hedge instead of inventing scene details (places, actions,
mood, weather, etc.). Soft prompt rules leak — see the lasagna/smoke-
alarm hallucination logged 2026-05-08 — so the verifier hard-checks
the constraint after generation. This test pins the contract.

Usage (from project root):
    python tests/test_no_elaboration_rule.py
"""
import os
import sys
import types
import unittest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# eva_config does `import torch` for two cudnn flags. Stub for offline run.
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

from eva_verifier_logic import (
    answer_violates_no_elaboration_rule,
    REASON_POLICY,
    _no_elab_extract_content_tokens,
    _no_elab_normalize_token,
)


class _StubEvidence:
    """Local TurnEvidence stand-in. The predicate only reads .source and
    .raw_text; we avoid `from eva_history import TurnEvidence` because
    eva_history transitively imports transformers (heavy ML dep) and
    this test is meant to run without the inference stack installed."""
    __slots__ = ("source", "raw_text")

    def __init__(self, source, raw_text):
        self.source = source
        self.raw_text = raw_text


class _StubAgent:
    """Minimal agent surface for the verifier predicate."""
    def __init__(self, turn_evidence=None):
        self.turn_evidence = turn_evidence or []


def _mem_evidence(raw_text):
    return _StubEvidence(source="memory", raw_text=raw_text)


_RECORDS_WITH_RULE = (
    "### [MEMORY MODULE DATA for 'Eva'] ###\n"
    "Record 1 [Lore][Subject:Eva][Topic:Cooking]: Eva sometimes helps Rosm "
    "in the kitchen.\n"
    "[NO ELABORATION RULE]: You may ONLY state facts that are literally "
    "written in the records above. Do NOT invent specific details such as: "
    "how someone looked, what they wore, weather, time, room descriptions."
)

_RECORDS_NO_RULE = (
    "### [MEMORY MODULE DATA for 'Eva'] ###\n"
    "Record 1 [Lore][Subject:Eva][Topic:Toy]: Eva's favorite toy has always "
    "been a cuddly bunny — soft, slightly worn at the ears, with one button "
    "eye that's been re-sewn twice."
)


class TestNoElaborationRulePredicate(unittest.TestCase):
    """Direct tests against answer_violates_no_elaboration_rule."""

    # --- Trigger gating -----------------------------------------------

    def test_rule_not_triggered_returns_false(self):
        """No evidence carries [NO ELABORATION RULE] -> always pass,
        even with a wildly fabricated answer."""
        agent = _StubAgent([_mem_evidence(_RECORDS_NO_RULE)])
        wild_answer = (
            "Hmph! Master once nearly burned a lasagna, set off the smoke "
            "alarm during a thunderstorm, while the radio played jazz."
        )
        self.assertFalse(
            answer_violates_no_elaboration_rule(agent, wild_answer, "do you cook?"),
            "Without [NO ELABORATION RULE] in evidence the check must not fire.",
        )

    def test_no_evidence_at_all_returns_false(self):
        agent = _StubAgent([])
        self.assertFalse(
            answer_violates_no_elaboration_rule(
                agent, "Anything goes here, lasagna and thunderstorms.",
                "tell me a story",
            ),
        )

    # --- Hedge bypass --------------------------------------------------

    def test_hedge_dont_have_specific_memory_passes(self):
        agent = _StubAgent([_mem_evidence(_RECORDS_WITH_RULE)])
        answer = (
            "Hmph — I don't have a specific memory about that, Master. "
            "I just remember being in the kitchen sometimes."
        )
        self.assertFalse(
            answer_violates_no_elaboration_rule(agent, answer, "what happened?"),
        )

    def test_hedge_specifics_arent_recorded_passes(self):
        agent = _StubAgent([_mem_evidence(_RECORDS_WITH_RULE)])
        answer = (
            "I remember it happened, but the specifics aren't recorded — "
            "lasagna and thunderstorms and smoke alarms aside."
        )
        self.assertFalse(
            answer_violates_no_elaboration_rule(agent, answer, "what happened?"),
        )

    def test_hedge_dont_remember_passes(self):
        agent = _StubAgent([_mem_evidence(_RECORDS_WITH_RULE)])
        answer = "Hmph, I don't really remember the specifics of that, Master."
        self.assertFalse(
            answer_violates_no_elaboration_rule(agent, answer, "tell me about it"),
        )

    # --- Violation cases ----------------------------------------------

    def test_lasagna_smoke_alarm_thunderstorm_fabrication_fails(self):
        """The 2026-05-08 regression case: rule triggered, answer
        invents lasagna + smoke alarm + thunderstorm + jazz radio
        without hedging. Must be caught."""
        agent = _StubAgent([_mem_evidence(_RECORDS_WITH_RULE)])
        answer = (
            "Hmph, Master — once you nearly burned a lasagna and set off "
            "the smoke alarm. Right in the middle of a thunderstorm too, "
            "with jazz on the radio. You should be embarrassed."
        )
        self.assertTrue(
            answer_violates_no_elaboration_rule(agent, answer, "anything fun in the kitchen?"),
            "lasagna/smoke/thunderstorm/jazz fabrication must trip the check.",
        )

    def test_supported_answer_passes(self):
        """Answer only echoes record content + persona stutters: no violation."""
        agent = _StubAgent([_mem_evidence(_RECORDS_WITH_RULE)])
        answer = "Hmph, of course I help Rosm in the kitchen sometimes, Master."
        self.assertFalse(
            answer_violates_no_elaboration_rule(agent, answer, "do you cook with rosm?"),
        )

    def test_single_unsupported_token_passes_under_threshold(self):
        """Threshold = 3. A single odd word should not trip the check."""
        agent = _StubAgent([_mem_evidence(_RECORDS_WITH_RULE)])
        # "carrots" is the only content word not in records or query.
        answer = "Hmph, of course I help Rosm in the kitchen — sometimes with carrots."
        self.assertFalse(
            answer_violates_no_elaboration_rule(agent, answer, "do you help in the kitchen?"),
        )

    def test_threshold_override_to_one_catches_single_token(self):
        """Lower threshold catches single-token inventions for stricter callers."""
        agent = _StubAgent([_mem_evidence(_RECORDS_WITH_RULE)])
        answer = "Hmph, I help Rosm in the kitchen — sometimes with carrots."
        self.assertTrue(
            answer_violates_no_elaboration_rule(
                agent, answer, "do you help in the kitchen?", min_unsupported=1,
            ),
        )

    def test_query_echo_does_not_count_as_invention(self):
        """Tokens echoed from the user's question are not unsupported."""
        agent = _StubAgent([_mem_evidence(_RECORDS_WITH_RULE)])
        # ballerina/wooden/Tchaikovsky are not in records, but they're in
        # the user's question -> query support, not invention.
        user_q = "did you ever get a wooden ballerina playing Tchaikovsky?"
        answer = (
            "Hmph, you mentioned a wooden ballerina playing Tchaikovsky — "
            "I don't have a specific memory about that, Master."
        )
        self.assertFalse(
            answer_violates_no_elaboration_rule(agent, answer, user_q),
        )

    # --- Empty / degenerate inputs ------------------------------------

    def test_empty_answer_returns_false(self):
        agent = _StubAgent([_mem_evidence(_RECORDS_WITH_RULE)])
        self.assertFalse(answer_violates_no_elaboration_rule(agent, "", "q"))
        self.assertFalse(answer_violates_no_elaboration_rule(agent, None, "q"))

    def test_non_string_answer_returns_false(self):
        agent = _StubAgent([_mem_evidence(_RECORDS_WITH_RULE)])
        self.assertFalse(answer_violates_no_elaboration_rule(agent, 42, "q"))


class TestNoElabHelpers(unittest.TestCase):
    """Lower-level helpers used by the predicate."""

    def test_normalize_strips_possessive_and_plural(self):
        self.assertEqual(_no_elab_normalize_token("Master's"), "master")
        self.assertEqual(_no_elab_normalize_token("ballerinas"), "ballerina")
        self.assertEqual(_no_elab_normalize_token("playing"), "play")
        self.assertEqual(_no_elab_normalize_token("burned"), "burn")

    def test_normalize_short_words_unchanged(self):
        # length-4 stems are kept whole — stripping would make them too tiny.
        self.assertEqual(_no_elab_normalize_token("toys"), "toys")
        self.assertEqual(_no_elab_normalize_token("eyes"), "eyes")

    def test_extract_filters_function_words(self):
        toks = _no_elab_extract_content_tokens(
            "this would have been the moment when Master remembered nothing"
        )
        # All listed tokens are in stopwords (modal/aux/persona/mental/placeholder).
        self.assertEqual(toks, set())

    def test_extract_keeps_scene_specifics(self):
        toks = _no_elab_extract_content_tokens(
            "lasagna burned in the kitchen during a thunderstorm"
        )
        # Scene-detail words (the kind the rule forbids inventing) survive.
        self.assertIn("lasagna", toks)
        self.assertIn("burn", toks)
        self.assertIn("kitchen", toks)
        self.assertIn("thunderstorm", toks)


class TestR32SubjectiveQuestionGating(unittest.TestCase):
    """R-3.2 (2026-05-13)：subjective / preference / state 问句应跳过 rule。

    复盘 2026-05-13 实跑 Turn 6/7：user 问 "do you want a new one?" 模型答持
    persona 偏好 ("Tch, why would I ever need a replacement?")——所有 token
    都不在 records 里，旧 rule 误报 → regen + canned 破 persona。
    """

    def test_do_you_want_skips_rule(self):
        agent = _StubAgent([_mem_evidence(_RECORDS_WITH_RULE)])
        # Persona-only answer: 大量不在 records 的 content tokens
        answer = (
            "Tch, why would I ever need a replacement? This one's perfect "
            "as-is. But if you're thinking of getting one for fun, go "
            "ahead—I won't complain too much."
        )
        # 旧逻辑会因 unsupported tokens >= 3 而 fail；R-3.2 应跳过
        self.assertFalse(
            answer_violates_no_elaboration_rule(
                agent, answer, "do you want a new one?",
            ),
            "subjective question 'do you want' 应跳过 no_elaboration_rule",
        )

    def test_do_you_like_skips_rule(self):
        agent = _StubAgent([_mem_evidence(_RECORDS_WITH_RULE)])
        answer = "Hmph, I love it—but pretending I don't makes the moment fun."
        self.assertFalse(
            answer_violates_no_elaboration_rule(
                agent, answer, "Do you like chocolate cake?",
            ),
        )

    def test_would_you_like_skips_rule(self):
        agent = _StubAgent([_mem_evidence(_RECORDS_WITH_RULE)])
        answer = "Maybe a small one—nothing extravagant, Master."
        self.assertFalse(
            answer_violates_no_elaboration_rule(
                agent, answer, "would you like a present?",
            ),
        )

    def test_are_you_state_question_skips_rule(self):
        agent = _StubAgent([_mem_evidence(_RECORDS_WITH_RULE)])
        answer = "Pretty good actually—getting some sun helped."
        self.assertFalse(
            answer_violates_no_elaboration_rule(
                agent, answer, "are you ok?",
            ),
        )

    def test_chinese_subjective_skips_rule(self):
        agent = _StubAgent([_mem_evidence(_RECORDS_WITH_RULE)])
        answer = "嗯，可能吧，不过Master你这样问我反而不太自在。"
        self.assertFalse(
            answer_violates_no_elaboration_rule(
                agent, answer, "你觉得怎么样?",
            ),
        )

    # ---- 关键反测：narrative invitation 仍应 fire ----
    def test_lasagna_still_caught_when_question_is_narrative(self):
        """与上面 lasagna_smoke_alarm 测试同义但放在 R-3.2 区块下，
        显式记录 'narrative-invitation 问句不被视为 subjective'。"""
        agent = _StubAgent([_mem_evidence(_RECORDS_WITH_RULE)])
        answer = (
            "Hmph, Master — once you nearly burned a lasagna and set off "
            "the smoke alarm. Right in the middle of a thunderstorm too, "
            "with jazz on the radio."
        )
        # "anything fun in X?" 是 narrative 开放式问句，不被视为 subjective
        # → rule 仍 fire → 抓住 invented lasagna/smoke/thunderstorm/jazz
        self.assertTrue(
            answer_violates_no_elaboration_rule(
                agent, answer, "anything fun in the kitchen?",
            ),
            "narrative-invitation 'anything fun' 不应被视为 subjective skip",
        )

    def test_why_question_not_treated_as_subjective(self):
        """'why X' 是解释 / 因果 ask，非主观偏好。rule 仍 fire。"""
        agent = _StubAgent([_mem_evidence(_RECORDS_WITH_RULE)])
        # 模型给一个含 unsupported tokens 的解释（仿造前一次实跑 Turn 7
        # "cake you'd owe me" 情况——content tokens lasagna/smoke 类似不存在）
        answer = (
            "Hmph, maybe I was distracted by lasagna and smoke alarms while "
            "thinking about thunderstorm season—your fault really."
        )
        # "why" 不被识别为 subjective → rule 仍 fire（threshold 3+）
        self.assertTrue(
            answer_violates_no_elaboration_rule(
                agent, answer, "why did you calculate it wrong before?",
            ),
        )


class TestReasonPolicyRegistration(unittest.TestCase):
    """Pin the new reason's policy entry — REASON_POLICY is the single
    source of truth for severity + fix_class."""

    def test_reason_registered_with_regenerate_fix(self):
        # 2026-05-14 Plan-A: regex verifier 全退役。该 reason 仍在
        # REASON_POLICY 里登记（telemetry 用），但 severity 降为 soft /
        # fix 改 canned_fallback——不再触发 regen 杀掉正确答案。
        # 真幻觉由 semantic_verifier_fail:fact_conflict_with_evidence /
        # pronoun_referent_mismatch 等 LLM-judge reason 接住。
        entry = REASON_POLICY.get("unsupported_specifics_under_no_elaboration_rule")
        self.assertIsNotNone(entry, "reason missing from REASON_POLICY")
        self.assertEqual(entry["severity"], "soft")
        self.assertEqual(entry["fix"], "canned_fallback")
        self.assertIn("canned", entry)
        self.assertTrue(entry["canned"], "canned message must be non-empty")


if __name__ == "__main__":
    unittest.main(verbosity=2)
