"""tests/test_verdict_ledger.py — R-3 Verdict / VerdictLedger 单元测试

R-3 (2026-05-13) 把"safe_fallback 释放哪个候选"从 caller 手动传
phase2_answer= 改为读 agent.verdict_ledger.best()。

复盘场景：2026-05-13 Turn 5
  - original  "55 days until my birthday"           regex fail (date_math)
  - regen     "my birthday is way later"           llm-judge fail (pronoun)
  - 旧逻辑：fallback 释放 regen → 用户看到比 original 更差答案
  - P0-3 patch: caller 改传 final_answer (original) 救场
  - R-3:        ledger.best() 按 (severity, reason_count, stage_order)
               自然选 original，无需 caller 知道 stage 差异

本测试用 stub 验证：
  - Verdict.severity / reason_count / reason_class 计算
  - VerdictLedger.best() 排序优先级
  - has_disagreement() 检测 regex vs llm 跨 verifier 不一致
  - safe_fallback 端到端：Turn 5 复盘 → 释放 original
"""
from __future__ import annotations

import os
import sys
import unittest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from eva_history import (  # noqa: E402
    Verdict,
    VerdictLedger,
    VerifyResult,
    _classify_verdict_reasons,
)


# ============================================================
# Reason 分类
# ============================================================
class TestClassifyReasons(unittest.TestCase):
    def test_regex_only(self):
        self.assertEqual(
            _classify_verdict_reasons(["date_math_target_date_mismatch"]),
            "regex",
        )

    def test_llm_semantic_only(self):
        self.assertEqual(
            _classify_verdict_reasons(["semantic_verifier_fail:pronoun_referent_mismatch"]),
            "llm_semantic",
        )

    def test_mixed(self):
        self.assertEqual(
            _classify_verdict_reasons([
                "date_math_target_date_mismatch",
                "semantic_verifier_fail:x",
            ]),
            "mixed",
        )

    def test_none_when_empty(self):
        self.assertEqual(_classify_verdict_reasons([]), "none")
        self.assertEqual(_classify_verdict_reasons(None), "none")


# ============================================================
# Verdict 数据契约
# ============================================================
class TestVerdict(unittest.TestCase):
    def test_severity_clean(self):
        v = Verdict("hi", "original", VerifyResult(ok=True))
        self.assertEqual(v.severity, 0)

    def test_severity_soft_fail(self):
        v = Verdict("hi", "original",
                    VerifyResult(ok=False, reasons=["x"], hard_fail=False))
        self.assertEqual(v.severity, 1)

    def test_severity_hard_fail(self):
        v = Verdict("hi", "original",
                    VerifyResult(ok=False, reasons=["x"], hard_fail=True))
        self.assertEqual(v.severity, 2)

    def test_reason_count(self):
        v = Verdict("hi", "original",
                    VerifyResult(ok=False, reasons=["a", "b", "c"], hard_fail=True))
        self.assertEqual(v.reason_count, 3)

    def test_reason_class_propagates(self):
        v = Verdict(
            "hi", "regen",
            VerifyResult(ok=False, reasons=["semantic_verifier_fail:x"], hard_fail=True),
        )
        self.assertEqual(v.reason_class, "llm_semantic")


# ============================================================
# VerdictLedger.best() 排序优先级
# ============================================================
class TestLedgerBest(unittest.TestCase):
    def test_clean_beats_failed(self):
        led = VerdictLedger()
        led.add(Verdict("bad", "original",
                        VerifyResult(ok=False, reasons=["x"], hard_fail=True)))
        led.add(Verdict("good", "regen", VerifyResult(ok=True)))
        # severity 0 vs 2 → clean 胜
        best = led.best()
        self.assertEqual(best.answer, "good")
        self.assertEqual(best.source_stage, "regen")

    def test_fewer_reasons_wins_on_same_severity(self):
        led = VerdictLedger()
        led.add(Verdict("a", "original",
                        VerifyResult(ok=False, reasons=["x", "y"], hard_fail=True)))
        led.add(Verdict("b", "regen",
                        VerifyResult(ok=False, reasons=["x"], hard_fail=True)))
        # 同 severity，reason_count 1 < 2 → "b" 胜
        self.assertEqual(led.best().answer, "b")

    def test_earlier_stage_wins_on_tie(self):
        """Turn 5 复盘：original 和 regen 都 hard_fail，各 1 reason。
        stage_order: original (0) < regen (2) → original 胜。"""
        led = VerdictLedger()
        led.add(Verdict("55 days", "original",
                        VerifyResult(ok=False, reasons=["date_math_target_date_mismatch"],
                                     hard_fail=True)))
        led.add(Verdict("my birthday way later", "regen",
                        VerifyResult(ok=False,
                                     reasons=["semantic_verifier_fail:pronoun_referent_mismatch"],
                                     hard_fail=True)))
        best = led.best()
        self.assertEqual(best.source_stage, "original")
        self.assertEqual(best.answer, "55 days")

    def test_empty_ledger_best_is_none(self):
        self.assertIsNone(VerdictLedger().best())


# ============================================================
# Disagreement 检测
# ============================================================
class TestLedgerDisagreement(unittest.TestCase):
    def test_regex_vs_llm_is_disagreement(self):
        led = VerdictLedger()
        led.add(Verdict("a", "original",
                        VerifyResult(ok=False, reasons=["date_math_target_date_mismatch"],
                                     hard_fail=True)))
        led.add(Verdict("b", "regen",
                        VerifyResult(ok=False,
                                     reasons=["semantic_verifier_fail:x"],
                                     hard_fail=True)))
        self.assertTrue(led.has_disagreement())

    def test_same_class_is_not_disagreement(self):
        led = VerdictLedger()
        led.add(Verdict("a", "original",
                        VerifyResult(ok=False, reasons=["regex_a"], hard_fail=True)))
        led.add(Verdict("b", "regen",
                        VerifyResult(ok=False, reasons=["regex_b"], hard_fail=True)))
        self.assertFalse(led.has_disagreement())

    def test_single_failed_is_not_disagreement(self):
        # 只 1 个 failed，不构成 disagreement
        led = VerdictLedger()
        led.add(Verdict("a", "original",
                        VerifyResult(ok=False, reasons=["regex_a"], hard_fail=True)))
        led.add(Verdict("b", "regen", VerifyResult(ok=True)))
        self.assertFalse(led.has_disagreement())


# ============================================================
# 集成：safe_fallback 通过 ledger.best() 释放
# ============================================================
class TestSafeFallbackUsesLedger(unittest.TestCase):
    """验证 safe_fallback_for_hard_verifier_failure 接通 ledger 路径。
    用最小 stub agent（不载模型），只暴露 verdict_ledger / history_manager。"""

    def _make_stub_agent(self, ledger=None):
        from types import SimpleNamespace
        hm = SimpleNamespace(current_turn=None, history=[], user_name="Rosm")
        return SimpleNamespace(
            verdict_ledger=ledger or VerdictLedger(),
            history_manager=hm,
            turn_evidence=[],
            last_memory=SimpleNamespace(observation=""),
            active_date_binding=None,
            # 给 _self_validate_date_calculation 用的 stub
            _extract_month_day_from_memory=lambda obs: None,
        )

    def test_turn5_scenario_releases_original(self):
        """复现 Turn 5：original (regex fail) + regen (llm fail) →
        fallback 应释放 original 的 "55 days"。"""
        from eva_verifier_logic import safe_fallback_for_hard_verifier_failure
        led = VerdictLedger()
        original_result = VerifyResult(
            ok=False,
            reasons=["date_math_target_date_mismatch"],
            hard_fail=True,
        )
        regen_result = VerifyResult(
            ok=False,
            reasons=["semantic_verifier_fail:pronoun_referent_mismatch"],
            hard_fail=True,
        )
        led.add(Verdict("55 days until my birthday", "original", original_result))
        led.add(Verdict("my birthday is way later", "regen", regen_result))
        agent = self._make_stub_agent(ledger=led)
        # caller 不传 phase2_answer，让 ledger 决定
        out = safe_fallback_for_hard_verifier_failure(
            agent, regen_result, latest_user_text="how many days until your birthday?",
        )
        # 期望释放 original 的 "55 days"（regen 的 LLM-judge reason 走 P3 policy）
        # P3 policy 是 "RELEASE answer"，best() 选了 original → 应释放 original
        self.assertIn("55 days", out, f"unexpected fallback output: {out!r}")

    def test_legacy_phase2_answer_param_still_works(self):
        """Backward-compat: 当 ledger 空（test stub 等场景），caller 传
        phase2_answer= 仍能 release。"""
        from eva_verifier_logic import safe_fallback_for_hard_verifier_failure
        agent = self._make_stub_agent(ledger=VerdictLedger())  # 空
        out = safe_fallback_for_hard_verifier_failure(
            agent,
            VerifyResult(ok=False,
                         reasons=["semantic_verifier_fail:x"],
                         hard_fail=True),
            latest_user_text="x",
            phase2_answer="legacy answer",
        )
        self.assertEqual(out, "legacy answer")

    def test_canned_when_dominant_not_semantic(self):
        """非 semantic 类 reason 时仍走 canned，不释放 ledger 中的 candidate。"""
        from eva_verifier_logic import safe_fallback_for_hard_verifier_failure
        led = VerdictLedger()
        result = VerifyResult(
            ok=False,
            reasons=["missing_web_evidence_for_external_or_current_request"],
            hard_fail=True,
        )
        led.add(Verdict("fabricated answer", "original", result))
        agent = self._make_stub_agent(ledger=led)
        out = safe_fallback_for_hard_verifier_failure(
            agent, result, latest_user_text="who composed X?",
        )
        # 不应该原样释放（避免幻觉），走 canned
        self.assertNotEqual(out, "fabricated answer")


if __name__ == "__main__":
    unittest.main()
