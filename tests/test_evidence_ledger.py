"""tests/test_evidence_ledger.py — R-4 Evidence Ledger 单元测试

覆盖 R-4 (2026-05-13) 引入的 `TurnEvidenceLedger` 类、5 个新写入点的契约、
以及 3 个读取口切到 ledger 后的行为。

不依赖 model / DeepSeek，纯本地：构造 stub agent + 直接调用 _add_turn_evidence
等 helper，验证 ledger 的写入 + verifier 读取一致性。
"""
from __future__ import annotations

import os
import re
import sys
import unittest
from types import SimpleNamespace

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from eva_history import TurnEvidence, TurnEvidenceLedger
from eva_verifier_logic import (
    current_turn_has_memorysearch_evidence,
    current_turn_has_remember_evidence,
    current_turn_has_forget_evidence,
)


# ============================================================
# Ledger class — pure data structure
# ============================================================
class TestTurnEvidenceLedgerClass(unittest.TestCase):
    """TurnEvidenceLedger 自身行为：list-compat + query API。"""

    def test_list_like_compat(self):
        led = TurnEvidenceLedger()
        self.assertEqual(len(led), 0)
        self.assertFalse(bool(led))
        led.append(TurnEvidence(source="memory"))
        self.assertEqual(len(led), 1)
        self.assertTrue(bool(led))
        self.assertEqual([e.source for e in led], ["memory"])
        self.assertEqual(led[0].source, "memory")
        led.clear()
        self.assertEqual(len(led), 0)

    def test_covers_subject_slot(self):
        led = TurnEvidenceLedger()
        led.append(TurnEvidence(source="memory", subject="Eva",
                                slot="toy", value="bunny", confidence="exact"))
        # 完全匹配
        self.assertTrue(led.covers(subject="Eva", slot="toy", min_tier="exact"))
        # subject 错
        self.assertFalse(led.covers(subject="Rosm", slot="toy"))
        # min_tier 太严：只有 related 时 exact-only 查询 miss
        led.clear()
        led.append(TurnEvidence(source="memory", subject="Eva", slot="toy",
                                confidence="related"))
        self.assertTrue(led.covers(subject="Eva", slot="toy", min_tier="related"))
        self.assertFalse(led.covers(subject="Eva", slot="toy", min_tier="exact"))

    def test_covers_topic(self):
        led = TurnEvidenceLedger()
        led.append(TurnEvidence(source="topic_dict", subject="Eva",
                                topic="Toy", confidence="topic"))
        self.assertTrue(led.covers(subject="Eva", topic="Toy", min_tier="topic"))
        # topic 命中默认 min_tier="topic"
        self.assertTrue(led.covers(topic="Toy"))
        # 但不构成 "related" 强度的 grounding
        self.assertFalse(led.covers(topic="Toy", min_tier="related"))

    def test_best_for_picks_highest_tier(self):
        led = TurnEvidenceLedger()
        led.append(TurnEvidence(source="memory", subject="Eva", slot="toy",
                                value="related-value", confidence="related"))
        led.append(TurnEvidence(source="memory", subject="Eva", slot="toy",
                                value="bunny", confidence="exact"))
        best = led.best_for("Eva", "toy")
        self.assertIsNotNone(best)
        self.assertEqual(best.value, "bunny")
        self.assertEqual(best.confidence, "exact")

    def test_by_source_filter(self):
        led = TurnEvidenceLedger()
        led.append(TurnEvidence(source="memory"))
        led.append(TurnEvidence(source="topic_dict"))
        led.append(TurnEvidence(source="notes_write"))
        self.assertEqual(len(led.by_source("memory")), 1)
        self.assertEqual(len(led.by_source("memory", "topic_dict")), 2)
        self.assertTrue(led.has_source("notes_write"))
        self.assertFalse(led.has_source("web"))


# ============================================================
# Reader contracts — verifier 三个 has_*_evidence 读取口
# ============================================================
def _stub_agent(ledger_items=None):
    """构造最小 stub agent，只带 turn_evidence ledger + history_manager 桩。"""
    led = TurnEvidenceLedger()
    for ev in ledger_items or []:
        led.append(ev)
    hm = SimpleNamespace(current_turn=None, history=[])
    return SimpleNamespace(history_manager=hm, turn_evidence=led)


class TestReaderContracts(unittest.TestCase):

    def test_memorysearch_reader_picks_memory_source(self):
        # source="memory" 是 MemorySearch tool 或 PRE PROBE 写的标记
        agent = _stub_agent([TurnEvidence(source="memory", subject="Eva",
                                          slot="toy", confidence="exact")])
        self.assertTrue(current_turn_has_memorysearch_evidence(agent))

    def test_memorysearch_reader_picks_topic_dict_source(self):
        # R-4 关键：topic_dict 命中也算 memory-grounded 的一种
        agent = _stub_agent([TurnEvidence(source="topic_dict", topic="Toy",
                                          confidence="topic")])
        self.assertTrue(current_turn_has_memorysearch_evidence(agent))

    def test_memorysearch_reader_false_when_only_unrelated(self):
        agent = _stub_agent([TurnEvidence(source="web")])
        self.assertFalse(current_turn_has_memorysearch_evidence(agent))

    def test_remember_reader_uses_notes_write(self):
        agent = _stub_agent([TurnEvidence(source="notes_write",
                                          record_ref="abc12345")])
        self.assertTrue(current_turn_has_remember_evidence(agent))
        # source="memory" 不算 remember
        agent = _stub_agent([TurnEvidence(source="memory")])
        self.assertFalse(current_turn_has_remember_evidence(agent))

    def test_forget_reader_uses_notes_delete(self):
        agent = _stub_agent([TurnEvidence(source="notes_delete",
                                          record_ref="abc12345")])
        self.assertTrue(current_turn_has_forget_evidence(agent))
        # notes_write != notes_delete
        agent = _stub_agent([TurnEvidence(source="notes_write")])
        self.assertFalse(current_turn_has_forget_evidence(agent))


# ============================================================
# Tool dispatch writes — RememberThis/ForgetMemory 落 ledger
# ============================================================
#
# 这两个测试不跑完整 ChatAgent（要载模型），改成直接验证 _add_turn_evidence
# 在正确的字段上落 ledger。集成层面的"工具调用 → 写 ledger"链路靠手动验证
# step_once 的修改点（见 eva_core.py 2440 附近）。
# ============================================================
class TestToolDispatchSchema(unittest.TestCase):
    """验证 RememberThis 成功后写入 ledger 的 evidence shape 符合 reader 契约。"""

    def test_notes_write_shape_matches_reader(self):
        led = TurnEvidenceLedger()
        # 模拟 step_once 在 [REMEMBERED] 分支会做的写入
        nid = "7a4a68da"
        led.append(TurnEvidence(
            source="notes_write",
            subject="Rosm",
            slot=None,
            value="Master has a meeting next Monday.",
            confidence="exact",
            raw_text=f"[REMEMBERED] Stored as Note #{nid} (entity=Rosm, topic=Meeting).",
            topic="Meeting",
            record_ref=nid,
            meta={"channel": "tool"},
        ))
        agent = SimpleNamespace(turn_evidence=led, history_manager=SimpleNamespace(current_turn=None))
        self.assertTrue(current_turn_has_remember_evidence(agent))
        # 同时也算 memorysearch 不通过（notes_write 不是 memory）
        self.assertFalse(current_turn_has_memorysearch_evidence(agent))

    def test_notes_delete_shape_matches_reader(self):
        led = TurnEvidenceLedger()
        nid = "7a4a68da"
        led.append(TurnEvidence(
            source="notes_delete",
            value=nid,
            confidence="exact",
            raw_text=f"[FORGOTTEN] Note #{nid} tombstoned.",
            record_ref=nid,
            meta={"channel": "tool", "reason": "canceled"},
        ))
        agent = SimpleNamespace(turn_evidence=led, history_manager=SimpleNamespace(current_turn=None))
        self.assertTrue(current_turn_has_forget_evidence(agent))


# ============================================================
# Regression scenarios — R-4 撤补丁后必须仍能处理的场景
# ============================================================
class TestR4RegressionScenarios(unittest.TestCase):
    """复现 2026-05-13 测试里被 P1-4 补丁压住的两个场景，
    验证撤补丁后 ledger 单源足以处理。"""

    def test_turn10_pre_probe_exact_covers_memory_check(self):
        """Turn 10 复现：'maybe you should search toy in your memory'

        PRE PROBE 已经命中 Toy topic + 注入 EXACT bunny lore。即便
        _explicit_memory_check_request 仍 True，ledger 里有 source="memory"
        + source="topic_dict" 的证据，current_turn_has_memorysearch_evidence
        应该返回 True，verifier 因此不会再 inject 一次 MemorySearch。
        """
        led = TurnEvidenceLedger()
        # PRE PROBE 走 _record_active_memory_evidence 写的两条
        led.append(TurnEvidence(source="topic_dict", subject="Eva",
                                topic="Toy", confidence="topic",
                                value="Toy"))
        led.append(TurnEvidence(source="memory", subject="Eva",
                                slot="toy", value="cuddly bunny",
                                confidence="exact",
                                raw_text="Eva's favorite toy has always been..."))
        agent = SimpleNamespace(turn_evidence=led,
                                history_manager=SimpleNamespace(current_turn=None))
        self.assertTrue(current_turn_has_memorysearch_evidence(agent))

    def test_turn11_setup_remember_no_slot_no_topic_no_fire(self):
        """Turn 11 复现：'Fine, can you help me to remember something?'

        这是引子句：用户还没说要记什么。PRE PROBE 跑了但找不到 topic/slot，
        ledger 是空的。撤掉 _is_setup_remember_phrasing 后，verifier 的
        check 凭"用户没问 asked_slots"这一条短路退出（详见
        eva_verifier_logic.py R-4 注释处的 if asked_slots 分支）。
        """
        # 测试 extract_memory_slots 对这种句式返回空列表
        from eva_slots import extract_memory_slots
        slots = extract_memory_slots("Fine, can you help me to remember something?")
        self.assertEqual(slots, [],
                         "setup-remember 句应该不抽到任何 MEMORY_SLOT_FIELDS")


if __name__ == "__main__":
    unittest.main()
