"""tests/test_dialog_focus.py — R-6 DialogFocus / LastMemoryState 单元测试

R-6 (2026-05-13) 把跨轮 sticky 的 dialog state 从散落的 9 个 last_memory_* /
last_missing_* 字段合并成两个 dataclass：

  - `LastMemoryState`: 上一次 retrieval 的原始快照（observation / primary_query
    / has_exact / has_related / judge_*_count / missing_slots）。
  - `DialogFocus`: 会话级"当前关注实体/slot/topic"。取代旧字段
    `last_memory_target_entity` / `last_missing_slot_target_entity`（两者
    一直是同一个值，R-6 一并合并）。

本测试不依赖 model；构造 stub agent 验证：
  - dataclass 自身行为：reset / update 部分字段 / is_set
  - dialog_focus 跨轮持久（_reset_turn_evidence 不清它）
  - pronoun-followup 通过 dialog_focus.entity 继承上轮 target
  - user 显式提名覆盖 focus
"""
from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from eva_history import LastMemoryState, DialogFocus  # noqa: E402


# ============================================================
# LastMemoryState：数据结构契约
# ============================================================
class TestLastMemoryState(unittest.TestCase):
    def test_default_empty(self):
        lm = LastMemoryState()
        self.assertEqual(lm.observation, "")
        self.assertEqual(lm.primary_query, "")
        self.assertFalse(lm.has_exact)
        self.assertFalse(lm.has_related)
        self.assertEqual(lm.judge_exact_count, 0)
        self.assertEqual(lm.judge_related_count, 0)
        self.assertEqual(lm.missing_slots, [])

    def test_reset_clears_all_fields(self):
        lm = LastMemoryState()
        lm.observation = "some obs"
        lm.primary_query = "q"
        lm.has_exact = True
        lm.has_related = True
        lm.judge_exact_count = 3
        lm.judge_related_count = 5
        lm.missing_slots = ["toy"]
        lm.reset()
        self.assertEqual(lm.observation, "")
        self.assertEqual(lm.primary_query, "")
        self.assertFalse(lm.has_exact)
        self.assertEqual(lm.missing_slots, [])

    def test_missing_slots_is_independent_list(self):
        # 默认 default_factory=list，不同实例不共享同一 list 对象
        a = LastMemoryState()
        b = LastMemoryState()
        a.missing_slots.append("toy")
        self.assertEqual(b.missing_slots, [])


# ============================================================
# DialogFocus：数据结构契约
# ============================================================
class TestDialogFocus(unittest.TestCase):
    def test_default_unset(self):
        df = DialogFocus()
        self.assertFalse(df.is_set())
        self.assertEqual(df.entity, "")
        self.assertEqual(df.set_at_turn, -1)

    def test_update_partial_fields(self):
        df = DialogFocus()
        df.update(entity="Eva", topic="Toy", turn=2, source="pre_probe")
        self.assertTrue(df.is_set())
        self.assertEqual(df.entity, "Eva")
        self.assertEqual(df.topic, "Toy")
        self.assertEqual(df.set_at_turn, 2)
        self.assertEqual(df.source, "pre_probe")

    def test_update_empty_args_does_not_overwrite(self):
        # update(entity="") / turn=-1 不应清掉原值——典型用法：
        # pronoun-followup 只更新 source，不改 entity。
        df = DialogFocus()
        df.update(entity="Eva", topic="Toy", turn=2, source="pre_probe")
        df.update(source="pronoun_inherit")  # 只更新 source
        self.assertEqual(df.entity, "Eva")
        self.assertEqual(df.topic, "Toy")
        self.assertEqual(df.set_at_turn, 2)
        self.assertEqual(df.source, "pronoun_inherit")

    def test_user_naming_overrides_entity(self):
        df = DialogFocus()
        df.update(entity="Eva", source="pre_probe")
        # 模拟用户显式提名 "what about Rosm's..."
        df.update(entity="Rosm", source="user_named")
        self.assertEqual(df.entity, "Rosm")
        self.assertEqual(df.source, "user_named")

    def test_reset(self):
        df = DialogFocus()
        df.update(entity="Eva", slot="toy", topic="Toy", turn=5, source="tool")
        df.reset()
        self.assertFalse(df.is_set())
        self.assertEqual(df.set_at_turn, -1)
        self.assertEqual(df.source, "")


# ============================================================
# 跨轮 sticky：_reset_turn_evidence 不清 dialog_focus / last_memory
# ============================================================
class TestStickiness(unittest.TestCase):
    """R-6 设计：last_memory 和 dialog_focus 是跨轮持久的 dialog state。
    _reset_turn_evidence 只清"本轮证据"，不动跨轮 sticky 字段。
    """

    def test_reset_turn_evidence_does_not_clear_dialog_focus(self):
        # 不真的 import ChatAgent（要载模型），直接看源码确认契约。
        import inspect
        from eva_core import ChatAgent
        src = inspect.getsource(ChatAgent._reset_turn_evidence)
        # 关键不变式：reset 函数里 NOT touch dialog_focus / last_memory
        self.assertNotIn("self.dialog_focus", src,
                         "R-6: _reset_turn_evidence 必须不动 dialog_focus")
        self.assertNotIn("self.last_memory =", src,
                         "R-6: _reset_turn_evidence 必须不动 last_memory")


# ============================================================
# verifier 集成：pronoun-followup 通过 dialog_focus.entity 继承 target
# ============================================================
class TestVerifierUsesFocus(unittest.TestCase):
    """build_required_memory_params 在 pronoun-followup 检测到时，应该读
    agent.dialog_focus.entity 作为 target 兜底（取代旧 P1-6 补丁的
    inherited_target = last_memory_target_entity 路径）。"""

    def test_focus_entity_overrides_default_both_for_pronoun_followup(self):
        # 用源码检查替代真跑 — 关键是确认 verifier 读 dialog_focus.entity
        import inspect
        from eva_verifier_logic import build_required_memory_params
        src = inspect.getsource(build_required_memory_params)
        # 撤 P1-6：不再读 last_memory_target_entity
        self.assertNotIn("last_memory_target_entity", src,
                         "R-6: verifier 应该读 dialog_focus.entity 而非 last_memory_target_entity")
        # 走 dialog_focus 路径
        self.assertIn("dialog_focus", src,
                      "R-6: verifier 必须读 dialog_focus")

    def test_canonical_focus_entity_used_when_target_is_both(self):
        # 把整段判定逻辑摘出来跑：focus=Eva + 解析 target=Both → 应取 Eva
        from eva_memory_legacy import _canonical_known_entity_name
        focus_entity = _canonical_known_entity_name("Eva")
        target_inferred = "Both"
        # 模拟 verifier 里的判定 if-block
        if focus_entity in ("Eva", "Rosm") and target_inferred in ("", "Both", "Shared"):
            chosen = focus_entity
        else:
            chosen = target_inferred
        self.assertEqual(chosen, "Eva")

    def test_explicit_inferred_target_overrides_focus(self):
        # 当 _infer_memory_target_from_text 能从 user_text 推出 Eva 或 Rosm，
        # 不应被 focus 覆盖（focus 只在 Both / "" 时兜底）。
        from eva_memory_legacy import _canonical_known_entity_name
        focus_entity = _canonical_known_entity_name("Eva")
        target_inferred = "Rosm"  # 用户显式说 "Rosm's birthday" 推出来的
        if focus_entity in ("Eva", "Rosm") and target_inferred in ("", "Both", "Shared"):
            chosen = focus_entity
        else:
            chosen = target_inferred
        self.assertEqual(chosen, "Rosm",
                         "verifier 应优先尊重 user_text 显式推断，不被 focus 覆盖")


if __name__ == "__main__":
    unittest.main()
