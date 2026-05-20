"""
test_session_isolation.py — Offline 验证 Discord 多用户隔离的正确性。

测试目标：eva_discord_sessions.py 里的 snapshot/restore + LRU 契约。
不加载 9B 模型，不连 Discord，纯逻辑测试。

跑法（项目根目录下）：
    D:/Anaconda/envs/py310/python.exe tests/test_session_isolation.py
"""

import os
import sys
import types
import unittest

# ============================================================
# Path + 离线 stub setup（遵循 tests/test_no_elaboration_rule.py 的模式）
# ============================================================
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# eva_config 在 import 时读 torch.backends.cudnn 的两个 flag。Stub 一下。
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

# eva_history 在 import 时 `from transformers import StoppingCriteria`。
# Stub 一个空 class 就行，本测试不实际触发 stopping criteria。
if "transformers" not in sys.modules:
    _tf_stub = types.ModuleType("transformers")

    class _StoppingCriteriaStub:  # noqa: N801 — 名字必须匹配 transformers
        pass

    _tf_stub.StoppingCriteria = _StoppingCriteriaStub
    sys.modules["transformers"] = _tf_stub

# eva_render 在 import 时 `from PIL import Image`。Stub 兜底（CI 环境可能没 Pillow）。
if "PIL" not in sys.modules:
    _pil = types.ModuleType("PIL")

    class _ImageStub:
        @staticmethod
        def open(*a, **kw): return None

    _pil.Image = _ImageStub
    sys.modules["PIL"] = _pil
    sys.modules["PIL.Image"] = _ImageStub  # 有些 `from PIL.Image import ...` 也走这

# ============================================================
# Subject under test
# ============================================================
from eva_discord_sessions import (
    UserSession,
    SessionStore,
    snapshot_agent_state,
    restore_agent_state,
    inject_resumed_turns,
    STICKY_SLOTS,
)


# ============================================================
# StubAgent —— 模拟 ChatAgent 的 5 个 sticky 槽
#
# 我们不需要真的 ChatAgent；snapshot/restore 只读这 5 个属性。restore
# 路径会调 HistoryManager()/LastMemoryState()/DialogFocus() 的构造函数，
# 那是真实的 eva_history dataclasses（轻量，stub 后可正常 import）。
# ============================================================
def _make_stub_agent():
    """构造一个有 5 个 sticky 槽的最小 agent。"""
    from eva_history import HistoryManager, LastMemoryState, DialogFocus
    a = types.SimpleNamespace()
    a.history_manager = HistoryManager()
    a.last_memory = LastMemoryState()
    a.dialog_focus = DialogFocus()
    a._recent_phase2_outputs = []
    a._recent_phase2_modes = []
    return a


def _fake_turn(agent, user_text: str, final_answer: str):
    """假装跑了一轮对话，往 history_manager 写一条 turn 记录。

    模拟 agent.run 的最小副作用：start_turn / add_assistant_step / finish_turn。
    """
    from eva_history import ConversationTurn  # 轻量 dataclass
    agent.history_manager.start_turn(user_text)
    # 最简单的：直接放一个含 answer 块的 assistant step
    agent.history_manager.add_assistant_step(f"<|answer|>{final_answer}<|end_react|>")
    agent.history_manager.finish_turn()


# ============================================================
# Tests
# ============================================================
class SnapshotRestoreRoundtripTest(unittest.TestCase):
    """Group 1: 单纯验证 snapshot/restore 是无损往返。"""

    def test_blank_restore_gives_fresh_objects(self):
        """restore_agent_state(agent, None) 应该把 5 个槽都换成新对象。"""
        agent = _make_stub_agent()
        # 故意污染 agent 的几个字段
        _fake_turn(agent, "hello", "hi back")
        agent.last_memory.observation = "stale obs"
        agent.dialog_focus.entity = "Eva"
        agent._recent_phase2_outputs = ["abc"]

        # 用 None 还原 → 等于建新 session
        restore_agent_state(agent, None)

        self.assertEqual(len(agent.history_manager.history), 0)
        self.assertEqual(agent.last_memory.observation, "")
        self.assertEqual(agent.dialog_focus.entity, "")
        self.assertEqual(agent._recent_phase2_outputs, [])
        self.assertEqual(agent._recent_phase2_modes, [])

    def test_snapshot_then_restore_roundtrip(self):
        """snapshot 出来再 restore 回去，5 个槽内容应一致。"""
        agent = _make_stub_agent()
        _fake_turn(agent, "what's my birthday?", "July 7th, Master.")
        _fake_turn(agent, "and yours?", "I share that day too.")
        agent.last_memory.observation = "EXACT memory hit"
        agent.last_memory.has_exact = True
        agent.dialog_focus.entity = "Rosm"
        agent.dialog_focus.slot = "birthday"
        agent._recent_phase2_outputs = ["o1", "o2"]
        agent._recent_phase2_modes = ["after_memory", "direct"]

        snap = snapshot_agent_state(agent)

        # 把 agent 状态搅烂
        restore_agent_state(agent, None)
        self.assertEqual(len(agent.history_manager.history), 0)

        # 再 restore 回去
        restore_agent_state(agent, snap)

        self.assertEqual(len(agent.history_manager.history), 2)
        self.assertEqual(agent.history_manager.history[0].user_content,
                         "what's my birthday?")
        self.assertTrue(agent.last_memory.has_exact)
        self.assertEqual(agent.last_memory.observation, "EXACT memory hit")
        self.assertEqual(agent.dialog_focus.entity, "Rosm")
        self.assertEqual(agent.dialog_focus.slot, "birthday")
        self.assertEqual(agent._recent_phase2_outputs, ["o1", "o2"])
        self.assertEqual(agent._recent_phase2_modes, ["after_memory", "direct"])


class DeepcopyIsolationTest(unittest.TestCase):
    """Group 2: 验证 snapshot 是深拷贝 —— 之后 mutate agent 不能污染快照。"""

    def test_mutating_agent_history_after_snapshot_doesnt_leak(self):
        """snapshot 后 agent 加新 turn，旧快照的 history 长度不变。"""
        agent = _make_stub_agent()
        _fake_turn(agent, "q1", "a1")
        snap = snapshot_agent_state(agent)

        _fake_turn(agent, "q2", "a2")
        # agent 现在有 2 条，但 snap 应该只有 1 条
        self.assertEqual(len(agent.history_manager.history), 2)
        self.assertEqual(len(snap["history_manager"].history), 1)

    def test_mutating_dialog_focus_after_snapshot_doesnt_leak(self):
        agent = _make_stub_agent()
        agent.dialog_focus.entity = "Eva"
        snap = snapshot_agent_state(agent)

        agent.dialog_focus.entity = "Rosm"
        # snapshot 应该还是 "Eva"
        self.assertEqual(snap["dialog_focus"].entity, "Eva")

    def test_recent_outputs_list_is_independent(self):
        """list copy 必须是新 list 对象，不能 share reference。"""
        agent = _make_stub_agent()
        agent._recent_phase2_outputs = ["a"]
        snap = snapshot_agent_state(agent)

        agent._recent_phase2_outputs.append("b")
        self.assertEqual(snap["recent_phase2_outputs"], ["a"])
        self.assertEqual(agent._recent_phase2_outputs, ["a", "b"])


class TwoUsersIndependentTest(unittest.TestCase):
    """Group 3: 模拟两个 Discord user 交替说话，互不串台。"""

    def test_user_a_history_doesnt_leak_into_user_b(self):
        agent = _make_stub_agent()  # 共享的 agent
        store = SessionStore()

        # ---- User A 来了一句 ----
        sess_a = store.get_or_create(111, "Rosm")
        restore_agent_state(agent, sess_a.state)
        _fake_turn(agent, "remember my birthday is Aug 1", "noted, Master")
        sess_a.state = snapshot_agent_state(agent)
        sess_a.turn_count += 1

        # ---- User B 进来：应该看不到 A 的对话 ----
        sess_b = store.get_or_create(222, "Bob")
        restore_agent_state(agent, sess_b.state)
        self.assertEqual(len(agent.history_manager.history), 0,
                         "User B 不应该看到 User A 的 history")
        _fake_turn(agent, "what time is it?", "I'm not sure")
        sess_b.state = snapshot_agent_state(agent)
        sess_b.turn_count += 1

        # ---- User A 回来：应该看到自己第 1 轮 ----
        restore_agent_state(agent, sess_a.state)
        self.assertEqual(len(agent.history_manager.history), 1)
        self.assertEqual(agent.history_manager.history[0].user_content,
                         "remember my birthday is Aug 1")

        # ---- User B 回来：应该看到自己的第 1 轮 ----
        restore_agent_state(agent, sess_b.state)
        self.assertEqual(len(agent.history_manager.history), 1)
        self.assertEqual(agent.history_manager.history[0].user_content,
                         "what time is it?")

    def test_same_user_two_turns_accumulates(self):
        agent = _make_stub_agent()
        store = SessionStore()

        sess = store.get_or_create(111, "Rosm")
        restore_agent_state(agent, sess.state)
        _fake_turn(agent, "hi", "hello")
        sess.state = snapshot_agent_state(agent)

        # 第二轮：history 里应该已经有 "hi"
        restore_agent_state(agent, sess.state)
        self.assertEqual(len(agent.history_manager.history), 1)
        _fake_turn(agent, "remember me", "always, Master")
        sess.state = snapshot_agent_state(agent)

        # 第三轮：应该有 2 条
        restore_agent_state(agent, sess.state)
        self.assertEqual(len(agent.history_manager.history), 2)

    def test_dialog_focus_sticky_per_user(self):
        """dialog_focus 是跨轮 sticky，但只 sticky 到该用户自己。"""
        agent = _make_stub_agent()
        store = SessionStore()

        # User A 把 focus 设到 Rosm
        sess_a = store.get_or_create(111, "Rosm")
        restore_agent_state(agent, sess_a.state)
        agent.dialog_focus.entity = "Rosm"
        agent.dialog_focus.slot = "birthday"
        sess_a.state = snapshot_agent_state(agent)

        # User B 来：应该看到空的 focus
        sess_b = store.get_or_create(222, "Bob")
        restore_agent_state(agent, sess_b.state)
        self.assertEqual(agent.dialog_focus.entity, "")
        self.assertEqual(agent.dialog_focus.slot, "")

        # User A 回来：focus 还在
        restore_agent_state(agent, sess_a.state)
        self.assertEqual(agent.dialog_focus.entity, "Rosm")
        self.assertEqual(agent.dialog_focus.slot, "birthday")


class SessionStoreLRUTest(unittest.TestCase):
    """Group 4: LRU 行为。"""

    def test_reset_removes_user(self):
        store = SessionStore()
        store.get_or_create(111, "Rosm")
        self.assertIn(111, store)
        self.assertTrue(store.reset(111))
        self.assertNotIn(111, store)
        # 第二次 reset 应该返回 False（已经没了）
        self.assertFalse(store.reset(111))

    def test_user_rename_updates_session(self):
        store = SessionStore()
        sess1 = store.get_or_create(111, "Rosm")
        sess2 = store.get_or_create(111, "Master Rosm")
        self.assertIs(sess1, sess2)
        self.assertEqual(sess2.user_name, "Master Rosm")

    def test_lru_eviction_kicks_out_oldest(self):
        """容量满了之后，最久没访问的应该被淘汰。"""
        store = SessionStore(max_sessions=3)
        store.get_or_create(1, "u1")
        store.get_or_create(2, "u2")
        store.get_or_create(3, "u3")
        self.assertEqual(len(store), 3)
        self.assertEqual(store.evictions, 0)

        # 触发 LRU：user 4 进来 → user 1 应该被淘汰
        store.get_or_create(4, "u4")
        self.assertEqual(len(store), 3)
        self.assertEqual(store.evictions, 1)
        self.assertNotIn(1, store)
        self.assertIn(2, store)
        self.assertIn(3, store)
        self.assertIn(4, store)

    def test_touch_protects_against_eviction(self):
        """touch 过的用户应该被移到队尾，避免被淘汰。"""
        store = SessionStore(max_sessions=3)
        store.get_or_create(1, "u1")
        store.get_or_create(2, "u2")
        store.get_or_create(3, "u3")

        # 手动 touch 用户 1，让它"年轻"
        store.touch(1)

        # 再来一个新用户 → 现在最老的是 2，应该淘汰 2 而不是 1
        store.get_or_create(4, "u4")
        self.assertIn(1, store, "touched 过的 user 1 不该被淘汰")
        self.assertNotIn(2, store, "user 2 应该是最老的，被淘汰")

    def test_get_or_create_touches_existing(self):
        """重复 get_or_create 同一个用户也算 touch。"""
        store = SessionStore(max_sessions=3)
        store.get_or_create(1, "u1")
        store.get_or_create(2, "u2")
        store.get_or_create(3, "u3")

        # 重新 get user 1 → 把它移到队尾
        store.get_or_create(1, "u1")

        store.get_or_create(4, "u4")
        self.assertIn(1, store)
        self.assertNotIn(2, store)


class CrashSafetyTest(unittest.TestCase):
    """Group 5: 推理崩溃中途，snapshot 仍能拿到部分状态。

    模拟 eva_discord.py 里 try/finally 包裹 snapshot 的行为：
    哪怕 agent.run 抛异常，已经发生的 history mutation 也能被存下来。
    """

    def test_partial_turn_still_snapshots(self):
        agent = _make_stub_agent()

        # 模拟 agent.run 走到一半：start_turn 已发生，finish_turn 还没
        agent.history_manager.start_turn("user input that will crash")
        # 模拟某种 mutation：last_memory 已经被写
        agent.last_memory.primary_query = "what crashed"

        # 模拟 finally 里 snapshot —— 不应该抛异常
        snap = snapshot_agent_state(agent)
        self.assertIsNotNone(snap["history_manager"].current_turn,
                             "current_turn 应该被快照保留")
        self.assertEqual(snap["last_memory"].primary_query, "what crashed")


class StickySlotsContractTest(unittest.TestCase):
    """Group 6: 元测试 —— 确保白名单 + UserSession 的字段集没漂移。"""

    def test_sticky_slots_list_matches_snapshot_keys(self):
        """STICKY_SLOTS 是文档/契约，snapshot 实际写的 key 必须对应。"""
        agent = _make_stub_agent()
        snap = snapshot_agent_state(agent)
        # snapshot dict 的 key 是去掉前导下划线的版本（_recent_phase2_outputs → recent_phase2_outputs）
        expected_keys = {
            "history_manager", "last_memory", "dialog_focus",
            "recent_phase2_outputs", "recent_phase2_modes",
        }
        self.assertEqual(set(snap.keys()), expected_keys)
        # 同时验证 STICKY_SLOTS 的长度对得上
        self.assertEqual(len(STICKY_SLOTS), len(expected_keys))


class InjectResumedTurnsTest(unittest.TestCase):
    """Group 7: bot 重启后从 Discord 拉历史还原对话上下文。"""

    def test_inject_within_keep_full_turns(self):
        """注入数量 ≤ KEEP_FULL_TURNS 时，全部保留在 history，compressed_kv 空。"""
        agent = _make_stub_agent()
        turns = [
            ("hi Eva", "Hi Master, how can I help today?"),
            ("what's the weather?", "Sunny and 22°C in Sydney."),
        ]
        count = inject_resumed_turns(agent, turns)
        self.assertEqual(count, 2)
        self.assertEqual(len(agent.history_manager.history), 2)
        self.assertEqual(len(agent.history_manager.compressed_kv), 0)
        # 验证 user_content 正确
        self.assertEqual(agent.history_manager.history[0].user_content, "hi Eva")
        # 验证 final answer 提取正确
        self.assertEqual(
            agent.history_manager.history[0].get_final_answer(),
            "Hi Master, how can I help today?",
        )

    def test_inject_exceeds_keep_full_compresses_oldest(self):
        """注入 4 轮（>KEEP_FULL_TURNS=3），最老的进 compressed_kv，3 轮留在 history。"""
        agent = _make_stub_agent()
        turns = [
            ("q1", "a1"),  # 应被压缩
            ("q2", "a2"),
            ("q3", "a3"),
            ("q4", "a4"),
        ]
        count = inject_resumed_turns(agent, turns)
        self.assertEqual(count, 4)
        self.assertEqual(len(agent.history_manager.history), 3)
        self.assertEqual(len(agent.history_manager.compressed_kv), 1)
        # 最老的 q1/a1 进了 compressed_kv 的 summary text
        self.assertIn("q1", agent.history_manager.compressed_kv[0])
        self.assertIn("a1", agent.history_manager.compressed_kv[0])

    def test_inject_skips_empty_pairs(self):
        """任一端空字符串的对应该被跳过，不写入 history。"""
        agent = _make_stub_agent()
        turns = [
            ("real question", "real answer"),
            ("", "no user text"),         # skip
            ("no answer", ""),             # skip
            ("another q", "another a"),
        ]
        count = inject_resumed_turns(agent, turns)
        self.assertEqual(count, 2)
        self.assertEqual(len(agent.history_manager.history), 2)

    def test_inject_into_empty_agent_then_swap_roundtrip(self):
        """注入后做 snapshot/restore 应保留历史（验证 deepcopy 不丢失）。"""
        agent = _make_stub_agent()
        turns = [("hello", "hi back"), ("what now", "let's chat")]
        inject_resumed_turns(agent, turns)

        snap = snapshot_agent_state(agent)
        restore_agent_state(agent, None)  # wipe
        self.assertEqual(len(agent.history_manager.history), 0)
        restore_agent_state(agent, snap)
        self.assertEqual(len(agent.history_manager.history), 2)
        self.assertEqual(
            agent.history_manager.history[1].get_final_answer(),
            "let's chat",
        )

    def test_user_name_in_summary_text(self):
        """compressed_kv 里的 summary 应该带正确的 user_name 前缀。"""
        agent = _make_stub_agent()
        agent.history_manager.set_user_name("Rosm")
        turns = [(f"q{i}", f"a{i}") for i in range(5)]  # 5 turns，2 个会被压缩
        inject_resumed_turns(agent, turns)
        self.assertEqual(len(agent.history_manager.compressed_kv), 2)
        # summary 文本格式：'Rosm: "q0" -> Eva: "a0"'
        self.assertTrue(
            agent.history_manager.compressed_kv[0].startswith('Rosm:'),
            f"got: {agent.history_manager.compressed_kv[0]!r}"
        )
        self.assertIn('Eva:', agent.history_manager.compressed_kv[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
