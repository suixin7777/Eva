"""tests/test_forget_query.py — R-2 ForgetMemory query-first dispatch

R-2 (2026-05-13) 把 ForgetMemory 从"只接受 record_id"扩成 query-first 调度。
本测试用 fake encoder + 真 NotesStore 验证：

  - record_id 路径仍然能用（向后兼容）
  - query 路径：唯一候选 → 直接 tombstone
  - query 路径：多候选 → [FORGET DISAMBIGUATE]
  - query 路径：0 候选 → [FORGET NOT FOUND]
  - query + topic 过滤
  - 低置信度 top1 → 拒删（避免误删无关 note）
  - 错误参数（无 query 无 record_id）→ 明确报错
"""
from __future__ import annotations

import hashlib
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from Memory_maker.notes_runtime import (
    NotesStore,
    execute_forget_memory,
    FORGET_MATCH_MIN_SCORE,
    FORGET_UNIQUE_GAP,
)


# ============================================================
# Encoder：对相关词组聚簇，无关词组完全正交。
# ============================================================
# 真 mpnet 在归一化后 cosine 大致 [0.3, 0.8]。我们让"meeting/会议/工作"组
# 聚到向量 e0 附近、"toy/bunny"组聚到 e1 附近、"birthday/cake"组聚到 e2
# 附近。这样同主题查询能命中阈值，跨主题查询低于阈值。
DIM = 32
GROUPS = {
    # 关键词 → 主向量 index
    "meeting": 0, "meet": 0, "monday": 0, "work": 0, "deadline": 0,
    "toy": 1, "bunny": 1, "plush": 1, "stuffed": 1,
    "birthday": 2, "cake": 2, "candles": 2, "party": 2,
}


def _basis(i: int) -> np.ndarray:
    v = np.zeros(DIM, dtype="float32")
    v[i] = 1.0
    return v


def fake_encoder(texts):
    out = []
    for t in texts:
        tl = t.lower()
        # 哪些 group 的关键词出现在 text 里？合并加权然后归一化。
        vec = np.zeros(DIM, dtype="float32")
        hits = 0
        for word, gi in GROUPS.items():
            if word in tl:
                vec += _basis(gi)
                hits += 1
        if hits == 0:
            # 落在一个不可达的高维分量上，与所有 group 正交
            h = hashlib.sha256(t.encode("utf-8")).digest()
            seed = int.from_bytes(h[:8], "big") & 0x7FFFFFFF
            rng = np.random.RandomState(seed)
            v = rng.randn(DIM - len(GROUPS)).astype("float32")
            vec[len(GROUPS):] = v
        n = np.linalg.norm(vec)
        out.append(vec / (n if n > 0 else 1.0))
    return np.stack(out).astype("float32")


# ============================================================
# Common fixture
# ============================================================
class _ForgetTestBase(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="evaforget_"))
        self.store = NotesStore(
            root=self.tmpdir, encoder=fake_encoder, dim=DIM,
            session_id="t-forget",
        )

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _add(self, content, topic="Meeting", entity="Rosm"):
        return self.store.add(
            vector_text=content, content=content,
            entity=entity, topic=topic, keywords="",
        )


# ============================================================
# 路径 A：record_id 老路径仍可用
# ============================================================
class TestForgetByRecordId(_ForgetTestBase):
    def test_record_id_path_tombstones_note(self):
        nid = self._add("Master has a meeting next Monday")
        obs = execute_forget_memory(self.store, {"record_id": nid, "reason": "canceled"})
        self.assertIn("[FORGOTTEN]", obs)
        self.assertIn(nid, obs)
        # 已 tombstone，再 search 不应返回
        self.assertEqual(self.store.search("meeting"), [])

    def test_record_id_malformed_returns_error(self):
        obs = execute_forget_memory(self.store, {"record_id": "BADBAD"})
        self.assertIn("[FORGET ERROR]", obs)
        self.assertIn("8-char hex", obs)

    def test_record_id_not_found_suggests_query(self):
        obs = execute_forget_memory(self.store, {"record_id": "deadbeef"})
        self.assertIn("[FORGET ERROR]", obs)
        # R-2：错误时建议改走 query 路径
        self.assertIn("query=", obs)


# ============================================================
# 路径 B：query 化匹配
# ============================================================
class TestForgetByQuery(_ForgetTestBase):
    def test_query_unique_match_tombstones(self):
        nid = self._add("Master has a meeting next Monday")
        # 同主题 query：encoder 把两者都聚到 e0，cosine 接近 1.0
        obs = execute_forget_memory(self.store, {
            "query": "the meeting next monday",
            "reason": "canceled",
        })
        self.assertIn("[FORGOTTEN]", obs)
        self.assertIn(nid, obs)
        # obs 里应注明 resolved_by_query
        self.assertIn("resolved from query", obs)

    def test_query_disambiguate_when_multiple_close(self):
        # 两条都在 meeting 簇，分数会很接近 → disambiguation
        self._add("Master has a meeting next Monday")
        self._add("Work meeting on deadline next monday")
        obs = execute_forget_memory(self.store, {
            "query": "meeting next monday",
            "reason": "canceled",
        })
        self.assertIn("[FORGET DISAMBIGUATE]", obs)
        # 应列出多个候选
        self.assertGreaterEqual(obs.count("[Note #"), 2)

    def test_query_not_found_when_no_match(self):
        # 库里只有 birthday 主题的 note，query 走 meeting 主题 → 分数极低
        self._add("Eva's birthday cake had 7 candles", topic="Birthday")
        obs = execute_forget_memory(self.store, {
            "query": "the meeting on monday",
            "reason": "canceled",
        })
        # 低于阈值会被当成 not found（refuse to delete）
        self.assertTrue(
            obs.startswith("[FORGET NOT FOUND]") or obs.startswith("[FORGET ERROR]"),
            f"unexpected obs: {obs}"
        )

    def test_query_topic_filter_narrows(self):
        # 同样 meeting 簇里两条，加 topic 过滤把 Work topic 排除
        self._add("Personal meeting next Monday", topic="Meeting")
        self._add("Work meeting on deadline", topic="Work")
        obs = execute_forget_memory(self.store, {
            "query": "meeting",
            "topic": "Meeting",  # 只匹配 Meeting topic 的那条
            "reason": "rescheduled",
        })
        # topic 过滤后变唯一候选 → 直接删
        self.assertIn("[FORGOTTEN]", obs)

    def test_low_confidence_refuses_to_delete(self):
        # 只有 birthday/cake 簇的记录；query 是无相关词 → encoder 给随机
        # 子空间向量，与 birthday 簇正交 → 分数 ~ 0，远低于阈值。
        self._add("Eva's birthday cake had 7 candles", topic="Birthday")
        obs = execute_forget_memory(self.store, {
            "query": "xyzqq random gibberish nothing",
            "reason": "test",
        })
        self.assertIn("NOT FOUND", obs)
        # 没删任何东西
        self.assertEqual(len(self.store.list_notes()), 1)


# ============================================================
# 参数错误
# ============================================================
class TestForgetParamErrors(unittest.TestCase):
    def test_no_store_returns_error(self):
        obs = execute_forget_memory(None, {})
        self.assertIn("Notes store is not active", obs)

    def test_no_query_no_record_id_returns_error(self):
        # 用 sentinel 假 store（不需要任何真行为，因为参数校验先于 store 调用）
        class _Fake: pass
        obs = execute_forget_memory(_Fake(), {})
        self.assertIn("Must pass either query=", obs)
        self.assertIn("record_id=", obs)


# ============================================================
# R-2.2 auto-correct：record_id 抄错时 runtime 用 fallback_context 救场
# ============================================================
class TestForgetAutoCorrect(_ForgetTestBase):
    """复现 2026-05-13 Turn 13 场景：模型按 SFT 训练 pattern 直接调
    ForgetMemory(record_id="...") 但 hex 抄错。runtime 应该用上下文
    fallback 找到正确 note 并 tombstone。"""

    def test_hallucinated_id_corrected_by_user_text(self):
        # 真实 note：meeting 主题
        real_id = self._add("Master has a meeting next Monday",
                            topic="Meeting", entity="Rosm")
        # 模型抄错了 hex（valid 格式但不存在的 id）
        wrong_id = "70374433"  # 真 Turn 13 复盘里幻觉出的 id
        obs = execute_forget_memory(self.store, {
            "record_id": wrong_id,
            "reason": "Meeting canceled",
        }, fallback_context="the meeting is canceled, forget it")
        # 应自动纠正
        self.assertIn("[FORGOTTEN]", obs)
        self.assertIn(real_id, obs)
        self.assertIn("auto-corrected", obs)
        # 真 note 已被 tombstone
        self.assertEqual(self.store.search("meeting"), [])

    def test_hallucinated_id_corrected_by_remembered_content(self):
        # 模拟 RememberThis 后立即 forget：fallback_context 带上 [REMEMBERED] tool obs
        real_id = self._add("Master has a meeting next Monday",
                            topic="Meeting", entity="Rosm")
        wrong_id = "deadbeef"
        # 模拟 step_once 拼出来的 fallback_context（user 文本 + RememberThis content）
        fallback = (
            "forget it  Master has a meeting next Monday."
            "  [REMEMBERED] Stored as Note #" + real_id + " (entity=Rosm, topic=Meeting)."
        )
        obs = execute_forget_memory(self.store, {
            "record_id": wrong_id,
            "reason": "canceled",
        }, fallback_context=fallback)
        self.assertIn("[FORGOTTEN]", obs)
        self.assertIn(real_id, obs)

    def test_no_fallback_context_returns_error(self):
        # 抄错 id + 没 fallback_context → 走老错误路径
        self._add("Master has a meeting next Monday", topic="Meeting")
        obs = execute_forget_memory(self.store, {
            "record_id": "deadbeef",
            "reason": "canceled",
        }, fallback_context="")
        self.assertIn("[FORGET ERROR]", obs)
        self.assertIn("deadbeef", obs)

    def test_malformed_id_with_context_auto_corrects(self):
        # 模型甚至可能产出不是 8-char hex 的串（"70374433" 8 位、其他可能更乱）。
        # 这种格式错也走 auto-correct。
        real_id = self._add("Master has a meeting next Monday", topic="Meeting")
        obs = execute_forget_memory(self.store, {
            "record_id": "NOT-A-VALID-ID",
            "reason": "canceled",
        }, fallback_context="forget the meeting")
        self.assertIn("[FORGOTTEN]", obs)
        self.assertIn(real_id, obs)

    def test_auto_correct_refuses_when_low_confidence(self):
        # 库里只有 birthday 主题；fallback context 是 "the meeting canceled"。
        # R-2.1 P1 加了 topic-consistency 校验：fallback 含 meeting 信号但
        # birthday note 的 topic="Birthday" 不匹配 → P1 跳过；P2 topic 过滤
        # 找不到 Meeting note → 跳过；P3 相似度对跨主题分数极低 → 也跳过。
        # 最终 fall through 到错误路径，避免误删 birthday note。
        self._add("Eva's birthday cake had 7 candles", topic="Birthday")
        obs = execute_forget_memory(self.store, {
            "record_id": "deadbeef",
            "reason": "canceled",
        }, fallback_context="forget the meeting canceled")
        # 应该报错而不是误删 birthday note
        self.assertIn("[FORGET ERROR]", obs)
        # 原 birthday note 未被删
        self.assertEqual(len(self.store.list_notes()), 1)

    @unittest.skip(
        "2026-05-13 Advisor cutover: this scenario (two meeting notes, user "
        "says 'forget the meeting') is now disambiguated by Advisor — it sees "
        "recent_notes with topic tags and writes the precise record_id into "
        "the per-turn hint. The local runtime no longer attempts topic-tag "
        "preferential matching. Kept skipped for rollback reference."
    )
    def test_auto_correct_picks_topic_tagged_when_two_meetings(self):
        meeting_nid = self._add("Personal meeting next Monday", topic="Meeting")
        work_nid = self._add("Work meeting on deadline", topic="Work")
        obs = execute_forget_memory(self.store, {
            "record_id": "deadbeef",
            "reason": "canceled",
        }, fallback_context="forget the meeting")
        self.assertIn("[FORGOTTEN]", obs)
        self.assertIn(meeting_nid, obs)
        self.assertNotIn(work_nid, obs)
        self.assertEqual(len(self.store.list_notes()), 1)

    # --------------------------------------------------------------
    # 用户提的边缘场景：fallback_context 里没有 [REMEMBERED] tag，
    # 完全靠 reason / 上轮 final_answer / 极短 user 文本来兜底
    # --------------------------------------------------------------
    def test_auto_correct_via_reason_only(self):
        """当 [REMEMBERED] 被压缩出 history、user 文本极短（"forget it"）时，
        模型自己写的 `reason` 字段经常包含 note 内容的复述，足以救场。"""
        real_id = self._add("Master has a meeting next Monday", topic="Meeting")
        # user 文本极短；fallback_context 只剩 reason 复述
        obs = execute_forget_memory(self.store, {
            "record_id": "70374433",
            "reason": "Meeting next Monday has been canceled",
        }, fallback_context="forget it  Meeting next Monday has been canceled")
        self.assertIn("[FORGOTTEN]", obs)
        self.assertIn(real_id, obs)

    def test_auto_correct_via_assistant_final_answer(self):
        """跨多轮 forget 场景：[REMEMBERED] 已被压缩，上一轮 Eva 的 final_answer
        通常复述了刚记的事实（'I've noted it down... meeting next Monday'）。
        eva_core 的 fallback_parts 收集了这一段。"""
        real_id = self._add("Master has a meeting next Monday", topic="Meeting")
        # 模拟 fallback_context 由 user_text + Eva 上轮 final_answer 拼成
        fallback = (
            "forget it  "
            "I've noted it down, Master. Don't forget to bring your laptop "
            "to the meeting next Monday."
        )
        obs = execute_forget_memory(self.store, {
            "record_id": "deadbeef",
            "reason": "canceled",
        }, fallback_context=fallback)
        self.assertIn("[FORGOTTEN]", obs)
        self.assertIn(real_id, obs)

    @unittest.skip(
        "2026-05-13 Advisor cutover: R-2.1 P1 (anaphoric+recent_adds LRU) "
        "fallback removed. Advisor now writes the correct record_id into the "
        "per-turn hint, so 'forget it' alone no longer needs runtime "
        "anaphoric inference. Kept skipped for rollback reference."
    )
    def test_auto_correct_no_overlap_falls_back_to_lru(self):
        nid = self._add("Master has a meeting next Monday", topic="Meeting")
        obs = execute_forget_memory(self.store, {
            "record_id": "deadbeef",
            "reason": "unused",
        }, fallback_context="forget it  unused")
        self.assertIn("[FORGOTTEN]", obs)
        self.assertIn(nid, obs)
        self.assertEqual(len(self.store.list_notes()), 0)


# ============================================================
# R-2.1 (2026-05-13)：抗 history-compaction 的 P0/P1/P2 分层
# ============================================================
# Bug 复盘：R-2 v1 完全依赖 fallback_context；当 history compaction 把
# [REMEMBERED] tool obs 压缩掉后，hex id 提取失败，相似度搜也容易因 noise
# 稀释跌破阈值。R-2.1 把 NotesStore.recent_adds + meta 字段过滤作为主信号源，
# 不再依赖 history 的完整性。
# ============================================================
@unittest.skip(
    "2026-05-13 Advisor cutover: R-2.1 multi-layer auto-correct collapsed to "
    "L0 (hex from context) + L1 (similarity search). P1 (anaphoric+recent_adds), "
    "P2 (event/topic uniqueness) layers removed because Advisor writes the "
    "correct record_id into the per-turn hint. The class is kept skipped for "
    "rollback reference and re-enablement if Advisor is disabled long-term."
)
class TestR21LayeredAutoCorrect(_ForgetTestBase):
    """R-2.1 _try_auto_correct 的 P0/P1/P2/P3 4 层。"""

    # P0: hex id from context
    def test_p0_hex_in_context_short_circuits(self):
        """fallback_context 含 [Note #xxx] 时，直接走 hex 提取，跳过相似度。"""
        nid = self._add("Personal meeting next Monday", topic="Meeting")
        # 模拟 fallback_context 含 [REMEMBERED] obs（compaction 前的 case）
        fallback = f"forget that  [REMEMBERED] Stored as Note #{nid}"
        obs = execute_forget_memory(self.store, {
            "record_id": "deadbeef",
            "reason": "rescheduled",
        }, fallback_context=fallback)
        self.assertIn("[FORGOTTEN]", obs)
        self.assertIn(nid, obs)
        # obs 应标 "extracted #xxx from recent session context"
        self.assertIn("extracted", obs)

    # P1: NotesStore.recent_adds + anaphoric — 抗 compaction 关键测试
    def test_p1_recent_adds_after_history_compaction(self):
        """模拟 history compaction：fallback_context 不含 [REMEMBERED] obs，
        只剩用户当前 turn 文本。recent_adds LRU 仍能让 auto-correct 找到目标。"""
        nid = self._add("Master has a meeting next Monday", topic="Meeting")
        # fallback_context 不含 hex id（被 compaction 抹掉）；仅 user_text
        fallback = "forget it"  # 极短 anaphoric
        obs = execute_forget_memory(self.store, {
            "record_id": "deadbeef",
            "reason": "canceled",
        }, fallback_context=fallback)
        # 应通过 P1 recent_adds 路径命中
        self.assertIn("[FORGOTTEN]", obs)
        self.assertIn(nid, obs)
        self.assertIn("most recently added note", obs)

    def test_p1_recent_adds_skips_when_no_anaphoric(self):
        """user 没用代词 forget 时，recent_adds 不应自动触发——避免误删。"""
        self._add("Master has a meeting next Monday", topic="Meeting")
        # 非 anaphoric: 用户直接陈述事实，没用 "it/that/the"
        fallback = "calculation result is 55 days"
        obs = execute_forget_memory(self.store, {
            "record_id": "deadbeef",
            "reason": "test",
        }, fallback_context=fallback)
        # 应该 fall through，没命中 → 老错误路径
        self.assertIn("[FORGET ERROR]", obs)

    def test_p1_recent_adds_lru_ordering(self):
        """LRU 维护：第二次 add 后，recent_adds[0] 是新 id；anaphoric forget
        应优先选最新那条。"""
        old_nid = self._add("Old meeting from yesterday", topic="Meeting")
        new_nid = self._add("New meeting tomorrow", topic="Meeting")
        # 验 LRU 顺序
        self.assertEqual(self.store.recent_adds[0], new_nid)
        self.assertEqual(self.store.recent_adds[1], old_nid)
        # forget it → 应删 new_nid（最新）
        fallback = "forget that"
        obs = execute_forget_memory(self.store, {
            "record_id": "deadbeef",
            "reason": "test",
        }, fallback_context=fallback)
        self.assertIn("[FORGOTTEN]", obs)
        self.assertIn(new_nid, obs)
        self.assertNotIn(old_nid, obs)
        # tombstone 后 recent_adds 同步剔出
        self.assertNotIn(new_nid, self.store.recent_adds)
        self.assertIn(old_nid, self.store.recent_adds)

    # P2: R-7 event.type / topic uniqueness
    def test_p2_unique_topic_match(self):
        """fallback_context 提到 meeting，库里 meeting topic 仅 1 条 →
        P2 唯一过滤命中，跳过相似度阈值游戏。"""
        nid = self._add("Standup tomorrow", topic="Meeting")
        # 用 birthday topic 测垫一条非 meeting 的，确认 unique 过滤
        # （fake_encoder 会让"meeting"和"birthday cake"完全正交，分数极低）
        # 用 force_forget anaphoric=False 让 P1 不触发，落到 P2
        # 但 P0 / P1 都 miss 的最简单方式是：non-anaphoric + 含 meeting 词
        fallback = "the work meeting"  # 含 "the meeting" 触发 anaphoric
        # 注意：此处会先触发 P1 anaphoric recent_adds，命中最新 add。
        # 测试 P2 需要 anaphoric=False 才能跳过 P1。
        obs = execute_forget_memory(self.store, {
            "record_id": "deadbeef",
            "reason": "canceled",
        }, fallback_context=fallback)
        self.assertIn("[FORGOTTEN]", obs)
        self.assertIn(nid, obs)

    def test_p2_unique_topic_when_anaphoric_fails(self):
        """构造场景：recent_adds 已被 tombstone（剩活 entry 但与查询主题不符），
        P2 topic 唯一过滤接管。"""
        # 加两个 note：meeting + pet
        meeting_nid = self._add("Standup tomorrow", topic="Meeting")
        pet_nid = self._add("Master adopted a cat named Peach", topic="Pet")
        # 把最新的 pet 先删掉
        self.store.tombstone(pet_nid, "test cleanup")
        # 此时 recent_adds 只剩 meeting_nid
        # forget "the meeting" → P1 anaphoric 命中 recent_adds[0] = meeting_nid
        fallback = "the meeting is canceled"
        obs = execute_forget_memory(self.store, {
            "record_id": "deadbeef",
            "reason": "canceled",
        }, fallback_context=fallback)
        self.assertIn("[FORGOTTEN]", obs)
        self.assertIn(meeting_nid, obs)

    # 整合：完全 compaction 模拟（极短 fallback + 多 note）
    def test_anaphoric_forget_with_short_fallback_chooses_newest_in_topic(self):
        """模拟："3 天前加了 meeting，今天 forget it" — fallback 只剩
        "forget it"，hex 全没。recent_adds 仍能找到 meeting 那条。"""
        # 在 meeting 之前先加几条无关 note
        self._add("Master likes chamomile tea", topic="Habits")
        self._add("Eva likes chocolate", topic="Food")
        meeting_nid = self._add("Master has a meeting next Monday at 2pm",
                                 topic="Meeting", entity="Rosm")
        # 此时 recent_adds[0] = meeting_nid
        fallback = "forget it"
        obs = execute_forget_memory(self.store, {
            "record_id": "deadbeef",
            "reason": "meeting canceled",
        }, fallback_context=fallback)
        self.assertIn("[FORGOTTEN]", obs)
        self.assertIn(meeting_nid, obs)

    def test_recent_adds_excludes_already_tombstoned(self):
        """删后 recent_adds 应同步剔出，避免悬挂引用。"""
        nid = self._add("test note", topic="Meeting")
        self.assertIn(nid, self.store.recent_adds)
        ok = self.store.tombstone(nid, "test")
        self.assertTrue(ok)
        self.assertNotIn(nid, self.store.recent_adds)


# ============================================================
# Sanity：常量阈值还在合理区间
# ============================================================
class TestForgetThresholds(unittest.TestCase):
    def test_min_score_in_sane_range(self):
        # 这俩常量是后续调参的旋钮——测试只是一个 trip wire。
        self.assertGreater(FORGET_MATCH_MIN_SCORE, 0.0)
        self.assertLess(FORGET_MATCH_MIN_SCORE, 1.0)
        self.assertGreater(FORGET_UNIQUE_GAP, 0.0)
        self.assertLess(FORGET_UNIQUE_GAP, 0.5)


if __name__ == "__main__":
    unittest.main()
