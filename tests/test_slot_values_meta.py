"""tests/test_slot_values_meta.py — R-1 meta.slot_values 单元测试

R-1 (2026-05-13)：slot value 抽取从 inference-time 正则迁移到 build-time
注入 meta.slot_values。covered:

  1. Reader: `_extract_slot_value_from_record` 优先读 meta.slot_values
  2. Lore corpus: 8.memory_optimized.jsonl 跑过迁移后含 slot_values
  3. NotesStore: RememberThis 工具填 slot_values → 写进 note meta
  4. MemorySearch render: SAVED NOTES 区域显示 slot_values
"""
from __future__ import annotations

import hashlib
import json
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

from eva_slots import _extract_slot_value_from_record  # noqa: E402
from Memory_maker.notes_runtime import NotesStore, execute_remember_this  # noqa: E402


# ============================================================
# Reader：优先 meta.slot_values
# ============================================================
class TestReaderPrefersMeta(unittest.TestCase):
    def test_meta_takes_precedence_over_prose(self):
        """即使 content 含老 regex 没覆盖的措辞，meta.slot_values 也能命中。"""
        rec = {
            "entity": "Eva",
            "topic": "Toy",
            "content": "Her cherished plush companion happened to be a tin soldier she rescued.",
            "meta": {
                "entity": "Eva",
                "topic": "Toy",
                "slot_values": {"toy": "tin soldier"},
            },
        }
        self.assertEqual(_extract_slot_value_from_record(rec, "Eva", "toy"),
                         "tin soldier")

    def test_falls_back_to_regex_when_meta_missing(self):
        """没 slot_values 时仍走原正则路径。"""
        rec = {
            "entity": "Eva",
            "topic": "Toy",
            "content": "Eva's favorite toy has always been a cuddly bunny — soft.",
            "meta": {"entity": "Eva", "topic": "Toy"},  # no slot_values
        }
        self.assertEqual(_extract_slot_value_from_record(rec, "Eva", "toy"),
                         "cuddly bunny")

    def test_falls_back_when_slot_value_empty_string(self):
        """slot_values 存在但值为空字符串时走 regex fallback。"""
        rec = {
            "entity": "Eva",
            "topic": "Toy",
            "content": "Eva's favorite toy is a music box.",
            "meta": {
                "entity": "Eva",
                "topic": "Toy",
                "slot_values": {"toy": ""},
            },
        }
        self.assertEqual(_extract_slot_value_from_record(rec, "Eva", "toy"),
                         "music box")


# ============================================================
# Lore corpus 迁移结果（实际跑过 migrate_slot_values.py 后的 jsonl）
# ============================================================
class TestLoreCorpusMigrated(unittest.TestCase):
    """读已迁移的 8.memory_optimized.jsonl，验证关键 slot 全部落进 meta。"""

    @classmethod
    def setUpClass(cls):
        jsonl = Path(_PROJECT_ROOT) / "Memory_maker" / "8.memory_optimized.jsonl"
        records = []
        with open(jsonl, encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                records.append(json.loads(ln))
        cls.records = records

    def _find_by(self, entity, topic):
        for r in self.records:
            m = r.get("meta", {}) or {}
            if m.get("entity") == entity and m.get("topic") == topic:
                return r
        return None

    def test_eva_toy_has_slot_values(self):
        rec = self._find_by("Eva", "Toy")
        self.assertIsNotNone(rec, "lore should have Eva/Toy record")
        sv = (rec.get("meta", {}) or {}).get("slot_values", {})
        self.assertEqual(sv.get("toy"), "cuddly bunny")

    def test_eva_birthday_has_slot_values(self):
        rec = self._find_by("Eva", "Birthday")
        sv = (rec.get("meta", {}) or {}).get("slot_values", {})
        self.assertEqual(sv.get("birthday"), "July 7th")

    def test_rosm_birthday_has_slot_values(self):
        rec = self._find_by("Rosm", "Birthday")
        sv = (rec.get("meta", {}) or {}).get("slot_values", {})
        self.assertEqual(sv.get("birthday"), "November 25th")

    def test_eva_full_name_has_slot_values(self):
        rec = self._find_by("Eva", "Identity")
        sv = (rec.get("meta", {}) or {}).get("slot_values", {})
        self.assertEqual(sv.get("full_name"), "Eva Louisa")


# ============================================================
# NotesStore.add() 接受 slot_values
# ============================================================
DIM = 32


def _fake_encoder(texts):
    vecs = []
    for t in texts:
        h = hashlib.sha256(t.encode("utf-8")).digest()
        seed = int.from_bytes(h[:8], "big") & 0x7FFFFFFF
        rng = np.random.RandomState(seed)
        v = rng.randn(DIM).astype("float32")
        n = np.linalg.norm(v)
        vecs.append(v / (n if n > 0 else 1.0))
    return np.stack(vecs).astype("float32")


class TestNotesStoreAcceptsSlotValues(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="evar1_"))
        self.store = NotesStore(
            root=self.tmpdir, encoder=_fake_encoder, dim=DIM, session_id="r1",
        )

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_add_with_slot_values(self):
        nid = self.store.add(
            vector_text="Master adopted an orange tabby cat named Peach.",
            content="Master adopted an orange tabby cat named Peach.",
            entity="Rosm", topic="Pet", keywords="cat, peach",
            slot_values={"pet_name": "Peach", "pet_species": "cat"},
        )
        # 反查 meta
        for m in self.store.metas:
            if m["note_id"] == nid:
                self.assertEqual(m.get("slot_values"),
                                 {"pet_name": "Peach", "pet_species": "cat"})
                return
        self.fail("note not found after add")

    def test_remember_this_forwards_slot_values_from_params(self):
        obs = execute_remember_this(self.store, {
            "content": "Master's birthday is November 25th.",
            "entity": "Rosm", "topic": "Birthday",
            "keywords": "birthday, november",
            "slot_values": {"birthday": "November 25th"},
        })
        self.assertIn("[REMEMBERED]", obs)
        # 反查 meta
        live = self.store.metas[-1]
        self.assertEqual(live.get("slot_values"), {"birthday": "November 25th"})

    def test_remember_this_without_slot_values_works(self):
        # 老 SFT 模型不填 slot_values 时仍正常入库
        obs = execute_remember_this(self.store, {
            "content": "Master likes chamomile tea before bed.",
            "entity": "Rosm", "topic": "Habits",
            "keywords": "tea, chamomile",
        })
        self.assertIn("[REMEMBERED]", obs)
        live = self.store.metas[-1]
        self.assertNotIn("slot_values", live)

    def test_invalid_slot_values_ignored_quietly(self):
        # 非 dict 不应崩，应当静默忽略
        obs = execute_remember_this(self.store, {
            "content": "Some fact.",
            "entity": "Rosm", "topic": "Misc", "keywords": "x",
            "slot_values": "not a dict",
        })
        self.assertIn("[REMEMBERED]", obs)
        live = self.store.metas[-1]
        self.assertNotIn("slot_values", live)


# ============================================================
# MemorySearch render: SAVED NOTES section 显示 slot_values
# ============================================================
class TestMemoryBlockRendersSlotValues(unittest.TestCase):
    def test_notes_with_slot_values_rendered(self):
        from eva_memory_legacy import _format_memory_records_block
        collected = {
            "target_entity": "Rosm",
            "records": [],
            "top1_score": 0.0,
            "notes": [
                {
                    "note_id": "abc12345",
                    "entity": "Rosm",
                    "topic": "Pet",
                    "content": "Master adopted an orange tabby cat named Peach.",
                    "rerank_score": 0.85,
                    "slot_values": {"pet_name": "Peach"},
                },
            ],
        }
        out = _format_memory_records_block(collected)
        self.assertIn(">>> SAVED NOTES", out)
        self.assertIn("[Note #abc12345]", out)
        # R-1: slot_values 子项应该渲染
        self.assertIn("pet_name: Peach", out)


# ============================================================
# R-1.1 (2026-05-13)：PRE PROBE 注入的 slot evidence 必须落 TurnEvidenceLedger
# ============================================================
# 复盘 2026-05-13 实跑 Turn 5/6 bug：PRE PROBE 已经把 toy slot 抽出来
# 注入到 packet 文本里，但 fake_retrieval_result["slot_evidence"] 是硬
# 编码空 dict —— 导致 _record_active_memory_evidence 走 R-4 兜底分支，
# 写一条 slot=None 的 evidence。verifier 的 _exact_memory_evidence_for
# (subject="Eva", slot="toy") 看不到 slot=toy 的 evidence，误报
# unsupported_exact_toy_claim 触发不必要的 MemorySearch 重跑。
#
# 修复：format_packet 返回 slot_evidence；_new_refresh_active_memory
# 透传到 fake_retrieval_result；_record_active_memory_evidence 现在
# 走 R-1 主分支写正确的 slot evidence。
class TestR11PreProbeSlotEvidenceLedger(unittest.TestCase):
    def test_active_memory_evidence_records_slot(self):
        """模拟 PRE PROBE 阶段：retrieval_result.slot_evidence 非空时，
        ledger 应写一条 source='memory' subject='Eva' slot='toy' confidence='exact'。"""
        from types import SimpleNamespace
        from eva_history import TurnEvidenceLedger
        from eva_core import ChatAgent

        # 构造最小 stub agent 走 _record_active_memory_evidence 路径
        stub = SimpleNamespace(turn_evidence=TurnEvidenceLedger())

        # 绑方法（含 R-4 helper / R-1.1 写入逻辑）
        for name in ("_record_active_memory_evidence", "_add_turn_evidence"):
            meth = getattr(ChatAgent, name, None)
            if meth is not None:
                setattr(stub, name, meth.__get__(stub))

        # PRE PROBE 应该构造的 retrieval_result —— R-1.1 后 slot_evidence 非空
        retrieval_result = {
            "text": (
                "### [MEMORY MODULE DATA for 'Eva'] ###\n"
                "[SLOT EVIDENCE for Eva]\n"
                "- toy: FOUND = cuddly bunny"
            ),
            "exact_answer_found": True,
            "related_evidence_found": False,
            "is_empty": False,
            "slot_evidence": {
                "toy": {
                    "value": "cuddly bunny",
                    "source": "Eva's favorite toy has always been a cuddly bunny...",
                    "entity": "Eva",
                    "topic": "Toy",
                },
            },
            "requested_slots": ["toy"],
        }
        stub._record_active_memory_evidence(
            retrieval_result=retrieval_result,
            target_entity="Eva",
            query="do you have a toy?",
        )

        # 验证 ledger 里有一条 slot='toy' subject='Eva' 的 exact evidence
        toy_evs = [e for e in stub.turn_evidence
                   if e.source == "memory" and e.slot == "toy"
                   and e.subject == "Eva" and e.confidence == "exact"]
        self.assertEqual(len(toy_evs), 1,
                         f"应有 1 条 slot=toy 的 exact evidence；ledger={list(stub.turn_evidence)}")
        self.assertEqual(toy_evs[0].value, "cuddly bunny")

    def test_active_memory_evidence_empty_slot_falls_back(self):
        """slot_evidence 为空 dict 时仍走 R-4 兜底（slot=None 通用 record_set
        evidence），与改动前行为兼容。"""
        from types import SimpleNamespace
        from eva_history import TurnEvidenceLedger
        from eva_core import ChatAgent

        stub = SimpleNamespace(turn_evidence=TurnEvidenceLedger())
        for name in ("_record_active_memory_evidence", "_add_turn_evidence"):
            meth = getattr(ChatAgent, name, None)
            if meth is not None:
                setattr(stub, name, meth.__get__(stub))

        retrieval_result = {
            "text": "### [MEMORY MODULE DATA for 'Eva'] ###\nRecord 1 ...",
            "exact_answer_found": True,
            "is_empty": False,
            "slot_evidence": {},  # 空：模拟 PRE PROBE 没抽到具体 slot
            "requested_slots": [],
        }
        stub._record_active_memory_evidence(
            retrieval_result=retrieval_result,
            target_entity="Eva",
            query="random toy-ish query",
        )

        # ledger 应有 R-4 兜底的 slot=None evidence
        ledger_items = list(stub.turn_evidence)
        self.assertTrue(any(
            e.source == "memory" and e.subject == "Eva" and e.slot is None
            for e in ledger_items
        ), f"R-4 兜底失效；ledger={ledger_items}")


if __name__ == "__main__":
    unittest.main()
