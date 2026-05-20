"""tests/test_event_schema.py — R-7 RememberThis event schema 单元测试

R-7 (2026-05-13)：给 RememberThis 加 event_date / event_time / event_type /
participants / expires_at 字段，覆盖"日程类事实"。所有字段落 meta.event
sub-dict。NotesStore 提供结构化查询 API：search_by_date /
search_by_event_type / expire_stale。

覆盖：
  1. Sanitizer 函数（_normalize_event_date / _time / _type / _expires_at）
  2. NotesStore.add() 落 meta.event
  3. execute_remember_this 透传 event_* 参数
  4. event_date 明示时跳过 _resolve_relative_dates prose 改写
  5. NotesStore.search_by_date 范围匹配
  6. NotesStore.search_by_event_type
  7. NotesStore.expire_stale tombstone 过期 note
  8. MemorySearch 渲染 [event] 行
"""
from __future__ import annotations

import hashlib
import os
import shutil
import sys
import tempfile
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from Memory_maker.notes_runtime import (  # noqa: E402
    NotesStore,
    execute_remember_this,
    _normalize_event_date,
    _normalize_event_time,
    _normalize_event_type,
    _normalize_expires_at,
    _build_event_dict,
)


# ============================================================
# Sanitizers
# ============================================================
class TestNormalizers(unittest.TestCase):
    def test_event_date_valid(self):
        self.assertEqual(_normalize_event_date("2026-05-18"), "2026-05-18")
        self.assertEqual(_normalize_event_date(" 2026-12-31 "), "2026-12-31")

    def test_event_date_invalid(self):
        self.assertEqual(_normalize_event_date(""), "")
        self.assertEqual(_normalize_event_date("not a date"), "")
        self.assertEqual(_normalize_event_date("2026-02-30"), "")   # 不存在的日期
        self.assertEqual(_normalize_event_date("26-05-18"), "")     # 非 YYYY
        self.assertEqual(_normalize_event_date(None), "")

    def test_event_time(self):
        self.assertEqual(_normalize_event_time("14:00"), "14:00")
        self.assertEqual(_normalize_event_time("9:30"), "09:30")
        self.assertEqual(_normalize_event_time("23:59"), "23:59")
        self.assertEqual(_normalize_event_time("24:00"), "")  # 越界
        self.assertEqual(_normalize_event_time("noon"), "")

    def test_event_type(self):
        self.assertEqual(_normalize_event_type("Meeting"), "meeting")
        self.assertEqual(_normalize_event_type(" appointment "), "appointment")
        self.assertEqual(_normalize_event_type(""), "")
        self.assertEqual(_normalize_event_type("x" * 50), "")  # 超 40 字符

    def test_expires_at_short_date_promoted(self):
        # YYYY-MM-DD 短形式应该被补成 ISO8601 timestamp
        self.assertEqual(_normalize_expires_at("2026-05-19"),
                         "2026-05-19T00:00:00Z")

    def test_expires_at_iso(self):
        self.assertEqual(_normalize_expires_at("2026-05-19T12:00:00Z"),
                         "2026-05-19T12:00:00Z")

    def test_expires_at_invalid(self):
        self.assertEqual(_normalize_expires_at("bad"), "")
        self.assertEqual(_normalize_expires_at(None), "")

    def test_build_event_dict_omits_empties(self):
        d = _build_event_dict(event_date="2026-05-18",
                              event_time="", event_type="", expires_at="")
        self.assertEqual(d, {"date": "2026-05-18"})
        d2 = _build_event_dict(event_date="", event_time="",
                               event_type="", expires_at="")
        self.assertEqual(d2, {})


# ============================================================
# NotesStore 入库
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


class _StoreBase(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="evar7_"))
        self.store = NotesStore(
            root=self.tmpdir, encoder=_fake_encoder, dim=DIM, session_id="r7",
        )

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)


class TestNotesStoreAcceptsEventFields(_StoreBase):
    def test_event_fields_persist_to_meta(self):
        nid = self.store.add(
            vector_text="Master has a meeting next Monday with the boss",
            content="Master has a meeting next Monday with the boss",
            entity="Rosm", topic="Meeting", keywords="meeting, work",
            event_date="2026-05-18", event_time="14:00",
            event_type="meeting", expires_at="2026-05-19",
            participants=["Rosm", "boss"],
        )
        m = next(x for x in self.store.metas if x["note_id"] == nid)
        self.assertEqual(m["event"]["date"], "2026-05-18")
        self.assertEqual(m["event"]["time"], "14:00")
        self.assertEqual(m["event"]["type"], "meeting")
        self.assertEqual(m["event"]["expires_at"], "2026-05-19T00:00:00Z")
        self.assertEqual(m["event"]["participants"], ["Rosm", "boss"])

    def test_no_event_field_no_event_dict(self):
        # 不传 event_* 时不应留空 event dict 在 meta
        nid = self.store.add(
            vector_text="Master likes chamomile tea",
            content="Master likes chamomile tea",
            entity="Rosm", topic="Habits", keywords="tea",
        )
        m = next(x for x in self.store.metas if x["note_id"] == nid)
        self.assertNotIn("event", m)

    def test_invalid_event_date_dropped(self):
        # bogus event_date 不该破坏 add；event dict 应该不带 date 子键
        nid = self.store.add(
            vector_text="x", content="x", entity="Rosm",
            topic="Test", keywords="x",
            event_date="not a date", event_type="meeting",
        )
        m = next(x for x in self.store.metas if x["note_id"] == nid)
        self.assertNotIn("date", m.get("event", {}))
        self.assertEqual(m["event"]["type"], "meeting")


class TestExecuteRememberThis(_StoreBase):
    def test_event_params_forwarded(self):
        obs = execute_remember_this(self.store, {
            "content": "Master has a meeting next Monday",
            "entity": "Rosm", "topic": "Meeting", "keywords": "meeting",
            "event_date": "2026-05-18", "event_type": "meeting",
            "participants": ["Rosm"],
        })
        self.assertIn("[REMEMBERED]", obs)
        m = self.store.metas[-1]
        self.assertEqual(m["event"]["date"], "2026-05-18")
        self.assertEqual(m["event"]["type"], "meeting")

    def test_explicit_event_date_skips_prose_normalization(self):
        """R-7：模型给了 event_date 时，content 里的 'next Monday' 不应
        被改写成 'next Monday (2026-05-18)' —— 结构化字段已是 source of
        truth，避免冗余。"""
        execute_remember_this(self.store, {
            "content": "Master has a meeting next Monday",
            "entity": "Rosm", "topic": "Meeting", "keywords": "meeting",
            "event_date": "2026-05-18",
        })
        last_content = self.store.contents[-1]
        # 不应叠加 "(2026-05-18)" 这种 prose 注释
        self.assertNotIn("(2026-05-18)", last_content)
        # 但 meta.event.date 是 2026-05-18
        self.assertEqual(self.store.metas[-1]["event"]["date"], "2026-05-18")

    def test_no_event_date_falls_back_to_prose_normalize(self):
        """无 event_date 时仍走 P2-7 _resolve_relative_dates 兜底（老 SFT
        模型行为保留）。"""
        execute_remember_this(self.store, {
            "content": "Master has a meeting next Monday",
            "entity": "Rosm", "topic": "Meeting", "keywords": "meeting",
            # 不传 event_date
        })
        last_content = self.store.contents[-1]
        # _resolve_relative_dates 会把 "next Monday" 改成 "next Monday (YYYY-MM-DD)"
        self.assertIn("next Monday", last_content)
        self.assertRegex(last_content, r"\(\d{4}-\d{2}-\d{2}\)")


# ============================================================
# 查询 API
# ============================================================
class TestQueryAPI(_StoreBase):
    def _seed(self):
        self.store.add(
            vector_text="meeting 1", content="meeting 1",
            entity="Rosm", topic="Meeting", keywords="m",
            event_date="2026-05-18", event_type="meeting",
        )
        self.store.add(
            vector_text="appointment", content="appointment",
            entity="Rosm", topic="Health", keywords="a",
            event_date="2026-05-20", event_type="appointment",
        )
        self.store.add(
            vector_text="no event", content="no event",
            entity="Eva", topic="Personality", keywords="p",
            # no event_*
        )

    def test_search_by_date_range(self):
        self._seed()
        hits = self.store.search_by_date("2026-05-17", "2026-05-19")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["meta"]["event"]["date"], "2026-05-18")

    def test_search_by_date_single_day(self):
        self._seed()
        hits = self.store.search_by_date("2026-05-20")  # 单日
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["meta"]["event"]["type"], "appointment")

    def test_search_by_event_type(self):
        self._seed()
        hits = self.store.search_by_event_type("meeting")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["meta"]["event"]["date"], "2026-05-18")

    def test_search_skips_tombstoned(self):
        self._seed()
        # 删第一条 meeting，再搜
        all_hits = self.store.search_by_event_type("meeting")
        self.store.tombstone(all_hits[0]["note_id"], "test")
        self.assertEqual(self.store.search_by_event_type("meeting"), [])

    def test_search_bad_args_returns_empty(self):
        # 不合法日期不抛错
        self.assertEqual(self.store.search_by_date("bad"), [])
        self.assertEqual(self.store.search_by_event_type(""), [])

    def test_expire_stale_tombstones_past_notes(self):
        # 创建一条已过期的 note
        nid_expired = self.store.add(
            vector_text="x", content="expired meeting", entity="Rosm",
            topic="Meeting", keywords="x",
            expires_at="2026-01-01",  # 已过期
        )
        nid_future = self.store.add(
            vector_text="y", content="future meeting", entity="Rosm",
            topic="Meeting", keywords="y",
            expires_at="2099-12-31",  # 未来
        )
        # 用一个固定 now 跑 expire
        now = datetime(2026, 6, 1, tzinfo=timezone.utc)
        expired = self.store.expire_stale(now=now)
        self.assertIn(nid_expired, expired)
        self.assertNotIn(nid_future, expired)
        # 已 tombstone 的不再 live
        live_ids = {n["note_id"] for n in self.store.list_notes()}
        self.assertNotIn(nid_expired, live_ids)
        self.assertIn(nid_future, live_ids)

    def test_expire_stale_default_now(self):
        # 不传 now 用 datetime.now(UTC)。过期是过去时间。
        self.store.add(
            vector_text="x", content="x", entity="Rosm",
            topic="x", keywords="x",
            expires_at="2020-01-01",
        )
        expired = self.store.expire_stale()
        self.assertEqual(len(expired), 1)


# ============================================================
# 渲染层
# ============================================================
class TestRenderEvent(unittest.TestCase):
    def test_event_line_appears_in_notes_block(self):
        from eva_memory_legacy import _format_memory_records_block
        collected = {
            "target_entity": "Rosm",
            "records": [],
            "top1_score": 0.0,
            "notes": [{
                "note_id": "abc12345",
                "entity": "Rosm",
                "topic": "Meeting",
                "content": "Master has a meeting next Monday with the boss",
                "rerank_score": 0.85,
                "event": {
                    "date": "2026-05-18",
                    "time": "14:00",
                    "type": "meeting",
                    "participants": ["Rosm", "boss"],
                },
            }],
        }
        out = _format_memory_records_block(collected)
        self.assertIn("[event]", out)
        self.assertIn("date=2026-05-18", out)
        self.assertIn("type=meeting", out)
        self.assertIn("participants=Rosm,boss", out)


if __name__ == "__main__":
    unittest.main()
