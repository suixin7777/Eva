"""Offline unit tests for Memory_maker/notes_runtime.py.

Pure faiss + numpy round-trip. No mpnet model is loaded — a deterministic
fake encoder maps each text to a hashed unit vector, so identical strings
get identical vectors and similar substrings get correlated vectors.

Run from project root:

    D:/Anaconda/envs/py310/python.exe tests/test_notes_runtime.py
"""
import hashlib
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

# ------------------------------------------------------------
# Path setup
# ------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from Memory_maker.notes_runtime import NotesStore  # noqa: E402


# ------------------------------------------------------------
# Fake encoder — deterministic, normalized
# ------------------------------------------------------------
DIM = 32  # small for speed


def _hash_to_vec(text: str, dim: int = DIM) -> np.ndarray:
    """Map text to a unit-norm float32 vector via SHA256-seeded RNG."""
    h = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(h[:8], "big") & 0x7FFFFFFF
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype("float32")
    n = np.linalg.norm(v)
    return v / (n if n > 0 else 1.0)


def fake_encoder(texts):
    return np.stack([_hash_to_vec(t) for t in texts]).astype("float32")


# ------------------------------------------------------------
# Tests
# ------------------------------------------------------------
class _BaseStoreTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="evanotes_"))
        self.store = NotesStore(
            root=self.tmpdir, encoder=fake_encoder, dim=DIM,
            session_id="t0",
        )

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    # helper
    def _add_simple(self, vt="Eva loves cuddly bunny",
                    content="Eva's plush bunny is her oldest toy.",
                    entity="Eva", topic="Toy", keywords="bunny,plush"):
        return self.store.add(
            vector_text=vt, content=content,
            entity=entity, topic=topic, keywords=keywords,
        )


class TestAddAndPersist(_BaseStoreTest):
    def test_add_returns_8_char_id(self):
        rid = self._add_simple()
        self.assertEqual(len(rid), 8)
        self.assertTrue(all(c in "0123456789abcdef" for c in rid))

    def test_add_persists_jsonl(self):
        rid = self._add_simple()
        with open(self.store.jsonl_path, encoding="utf-8") as f:
            lines = [json.loads(ln) for ln in f if ln.strip()]
        self.assertEqual(len(lines), 1)
        rec = lines[0]
        self.assertEqual(rec["meta"]["note_id"], rid)
        self.assertEqual(rec["meta"]["entity"], "Eva")
        self.assertEqual(rec["meta"]["topic"], "Toy")
        self.assertFalse(rec["meta"]["deleted"])
        self.assertIn("created_at", rec["meta"])
        self.assertTrue(rec["meta"]["origin"].startswith("session_"))

    def test_add_persists_index_and_meta(self):
        self._add_simple()
        self.assertTrue(self.store.index_path.exists())
        self.assertTrue(self.store.content_path.exists())
        self.assertTrue(self.store.meta_path.exists())
        self.assertEqual(self.store.index.ntotal, 1)
        self.assertEqual(len(self.store.contents), 1)
        self.assertEqual(len(self.store.metas), 1)

    def test_add_keywords_string_or_list(self):
        rid_a = self.store.add(vector_text="A", content="A.",
                               entity="Eva", topic="Toy",
                               keywords="bunny, plush, fox")
        rid_b = self.store.add(vector_text="B", content="B.",
                               entity="Eva", topic="Toy",
                               keywords=["bunny", "plush"])
        for rid in (rid_a, rid_b):
            m = next(m for m in self.store.metas if m["note_id"] == rid)
            self.assertIsInstance(m["keywords"], list)
            self.assertTrue(all(isinstance(k, str) for k in m["keywords"]))

    def test_add_rejects_bad_entity(self):
        with self.assertRaises(ValueError):
            self.store.add(vector_text="x", content="x.",
                           entity="Bob", topic="Toy", keywords="")

    def test_add_rejects_empty_text(self):
        with self.assertRaises(ValueError):
            self.store.add(vector_text="", content="x.",
                           entity="Eva", topic="Toy", keywords="")

    def test_audit_log_records_add(self):
        rid = self._add_simple()
        with open(self.store.audit_path, encoding="utf-8") as f:
            log = f.read()
        self.assertIn(f"ADD {rid}", log)


class TestSearch(_BaseStoreTest):
    def test_search_empty_store_returns_empty_list(self):
        self.assertEqual(self.store.search("anything"), [])

    def test_search_finds_added_note(self):
        rid = self._add_simple(vt="Eva loves cuddly bunny")
        results = self.store.search("Eva loves cuddly bunny", top_k=5)
        self.assertGreaterEqual(len(results), 1)
        self.assertEqual(results[0]["note_id"], rid)
        self.assertGreater(results[0]["score"], 0.99)  # exact text → cosine ~1

    def test_search_excludes_tombstoned(self):
        rid = self._add_simple()
        self.assertEqual(len(self.store.search("cuddly bunny")), 1)
        self.assertTrue(self.store.tombstone(rid, "user asked to forget"))
        self.assertEqual(self.store.search("cuddly bunny"), [])

    def test_search_top_k_limit(self):
        for i in range(5):
            self._add_simple(vt=f"text-{i}", content=f"c-{i}")
        # All distinct — search any query, top_k=2 returns at most 2.
        results = self.store.search("text-0", top_k=2)
        self.assertLessEqual(len(results), 2)


class TestTombstone(_BaseStoreTest):
    def test_tombstone_unknown_id_returns_false(self):
        self.assertFalse(self.store.tombstone("deadbeef", "n/a"))

    def test_tombstone_twice_returns_false_second_time(self):
        rid = self._add_simple()
        self.assertTrue(self.store.tombstone(rid, "first"))
        self.assertFalse(self.store.tombstone(rid, "again"))

    def test_tombstone_persists_after_dump(self):
        rid = self._add_simple()
        self.store.tombstone(rid, "wrong")
        with open(self.store.meta_path, encoding="utf-8") as f:
            metas = json.load(f)
        self.assertTrue(metas[0]["deleted"])
        self.assertEqual(metas[0]["deleted_reason"], "wrong")
        self.assertIn("deleted_at", metas[0])

    def test_audit_log_records_delete(self):
        rid = self._add_simple()
        self.store.tombstone(rid, "user changed mind")
        with open(self.store.audit_path, encoding="utf-8") as f:
            log = f.read()
        self.assertIn(f"DELETE {rid}", log)
        self.assertIn("user changed mind", log)


class TestReload(_BaseStoreTest):
    def test_reload_preserves_notes_and_tombstone(self):
        rid_a = self._add_simple(vt="A", content="A.")
        rid_b = self._add_simple(vt="B", content="B.")
        self.store.tombstone(rid_a, "drop A")

        # Reopen with a fresh store pointing at same dir.
        reopened = NotesStore(root=self.tmpdir, encoder=fake_encoder, dim=DIM)
        self.assertEqual(reopened.index.ntotal, 2)
        self.assertEqual(len(reopened.contents), 2)
        self.assertEqual(len(reopened.metas), 2)
        # Tombstone preserved
        self.assertTrue(reopened.metas[0]["deleted"])
        self.assertFalse(reopened.metas[1]["deleted"])
        # Search excludes tombstoned
        results = reopened.search("A")
        self.assertNotIn(rid_a, [r["note_id"] for r in results])

    def test_reload_uses_existing_index_dim(self):
        # Construct with dim=32, save, reopen with dim=64 hint — should
        # adopt the saved index's dim, not the constructor argument.
        self._add_simple()
        reopened = NotesStore(root=self.tmpdir, encoder=fake_encoder, dim=64)
        self.assertEqual(reopened.dim, DIM)


class TestStatusAndList(_BaseStoreTest):
    def test_status_counts(self):
        rid_a = self._add_simple(vt="A", content="A.")
        self._add_simple(vt="B", content="B.")
        self.store.tombstone(rid_a, "")
        s = self.store.status()
        self.assertEqual(s["total"], 2)
        self.assertEqual(s["live"], 1)
        self.assertEqual(s["deleted"], 1)
        self.assertEqual(s["index_ntotal"], 2)

    def test_list_notes_excludes_deleted_by_default(self):
        rid_a = self._add_simple(vt="A", content="A content")
        self._add_simple(vt="B", content="B content")
        self.store.tombstone(rid_a, "")
        live = self.store.list_notes()
        self.assertEqual(len(live), 1)
        all_ = self.store.list_notes(include_deleted=True)
        self.assertEqual(len(all_), 2)


class TestCompact(_BaseStoreTest):
    def test_compact_drops_tombstoned(self):
        rid_a = self._add_simple(vt="A", content="A.")
        rid_b = self._add_simple(vt="B", content="B.")
        rid_c = self._add_simple(vt="C", content="C.")
        self.store.tombstone(rid_a, "")
        self.store.tombstone(rid_c, "")
        before = self.store.index.ntotal
        result = self.store.compact()
        self.assertEqual(before, 3)
        self.assertEqual(result["before"], 3)
        self.assertEqual(result["after"], 1)
        self.assertEqual(result["dropped"], 2)
        self.assertEqual(self.store.index.ntotal, 1)
        # Surviving note should be the only live one (rid_b).
        self.assertEqual(self.store.metas[0]["note_id"], rid_b)
        # Dropped note_ids are gone from any search result.
        all_results = self.store.search("anything", top_k=10)
        seen_ids = {r["note_id"] for r in all_results}
        self.assertNotIn(rid_a, seen_ids)
        self.assertNotIn(rid_c, seen_ids)
        self.assertEqual(seen_ids, {rid_b})

    def test_compact_when_nothing_deleted_is_noop_in_count(self):
        self._add_simple(vt="A", content="A.")
        self._add_simple(vt="B", content="B.")
        result = self.store.compact()
        self.assertEqual(result["dropped"], 0)
        self.assertEqual(self.store.index.ntotal, 2)


class TestDiscard(_BaseStoreTest):
    def test_discard_clears_state(self):
        self._add_simple()
        self._add_simple(vt="other")
        self.store.discard()
        self.assertEqual(self.store.index.ntotal, 0)
        self.assertEqual(self.store.contents, [])
        self.assertEqual(self.store.metas, [])
        # Source files removed
        self.assertFalse(self.store.jsonl_path.exists())
        # Audit log archived (renamed), not the same file
        archived = list(self.tmpdir.glob("audit_*.log"))
        self.assertEqual(len(archived), 1)

    def test_discard_then_add_starts_clean(self):
        self._add_simple()
        self.store.discard()
        rid = self._add_simple(vt="post-discard")
        self.assertEqual(self.store.index.ntotal, 1)
        self.assertEqual(self.store.metas[0]["note_id"], rid)


# ============================================================
# Integration with eva_memory_legacy (formatter + attach helper)
# ============================================================
# Importing eva_memory_legacy pulls in eva_config which imports torch
# for cudnn flags. Stub torch the same way other tests do — keeps this
# suite runnable on machines without the heavy ML stack installed.
import types

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

from eva_memory_legacy import (  # noqa: E402
    _attach_user_notes,
    _format_memory_records_block,
)


class TestRetrievalIntegration(unittest.TestCase):
    """Notes flow into the formatter with a [Note #...] tag in their own
    `>>> SAVED NOTES <<<` section, never merged with lore-corpus records."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="evanotes_integ_"))
        self.store = NotesStore(
            root=self.tmpdir, encoder=fake_encoder, dim=DIM,
            session_id="integ_t",
        )

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_attach_notes_into_empty_pool(self):
        rid = self.store.add(
            vector_text="Master likes black coffee",
            content="He drinks it every morning, no sugar.",
            entity="Rosm", topic="Likes", keywords="coffee, black",
        )
        collected = {"target_entity": "Rosm", "keywords": [],
                     "search_query": "what does master drink",
                     "records": [], "top1_score": 0.0, "error": None}
        out = _attach_user_notes(
            collected, self.store, "Master likes black coffee",
        )
        # Notes live in their OWN bucket — never merged with lore-corpus records.
        self.assertEqual(out["records"], [])
        self.assertEqual(len(out["notes"]), 1)
        rec = out["notes"][0]
        self.assertEqual(rec["note_id"], rid)
        self.assertEqual(rec["entity"], "Rosm")
        self.assertEqual(rec["category"], "Lore")
        self.assertEqual(rec["topic"], "Likes")
        self.assertGreater(rec["rerank_score"], 0.99)

    def test_attach_no_notes_store_is_noop(self):
        collected = {"records": [], "top1_score": 0.0}
        out = _attach_user_notes(collected, None, "anything")
        self.assertEqual(out["records"], [])
        self.assertNotIn("notes", out)

    def test_attach_empty_query_is_noop(self):
        self.store.add(vector_text="x", content="x.", entity="Eva",
                       topic="Toy", keywords="x")
        collected = {"records": [], "top1_score": 0.0}
        out = _attach_user_notes(collected, self.store, "")
        self.assertEqual(out["records"], [])
        self.assertNotIn("notes", out)

    def test_attach_excludes_tombstoned(self):
        rid_keep = self.store.add(vector_text="alpha note",
                                  content="A.", entity="Eva",
                                  topic="Toy", keywords="a")
        rid_drop = self.store.add(vector_text="alpha note",
                                  content="dup A.", entity="Eva",
                                  topic="Toy", keywords="a")
        self.store.tombstone(rid_drop, "duplicate")
        collected = {"records": [], "top1_score": 0.0}
        out = _attach_user_notes(collected, self.store, "alpha note")
        ids = {r["note_id"] for r in out.get("notes", [])}
        self.assertIn(rid_keep, ids)
        self.assertNotIn(rid_drop, ids)

    def test_attach_does_not_touch_existing_records(self):
        prod = {
            "content": "old lore record",
            "entity": "Eva", "category": "Lore", "topic": "Toy",
            "rerank_score": 5.0, "low_confidence": False,
        }
        rid = self.store.add(vector_text="freshly remembered fact",
                             content="brand new.", entity="Eva",
                             topic="Toy", keywords="fresh")
        collected = {"records": [prod], "top1_score": 5.0}
        out = _attach_user_notes(
            collected, self.store, "freshly remembered fact",
        )
        # Lore-corpus records bucket is untouched.
        self.assertEqual(len(out["records"]), 1)
        self.assertEqual(out["records"][0]["content"], "old lore record")
        # Note lives in its own bucket regardless of cosine vs rerank.
        self.assertEqual(len(out["notes"]), 1)
        self.assertEqual(out["notes"][0]["note_id"], rid)

    def test_attach_filters_below_min_cosine(self):
        # Force a high min_cosine — even an exact-text match (cosine 1.0)
        # passes; a different-text query (cosine < 1.0) is filtered.
        # Avoids flakiness from the hash-based fake encoder's variance.
        self.store.add(vector_text="alpha note",
                       content="A.", entity="Eva", topic="Toy",
                       keywords="a")
        collected = {"records": [], "top1_score": 0.0}
        out_match = _attach_user_notes(
            collected, self.store, "alpha note", min_cosine=0.99,
        )
        self.assertEqual(len(out_match.get("notes", [])), 1)

        collected2 = {"records": [], "top1_score": 0.0}
        out_no_match = _attach_user_notes(
            collected2, self.store, "totally different query",
            min_cosine=0.99,
        )
        self.assertFalse(out_no_match.get("notes"))

    def test_format_block_renders_notes_section(self):
        rid = self.store.add(
            vector_text="Master loves jazz",
            content="He plays it every evening on vinyl.",
            entity="Rosm", topic="Music", keywords="jazz, vinyl",
        )
        collected = {"target_entity": "Rosm", "keywords": [],
                     "search_query": "Master loves jazz",
                     "records": [], "top1_score": 0.0, "error": None}
        collected = _attach_user_notes(
            collected, self.store, "Master loves jazz",
        )
        block = _format_memory_records_block(collected)
        self.assertIn(">>> SAVED NOTES", block)
        self.assertIn(f"[Note #{rid}]", block)
        self.assertIn("[Subject: Rosm]", block)
        self.assertIn("[Topic: Music]", block)
        self.assertIn(">>> END SAVED NOTES <<<", block)

    def test_format_block_notes_section_appears_alongside_lore_records(self):
        # Lore records present + a note matching the query —
        # both should render, in their own sections.
        rid = self.store.add(
            vector_text="Master adopted an orange cat named Peach",
            content="Master adopted an orange cat named Peach.",
            entity="Rosm", topic="Pet", keywords="cat, peach, orange",
        )
        collected = {
            "target_entity": "Both", "keywords": [],
            "search_query": "Master adopted an orange cat named Peach",
            "records": [
                {"content": "old lore record about pleasure ground",
                 "entity": "Shared", "category": "Event", "topic": "Date",
                 "rerank_score": 4.0, "low_confidence": False},
            ],
            "top1_score": 4.0, "error": None,
        }
        collected = _attach_user_notes(
            collected, self.store,
            "Master adopted an orange cat named Peach",
        )
        block = _format_memory_records_block(collected)
        # Lore record still rendered
        self.assertIn("pleasure ground", block)
        # Note rendered in its own section
        self.assertIn(">>> SAVED NOTES", block)
        self.assertIn(f"[Note #{rid}]", block)
        self.assertIn("Peach", block)

    def test_format_block_notes_section_when_no_lore_records(self):
        # Lore retrieval empty + note matches → render notes-only block
        # instead of "no relevant records found".
        rid = self.store.add(
            vector_text="A unique fact about widgets",
            content="A unique fact about widgets.",
            entity="Eva", topic="Lore", keywords="widget",
        )
        collected = {"target_entity": "Eva", "keywords": [],
                     "search_query": "A unique fact about widgets",
                     "records": [], "top1_score": 0.0, "error": None}
        collected = _attach_user_notes(
            collected, self.store, "A unique fact about widgets",
        )
        block = _format_memory_records_block(collected)
        self.assertNotIn("No relevant records found", block)
        self.assertIn(f"[Note #{rid}]", block)
        self.assertIn(">>> SAVED NOTES", block)

    def test_format_block_no_notes_unchanged_format(self):
        # When notes_store yields nothing, formatter output looks identical
        # to the original (no stray [Note #] tag, no notes section).
        collected = {
            "target_entity": "Eva", "keywords": [],
            "search_query": "name?",
            "records": [{
                "content": "Eva Louisa is her full name.",
                "entity": "Eva", "category": "Lore", "topic": "Identity",
                "rerank_score": 0.9, "low_confidence": False,
            }],
            "top1_score": 0.9, "error": None,
        }
        block = _format_memory_records_block(collected)
        self.assertNotIn("[Note #", block)
        self.assertNotIn("SAVED NOTES", block)
        self.assertIn("[Subject: Eva]", block)


# ============================================================
# Tool dispatch helpers (RememberThis / ForgetMemory)
# ============================================================
from Memory_maker.notes_runtime import (  # noqa: E402
    execute_remember_this,
    execute_forget_memory,
)


class TestExecuteRememberThis(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="evanotes_remember_"))
        self.store = NotesStore(
            root=self.tmpdir, encoder=fake_encoder, dim=DIM,
            session_id="remember_t",
        )

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_happy_path_returns_remembered_obs_with_id(self):
        params = {
            "content": "Master likes black coffee, no sugar.",
            "entity": "Rosm",
            "topic": "Likes",
            "keywords": "coffee, black, drinks",
        }
        obs = execute_remember_this(self.store, params)
        self.assertTrue(obs.startswith("[REMEMBERED]"))
        # Note actually written
        self.assertEqual(self.store.index.ntotal, 1)
        # Returned obs contains the note_id
        rid = self.store.metas[0]["note_id"]
        self.assertIn(f"Note #{rid}", obs)
        self.assertIn("ForgetMemory", obs)  # tells model how to undo

    def test_no_notes_store_returns_error_obs(self):
        obs = execute_remember_this(None, {"content": "x", "entity": "Eva",
                                           "topic": "Toy", "keywords": ""})
        self.assertTrue(obs.startswith("[REMEMBER ERROR]"))
        self.assertIn("not active", obs.lower())

    def test_missing_content_returns_error(self):
        obs = execute_remember_this(self.store, {
            "content": "", "entity": "Eva", "topic": "Toy", "keywords": "",
        })
        self.assertTrue(obs.startswith("[REMEMBER ERROR]"))
        self.assertIn("content", obs)
        self.assertEqual(self.store.index.ntotal, 0)  # nothing written

    def test_missing_topic_returns_error(self):
        obs = execute_remember_this(self.store, {
            "content": "x", "entity": "Eva", "topic": "", "keywords": "",
        })
        self.assertTrue(obs.startswith("[REMEMBER ERROR]"))
        self.assertIn("topic", obs)
        self.assertEqual(self.store.index.ntotal, 0)

    def test_invalid_entity_returns_error(self):
        obs = execute_remember_this(self.store, {
            "content": "x", "entity": "Bob", "topic": "Toy", "keywords": "",
        })
        self.assertTrue(obs.startswith("[REMEMBER ERROR]"))
        self.assertIn("entity", obs)
        self.assertEqual(self.store.index.ntotal, 0)

    def test_invalid_category_returns_error(self):
        obs = execute_remember_this(self.store, {
            "content": "x", "entity": "Eva", "topic": "Toy",
            "keywords": "", "category": "Junk",
        })
        self.assertTrue(obs.startswith("[REMEMBER ERROR]"))
        self.assertIn("category", obs)

    def test_none_params_treated_as_empty(self):
        obs = execute_remember_this(self.store, None)
        self.assertTrue(obs.startswith("[REMEMBER ERROR]"))


class TestExecuteForgetMemory(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="evanotes_forget_"))
        self.store = NotesStore(
            root=self.tmpdir, encoder=fake_encoder, dim=DIM,
            session_id="forget_t",
        )
        self.rid = self.store.add(
            vector_text="Eva enjoys baking macarons",
            content="She made lemon ones last weekend.",
            entity="Eva", topic="Food", keywords="bake, macaron",
        )

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_happy_path_returns_forgotten_obs(self):
        obs = execute_forget_memory(self.store, {
            "record_id": self.rid, "reason": "test wrong",
        })
        self.assertTrue(obs.startswith("[FORGOTTEN]"))
        self.assertIn(self.rid, obs)
        # Tombstone written
        self.assertTrue(self.store.metas[0]["deleted"])

    def test_no_notes_store_returns_error_obs(self):
        obs = execute_forget_memory(None, {
            "record_id": "abcd1234", "reason": "x",
        })
        self.assertTrue(obs.startswith("[FORGET ERROR]"))
        self.assertIn("not active", obs.lower())

    def test_missing_record_id_returns_error(self):
        obs = execute_forget_memory(self.store, {"record_id": "", "reason": ""})
        self.assertTrue(obs.startswith("[FORGET ERROR]"))
        self.assertIn("record_id", obs)
        self.assertFalse(self.store.metas[0]["deleted"])

    def test_malformed_record_id_returns_error(self):
        # 7 chars instead of 8
        obs = execute_forget_memory(self.store, {
            "record_id": "abcdef0", "reason": "x",
        })
        self.assertTrue(obs.startswith("[FORGET ERROR]"))
        self.assertIn("8-char", obs)

    def test_unknown_record_id_returns_error(self):
        obs = execute_forget_memory(self.store, {
            "record_id": "deadbeef", "reason": "x",
        })
        self.assertTrue(obs.startswith("[FORGET ERROR]"))
        self.assertIn("No live note", obs)

    def test_double_forget_returns_error_second_time(self):
        # First forget succeeds
        obs1 = execute_forget_memory(self.store, {
            "record_id": self.rid, "reason": "first",
        })
        self.assertTrue(obs1.startswith("[FORGOTTEN]"))
        # Second forget on same id: tombstone() returns False → error
        obs2 = execute_forget_memory(self.store, {
            "record_id": self.rid, "reason": "again",
        })
        self.assertTrue(obs2.startswith("[FORGET ERROR]"))


class TestEndToEndRememberAndRetrieve(unittest.TestCase):
    """The intended use case: model calls RememberThis, then a later
    MemorySearch surfaces the saved note tagged with [Note #...], and
    a later ForgetMemory removes it from retrieval."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="evanotes_e2e_"))
        self.store = NotesStore(
            root=self.tmpdir, encoder=fake_encoder, dim=DIM,
            session_id="e2e_t",
        )

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_remember_then_retrieve_then_forget_then_gone(self):
        # 1) Model calls RememberThis
        obs_remember = execute_remember_this(self.store, {
            "content": "Master mentioned today his favorite color is teal",
            "entity": "Rosm", "topic": "Likes",
            "keywords": "color, teal, favorite",
        })
        self.assertTrue(obs_remember.startswith("[REMEMBERED]"))
        rid = self.store.metas[0]["note_id"]

        # 2) Later turn: MemorySearch — _attach_user_notes + formatter
        from eva_memory_legacy import (
            _attach_user_notes, _format_memory_records_block,
        )
        collected = {"target_entity": "Rosm", "keywords": [],
                     "search_query": "Master's favorite color",
                     "records": [], "top1_score": 0.0, "error": None}
        collected = _attach_user_notes(
            collected, self.store, "Master mentioned today his favorite color is teal",
        )
        block = _format_memory_records_block(collected)
        self.assertIn(f"[Note #{rid}]", block)
        self.assertIn("teal", block)

        # 3) Model calls ForgetMemory
        obs_forget = execute_forget_memory(self.store, {
            "record_id": rid, "reason": "Master corrected himself",
        })
        self.assertTrue(obs_forget.startswith("[FORGOTTEN]"))

        # 4) Subsequent retrieval no longer surfaces it
        collected2 = {"target_entity": "Rosm", "keywords": [],
                      "search_query": "Master's favorite color",
                      "records": [], "top1_score": 0.0, "error": None}
        collected2 = _attach_user_notes(
            collected2, self.store, "Master mentioned today his favorite color is teal",
        )
        block2 = _format_memory_records_block(collected2)
        self.assertNotIn(f"[Note #{rid}]", block2)


# ============================================================
# Verifier helpers (remember/forget evidence detection +
# recent note_id scan)
# ============================================================
from types import SimpleNamespace

from eva_verifier_logic import (  # noqa: E402
    current_turn_has_remember_evidence,
    current_turn_has_forget_evidence,
    find_recent_note_id,
    REASON_POLICY,
)


def _mk_turn(steps):
    """Build a minimal turn-shaped object for verifier helpers."""
    return SimpleNamespace(assistant_steps=list(steps))


def _mk_agent(current_turn=None, history=None, turn_evidence=None):
    """Build a stub agent. R-4 (2026-05-13)：current_turn_has_*_evidence
    现在读 turn_evidence ledger，不再读 history_manager.current_turn 字符串。
    保留 history_manager 字段以兼容仍 grep tool_step 的其他测试 helper
    （find_recent_note_id 等仍走 history 路径）。
    """
    hm = SimpleNamespace(
        current_turn=current_turn,
        history=list(history or []),
    )
    return SimpleNamespace(history_manager=hm, turn_evidence=turn_evidence or [])


class TestRememberForgetEvidence(unittest.TestCase):
    """R-4：current_turn_has_remember/forget_evidence 改为读 turn_evidence
    ledger 中的 source="notes_write"/"notes_delete" 条目。"""

    def _ev(self, source, **kw):
        from eva_history import TurnEvidence
        return TurnEvidence(source=source, **kw)

    def test_remember_evidence_true_when_ledger_has_notes_write(self):
        agent = _mk_agent(turn_evidence=[self._ev("notes_write", record_ref="abc12345")])
        self.assertTrue(current_turn_has_remember_evidence(agent))

    def test_remember_evidence_false_when_no_notes_write(self):
        agent = _mk_agent(turn_evidence=[])
        self.assertFalse(current_turn_has_remember_evidence(agent))

    def test_remember_evidence_false_when_only_other_source(self):
        agent = _mk_agent(turn_evidence=[self._ev("memory")])
        self.assertFalse(current_turn_has_remember_evidence(agent))

    def test_remember_evidence_false_when_no_current_turn(self):
        # 即便 history_manager.current_turn 为 None，只要 ledger 没有
        # notes_write，结果就是 False（与旧行为等价）。
        agent = _mk_agent(current_turn=None, turn_evidence=[])
        self.assertFalse(current_turn_has_remember_evidence(agent))

    def test_forget_evidence_true_when_ledger_has_notes_delete(self):
        agent = _mk_agent(turn_evidence=[self._ev("notes_delete", record_ref="abc12345")])
        self.assertTrue(current_turn_has_forget_evidence(agent))

    def test_forget_evidence_false_when_no_notes_delete(self):
        agent = _mk_agent(turn_evidence=[self._ev("notes_write")])
        self.assertFalse(current_turn_has_forget_evidence(agent))


class TestFindRecentNoteId(unittest.TestCase):
    def test_finds_id_in_current_turn(self):
        turn = _mk_turn([
            {"role": "tool", "content":
                "### [MEMORY MODULE DATA for 'Rosm'] ###\n"
                ">>> SAVED NOTES ...\n"
                "  Note 1 [Note #cafef00d] [Subject: Rosm] [Topic: Pet]: ...\n"},
        ])
        agent = _mk_agent(current_turn=turn)
        self.assertEqual(find_recent_note_id(agent), "cafef00d")

    def test_finds_id_in_recent_history_when_current_turn_blank(self):
        prior = _mk_turn([
            {"role": "tool", "content":
                "Note 1 [Note #1f2864e5] [Subject: Rosm] [Topic: Pet]: ..."},
        ])
        current = _mk_turn([
            {"role": "assistant", "content": "<think>...</think><|answer|>ok<|end_react|>"},
        ])
        agent = _mk_agent(current_turn=current, history=[prior])
        self.assertEqual(find_recent_note_id(agent), "1f2864e5")

    def test_returns_most_recent_when_multiple(self):
        turn = _mk_turn([
            {"role": "tool", "content": "[Note #aaaaaaaa] earlier"},
            {"role": "assistant", "content": "blah"},
            {"role": "tool", "content": "[Note #bbbbbbbb] later"},
        ])
        agent = _mk_agent(current_turn=turn)
        # Reverse iteration → finds the LATER tool step first.
        self.assertEqual(find_recent_note_id(agent), "bbbbbbbb")

    def test_returns_none_when_no_note_tag(self):
        turn = _mk_turn([
            {"role": "tool", "content":
                "### [MEMORY MODULE DATA for 'Eva'] ###\n"
                "Record 1 [Lore] [Subject: Eva] [Topic: Toy]: ..."},
        ])
        agent = _mk_agent(current_turn=turn)
        self.assertIsNone(find_recent_note_id(agent))

    def test_returns_none_when_no_history_at_all(self):
        agent = _mk_agent(current_turn=None, history=[])
        self.assertIsNone(find_recent_note_id(agent))

    def test_max_turns_limits_history_lookback(self):
        old = _mk_turn([{"role": "tool",
                         "content": "[Note #00000000] very old"}])
        # Pad with 5 turns so old falls outside max_turns=2.
        pads = [_mk_turn([{"role": "tool", "content": "neutral"}])
                for _ in range(5)]
        agent = _mk_agent(current_turn=None, history=[old] + pads)
        self.assertIsNone(find_recent_note_id(agent, max_turns=2))

    def test_live_store_fallback_when_history_misses(self):
        """When the model answered prior turns from in-context recall,
        no [Note #...] tag ever got rendered to tool_outputs. The
        live-store fallback should kick in and find the note by
        cosine match against latest_user_text."""
        # Empty history (no tags anywhere).
        tmpdir = Path(tempfile.mkdtemp(prefix="evanotes_fallback_"))
        try:
            store = NotesStore(
                root=tmpdir, encoder=fake_encoder, dim=DIM,
                session_id="fallback_t",
            )
            rid = store.add(
                vector_text="forget about the cat",  # exact-match query → cosine ~1.0
                content="Master adopted an orange tabby cat named Peach.",
                entity="Rosm", topic="Pet", keywords="cat, orange",
            )
            agent = SimpleNamespace(
                history_manager=SimpleNamespace(current_turn=None, history=[]),
                memory_state={"notes_store": store},
            )
            # No latest_user_text → still returns None (refuse to guess).
            self.assertIsNone(find_recent_note_id(agent))
            # With latest_user_text matching the note → fallback hits.
            self.assertEqual(
                find_recent_note_id(
                    agent, latest_user_text="forget about the cat",
                ),
                rid,
            )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_history_scan_takes_precedence_over_fallback(self):
        """If a [Note #...] tag is in history AND a different note lives
        in the store, history wins (fast path is authoritative).
        """
        tmpdir = Path(tempfile.mkdtemp(prefix="evanotes_precedence_"))
        try:
            store = NotesStore(
                root=tmpdir, encoder=fake_encoder, dim=DIM,
                session_id="precedence_t",
            )
            store.add(vector_text="A", content="A.",
                      entity="Eva", topic="Toy", keywords="a")
            turn_with_tag = _mk_turn([
                {"role": "tool", "content": "[Note #11111111] from history"},
            ])
            agent = SimpleNamespace(
                history_manager=SimpleNamespace(
                    current_turn=turn_with_tag, history=[]),
                memory_state={"notes_store": store},
            )
            # History tag wins even though the live store has a different id.
            self.assertEqual(
                find_recent_note_id(
                    agent, latest_user_text="anything",
                ),
                "11111111",
            )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_fallback_with_no_notes_store_returns_none(self):
        """latest_user_text is given but agent has no notes_store — bail."""
        agent = _mk_agent(current_turn=None, history=[])
        agent.memory_state = {}  # no notes_store key
        self.assertIsNone(find_recent_note_id(
            agent, latest_user_text="forget about the cat",
        ))

    def test_fallback_with_empty_store_returns_none(self):
        """notes_store exists but has zero live notes — bail."""
        tmpdir = Path(tempfile.mkdtemp(prefix="evanotes_emptystore_"))
        try:
            store = NotesStore(
                root=tmpdir, encoder=fake_encoder, dim=DIM,
                session_id="empty_t",
            )
            agent = SimpleNamespace(
                history_manager=SimpleNamespace(current_turn=None, history=[]),
                memory_state={"notes_store": store},
            )
            self.assertIsNone(find_recent_note_id(
                agent, latest_user_text="forget about the cat",
            ))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestReasonPolicyRegistration(unittest.TestCase):
    # 2026-05-13 Advisor cutover: severity 从 "hard" 降为 "soft"，fix 从
    # "inject_tool" 改为 "canned_fallback"。Advisor 现在在 prompt 顶层告诉
    # Eva 该调哪个工具，事后侦测+重新注入工具的需求被取代。这两条 reason
    # 仍注册在 REASON_POLICY 里——但只做 telemetry 告警，不再触发硬失败。
    def test_explicit_forget_reason_registered(self):
        self.assertIn("explicit_forget_request_not_handled", REASON_POLICY)
        entry = REASON_POLICY["explicit_forget_request_not_handled"]
        self.assertEqual(entry["severity"], "soft")
        self.assertEqual(entry["fix"], "canned_fallback")
        self.assertTrue(entry["canned"])

    def test_explicit_remember_reason_registered(self):
        self.assertIn("explicit_remember_request_not_handled", REASON_POLICY)
        entry = REASON_POLICY["explicit_remember_request_not_handled"]
        self.assertEqual(entry["severity"], "soft")
        self.assertEqual(entry["fix"], "canned_fallback")
        self.assertTrue(entry["canned"])


# ============================================================
# Heuristic RememberThis param extractor
# ============================================================
from eva_verifier_logic import (  # noqa: E402
    extract_remember_params_from_user_text,
)


class TestExtractRememberParams(unittest.TestCase):
    def test_classic_remember_this_form(self):
        params = extract_remember_params_from_user_text(
            "Eva, remember this: I just adopted an orange tabby cat named Peach."
        )
        self.assertIsNotNone(params)
        self.assertEqual(params["entity"], "Rosm")  # first-person → Rosm
        self.assertEqual(params["topic"], "Pet")     # "cat" → Pet
        # First-person normalized to Master
        self.assertIn("Master", params["content"])
        self.assertNotIn(" I ", " " + params["content"] + " ")
        self.assertIn("Peach", params["content"])
        # Keywords contain salient nouns
        self.assertIn("peach", params["keywords"])
        self.assertIn("cat", params["keywords"])

    def test_remember_that_form(self):
        params = extract_remember_params_from_user_text(
            "remember that the meeting is at 3pm tomorrow"
        )
        self.assertIsNotNone(params)
        self.assertIn("meeting", params["content"])
        # No first-person → Shared (or default)
        self.assertEqual(params["entity"], "Shared")

    def test_dont_forget_form(self):
        params = extract_remember_params_from_user_text(
            "Don't forget that I'm vegetarian and dislike onions."
        )
        self.assertIsNotNone(params)
        self.assertEqual(params["entity"], "Rosm")
        # I'm → Master is
        self.assertIn("Master is", params["content"])

    def test_chinese_remember_form(self):
        params = extract_remember_params_from_user_text(
            "记一下，我刚买了一只叫桃桃的橘猫"
        )
        self.assertIsNotNone(params)
        self.assertEqual(params["entity"], "Rosm")  # 我 → first-person
        self.assertEqual(params["topic"], "Pet")     # 猫 → Pet
        self.assertIn("桃桃", params["content"])

    def test_empty_input_returns_none(self):
        self.assertIsNone(extract_remember_params_from_user_text(""))
        self.assertIsNone(extract_remember_params_from_user_text(None))
        self.assertIsNone(extract_remember_params_from_user_text("   "))

    def test_only_preamble_returns_none(self):
        # "remember this:" with nothing after should yield empty content → None
        self.assertIsNone(extract_remember_params_from_user_text("remember this:"))
        self.assertIsNone(extract_remember_params_from_user_text("don't forget that"))

    def test_topic_guess_food(self):
        params = extract_remember_params_from_user_text(
            "Note this down: I really enjoy lemon cake"
        )
        self.assertIn(params["topic"], {"Food", "Likes"})  # both reasonable

    def test_topic_guess_birthday(self):
        params = extract_remember_params_from_user_text(
            "Please remember that my birthday is July 7th"
        )
        self.assertEqual(params["topic"], "Birthday")

    def test_keywords_dedup_and_capped(self):
        params = extract_remember_params_from_user_text(
            "Remember: cat cat cat dog dog fish bird hamster turtle parrot snake"
        )
        kw_list = [k.strip() for k in params["keywords"].split(",")]
        self.assertEqual(len(kw_list), len(set(kw_list)))  # dedup
        self.assertLessEqual(len(kw_list), 6)               # capped

    def test_no_first_person_yields_shared_entity(self):
        params = extract_remember_params_from_user_text(
            "Remember that the meeting room got renovated last week."
        )
        self.assertEqual(params["entity"], "Shared")
        # No first-person normalization should happen
        self.assertNotIn("Master", params["content"])


# ============================================================
# Slot extractor: pet/animal name queries must not trigger
# `full_name` slot (it's for human full names like Rosmarinus, not
# pet names like Peach). Same negative-context pattern as the existing
# place/venue suppression in eva_memory_legacy._detect_requested_slot_fields.
# ============================================================
from eva_memory_legacy import _detect_requested_slot_fields  # noqa: E402


class TestSlotDetectionPetSuppression(unittest.TestCase):
    """User asks about a pet's name, slot extractor used to pick 'name'
    → full_name → inject `[MISSING SLOTS]: full_name — answer must say
    these are not recorded.` contradicting any saved note. Fix: extend
    full_name_blocked to cover pet/animal contexts."""

    def test_name_of_cat_does_not_request_full_name(self):
        slots = _detect_requested_slot_fields("what was the name of that cat?")
        self.assertNotIn("full_name", slots)

    def test_cats_name_does_not_request_full_name(self):
        slots = _detect_requested_slot_fields("what's the cat's name?")
        self.assertNotIn("full_name", slots)

    def test_pet_name_does_not_request_full_name(self):
        slots = _detect_requested_slot_fields("can you remind me of the pet name?")
        self.assertNotIn("full_name", slots)

    def test_its_name_does_not_request_full_name(self):
        slots = _detect_requested_slot_fields(
            "Did I tell you about my new dog? What was its name?"
        )
        self.assertNotIn("full_name", slots)

    def test_chinese_cat_name_does_not_request_full_name(self):
        slots = _detect_requested_slot_fields("我家那只猫的名字是什么？")
        self.assertNotIn("full_name", slots)

    # ---- preserved positives (must still detect full_name) ----
    def test_your_name_still_requests_full_name(self):
        """Person-name queries must still trigger full_name slot extraction."""
        slots = _detect_requested_slot_fields("what's your full name?")
        self.assertIn("full_name", slots)

    def test_my_real_name_still_requests_full_name(self):
        slots = _detect_requested_slot_fields("do you remember my real name?")
        self.assertIn("full_name", slots)

    def test_existing_place_block_preserved(self):
        """Pre-existing block: 'name of the museum' must not trigger full_name."""
        slots = _detect_requested_slot_fields(
            "what was the name of that museum we visited?"
        )
        self.assertNotIn("full_name", slots)


# ============================================================
# Missing-slot warning suppression when notes present
# ============================================================
from eva_memory_legacy import memory_block_has_notes  # noqa: E402


class TestMemoryBlockHasNotes(unittest.TestCase):
    def test_true_when_marker_present(self):
        obs = (
            "### [MEMORY MODULE DATA for 'Rosm'] ###\n"
            "Record 1 [Lore] [Subject: Shared] ...\n\n"
            ">>> SAVED NOTES (pass record_id ...) <<<\n"
            "  Note 1 [Note #abc12345] ... \n"
            ">>> END SAVED NOTES <<<"
        )
        self.assertTrue(memory_block_has_notes(obs))

    def test_false_when_only_lore_records(self):
        obs = (
            "### [MEMORY MODULE DATA for 'Eva'] ###\n"
            "Record 1 [Lore] [Subject: Eva] [Topic: Toy]: cuddly bunny ...\n"
        )
        self.assertFalse(memory_block_has_notes(obs))

    def test_false_when_obs_empty_or_none(self):
        self.assertFalse(memory_block_has_notes(""))
        self.assertFalse(memory_block_has_notes(None))

    def test_false_when_no_relevant_records_message(self):
        obs = (
            "### [MEMORY MODULE DATA for 'Eva'] ###\n"
            "No relevant records found.\n"
        )
        self.assertFalse(memory_block_has_notes(obs))


class TestNotesSuppressMissingSlotWarning(unittest.TestCase):
    """Reproduce the failure mode: a saved note contains the answer the
    user asked for, but lore-corpus slot extraction can't see it. Without
    suppression the model gets a [MISSING SLOTS] directive that contradicts
    the note. With suppression, the note surfaces alone and the model can
    read it.
    """

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="evanotes_suppress_"))
        self.store = NotesStore(
            root=self.tmpdir, encoder=fake_encoder, dim=DIM,
            session_id="suppress_t",
        )

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _build_obs_with_notes_section(self):
        """Build a MemorySearch-shaped observation containing a note."""
        rid = self.store.add(
            vector_text="Master adopted an orange tabby cat named Peach",
            content="Master just adopted an orange tabby cat named Peach.",
            entity="Rosm", topic="Pet", keywords="cat, orange, peach",
        )
        from eva_memory_legacy import (
            _attach_user_notes, _format_memory_records_block,
        )
        collected = {
            "target_entity": "Rosm", "keywords": [],
            "search_query": "name of cat",
            "records": [{
                "content": "If Eva had a pet, she's certain it'd be a cat.",
                "entity": "Shared", "category": "Lore", "topic": "Pet",
                "rerank_score": 4.5, "low_confidence": False,
            }],
            "top1_score": 4.5, "error": None,
        }
        collected = _attach_user_notes(
            collected, self.store, "Master adopted an orange tabby cat named Peach",
        )
        return rid, _format_memory_records_block(collected)

    def test_obs_contains_notes_section(self):
        _, obs = self._build_obs_with_notes_section()
        self.assertTrue(memory_block_has_notes(obs))

    def test_suppression_logic_drops_missing_slot_warning(self):
        """Mirror the dispatch-site conditional. When obs has notes,
        a non-empty slot_note must NOT be appended."""
        _, obs = self._build_obs_with_notes_section()
        slot_note = (
            "\n[MISSING SLOTS]: full_name — answer must say "
            "these are not recorded."
        )
        # Emulate the dispatch-site guard (eva_core.py / eva_verifier_logic.py)
        if slot_note and not memory_block_has_notes(obs):
            obs = obs + slot_note
        self.assertNotIn("[MISSING SLOTS]", obs)

    def test_suppression_does_not_apply_when_no_notes(self):
        """When the observation has only lore records, the warning
        should still be appended — the suppression must not blanket-fire."""
        prod_only_obs = (
            "### [MEMORY MODULE DATA for 'Rosm'] ###\n"
            "Record 1 [Lore] [Subject: Rosm] [Topic: Identity]: ...\n"
        )
        slot_note = (
            "\n[MISSING SLOTS]: full_name — answer must say "
            "these are not recorded."
        )
        if slot_note and not memory_block_has_notes(prod_only_obs):
            prod_only_obs = prod_only_obs + slot_note
        self.assertIn("[MISSING SLOTS]", prod_only_obs)


if __name__ == "__main__":
    unittest.main(verbosity=2)
