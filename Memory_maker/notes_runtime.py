"""notes_runtime.py — User notes memory store (production).

Provides `NotesStore`: a runtime-mutable, FAISS-backed memory namespace
parallel to the production lore corpus at `Memory/`. Used by the
`RememberThis` / `ForgetMemory` tools so Eva can persist user-told facts
across the session, surface them in subsequent `MemorySearch` results,
and forget them on request.

Two-layer memory architecture:
  - `Memory/`     — hand-curated lore corpus, written by Memory_maker
                    offline tools (`rewrite_memory.py` → `Memory.py`).
                    Read-only at runtime.
  - `Notes/`      — runtime user-told facts, written by RememberThis at
                    inference time. Tombstoned by ForgetMemory.

设计要点：

  - JSONL 是真源；任何写操作先追加 JSONL，再更新内存中的 index/contents/metas，
    最后 dump 到磁盘。
  - 删除走 tombstone（`meta.deleted = True`），FAISS 索引不动；search 时过滤。
  - `compact()` 在 tombstone 占比过高时从存活记录重建 index。
  - 共用主库的 mpnet encoder（外部传入），避免双载模型；测试可注入 fake encoder。
  - 单进程使用，不加文件锁。
"""
from __future__ import annotations

import json
import re
import shutil
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable, Optional

import faiss
import numpy as np

# 所有时间走 eva_time.get_current_time。notes_runtime 在 Memory_maker 子模块下，
# 要先把项目根加到 sys.path 才能 import。
import sys as _sys
import os as _os
_PROJECT_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _PROJECT_ROOT not in _sys.path:
    _sys.path.insert(0, _PROJECT_ROOT)
from eva_time import get_current_time


# Canonical UTC 时间戳生成器（notes created_at / deleted_at / audit 都用这个）。
# 跨时区可移植，故强制 UTC，不受 EVA_TIMEZONE 影响。
_ISO = lambda: get_current_time("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


# R-7 (2026-05-13): 解析 + 规范化 event 字段。这些 sanitize 函数都不抛错，
# 不合法的输入静默丢弃——R-7 设计原则：event 字段是 hint，错了不破坏写入。
_DATE_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
_TIME_RE = re.compile(r"^(\d{1,2}):(\d{2})(?::\d{2})?$")


def _normalize_event_date(s: str) -> str:
    """返回规范化的 YYYY-MM-DD 字符串，或空串（无效输入）。"""
    if not isinstance(s, str):
        return ""
    s = s.strip()
    if not s:
        return ""
    m = _DATE_RE.match(s)
    if not m:
        return ""
    try:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        # 用 datetime 校验合法性（处理 2026-02-30 等）
        datetime(year=y, month=mo, day=d)
        return f"{y:04d}-{mo:02d}-{d:02d}"
    except ValueError:
        return ""


def _normalize_event_time(s: str) -> str:
    """返回规范化的 HH:MM 字符串，或空串。"""
    if not isinstance(s, str):
        return ""
    s = s.strip()
    if not s:
        return ""
    m = _TIME_RE.match(s)
    if not m:
        return ""
    try:
        h, mi = int(m.group(1)), int(m.group(2))
        if not (0 <= h <= 23 and 0 <= mi <= 59):
            return ""
        return f"{h:02d}:{mi:02d}"
    except ValueError:
        return ""


def _normalize_event_type(s: str) -> str:
    """event_type 是自由 string，仅做 trim + 长度限制。"""
    if not isinstance(s, str):
        return ""
    s = s.strip().lower()
    if not s or len(s) > 40:
        return ""
    return s


def _normalize_expires_at(s: str) -> str:
    """ISO8601 timestamp 或 YYYY-MM-DD（视为当天 00:00 UTC）；空 / 无效返回空。"""
    if not isinstance(s, str):
        return ""
    s = s.strip()
    if not s:
        return ""
    # 接受 YYYY-MM-DD 短形式：补到日期 + UTC 凌晨
    if _DATE_RE.match(s):
        return s + "T00:00:00Z"
    # 严格 ISO8601：尝试 datetime.fromisoformat（Python 3.11+ 接受 Z 后缀）
    try:
        # 兼容 "Z" 结尾
        probe = s.replace("Z", "+00:00")
        datetime.fromisoformat(probe)
        return s
    except ValueError:
        return ""


def _build_event_dict(*, event_date: str, event_time: str,
                       event_type: str, expires_at: str) -> dict:
    """把四个 event_* 字段聚成 meta.event sub-dict。所有字段空 → 返回空 dict。"""
    d = _normalize_event_date(event_date)
    t = _normalize_event_time(event_time)
    et = _normalize_event_type(event_type)
    ex = _normalize_expires_at(expires_at)
    out = {}
    if d:
        out["date"] = d
    if t:
        out["time"] = t
    if et:
        out["type"] = et
    if ex:
        out["expires_at"] = ex
    return out


# ============================================================
# Encoder helper
# ============================================================
EncoderFn = Callable[[list[str]], np.ndarray]
"""Encoder signature: list[str] -> (n, dim) float32 with L2-normalized rows."""


def _default_encoder_factory(model_name: str = "sentence-transformers/all-mpnet-base-v2") -> EncoderFn:
    """Lazy-load mpnet on first call; reuse the same model afterwards."""
    from sentence_transformers import SentenceTransformer  # heavy import
    model = SentenceTransformer(model_name)

    def _encode(texts: list[str]) -> np.ndarray:
        emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(emb, dtype="float32")
    return _encode


# ============================================================
# Store
# ============================================================
class NotesStore:
    """FAISS-backed user-notes store. The model writes here via
    RememberThis and tombstones here via ForgetMemory; retrieval
    surfaces live notes in a dedicated section of MemorySearch output.
    """

    INDEX_FILE = "notes.index"
    CONTENT_FILE = "notes_content.json"
    META_FILE = "notes_meta.json"
    JSONL_FILE = "notes.jsonl"
    AUDIT_FILE = "audit.log"

    def __init__(
        self,
        root: Path | str,
        encoder: Optional[EncoderFn] = None,
        dim: int = 768,
        session_id: Optional[str] = None,
    ):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.dim = dim
        self.encoder = encoder  # may be None until first _encode()
        self.session_id = session_id or get_current_time().strftime("%Y%m%d_%H%M%S")
        self._load_or_init()

    # ---------- paths ----------
    @property
    def index_path(self) -> Path: return self.root / self.INDEX_FILE
    @property
    def content_path(self) -> Path: return self.root / self.CONTENT_FILE
    @property
    def meta_path(self) -> Path: return self.root / self.META_FILE
    @property
    def jsonl_path(self) -> Path: return self.root / self.JSONL_FILE
    @property
    def audit_path(self) -> Path: return self.root / self.AUDIT_FILE

    # ---------- bootstrap ----------
    def _load_or_init(self) -> None:
        if self.index_path.exists() and self.meta_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            with open(self.content_path, encoding="utf-8") as f:
                self.contents: list[str] = json.load(f)
            with open(self.meta_path, encoding="utf-8") as f:
                self.metas: list[dict] = json.load(f)
            if self.index.ntotal != len(self.contents) or len(self.contents) != len(self.metas):
                raise RuntimeError(
                    f"corrupt notes store: index.ntotal={self.index.ntotal}, "
                    f"contents={len(self.contents)}, metas={len(self.metas)}"
                )
            self.dim = self.index.d
        else:
            self.index = faiss.IndexFlatIP(self.dim)
            self.contents = []
            self.metas = []
            self._dump_runtime()
        # R-2.1 (2026-05-13): session-level LRU of recently added note ids
        # (newest-first). 抗 history_manager.history compaction —— 当
        # history 摘要化把 [REMEMBERED] tool obs 丢了之后，runtime 仍能在
        # NotesStore 自身的内存表里查"最近加的那条 note"作为 ForgetMemory
        # auto-correct 的兜底信号。
        # 不持久化到 JSONL：只是 session 级 anaphoric forget 用的 hint，
        # 跨会话语义已经被 history compaction 失效，重新加载也没意义。
        self.recent_adds: list[str] = []
        self._RECENT_ADDS_CAP = 20

    # ---------- public API ----------
    def add(
        self,
        vector_text: str,
        content: str,
        entity: str,
        topic: str,
        keywords: str | list[str],
        category: str = "Lore",
        secondary_topics: Optional[list[str]] = None,
        participants: Optional[list[str]] = None,
        slot_values: Optional[dict] = None,
        event_date: str = "",
        event_time: str = "",
        event_type: str = "",
        expires_at: str = "",
    ) -> str:
        """Add a note. Returns the 8-char note_id.

        R-1 (2026-05-13)：新增 `slot_values` 可选字段。模型 SFT 训练后
        可以在 RememberThis 调用时填入结构化 slot——例如
        ``slot_values={"pet_name": "Peach"}``——runtime 把它存进 meta，
        后续 MemorySearch 走 slot 路径直接命中，不需要 inference-time 跑正则。

        R-7 (2026-05-13)：新增 event-* 字段，给"日程类事实"做结构化标注。
        与 slot_values 互补——slot_values 装无时间维度的事实（pet_name /
        diet_restriction），event_* 装时间相关的（meeting / appointment /
        birthday 当作事件时）：

          event_date: ISO8601 日期 "YYYY-MM-DD"，模型从 "next Monday" 等
                      相对表述里解析出绝对日期后填入。
          event_time: "HH:MM" 24h 制。
          event_type: 自由 string，建议短小——"meeting" / "appointment" /
                      "deadline" / "birthday" 等。
          expires_at: ISO8601 timestamp。一旦过期，`expire_stale()` 会自动
                      tombstone 这条 note。

        这些字段统一落 meta.event 子 dict（保持 meta 顶层干净）。
        """
        if not vector_text or not content:
            raise ValueError("vector_text and content are required")
        if entity not in ("Eva", "Rosm", "Shared"):
            raise ValueError(f"entity must be Eva|Rosm|Shared, got {entity!r}")
        if category not in ("Lore", "Event"):
            raise ValueError(f"category must be Lore|Event, got {category!r}")

        kw_list = (
            [k.strip() for k in keywords.split(",") if k.strip()]
            if isinstance(keywords, str) else list(keywords)
        )

        note_id = uuid.uuid4().hex[:8]
        meta = {
            "entity": entity,
            "category": category,
            "topic": topic,
            "participants": participants or (
                [entity] if entity in ("Eva", "Rosm") else ["Eva", "Rosm"]
            ),
            "keywords": kw_list,
            "note_id": note_id,
            "origin": f"session_{self.session_id}",
            "created_at": _ISO(),
            "deleted": False,
        }
        if secondary_topics:
            meta["secondary_topics"] = list(secondary_topics)
        # R-1: slot_values 只 sanitize 成 {str: str}，避免不合规嵌套
        if slot_values and isinstance(slot_values, dict):
            cleaned_slots = {}
            for k, v in slot_values.items():
                if isinstance(k, str) and isinstance(v, str) and v.strip():
                    cleaned_slots[k.strip()] = v.strip()
            if cleaned_slots:
                meta["slot_values"] = cleaned_slots

        # R-7: event 字段统一落 meta.event 子 dict。每个字段都 sanitize。
        event_dict = _build_event_dict(
            event_date=event_date, event_time=event_time,
            event_type=event_type, expires_at=expires_at,
        )
        if event_dict:
            # 如果传了 participants 且是 list，event 也带上（不污染顶层 meta.participants）
            if participants and isinstance(participants, list):
                p_clean = [p.strip() for p in participants if isinstance(p, str) and p.strip()]
                if p_clean:
                    event_dict["participants"] = p_clean
            meta["event"] = event_dict

        # Encode + add to FAISS first; if encoding fails, nothing else mutates.
        emb = self._encode([vector_text])
        self._assert_dim(emb.shape[1])
        self.index.add(emb)
        self.contents.append(content)
        self.metas.append(meta)

        # Persist source-of-truth JSONL (same schema as 8.memory_optimized.jsonl
        # plus per-note runtime fields).
        record = {
            "vector_text": vector_text,
            "content": content,
            "meta": dict(meta),  # copy
        }
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        self._dump_runtime()
        self._audit("ADD", note_id, f"entity={entity} topic={topic}")
        # R-2.1: 维护 session LRU. newest at index 0.
        self.recent_adds.insert(0, note_id)
        del self.recent_adds[self._RECENT_ADDS_CAP:]
        return note_id

    def tombstone(self, note_id: str, reason: str = "") -> bool:
        """Mark note as deleted. Returns True if found and tombstoned."""
        for m in self.metas:
            if m.get("note_id") == note_id and not m.get("deleted"):
                m["deleted"] = True
                m["deleted_reason"] = reason or "(no reason given)"
                m["deleted_at"] = _ISO()
                self._dump_runtime()
                self._audit("DELETE", note_id, reason or "(no reason)")
                # R-2.1: 同步剔出 LRU 避免悬挂引用
                try:
                    self.recent_adds.remove(note_id)
                except ValueError:
                    pass
                return True
        return False

    # ---------- R-7: 结构化查询 API ----------
    def search_by_date(self, start_iso: str, end_iso: str = "") -> list[dict]:
        """返回 event.date 落在 [start, end] 闭区间内的 live note。

        end_iso 为空时退化为单日查询（== start_iso）。
        日期不合法直接返回空列表（不抛错）。
        """
        s = _normalize_event_date(start_iso)
        e = _normalize_event_date(end_iso) if end_iso else s
        if not s or not e:
            return []
        out = []
        for idx, m in enumerate(self.metas):
            if m.get("deleted"):
                continue
            ev = m.get("event") or {}
            d = ev.get("date") or ""
            if s <= d <= e:  # ISO 8601 字典序 == 时序，安全
                out.append({
                    "note_id": m.get("note_id"),
                    "content": self.contents[idx],
                    "meta": m,
                })
        return out

    def search_by_event_type(self, event_type: str) -> list[dict]:
        """返回 event.type == event_type 的 live note。"""
        et = _normalize_event_type(event_type)
        if not et:
            return []
        out = []
        for idx, m in enumerate(self.metas):
            if m.get("deleted"):
                continue
            ev = m.get("event") or {}
            if (ev.get("type") or "") == et:
                out.append({
                    "note_id": m.get("note_id"),
                    "content": self.contents[idx],
                    "meta": m,
                })
        return out

    def expire_stale(self, now=None, *, reason: str = "expired") -> list[str]:
        """Tombstone 所有 event.expires_at < now 的 live note。

        Args:
            now: datetime 对象（默认 datetime.now(UTC)）。
            reason: 写进 audit log 的删除原因。
        Returns:
            被 tombstone 的 note_id 列表。
        """
        if now is None:
            now = get_current_time("UTC")
        # 把 now 规范成 ISO8601 UTC 字符串再做字典序比较——避免 naive vs aware
        # 比较失败。
        if now.tzinfo is None:
            now_iso = now.strftime("%Y-%m-%dT%H:%M:%SZ")
        else:
            now_iso = now.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        expired = []
        for m in self.metas:
            if m.get("deleted"):
                continue
            ev = m.get("event") or {}
            exp = ev.get("expires_at") or ""
            if not exp:
                continue
            # ISO8601 字符串可直接字典序比较（前提是都 UTC）
            exp_norm = exp.replace("Z", "Z")  # noop, 强调
            if exp_norm < now_iso:
                expired.append(m.get("note_id"))
        for nid in expired:
            self.tombstone(nid, reason=reason)
        return expired
    # ---------- /R-7 ----------

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        """Search live (non-tombstoned) notes.

        Returns list of {score, idx, note_id, content, meta} sorted by score desc.
        """
        if self.index.ntotal == 0:
            return []
        emb = self._encode([query])
        # Over-fetch to allow filtering; cap at ntotal.
        actual_k = min(max(top_k * 2, top_k + 5), self.index.ntotal)
        D, I = self.index.search(emb, actual_k)
        out: list[dict] = []
        for score, idx in zip(D[0], I[0]):
            if idx == -1:
                continue
            meta = self.metas[idx]
            if meta.get("deleted"):
                continue
            out.append({
                "score": float(score),
                "idx": int(idx),
                "note_id": meta["note_id"],
                "content": self.contents[idx],
                "meta": meta,
            })
            if len(out) >= top_k:
                break
        return out

    def status(self) -> dict:
        """Quick counts for CLI / debugging."""
        live = sum(1 for m in self.metas if not m.get("deleted"))
        deleted = len(self.metas) - live
        return {
            "session_id": self.session_id,
            "total": len(self.metas),
            "live": live,
            "deleted": deleted,
            "index_ntotal": int(self.index.ntotal),
            "root": str(self.root),
        }

    def list_notes(self, include_deleted: bool = False) -> list[dict]:
        out = []
        for content, meta in zip(self.contents, self.metas):
            if not include_deleted and meta.get("deleted"):
                continue
            out.append({
                "note_id": meta["note_id"],
                "entity": meta.get("entity"),
                "topic": meta.get("topic"),
                "deleted": bool(meta.get("deleted")),
                "content_preview": content[:80],
            })
        return out

    def compact(self) -> dict:
        """Rebuild FAISS index from live notes, dropping tombstoned ones.

        Re-encodes vector_text from the JSONL source (the runtime in-memory
        list doesn't store vector_text — only content). Returns
        {before, after, dropped}.
        """
        before = self.index.ntotal
        # Read JSONL and pair up by note_id
        by_id = {}
        with open(self.jsonl_path, encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                rec = json.loads(ln)
                rid = rec.get("meta", {}).get("note_id")
                if rid:
                    by_id[rid] = rec  # last occurrence wins (no-op for our flow)

        live_records = []
        for m, c in zip(self.metas, self.contents):
            if m.get("deleted"):
                continue
            rid = m["note_id"]
            jrec = by_id.get(rid)
            if not jrec:
                # JSONL missing — skip but don't crash; audit it.
                self._audit("COMPACT_SKIP", rid, "missing in JSONL")
                continue
            live_records.append((jrec["vector_text"], c, m))

        new_index = faiss.IndexFlatIP(self.dim)
        new_contents: list[str] = []
        new_metas: list[dict] = []
        if live_records:
            vtexts = [r[0] for r in live_records]
            emb = self._encode(vtexts)
            self._assert_dim(emb.shape[1])
            new_index.add(emb)
            for _, c, m in live_records:
                new_contents.append(c)
                new_metas.append(m)

        self.index = new_index
        self.contents = new_contents
        self.metas = new_metas
        self._dump_runtime()
        # Rewrite JSONL too (drop tombstoned source rows)
        with open(self.jsonl_path, "w", encoding="utf-8") as f:
            for vt, c, m in live_records:
                f.write(json.dumps(
                    {"vector_text": vt, "content": c, "meta": dict(m)},
                    ensure_ascii=False,
                ) + "\n")

        after = self.index.ntotal
        self._audit("COMPACT", "-", f"before={before} after={after}")
        return {"before": before, "after": after, "dropped": before - after}

    def discard(self) -> None:
        """Wipe runtime files. Audit log is renamed (kept for forensics)."""
        if self.audit_path.exists():
            stamp = get_current_time().strftime("%Y%m%d_%H%M%S")
            archived = self.root / f"audit_{stamp}.log"
            self.audit_path.rename(archived)
        for p in (self.index_path, self.content_path, self.meta_path, self.jsonl_path):
            if p.exists():
                p.unlink()
        # Reset in-memory state
        self.index = faiss.IndexFlatIP(self.dim)
        self.contents = []
        self.metas = []
        self._dump_runtime()
        self._audit("DISCARD", "-", "store wiped")

    # ---------- internals ----------
    def _encode(self, texts: list[str]) -> np.ndarray:
        if self.encoder is None:
            self.encoder = _default_encoder_factory()
        emb = self.encoder(texts)
        emb = np.asarray(emb, dtype="float32")
        if emb.ndim != 2:
            raise ValueError(f"encoder returned shape {emb.shape}, expected 2D")
        return emb

    def _assert_dim(self, d: int) -> None:
        if d != self.dim:
            raise ValueError(
                f"encoder dim mismatch: store dim={self.dim}, encoder produced {d}"
            )

    def _dump_runtime(self) -> None:
        faiss.write_index(self.index, str(self.index_path))
        with open(self.content_path, "w", encoding="utf-8") as f:
            json.dump(self.contents, f, ensure_ascii=False)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metas, f, ensure_ascii=False)

    def _audit(self, op: str, note_id: str, detail: str) -> None:
        with open(self.audit_path, "a", encoding="utf-8") as f:
            f.write(f"{_ISO()} {op} {note_id} {detail}\n")


# ============================================================
# Tool dispatch helpers — used by eva_core to execute the
# `RememberThis` / `ForgetMemory` tool calls the model emits.
#
# Pulled out of eva_core so they can be unit-tested without spinning
# up a full ChatAgent (model + processor + memory_state). The eva_core
# dispatch site is a thin wrapper: validate guard → call helper →
# write returned string into history as the tool observation.
# ============================================================

_VALID_ENTITIES = ("Eva", "Rosm", "Shared")
_VALID_CATEGORIES = ("Lore", "Event")


# P2-7 修复（2026-05-13 Turn 12）：用户说 "meeting next Monday, remember it"
# 时，RememberThis 把 "next Monday" 原样存了下来。一周后这条 note 字面
# 含义就漂移了。下面把常见的相对日期短语在写库前归一化成绝对日期。
# 保守覆盖：英文的 next/this <weekday>、tomorrow/yesterday、in N day(s) /
# week(s)，以及对应的中文形式（明天/后天/下周一 等）。复杂表达
# （"the Monday after next"、"end of next month"）留给后续 LLM normalize。

_WEEKDAY_INDEX_EN = {
    "monday": 0, "mon": 0,
    "tuesday": 1, "tue": 1, "tues": 1,
    "wednesday": 2, "wed": 2,
    "thursday": 3, "thu": 3, "thur": 3, "thurs": 3,
    "friday": 4, "fri": 4,
    "saturday": 5, "sat": 5,
    "sunday": 6, "sun": 6,
}

_WEEKDAY_INDEX_ZH = {
    "周一": 0, "星期一": 0, "礼拜一": 0,
    "周二": 1, "星期二": 1, "礼拜二": 1,
    "周三": 2, "星期三": 2, "礼拜三": 2,
    "周四": 3, "星期四": 3, "礼拜四": 3,
    "周五": 4, "星期五": 4, "礼拜五": 4,
    "周六": 5, "星期六": 5, "礼拜六": 5,
    "周日": 6, "周天": 6, "星期日": 6, "星期天": 6, "礼拜天": 6,
}


def _resolve_relative_dates(text: str, today=None) -> str:
    """Rewrite common relative date phrases in `text` to absolute YYYY-MM-DD.

    Returns the rewritten text. Original phrase is preserved with the
    absolute date appended in parens — "next Monday (2026-05-18)" — so
    the prose stays readable and humans can verify the resolution.

    Conservative on purpose: only handles patterns we can confidently
    map. Anything ambiguous is left untouched.
    """
    if not isinstance(text, str) or not text.strip():
        return text
    today = today or get_current_time().date()

    def _next_weekday(base_date, target_dow, *, this_or_next: str):
        """Resolve 'this/next <weekday>' relative to base_date.

        Conventions:
          - 'next <weekday>' = upcoming occurrence strictly in the future.
            If today IS that weekday, jump to the one a week later.
          - 'this <weekday>' = upcoming occurrence within current week;
            if it's already past, same as 'next'.
        """
        diff = (target_dow - base_date.weekday()) % 7
        if this_or_next == "next" and diff == 0:
            diff = 7
        if this_or_next == "this" and diff == 0:
            diff = 0  # today itself; resolve to today
        return base_date + timedelta(days=diff)

    def _en_repl(m):
        qualifier = (m.group(1) or "").lower()       # 'next' / 'this' / 'last' / ''
        weekday = (m.group(2) or "").lower().rstrip(".")
        if weekday not in _WEEKDAY_INDEX_EN:
            return m.group(0)
        target_dow = _WEEKDAY_INDEX_EN[weekday]
        if qualifier == "last":
            diff = (today.weekday() - target_dow) % 7 or 7
            d = today - timedelta(days=diff)
        else:
            d = _next_weekday(today, target_dow, this_or_next=qualifier or "next")
        return f"{m.group(0)} ({d.isoformat()})"

    text = re.sub(
        r"\b(next|this|last)?\s*("
        r"monday|mon|tuesday|tue|tues|wednesday|wed|thursday|thu|thur|thurs|"
        r"friday|fri|saturday|sat|sunday|sun"
        r")\b\.?",
        _en_repl, text, flags=re.IGNORECASE,
    )

    # tomorrow / yesterday / today
    # 顺序很关键：先处理"day after tomorrow"（多词短语），再处理"tomorrow"，
    # 否则后者会在前者已经替换的文本里再叠一遍 (2026-05-14)。
    one = today + timedelta(days=1)
    two = today + timedelta(days=2)
    text = re.sub(r"\bthe\s+day\s+after\s+tomorrow\b",
                  f"the day after tomorrow ({two.isoformat()})",
                  text, flags=re.IGNORECASE)
    # 负向回顾：避免吃到上一步已经替换出来的 "...tomorrow (date)..." 里的 tomorrow
    text = re.sub(r"(?<!after\s)\btomorrow\b(?!\s*\()",
                  f"tomorrow ({one.isoformat()})",
                  text, flags=re.IGNORECASE)
    text = re.sub(r"\byesterday\b(?!\s*\()",
                  f"yesterday ({(today - timedelta(days=1)).isoformat()})",
                  text, flags=re.IGNORECASE)

    # in N day(s) / N week(s)
    def _in_days(m):
        n = int(m.group(1))
        d = today + timedelta(days=n)
        return f"{m.group(0)} ({d.isoformat()})"

    def _in_weeks(m):
        n = int(m.group(1))
        d = today + timedelta(weeks=n)
        return f"{m.group(0)} ({d.isoformat()})"

    text = re.sub(r"\bin\s+(\d{1,3})\s+days?\b", _in_days, text, flags=re.IGNORECASE)
    text = re.sub(r"\bin\s+(\d{1,3})\s+weeks?\b", _in_weeks, text, flags=re.IGNORECASE)

    # 中文：明天 / 后天 / 昨天 / 下周一 / 这周三 / 上周五
    text = text.replace("明天", f"明天({one.isoformat()})") if "明天(" not in text else text
    text = text.replace("后天", f"后天({two.isoformat()})") if "后天(" not in text else text
    yest = (today - timedelta(days=1)).isoformat()
    text = text.replace("昨天", f"昨天({yest})") if "昨天(" not in text else text

    def _zh_repl(m):
        qualifier = m.group(1) or "下"  # 下/这/上 — default 下
        weekday_zh = m.group(2)
        target_dow = _WEEKDAY_INDEX_ZH.get(weekday_zh)
        if target_dow is None:
            return m.group(0)
        if qualifier == "上":
            diff = (today.weekday() - target_dow) % 7 or 7
            d = today - timedelta(days=diff)
        else:
            this_or_next = "next" if qualifier == "下" else "this"
            d = _next_weekday(today, target_dow, this_or_next=this_or_next)
        return f"{m.group(0)}({d.isoformat()})"

    text = re.sub(
        r"(下|这|上)?\s*(周一|周二|周三|周四|周五|周六|周日|周天|"
        r"星期一|星期二|星期三|星期四|星期五|星期六|星期日|星期天|"
        r"礼拜一|礼拜二|礼拜三|礼拜四|礼拜五|礼拜六|礼拜天)",
        _zh_repl, text,
    )

    return text


def execute_remember_this(notes_store: Optional[NotesStore],
                          params: Optional[dict]) -> str:
    """Execute the `RememberThis` tool. Returns the observation string
    that should be written back into history (visible to the model on
    the next generation step).

    Errors are returned as `[REMEMBER ERROR] ...` strings rather than
    raised, because the model recovers better from a textual error
    block than from a tool-execution exception.
    """
    if notes_store is None:
        return ("[REMEMBER ERROR] Notes store is not active. "
                "RememberThis is unavailable in this session.")

    params = params or {}
    content = str(params.get("content", "") or "").strip()
    entity = str(params.get("entity", "") or "").strip()
    topic = str(params.get("topic", "") or "").strip()
    keywords = params.get("keywords", "") or ""
    category = str(params.get("category", "Lore") or "Lore").strip() or "Lore"
    # R-1: 可选 slot_values 字段。模型 SFT 学过结构化 slot 后会主动填入；
    # 老 SFT 模型不填也能 work（runtime 默认走 prose-only）。
    raw_slots = params.get("slot_values")
    slot_values = raw_slots if isinstance(raw_slots, dict) else None
    # R-7: event 字段。模型解析出绝对日期 / 类型后填入；
    # 老 SFT 模型不填则 _resolve_relative_dates fallback 仍兜底（在 prose 里补）。
    event_date = str(params.get("event_date", "") or "").strip()
    event_time = str(params.get("event_time", "") or "").strip()
    event_type = str(params.get("event_type", "") or "").strip()
    expires_at = str(params.get("expires_at", "") or "").strip()
    raw_parts = params.get("participants")
    participants = raw_parts if isinstance(raw_parts, list) else None

    if not content:
        return "[REMEMBER ERROR] Missing required field: content."
    if entity not in _VALID_ENTITIES:
        return (f"[REMEMBER ERROR] entity must be one of "
                f"{list(_VALID_ENTITIES)}, got {entity!r}.")
    if not topic:
        return "[REMEMBER ERROR] Missing required field: topic."
    if category not in _VALID_CATEGORIES:
        return (f"[REMEMBER ERROR] category must be one of "
                f"{list(_VALID_CATEGORIES)}, got {category!r}.")

    # R-7：当模型 EXPLICIT 提供 event_date 时，跳过 prose 内 _resolve_relative_dates
    # 改写（结构化字段已经是绝对日期的 source of truth，无需再叠 prose 注释）。
    # 模型没填 event_date 时仍跑 fallback——P2-7 老行为保留，覆盖老 SFT 模型。
    structured_date_provided = bool(_normalize_event_date(event_date))
    if not structured_date_provided:
        content = _resolve_relative_dates(content)

    try:
        note_id = notes_store.add(
            vector_text=content,
            content=content,
            entity=entity,
            topic=topic,
            keywords=keywords,
            category=category,
            slot_values=slot_values,
            event_date=event_date,
            event_time=event_time,
            event_type=event_type,
            expires_at=expires_at,
            participants=participants,
        )
    except ValueError as e:
        return f"[REMEMBER ERROR] {e}"
    except Exception as e:
        return f"[REMEMBER ERROR] Failed to add note: {type(e).__name__}: {e}"

    return (
        f"[REMEMBERED] Stored as Note #{note_id} "
        f"(entity={entity}, topic={topic}). "
        f"This note will surface in future MemorySearch results until "
        f"you call ForgetMemory(record_id=\"{note_id}\", reason=\"...\")."
    )


# R-2 (2026-05-13) ForgetMemory 匹配阈值。这些常量分离出来便于调参。
#   FORGET_MATCH_MIN_SCORE: 候选必须 >= 此分数才算"够像"。NotesStore 用
#     mpnet + IndexFlatIP，向量归一化后分数就是 cosine ∈ [-1, 1]；同义改写
#     大致在 0.4~0.7。0.35 留一定 recall，但能挡掉完全跑题的查询。
#   FORGET_UNIQUE_GAP: top1 比 top2 高出多少才算"确定无歧义"。0.15 来自
#     mpnet 在 Notes 上的经验值——同义两句通常差距 < 0.10，主题差异 > 0.20。
FORGET_MATCH_MIN_SCORE = 0.35
FORGET_UNIQUE_GAP = 0.15


def _is_valid_note_id(s: str) -> bool:
    return bool(s) and len(s) == 8 and all(c in "0123456789abcdef" for c in s)


def _filter_by_topic(hits: list, topic: str) -> list:
    """R-2：可选的 topic 二次过滤。空字符串就不过滤。"""
    if not topic:
        return hits
    needle = topic.strip().lower()
    return [h for h in hits if (h.get("meta", {}).get("topic", "") or "").lower() == needle]


def _format_disambiguation(hits: list, query: str) -> str:
    """R-2：多候选 / 低置信度时，回出一段 disambiguation 让模型补条件再调。"""
    lines = [
        f"[FORGET DISAMBIGUATE] Multiple notes match query {query!r} "
        f"(top {len(hits)} shown). Re-call ForgetMemory with a more specific "
        f"query, add topic=, or pick record_id= directly:",
    ]
    for h in hits:
        nid = h.get("note_id", "?")
        meta = h.get("meta", {}) or {}
        ent = meta.get("entity", "?")
        topic = meta.get("topic", "-")
        score = h.get("score", 0.0)
        preview = (h.get("content") or "").replace("\n", " ")[:80]
        lines.append(
            f"  - [Note #{nid}] (score={score:.2f}) [Subject: {ent}] "
            f"[Topic: {topic}]: {preview}"
        )
    return "\n".join(lines)


# ============================================================
# DELETED 2026-05-14 Plan-A Cleanup:
# R-2.1 anaphoric forget detection + topic/event_type inference helpers
# (_R21_ANAPHORIC_*_RE, _is_anaphoric_forget_intent, _R21_KEYWORD_TO_*,
# _infer_event_type_and_topic) — they were the P1/P2 layers of the old
# multi-tier ForgetMemory auto-correct. Advisor now writes the exact
# record_id into per-turn hints (suggested_calls[i].args.record_id),
# so runtime heuristic inference is no longer needed. The simplified
# _try_auto_correct keeps L0 (hex from context) + L1 (similarity search)
# only. To roll back, revive these symbols from git history alongside
# notes_runtime.py at 2026-05-13.
# ============================================================


def _extract_recent_note_id_from_context(notes_store: NotesStore,
                                          fallback_context: str) -> str:
    """R-2.1 P2 层：从 fallback_context 抓 hex id（前期会话有效）。"""
    if not fallback_context:
        return ""
    ids = re.findall(r"(?:Note\s*)?#([0-9a-f]{8})", fallback_context)
    if not ids:
        return ""
    live_ids = {m.get("note_id") for m in notes_store.metas if not m.get("deleted")}
    for nid in ids:
        if nid in live_ids:
            return nid
    return ""


def _try_auto_correct(notes_store: NotesStore,
                      *,
                      failed_id: str,
                      reason: str,
                      fallback_context: str,
                      topic_filter: str = "") -> Optional[str]:
    """模型给的 record_id 找不到时，runtime 兜底找正确的 note。

    2026-05-13 Advisor cutover：原 R-2.1 P0/P1/P2/P3 四层 fallback 删到只剩
    两层。Advisor 会把正确的 record_id 写进本轮建议块里，模型大概率会
    抄对——所以"猜对的代价"不再需要全套启发式。

    简化后只保留：
      L0: 从 fallback_context 抓 hex id（advisor 把 #abc12345 写进了建议
          块/提示块的概率很高，model 在 history compaction 之前的 obs 也
          通常含 [Note #...] 标签）
      L1: similarity search on fallback_context（advisor 漏写时的最低保底）

    删除：
      - anaphoric forget intent + recent_adds LRU 推断（advisor 已经按上下文
        判定该删哪条，runtime 不该再二次猜测）
      - R-7 event.type / topic 唯一过滤（同上）
      - topic-consistency 校验（同上）

    任何一层命中即返回 obs；都 miss 返回 None 让上游报错。
    """
    if notes_store is None:
        return None

    # ---------- L0: hex id from fallback_context ----------
    # 这层最稳：advisor / 提示块 / 历史 obs 里若提到 [Note #abc12345]，
    # 直接抄。常见情况：用户说"忘了那条记忆"，advisor 在建议里写
    # "suggested record_id: #abc12345"——model 抄过去时 hex 错位的话，
    # runtime 仍能从同一段 fallback_context 里抓回来。
    ctx_id = _extract_recent_note_id_from_context(notes_store, fallback_context)
    if ctx_id and ctx_id != failed_id:
        try:
            ok = notes_store.tombstone(ctx_id, reason)
        except Exception:
            ok = False
        if ok:
            return (f"[FORGOTTEN] Note #{ctx_id} tombstoned (auto-corrected: "
                    f"your record_id={failed_id!r} was not a live note; runtime "
                    f"extracted #{ctx_id} from recent session context). "
                    f"Reason: {reason or '(no reason given)'}. "
                    f"It will no longer appear in MemorySearch results.")

    # ---------- L1: similarity search on fallback_context ----------
    # 最低保底：advisor 漏写 hex 时，用 fallback_context 文本跑语义搜索。
    # 保留 unique-gap + min-score 阈值，避免误删不相关 note。
    if not fallback_context or not fallback_context.strip():
        return None
    try:
        hits = notes_store.search(fallback_context, top_k=5)
    except Exception:
        return None
    hits = _filter_by_topic(hits, topic_filter)
    if not hits:
        return None
    hits.sort(key=lambda h: h.get("score", 0.0), reverse=True)
    top1 = hits[0]
    top1_score = float(top1.get("score", 0.0))
    if top1_score < FORGET_MATCH_MIN_SCORE:
        return None
    if len(hits) >= 2:
        gap = top1_score - float(hits[1].get("score", 0.0))
        if gap < FORGET_UNIQUE_GAP:
            return None
    target_id = top1.get("note_id", "")
    if not _is_valid_note_id(target_id):
        return None
    try:
        ok = notes_store.tombstone(target_id, reason)
    except Exception:
        return None
    if not ok:
        return None
    return (f"[FORGOTTEN] Note #{target_id} tombstoned (auto-corrected: "
            f"your record_id={failed_id!r} was not a live note; runtime "
            f"resolved the target from session context, score={top1_score:.2f}). "
            f"Reason: {reason or '(no reason given)'}. "
            f"It will no longer appear in MemorySearch results.")


def execute_forget_memory(notes_store: Optional[NotesStore],
                          params: Optional[dict],
                          *,
                          fallback_context: str = "") -> str:
    """Execute the `ForgetMemory` tool. Returns the observation string.

    R-2 (2026-05-13)：双路径 + runtime intercept auto-correct。

      路径 A — 显式 record_id（训练分布的标准 pattern）：
        模型按 SFT 学到的形式调 ForgetMemory(record_id="xxxx", reason="...")。
        valid + 命中 live note → 直接 tombstone。

      路径 A 兜底 — record_id 不存在 / 格式错时的 auto-correct：
        模型可能幻觉 id（参见 2026-05-13 测试 Turn 13：模型写 "70374433"，
        实际是 "7a4a68da"）。此时 runtime 不直接 fail，先用 `fallback_context`
        （由 eva_core 传入：本轮 user 文本 + 最近 RememberThis content）
        跑 NotesStore.search 找正确候选；唯一命中即纠正后 tombstone，
        在 observation 里告知 "auto-corrected from <bad_id>"。
        这种"runtime 帮 model 对账"绕过了 SFT 强制 record_id 路径却 hex 抄错
        的失败模式，且不需要 retrain。

      路径 B — query / topic 化匹配（明确意图给 query 时走）：
        给了 query → 直接 search → unique 即删，否则 disambiguation。

      两个都没给 + 没 fallback_context：error。
    """
    if notes_store is None:
        return ("[FORGET ERROR] Notes store is not active. "
                "ForgetMemory is unavailable in this session.")

    params = params or {}
    note_id = str(params.get("record_id", "") or "").strip()
    query = str(params.get("query", "") or "").strip()
    topic_filter = str(params.get("topic", "") or "").strip()
    reason = str(params.get("reason", "") or "").strip()

    # ---------- 路径 A：显式 record_id（训练分布主路径） ----------
    if note_id:
        if _is_valid_note_id(note_id):
            try:
                ok = notes_store.tombstone(note_id, reason)
            except Exception as e:
                return f"[FORGET ERROR] Failed to tombstone: {type(e).__name__}: {e}"
            if ok:
                return (f"[FORGOTTEN] Note #{note_id} tombstoned. "
                        f"Reason: {reason or '(no reason given)'}. "
                        f"It will no longer appear in MemorySearch results.")
            # valid 但不存在 → 尝试 auto-correct
            corrected = _try_auto_correct(
                notes_store, failed_id=note_id, reason=reason,
                fallback_context=fallback_context, topic_filter=topic_filter,
            )
            if corrected is not None:
                return corrected
            return (
                f"[FORGET ERROR] No live note with id {note_id}. "
                f"It may have already been deleted, or the id is wrong. "
                f"Re-call with query=\"<short description>\" or use a "
                f"record_id from a recent [Note #...] tag."
            )
        # 格式错 → 也尝试 auto-correct（可能模型 hex 抄成了别的串）
        corrected = _try_auto_correct(
            notes_store, failed_id=note_id, reason=reason,
            fallback_context=fallback_context, topic_filter=topic_filter,
        )
        if corrected is not None:
            return corrected
        return (f"[FORGET ERROR] record_id must be the 8-char hex id from a "
                f"[Note #...] tag, got {note_id!r}.")

    # ---------- 路径 B：query / topic 化匹配 ----------
    if not query and not topic_filter:
        return ("[FORGET ERROR] Must pass either query=<short description> "
                "or record_id=<8-char hex>. Example: "
                "ForgetMemory(query=\"the meeting next monday\", reason=\"canceled\").")

    # 没有 query 只有 topic：直接列出该 topic 下所有 note 让模型选。
    effective_query = query or topic_filter

    try:
        hits = notes_store.search(effective_query, top_k=5)
    except Exception as e:
        return f"[FORGET ERROR] search failed: {type(e).__name__}: {e}"
    hits = _filter_by_topic(hits, topic_filter)

    if not hits:
        return (
            f"[FORGET NOT FOUND] No live note matches query {effective_query!r}"
            f"{f' under topic={topic_filter!r}' if topic_filter else ''}. "
            f"Check spelling or list recent notes via MemorySearch first."
        )

    # 按分数排序确保 top1/top2 比较稳定。
    hits.sort(key=lambda h: h.get("score", 0.0), reverse=True)
    top1 = hits[0]
    top1_score = float(top1.get("score", 0.0))

    if top1_score < FORGET_MATCH_MIN_SCORE:
        # 即使有结果，top1 也不够强 — 视为 not found，避免误删。
        return (
            f"[FORGET NOT FOUND] Best match score {top1_score:.2f} < "
            f"{FORGET_MATCH_MIN_SCORE:.2f}; refusing to delete to avoid "
            f"deleting an unrelated note. Refine your query or pass record_id."
        )

    # 唯一性判定：only 1 hit OR top1 比 top2 高出 GAP。
    is_unique = True
    if len(hits) >= 2:
        gap = top1_score - float(hits[1].get("score", 0.0))
        if gap < FORGET_UNIQUE_GAP:
            is_unique = False

    if not is_unique:
        # 多候选 → 列出让模型再选。
        return _format_disambiguation(hits[:3], effective_query)

    target_id = top1.get("note_id", "")
    if not _is_valid_note_id(target_id):
        return (f"[FORGET ERROR] Internal: matched note has malformed id "
                f"{target_id!r}.")

    try:
        ok = notes_store.tombstone(target_id, reason)
    except Exception as e:
        return f"[FORGET ERROR] Failed to tombstone: {type(e).__name__}: {e}"
    if not ok:
        return (f"[FORGET ERROR] Matched note #{target_id} could not be "
                f"tombstoned (already deleted?). Try MemorySearch first.")

    return (f"[FORGOTTEN] Note #{target_id} tombstoned (resolved from "
            f"query {effective_query!r}, score={top1_score:.2f}). "
            f"Reason: {reason or '(none given)'}. "
            f"It will no longer appear in MemorySearch results.")
