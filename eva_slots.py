"""
eva_slots.py — Slot extraction & evidence layer.

Extracted from eva_core.py and eva_memory_legacy.py during the post-Plan-B
cleanup. This module owns slot-level memory parsing — turning a slot name
(birthday/full_name/age/toy) and a record's prose content into a clean
extracted value, plus turn-side helpers for figuring out which slots the
user is asking about and what the model has actually grounded.

Two groups of functions:

1. From-text value extractors (formerly module-level in eva_memory_legacy):
       _extract_birthday_value_from_text
       _extract_full_name_value_from_text
       _extract_age_value_from_text
       _extract_toy_value_from_text
       _extract_slot_value_from_record   (dispatch on slot name)

2. Turn-side slot bookkeeping (formerly methods on ChatAgent — they only
   read MEMORY_SLOT_FIELDS, never self state, so they were syntactically
   methods but semantically pure functions):
       extract_memory_slots
       parse_slot_evidence_from_text
       build_missing_slot_note_from_missing

Why split this out:
- Slot logic was the only sub-system spread across two files (eva_core
  for "what slots is the user asking about" + eva_memory_legacy for
  "how do you read that slot's value out of a record"). Co-locating it
  makes the slot pipeline easier to audit.
- The ChatAgent methods touch zero self state; they were free-standing
  functions in disguise. Moving them shrinks ChatAgent without changing
  any behaviour.

Dependency direction:
    eva_memory_legacy ──> eva_slots
We import three low-level helpers (_param_to_text, _canonical_known_entity_name,
_clean_extracted_value) and one record-level guard (_record_can_support_target_slot)
from eva_memory_legacy. eva_memory_legacy never imports from eva_slots —
the slot value extractors used to live INSIDE eva_memory_legacy, so its
existing callers (_attach_slot_evidence_to_collection) now reach back via
this module instead.
"""

import re

from eva_config import MEMORY_SLOT_FIELDS, SLOT_APPLICABLE_SUBJECTS
from eva_memory_legacy import (
    _param_to_text,
    _canonical_known_entity_name,
    _clean_extracted_value,
    _record_can_support_target_slot,
    _phrase_matches_text,
    _normalize_match_text,
)


__all__ = [
    # Slot value extractors (from eva_memory_legacy)
    "_extract_birthday_value_from_text",
    "_extract_full_name_value_from_text",
    "_extract_age_value_from_text",
    "_extract_toy_value_from_text",
    "_extract_slot_value_from_record",
    # Turn-side slot bookkeeping (from ChatAgent)
    "extract_memory_slots",
    "parse_slot_evidence_from_text",
    "build_missing_slot_note_from_missing",
]


# ============================================================
# Value-from-text extractors
# ============================================================
# Each takes (content_text, target_entity) and returns either the cleaned
# slot value as a string, or "" when the prose does not contain a usable
# value for that target.
#
# target_entity follows the canonical "Eva" / "Rosm" / "Both" / "Shared"
# convention; "Both" / unknown means the extractor will use generic
# patterns rather than entity-specific prefixes.
# ============================================================
def _extract_birthday_value_from_text(content, target_entity):
    text = _param_to_text(content)
    if not text:
        return ""
    date_pat = r"([A-Z][a-z]+\.?\s+\d{1,2}(?:st|nd|rd|th)?)"
    target = _canonical_known_entity_name(target_entity or "Both")
    if target == "Rosm":
        prefixes = [r"Rosm(?:'s)?", r"Master(?:'s)?", r"Eva's creator(?:'s)?"]
    elif target == "Eva":
        prefixes = [r"Eva(?:'s)?", r"the maid(?:'s)?"]
    else:
        prefixes = []
    for pref in prefixes:
        m = re.search(rf"\b{pref}\s+birthday\s+(?:is|:|falls\s+on)\s+{date_pat}", text, re.I)
        if m:
            return _clean_extracted_value(m.group(1))
    m = re.search(rf"\bbirthday\s+(?:is|:|falls\s+on)\s+{date_pat}", text, re.I)
    if m:
        return _clean_extracted_value(m.group(1))
    return ""


def _extract_full_name_value_from_text(content, target_entity):
    text = _param_to_text(content)
    if not text:
        return ""
    name_pat = r"([A-Z][A-Za-z][A-Za-z0-9_ '\-]{1,80})"
    target = _canonical_known_entity_name(target_entity or "Both")
    if target == "Rosm":
        patterns = [
            rf"\bRosm(?:'s)?\s+(?:full\s+)?(?:real\s+)?name\s+(?:is|:)\s+{name_pat}",
            rf"\bEva's creator(?:'s)?\s+(?:full\s+)?(?:real\s+)?name\s+(?:is|:)\s+{name_pat}",
            rf"\bcreator(?:'s)?\s+(?:full\s+)?(?:real\s+)?name\s+(?:is|:)\s+{name_pat}",
        ]
    elif target == "Eva":
        patterns = [
            rf"\bEva(?:'s)?\s+(?:full\s+)?(?:real\s+)?name\s+(?:is|:)\s+{name_pat}",
            rf"\bher\s+(?:full\s+)?(?:real\s+)?name\s+(?:is|:)\s+{name_pat}",
        ]
    else:
        patterns = [rf"\b(?:full\s+)?(?:real\s+)?name\s+(?:is|:)\s+{name_pat}"]
    for pat in patterns:
        m = re.search(pat, text, re.I)
        if m:
            val = _clean_extracted_value(m.group(1))
            if val.lower() in {"eva", "rosm", "master", "creator"}:
                continue
            return val
    return ""


def _extract_age_value_from_text(content, target_entity):
    text = _param_to_text(content)
    if not text:
        return ""
    target = _canonical_known_entity_name(target_entity or "Both")
    if target == "Rosm":
        prefixes = [r"Rosm(?:'s)?", r"Master(?:'s)?", r"Eva's creator(?:'s)?"]
    elif target == "Eva":
        prefixes = [r"Eva(?:'s)?"]
    else:
        prefixes = []
    for pref in prefixes:
        m = re.search(rf"\b{pref}\s+age\s+(?:is|:)\s+(\d{{1,3}})\b", text, re.I)
        if m:
            return m.group(1)
        m = re.search(rf"\b{pref}\s+is\s+(\d{{1,3}})\s+years\s+old\b", text, re.I)
        if m:
            return m.group(1)
    return ""


def _extract_toy_value_from_text(content, target_entity):
    """Extract Eva/Rosm toy slot values from memory record prose.

    R-1 (2026-05-13) 之后这个 extractor 是 **fallback only**——主路径是
    `record.meta.slot_values["toy"]`（由 `generate/migrate_slot_values.py`
    build-time 注入，或由 RememberThis 工具调用时填）。本函数仅在以下
    场景跑：
      - 老 jsonl / 升级前的 record 没填 meta.slot_values
      - NotesStore note 由旧 SFT 模型创建（不知道 slot_values 参数）

    设计 trade-off：保留正则的动词族 + em-dash 停止符（原 P0-1 补丁），
    让 fallback 鲁棒一点；但**长期**新增 lore 不应依赖此正则——build-time
    迁移脚本是 canonical source。

    覆盖的动词形态：
      - "Eva's favorite toy was a cuddly bunny"
      - "Eva's favorite toy has always been a cuddly bunny"
      - "Eva's favorite toy used to be a music box"
    句尾停止符包含 em-dash / en-dash / ASCII hyphen，避免把
    "cuddly bunny — soft" 抓进 value。
    """
    text = _param_to_text(content)
    if not text:
        return ""
    target = _canonical_known_entity_name(target_entity or "Both")
    # 动词族：copular + perfect 形态。括号内不留 capturing group，保持上层
    # m.group(1) 仍然是值短语。
    link = r"(?:was|is|:|has\s+(?:always\s+)?been|had\s+been|used\s+to\s+be)"
    # 值短语停止符：原先只在 ,.;! 处停，现在追加 em-dash / en-dash / ASCII
    # hyphen-space，避免把破折号后的从句污染进 value。
    tail = r"((?:(?!\s+[—–-]\s+)[^,.;!])+)"
    if target == "Eva":
        patterns = [
            rf"\bEva(?:'s)?\s+(?:favorite\s+)?(?:childhood\s+)?toy\s+{link}\s+(?:a|an|the)?\s*{tail}",
            rf"\bher\s+(?:favorite\s+)?(?:childhood\s+)?toy\s+{link}\s+(?:a|an|the)?\s*{tail}",
            rf"\bfavorite\s+(?:childhood\s+)?toy\s+{link}\s+(?:a|an|the)?\s*{tail}",
        ]
    elif target == "Rosm":
        patterns = [
            rf"\bRosm(?:'s)?\s+(?:favorite\s+)?(?:childhood\s+)?toy\s+{link}\s+(?:a|an|the)?\s*{tail}",
            rf"\bhis\s+(?:favorite\s+)?(?:childhood\s+)?toy\s+{link}\s+(?:a|an|the)?\s*{tail}",
        ]
    else:
        patterns = [
            rf"\b(?:favorite\s+)?(?:childhood\s+)?toy\s+{link}\s+(?:a|an|the)?\s*{tail}",
        ]
    for pat in patterns:
        m = re.search(pat, text, re.I)
        if m:
            return _clean_extracted_value(m.group(1))
    return ""


def _extract_slot_value_from_record(record, target_entity, slot):
    """Dispatch on slot name to the right value extractor.

    R-1 (2026-05-13): 优先读 record.meta.slot_values[slot] —— build-time
    抽取（LLM 或 regex）已经把 value 锁进 meta，inference-time 不需要再跑
    正则。只有 meta 没填时才走正则 fallback（主要为 user notes 兜底）。

    Returns "" when:
      - slot is not a tracked MEMORY_SLOT_FIELD,
      - the record is not bound to the requested target_entity (subject
        canonicalisation guarded by _record_can_support_target_slot), or
      - meta.slot_values[slot] is missing AND none of the regex patterns matched.
    """
    if slot not in MEMORY_SLOT_FIELDS:
        return ""
    if not _record_can_support_target_slot(record, target_entity, slot):
        return ""

    # R-1: meta.slot_values 优先路径
    meta_slots = (record.get("meta", {}) or {}).get("slot_values", {}) or {}
    if isinstance(meta_slots, dict):
        val = meta_slots.get(slot)
        if val and isinstance(val, str) and val.strip():
            return _clean_extracted_value(val)

    content = record.get("content", "") or ""
    if slot == "birthday":
        return _extract_birthday_value_from_text(content, target_entity)
    if slot == "full_name":
        return _extract_full_name_value_from_text(content, target_entity)
    if slot == "age":
        return _extract_age_value_from_text(content, target_entity)
    if slot == "toy":
        return _extract_toy_value_from_text(content, target_entity)
    return ""


# ============================================================
# Turn-side slot bookkeeping
# ============================================================
# Formerly ChatAgent methods. They are pure functions: zero `self` state,
# only read MEMORY_SLOT_FIELDS. ChatAgent re-exposes them as wrapper
# methods so existing call sites (`self._extract_memory_slots(...)`) keep
# working, but new code can import these directly.
# ============================================================
def extract_memory_slots(text, encoder=None):
    """Precise value slots requested by the current turn.

    Only MEMORY_SLOT_FIELDS are slot-covered. Broad domain fields such as
    gaming/interests/project are intentionally excluded; they can retrieve
    related evidence but cannot create a missing/exact slot requirement.

    Subject-aware (2026-05-11): person-only slots (full_name / birthday /
    age / toy) are gated through `eva_subject_classifier.is_person_subject`
    so queries like "what was the cat's name?" no longer fire `full_name`.
    See SLOT_APPLICABLE_SUBJECTS in eva_config and the eva_subject_classifier
    module for details.

    Args:
        text: raw user query.
        encoder: optional callable for the subject classifier's Layer 2
            embedding nearest-neighbor. None falls back to Layer 1 regex
            (still catches the common pet/place/object cases).
    """
    q_norm = _normalize_match_text(text or "")
    if not q_norm:
        return []

    from eva_subject_classifier import is_person_subject
    person_query = is_person_subject(text, encoder=encoder)

    slots = []
    for slot, aliases in MEMORY_SLOT_FIELDS.items():
        applicable = SLOT_APPLICABLE_SUBJECTS.get(slot, set())
        if applicable == {"Person"} and not person_query:
            continue
        if any(_phrase_matches_text(alias, q_norm) for alias in [slot, *aliases]):
            slots.append(slot)
    return list(dict.fromkeys(slots))


def parse_slot_evidence_from_text(observation_text):
    """Read 'FOUND' lines back out of a memory observation block.

    Memory observations the model receives include lines like:
        - birthday: FOUND = July 7th
    This re-parses that structured tail into a {slot: value} dict so the
    verifier can check what the model actually had grounding for.
    """
    evidence = {}
    if not observation_text:
        return evidence
    for m in re.finditer(r"^\s*-\s*([a-z_]+)\s*:\s*FOUND\s*=\s*(.+?)\s*$",
                         str(observation_text), re.M):
        slot = m.group(1).strip()
        value = m.group(2).strip()
        if slot in MEMORY_SLOT_FIELDS and value:
            evidence[slot] = value
    return evidence


def build_missing_slot_note_from_missing(missing):
    """Build the trailing note that warns the model about ungrounded slots."""
    missing = [s for s in (missing or []) if s in MEMORY_SLOT_FIELDS]
    if not missing:
        return ""
    return (f"\n[MISSING SLOTS]: {', '.join(missing)} — answer must say "
            "these are not recorded.")
