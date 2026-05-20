"""
eva_memory_legacy.py — FAISS+BM25+rerank memory retrieval pipeline.

Extracted from eva_core.py during the P2 step-7 refactor. This module owns
the entire long-term-memory retrieval stack — every helper that turns a
free-text query plus a target_entity hint into a ranked, slot-augmented,
formatted block of evidence for the model to read inside a tool_output.

Architecture:
- load_memory_resources()         Boot-time loader for FAISS index + JSONL.
- _collect_memory_records(...)    Hybrid FAISS+BM25 fetch + CrossEncoder
                                  rerank, with stem matching, entity bias,
                                  and field/keyword evidence boosts.
- _merge_memory_collections(...)  De-dup and merge multi-subquery results.
- run_memory_search(...)          Public entry: compound-question split +
                                  per-subquery probe + slot evidence +
                                  text formatting. Optionally runs an
                                  external judge_fn for EXACT/RELATED tiering.

(Slot-level value extractors moved to eva_slots.py during cleanup —
_attach_slot_evidence_to_collection still calls into them via lazy import.)

This is the *legacy* side of the memory layer — it predates the topic-based
PRE PROBE in eva_memory_v2.MemoryModule. The two coexist: the v2 module
calls into these helpers (run_memory_search, _collect_memory_records,
_canonical_target_entity, etc.) for the actual retrieval. Do not delete
this file unless v2 has fully absorbed every code path.

All constants come from eva_config (RERANK_TOP_K, BM25_WEIGHT,
MEMORY_SLOT_FIELDS, MEMORY_FIELD_SYNONYMS, etc.). All low-level
text-canonicalisation helpers are self-contained here.
"""

import os
import re
import json
import numpy as np

try:
    import faiss
except ImportError:
    faiss = None
    print("[WARN] faiss not installed. MemorySearch will act as Mock.")

from rank_bm25 import BM25Okapi

# All retrieval thresholds, slot/domain field tables, and pattern lists live
# in eva_config — pull them in wholesale so the legacy code below sees the
# same names it had inside eva_core.
from eva_config import *  # noqa: F401, F403


def _list_to_text(value, sep=" "):
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return sep.join(str(x).strip() for x in value if str(x).strip())
    return str(value).strip()


def _record_search_text(record):
    meta = record.get("meta", {}) or {}
    pieces = [
        str(record.get("vector_text", "")),
        str(record.get("content", "")),
        str(meta.get("entity", "")),
        str(meta.get("category", "")),
        str(meta.get("topic", "")),
        _list_to_text(meta.get("participants", [])),
        _list_to_text(meta.get("keywords", [])),
    ]
    return " ".join(p for p in pieces if p).strip()


def _record_rerank_text(record):
    meta = record.get("meta", {}) or {}
    return (
        f"Vector text: {record.get('vector_text', '')}\n"
        f"Content: {record.get('content', '')}\n"
        f"Entity: {meta.get('entity', '')}\n"
        f"Category: {meta.get('category', '')}\n"
        f"Topic: {meta.get('topic', '')}\n"
        f"Participants: {_list_to_text(meta.get('participants', []), sep=', ')}\n"
        f"Keywords: {_list_to_text(meta.get('keywords', []), sep=', ')}"
    ).strip()


def _record_display_content(record):
    content = str(record.get("content", "") or "").strip()
    if content:
        return content
    return str(record.get("vector_text", "") or "").strip()


def load_memory_resources(index_path, content_path):
    if not (os.path.exists(index_path) and os.path.exists(content_path)):
        print(f"[WARN] Memory files not found at {index_path} or {content_path}")
        return {}
    index = faiss.read_index(index_path)
    with open(content_path, "r", encoding="utf-8") as f:
        content_data = f.read().strip()
        raw = (
            json.loads(content_data)
            if content_data.startswith("[")
            else [json.loads(line) for line in content_data.split("\n") if line.strip()]
        )

    # Format dispatch:
    #   - Old format (pre-2026-05-08): list[dict] with full records inline.
    #   - New format written by Memory_maker/Memory.py: list[str] of contents,
    #     with meta living in a sibling `memory_meta.json`. Reconstruct full
    #     dicts so downstream code (`_record_search_text`, BM25 tokenizer,
    #     entity-bonus scorer) sees the same shape either way.
    if raw and isinstance(raw[0], dict):
        db_records = raw
    else:
        meta_path = os.path.join(os.path.dirname(content_path), "memory_meta.json")
        metas: list = []
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                metas = json.load(f)
        if len(metas) != len(raw):
            print(f"[WARN] memory_meta.json missing or length mismatch "
                  f"(contents={len(raw)}, metas={len(metas)}). Records will "
                  f"have empty meta → entity/topic bonuses degrade.")
            metas = metas + [{} for _ in range(len(raw) - len(metas))]
            metas = metas[:len(raw)]
        db_records = [
            {"vector_text": c, "content": c, "meta": m}
            for c, m in zip(raw, metas)
        ]
    if hasattr(index, "ntotal") and index.ntotal != len(db_records):
        print(f"[WARN] FAISS index size ({index.ntotal}) != memory records ({len(db_records)}).")
    stopwords = {
        "the", "is", "at", "in", "on", "to", "of", "and", "or", "for", "an",
        "it", "do", "did", "was", "has", "had", "be", "are",
        "my", "me", "we", "us", "our", "you", "your", "his", "her", "its",
        "with", "what", "when", "where", "how", "why", "who", "whom", "whose",
        "a", "this", "that", "these", "those", "so", "then",
    }
    corpus_tokens = []
    for record in db_records:
        combined_text = _record_search_text(record).lower()
        tokens = [w for w in re.findall(r"\b\w{2,}\b", combined_text) if w not in stopwords]
        # which can be matched by query token 'game' -> 'gam'.
        # Must apply identically here and in _tokenize_memory_query.
        tokens = _stem_tokens(tokens)
        corpus_tokens.append(tokens)
    bm25_model = BM25Okapi(corpus_tokens)
    print(f"[Memory V6.4] Loaded {len(db_records)} records. FAISS + stemmed BM25 ready.")
    return {"index": index, "db_records": db_records, "bm25": bm25_model, "stopwords": stopwords}


def _normalize_scores(scores):
    if len(scores) == 0:
        return scores
    scores = np.asarray(scores, dtype=np.float32)
    valid_mask = np.isfinite(scores)
    if not valid_mask.any():
        return np.zeros_like(scores, dtype=np.float32)
    valid_scores = scores[valid_mask]
    min_s, max_s = np.min(valid_scores), np.max(valid_scores)
    out = np.zeros_like(scores, dtype=np.float32)
    if max_s - min_s == 0:
        return out
    out[valid_mask] = (valid_scores - min_s) / (max_s - min_s)
    return out


ENTITY_ALIASES = {
    "rosm": "Rosm", "master": "Rosm", "creator": "Rosm",
    "eva": "Eva", "maid": "Eva",
    "shared": "Shared", "both": "Both", "all": "Both",
}

USER_ENTITY_ALIASES = {"user", "guest", "me", "myself", "i", "current_user", "current user"}


def _param_to_text(value):
    if value is None: return ""
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(x).strip() for x in value if str(x).strip())
    return str(value).strip()


def _canonical_known_entity_name(name):
    raw = _param_to_text(name)
    if not raw: return raw
    key = raw.lower().strip()
    return ENTITY_ALIASES.get(key, raw)


def _canonical_target_entity(raw_entity, current_user="Guest"):
    raw = _param_to_text(raw_entity) or "Both"
    key = raw.lower().strip()
    if key in USER_ENTITY_ALIASES:
        current = _param_to_text(current_user) or "Guest"
        return _canonical_known_entity_name(current)
    return _canonical_known_entity_name(raw)


def _canonical_record_entity(raw_entity):
    raw = _param_to_text(raw_entity) or "Unknown"
    return _canonical_known_entity_name(raw)


def _split_compound_question(text, min_part_words=2):
    if not isinstance(text, str) or not text.strip():
        return []
    text = text.strip()
    parts = re.split(r"(?<=[?!])\s+", text)
    expanded = []
    for p in parts:
        sub = re.split(r"\s+(?:and|or|also|plus|but)\s+", p, flags=re.IGNORECASE)
        expanded.extend(sub)
    seen = set()
    out = []
    original_normalized = text.strip("?.!,;:").lower()
    for p in expanded:
        cleaned = p.strip().strip("?.!,;:")
        cleaned = re.sub(r"^\s*(?:and|or|also|plus|but|so|then)\s+", "", cleaned, flags=re.IGNORECASE).strip()
        if not cleaned: continue
        if len(cleaned.split()) < min_part_words: continue
        if cleaned.lower() == original_normalized: continue
        key = cleaned.lower()
        if key in seen: continue
        seen.add(key)
        out.append(cleaned)
    return out


def _normalize_match_text(value):
    """Normalize text for generic lexical/metadata matching."""
    text = _param_to_text(value).lower()
    text = re.sub(r"[_\-–—/]+", " ", text)
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _field_aliases(field):
    aliases = [field]
    aliases.extend(MEMORY_FIELD_SYNONYMS.get(field, []))
    seen = set()
    out = []
    for alias in aliases:
        norm = _normalize_match_text(alias)
        if norm and norm not in seen:
            seen.add(norm)
            out.append(norm)
    return out


def _all_memory_field_terms():
    terms = []
    for field in MEMORY_FIELD_SYNONYMS.keys():
        terms.extend(_field_aliases(field))
    return list(dict.fromkeys(t for t in terms if t))


def _phrase_matches_text(phrase, text_norm):
    phrase_norm = _normalize_match_text(phrase)
    if not phrase_norm or not text_norm:
        return False
    # For English/alnum phrases, require token boundaries. For CJK, substring is acceptable.
    if re.search(r"[a-z0-9]", phrase_norm):
        return bool(re.search(rf"(?<![a-z0-9]){re.escape(phrase_norm)}(?![a-z0-9])", text_norm))
    return phrase_norm in text_norm


def _contains_memory_field_terms(text):
    q_norm = _normalize_match_text(text)
    if not q_norm:
        return False
    return any(_phrase_matches_text(term, q_norm) for term in _all_memory_field_terms())


def _detect_memory_fields(text):
    """Detect remembered/profile fields from the query using MEMORY_FIELD_SYNONYMS.

    This is intentionally generic: adding a new field to MEMORY_FIELD_SYNONYMS
    automatically enables field detection, field subqueries,
    and structured metadata matching. No Topic-specific if/else should be needed.
    """
    if not isinstance(text, str):
        return []
    q_norm = _normalize_match_text(text)
    if not q_norm:
        return []
    fields = []
    for field in MEMORY_FIELD_SYNONYMS.keys():
        if any(_phrase_matches_text(alias, q_norm) for alias in _field_aliases(field)):
            fields.append(field)
    return list(dict.fromkeys(fields))


def _build_display_keywords_from_query(query, target_entity="Both", current_user="Guest", limit=16):
    """Build human-readable keywords for logs/tool arguments.

    v8: these keywords are also mirrored by structured keyword groups inside
    retrieval. The display list itself is still just for logging/tool clarity.
    Scoring uses _build_structured_keywords_from_query +
    _safe_keyword_evidence_score, not a flat keyword bag.
    """
    structured = _build_structured_keywords_from_query(
        query=query,
        target_entity=target_entity,
        current_user=current_user,
        raw_keywords="",
    )
    display = structured.get("display") or []
    return display[:limit]


def _query_token_stopwords():
    return {
        "the", "a", "an", "is", "are", "was", "were", "do", "does", "did",
        "i", "you", "your", "yourself", "my", "me", "we", "us", "our", "it",
        "that", "this", "these", "those", "to", "of", "in", "on", "for", "from",
        "what", "which", "when", "where", "how", "why", "who", "whom", "whose",
        "so", "and", "or", "also", "plus", "but", "can", "could", "would", "will",
        "some", "give", "please", "with", "about", "any", "other", "new", "recommend",
        "recommendation", "recommendations", "similar", "kind", "kinds", "type", "types",
        "memory", "memories", "record", "records", "database", "db", "lore", "topic",
        "field", "search", "seach", "check", "verify", "confirm", "name", "names",
    }


def _field_terms_for_keywords(field, max_aliases=8):
    terms = [field]
    terms.extend(MEMORY_FIELD_SYNONYMS.get(field, []))
    out, seen = [], set()
    for t in terms:
        norm = _normalize_match_text(t)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(norm)
        if len(out) >= max_aliases:
            break
    return out


def _build_structured_keywords_from_query(query, target_entity="Both", current_user="Guest", raw_keywords=""):
    """Build safe keyword evidence groups for MemorySearch-internal scoring.

    v19 cleanup:
    - This function is NOT used for current-turn intent routing.
    - It no longer harvests arbitrary tokens from the whole user sentence
      (e.g. should/focus/get/answer/maybe), because that polluted retrieval.
    - It only uses configured memory fields and explicit raw_keywords passed
      by a MemorySearch call.
    - Slot/domain distinction is preserved: slot fields support exact coverage;
      domain fields support related/domain ranking.
    """
    q = _param_to_text(query)
    raw_kw = _param_to_text(raw_keywords)
    fields = _detect_memory_fields(f"{q} {raw_kw}".strip())

    domain_fields = [f for f in fields if f in MEMORY_DOMAIN_FIELDS]
    slot_fields = [f for f in fields if f in MEMORY_SLOT_FIELDS]
    attribute_fields = [f for f in fields if f in MEMORY_ATTRIBUTE_FIELDS]

    # If a domain is present (gaming/interests/project), it gates attribute
    # words like favorite/like. Pure slots (birthday/full_name) remain required.
    required_fields = []
    optional_fields = []
    if domain_fields:
        required_fields.extend(domain_fields)
        optional_fields.extend(attribute_fields)
        required_fields.extend([f for f in slot_fields if f not in required_fields])
    else:
        required_fields.extend(slot_fields)
        required_fields.extend(attribute_fields)

    required, optional, display = [], [], []

    target = _canonical_target_entity(target_entity, current_user=current_user)
    if target in ("Eva", "Rosm", "Shared"):
        required.append(target)
        display.append(target)
    elif target == "Both":
        display.extend(["Eva", "Rosm", "shared"])

    for f in required_fields:
        terms = _field_terms_for_keywords(f)
        required.extend(terms)
        display.extend(terms[:5])
    for f in optional_fields:
        terms = _field_terms_for_keywords(f)
        optional.extend(terms)
        display.extend(terms[:5])

    # Explicit MemorySearch keywords are allowed, but still filtered. They are
    # treated as required evidence because they were intentionally supplied by
    # the model/user for this MemorySearch call, unlike arbitrary query tokens.
    stop = _query_token_stopwords() | {"should", "focus", "answer", "maybe", "advise", "tell", "news"}
    raw_parts = re.split(r"[,;\n/|]+", raw_kw)
    for part in raw_parts:
        term = _normalize_match_text(part)
        if not term or term in stop:
            continue
        # Reject very generic single verbs/adverbs. Keep multi-word phrases and
        # configured field/entity terms.
        toks = re.findall(r"\b[a-z0-9]{2,}\b", term)
        if len(toks) == 1 and toks[0] in stop:
            continue
        if term not in required:
            required.append(term)
            display.append(term)

    def dedupe(seq, limit=SAFE_KEYWORD_MAX_TERMS_PER_GROUP):
        out, seen = [], set()
        for item in seq:
            item = _normalize_match_text(item)
            if not item or item in seen:
                continue
            seen.add(item)
            out.append(item)
            if limit and len(out) >= limit:
                break
        return out

    required = dedupe(required)
    optional = dedupe(optional)
    display = dedupe(display, limit=32)

    return {
        "required": required,
        "optional": optional,
        "display": display,
        "fields": list(dict.fromkeys(fields)),
        "slot_fields": list(dict.fromkeys(slot_fields)),
        "domain_fields": list(dict.fromkeys(domain_fields)),
        "required_fields": list(dict.fromkeys(required_fields)),
        "optional_fields": list(dict.fromkeys(optional_fields)),
        "has_domain_gate": bool(domain_fields and attribute_fields),
    }

def _record_keyword_evidence_text(record):
    meta = record.get("meta", {}) or {}
    pieces = [
        str(record.get("vector_text", "")),
        str(record.get("content", "")),
        str(meta.get("category", "")),
        str(meta.get("topic", "")),
        _list_to_text(meta.get("keywords", [])),
    ]
    return _normalize_match_text(" ".join(p for p in pieces if p))


def _term_matches_record_text(term, record_text_norm, record_stems=None):
    term_norm = _normalize_match_text(term)
    if not term_norm or not record_text_norm:
        return False
    if _phrase_matches_text(term_norm, record_text_norm):
        return True
    toks = re.findall(r"\b[a-z0-9]{2,}\b", term_norm)
    if len(toks) == 1:
        if record_stems is None:
            record_stems = set(_stem_tokens(re.findall(r"\b\w{2,}\b", record_text_norm)))
        return _stem_token(toks[0]) in record_stems
    return False


def _matched_keyword_terms(terms, record_text_norm):
    if not terms:
        return []
    record_stems = set(_stem_tokens(re.findall(r"\b\w{2,}\b", record_text_norm)))
    matched = []
    for t in terms:
        if _term_matches_record_text(t, record_text_norm, record_stems=record_stems):
            matched.append(_normalize_match_text(t))
    return list(dict.fromkeys(matched))


def _safe_keyword_evidence_score(record, structured_keywords):
    if not ENABLE_SAFE_KEYWORD_EVIDENCE or not structured_keywords:
        return 0.0, [], {}

    required_terms = structured_keywords.get("required") or []
    optional_terms = structured_keywords.get("optional") or []
    has_domain_gate = bool(structured_keywords.get("has_domain_gate"))

    text_norm = _record_keyword_evidence_text(record)
    required_hits = _matched_keyword_terms(required_terms, text_norm)
    optional_hits = _matched_keyword_terms(optional_terms, text_norm)

    reasons = []
    score = 0.0

    if required_terms:
        if required_hits:
            score += min(0.32, SAFE_KEYWORD_REQUIRED_HIT_BONUS * len(required_hits))
            reasons.append("kw_required:" + "/".join(required_hits[:5]))
            if optional_hits:
                score += min(0.12, SAFE_KEYWORD_OPTIONAL_HIT_BONUS * len(optional_hits))
                reasons.append("kw_optional_after_required:" + "/".join(optional_hits[:5]))
        else:
            if has_domain_gate and optional_hits:
                score += SAFE_KEYWORD_GATE_FAIL_PENALTY
                reasons.append("kw_gate_failed_optional_only:" + "/".join(optional_hits[:5]))
    elif optional_hits:
        score += min(0.16, SAFE_KEYWORD_OPTIONAL_HIT_BONUS * len(optional_hits))
        reasons.append("kw_optional:" + "/".join(optional_hits[:5]))

    score = max(SAFE_KEYWORD_GATE_FAIL_PENALTY, min(SAFE_KEYWORD_MAX_BONUS, score))
    debug = {
        "required": required_terms,
        "optional": optional_terms,
        "required_hits": required_hits,
        "optional_hits": optional_hits,
        "has_domain_gate": has_domain_gate,
    }
    return score, list(dict.fromkeys(reasons)), debug


def _entity_for_query_text(query, target_entity, current_user="Guest"):
    target = _canonical_target_entity(target_entity, current_user=current_user)
    if target == "Rosm": ent = "Rosm"
    elif target == "Eva": ent = "Eva"
    elif target in ("Both", "Shared"): ent = "Rosm Eva shared"
    else: ent = target or "memory"
    q = _param_to_text(query)
    q_lower = q.lower()
    if any(x in q_lower for x in ["rosm", "eva", "master", "creator"]):
        return q
    return f"{ent} {q}".strip()


def _build_memory_query_variants_for_tool_call(query, target_entity, current_user="Guest"):
    """Build semantic query variants without forced keyword expansion.

    Use the original query plus natural split sub-questions only. Relevance is
    handled later by FAISS/BM25, metadata evidence, CrossEncoder rerank, and the
    forced-choice Memory Judge.
    """
    query = _param_to_text(query)
    if not query:
        return []
    variants = [query]
    variants.extend(_split_compound_question(query))
    out = []
    seen = set()
    for v in variants:
        v = (v or "").strip()
        key = v.lower()
        if v and key not in seen:
            seen.add(key)
            out.append(v)
    return out


def _stem_token(tok):
    """
    Light morphological normalization.

    Without this, BM25 token "gaming" in record keywords does not match
    BM25 query token "game", and the record gets near-zero retrieval score.
    Trace: a record with keywords=["Ballet","Gaming","Sketching","Doodling"]
    was missed across 4 query turns asking about games.

    Rules (intentionally conservative — over-stemming hurts more than it helps):
      ies   -> y    (memories -> memory, stories -> story)
      sses  -> ss   (kisses -> kiss; preserve original ss)
      ing   -> ""   (gaming -> gam, sketching -> sketch) only if stem >= 4 chars
      ed    -> ""   (loved -> lov, kissed -> kiss) only if stem >= 4 chars
      s     -> ""   (games -> game, books -> book) only if stem >= 4 chars
                    skip irregulars where final 's' is part of the lemma
    Stems are not real lemmas (gam/lov are stubs) but they are CONSISTENT
    between corpus and query, which is all BM25 needs.
    """
    if not tok or len(tok) <= 3:
        return tok
    t = tok.lower()
    if t.endswith("ies") and len(t) > 4:
        return t[:-3] + "y"
    if t.endswith("sses"):
        return t[:-2]
    if t.endswith("ing") and len(t) >= 6:
        return t[:-3]
    if t.endswith("ed") and len(t) >= 5:
        return t[:-2]
    # 's' suffix: only strip if not 'ss' / 'us' / 'is' (those are usually the lemma)
    if t.endswith("s") and not t.endswith(("ss", "us", "is")) and len(t) >= 5:
        return t[:-1]
    return t


def _stem_tokens(tokens):
    return [_stem_token(t) for t in tokens]


def _tokenize_memory_query(text, stopwords):
    raw = [w.lower() for w in re.findall(r"\b\w{2,}\b", text) if w.lower() not in stopwords]
    # matches corpus token 'gaming' (-> 'gam' on both sides).
    return _stem_tokens(raw)


def _inject_entity_pronouns(query, target_entity):
    query = _param_to_text(query)
    target_entity = _param_to_text(target_entity) or "Both"
    entity_aliases = {
        "Rosm": ["rosm", "master", "creator"],
        "Eva": ["eva", "maid"],
        "Shared": ["shared"],
        "Both": ["both", "shared", "rosm", "eva"],
    }
    query_lower = query.lower()
    aliases = entity_aliases.get(target_entity, [target_entity.lower()])
    if any(alias and alias in query_lower for alias in aliases):
        return query
    pronoun_map = {
        r"\bmy\b": f"{target_entity}'s",
        r"\byour\b": f"{target_entity}'s",
        r"\bthe user\b": target_entity,
        r"\bthe master\b": target_entity,
    }
    injected_query = query
    for pattern, replacement in pronoun_map.items():
        injected_query, count = re.subn(pattern, replacement, injected_query, flags=re.IGNORECASE)
        if count > 0: break
    return injected_query


def _is_cross_entity_query(search_query, current_user="Guest"):
    q = _param_to_text(search_query).lower()
    current = _param_to_text(current_user).lower()
    has_eva = any(x in q for x in ["eva", "maid"])
    has_rosm_or_user = any(x in q for x in ["rosm", "master", "creator"])
    if current and current not in {"guest", "user"}:
        has_rosm_or_user = has_rosm_or_user or (current in q)
    return has_eva and has_rosm_or_user


def _detect_record_fields(record):
    """Detect which configured fields are represented by a record.

    Uses the full searchable record text plus structured metadata. This makes
    the mechanism work for any future field added to MEMORY_FIELD_SYNONYMS,
    rather than only hard-coded fields such as Interests/Gaming.
    """
    meta = record.get("meta", {}) or {}
    metadata_text = " ".join([
        _param_to_text(meta.get("entity", "")),
        _param_to_text(meta.get("category", "")),
        _param_to_text(meta.get("topic", "")),
        _list_to_text(meta.get("participants", [])),
        _list_to_text(meta.get("keywords", [])),
    ])
    text_norm = _normalize_match_text(f"{_record_search_text(record)} {metadata_text}")
    fields = []
    for field in MEMORY_FIELD_SYNONYMS.keys():
        if any(_phrase_matches_text(alias, text_norm) for alias in _field_aliases(field)):
            fields.append(field)
    return list(dict.fromkeys(fields))


def _field_match_bonus(record, query):
    """Small schema-level field overlap bonus.

    v5 removes the old query-specific keyword-bag path. Field overlap is only a
    weak retrieval hint; the forced-choice judge is responsible for rejecting
    same-attribute but wrong-domain records such as favorite-food for
    favorite-game.
    """
    query_fields = _detect_memory_fields(query)
    record_fields = _detect_record_fields(record)
    if not query_fields or not record_fields:
        return 0.0
    overlap = set(query_fields) & set(record_fields)
    if not overlap:
        return 0.0
    return min(0.24, 0.08 * len(overlap))


def _metadata_values(record):
    meta = record.get("meta", {}) or {}
    values = {
        "entity": [_param_to_text(meta.get("entity", ""))],
        "category": [_param_to_text(meta.get("category", ""))],
        "topic": [_param_to_text(meta.get("topic", ""))],
        "participants": list(meta.get("participants", []) or []),
        "keywords": list(meta.get("keywords", []) or []),
    }
    return values


def _metadata_text(record):
    values = _metadata_values(record)
    pieces = []
    for vals in values.values():
        pieces.extend(_param_to_text(v) for v in vals if _param_to_text(v))
    return _normalize_match_text(" ".join(pieces))


def _match_token_set(text):
    norm = _normalize_match_text(text)
    toks = re.findall(r"\b\w{2,}\b", norm)
    stop = {
        "the", "and", "for", "with", "your", "you", "my", "me", "our", "ours",
        "memory", "memories", "record", "records", "database", "search", "seach",
        "check", "verify", "confirm", "please", "about", "what", "when", "where",
        "how", "why", "who", "this", "that", "topic", "field", "category", "entity",
    }
    return set(_stem_tokens([t for t in toks if t not in stop]))


def _metadata_match_bonus(record, query, target_entity="Both", detected_fields=None):
    """Generic structured metadata bonus for memory records.

    This is deliberately NOT a Topic=Interests special case. It treats every
    record's meta.entity/category/topic/participants/keywords as a second
    retrieval channel. Any future topic or keyword can be protected when the
    user's query explicitly overlaps with the metadata.
    """
    values = _metadata_values(record)
    q_norm = _normalize_match_text(query)
    target = _canonical_known_entity_name(target_entity)
    entity = _canonical_record_entity(values["entity"][0] if values["entity"] else "Unknown")
    meta_norm = _metadata_text(record)
    fields = set(detected_fields or _detect_memory_fields(query))

    bonus = 0.0
    reasons = []

    # Entity alignment is generic, but entity-only evidence is not enough to
    # preserve low-confidence records by itself.
    if target and target not in ("Both", "Shared") and entity == target:
        bonus += 0.22
        reasons.append("meta_entity_exact")
    elif target in ("Both", "Shared") and entity in ("Eva", "Rosm", "Shared"):
        bonus += 0.06
        reasons.append("meta_entity_broad")

    # Exact metadata phrase matches: topic/category/keywords/participants.
    for kind in ("topic", "category", "participants", "keywords"):
        for raw in values.get(kind, []):
            val = _normalize_match_text(raw)
            if not val:
                continue
            if _phrase_matches_text(val, q_norm):
                weight = {
                    "topic": 0.48,
                    "category": 0.22,
                    "participants": 0.18,
                    "keywords": 0.32,
                }.get(kind, 0.15)
                bonus += weight
                reasons.append(f"meta_{kind}_phrase:{val}")

    # Field-to-metadata alignment: if the query detected a configured field and
    # that field's name/synonyms appear in topic/keywords/category/participants,
    # boost it. This works for Interests today and any new fields tomorrow.
    for field in fields:
        aliases = _field_aliases(field)
        if any(_phrase_matches_text(alias, meta_norm) for alias in aliases):
            bonus += 0.34
            reasons.append(f"meta_field_alignment:{field}")

    # Token-overlap fallback between query and metadata. This helps when phrases
    # differ slightly, while still requiring structured metadata evidence.
    q_tokens = _match_token_set(query)
    m_tokens = _match_token_set(meta_norm)
    overlap = q_tokens & m_tokens
    if len(overlap) >= 2:
        bonus += min(0.24, 0.08 * len(overlap))
        reasons.append("meta_token_overlap:" + "/".join(sorted(list(overlap))[:5]))
    elif len(overlap) == 1 and fields:
        bonus += 0.08
        reasons.append("meta_token_overlap:" + next(iter(overlap)))

    return min(1.25, bonus), list(dict.fromkeys(reasons))


def _record_has_strong_metadata_evidence(record):
    """Whether metadata evidence is strong enough to protect a record.

    Entity-only evidence is intentionally excluded; otherwise every Eva record
    could be protected for an Eva query. Strong evidence must come from topic,
    category/keyword/participant phrase match, field alignment, or enough
    generic metadata overlap.
    """
    bonus = float(record.get("metadata_bonus", 0.0) or 0.0)
    reasons = record.get("metadata_reasons", []) or []
    if bonus < 0.32:
        return False
    strong_prefixes = (
        "meta_topic_phrase:",
        "meta_category_phrase:",
        "meta_keywords_phrase:",
        "meta_participants_phrase:",
        "meta_field_alignment:",
        "meta_token_overlap:",
    )
    return any(str(r).startswith(strong_prefixes) for r in reasons)

def _collection_has_strong_metadata_evidence(collected):
    for rec in (collected.get("records") or []):
        if _record_has_strong_metadata_evidence(rec):
            return True
    return False


def _retrieval_record_limit():
    """How many broad-retrieval candidates to keep before record labeling.

    Retrieval should be wide enough to let the labeler see candidates that
    might be subject-mismatched or topic-mismatched (and thus filtered out
    as WRONG). The final formatter still only displays
    MEMORY_JUDGE_KEEP_TOP_K records.
    """
    return max(FINAL_TOP_K, MEMORY_JUDGE_TOP_K)


def _truncate_for_judge(text, max_chars):
    text = _param_to_text(text)
    if max_chars and len(text) > max_chars:
        return text[:max_chars].rstrip() + "..."
    return text

def _infer_memory_target_from_text(text, default_target="Both", current_user="Guest"):
    """Infer memory target from pronouns plus detected memory/profile fields.

    Generic rule:
      - Explicit Eva/Rosm names win.
      - In Eva's dialogue, "your/you/yourself" + any configured memory field
        points to Eva.
      - "my/me/I" + any configured memory field points to the current user.

    Adding a new field to MEMORY_FIELD_SYNONYMS automatically participates in
    this target inference; no topic-specific rule is required.
    """
    q = _normalize_match_text(text)
    if not q:
        return default_target

    if re.search(r"(?<![a-z0-9])(eva|maid)(?![a-z0-9])", q):
        return "Eva"
    if re.search(r"(?<![a-z0-9])(rosm|master|creator)(?![a-z0-9])", q):
        return "Rosm"

    has_field = bool(_detect_memory_fields(text)) or _contains_memory_field_terms(text)
    has_memory_word = bool(re.search(r"(?<![a-z0-9])(memory|memories|record|records|lore|profile|database|db|topic|field)(?![a-z0-9])", q))

    current = _canonical_known_entity_name(current_user or "Guest")

    # First-person ownership should beat incidental second-person wording,
    # but only when it is actually possessive/self-referential.  A bare "I"
    # in "would you like I send a toy" must NOT steal the target away from
    # "your toy" / "your birthday gift" (Eva).
    if current != "Guest" and (has_field or has_memory_word):
        owns_field = bool(re.search(r"(?<![a-z0-9])(my|mine|myself)(?![a-z0-9])", q))
        self_fact_verb = bool(re.search(
            r"(?<![a-z0-9])i\s+(?:like|love|prefer|enjoy|remember|forgot|said|told|mentioned|"
            r"gave|messed|visited|played|have|had|am|was)(?![a-z0-9])",
            q,
        ))
        if owns_field or self_fact_verb:
            return current

    if (has_field or has_memory_word) and re.search(r"(?<![a-z0-9])(your|yourself|you)(?![a-z0-9])", q):
        return "Eva"

    return default_target

def _compute_entity_bonus(db_ent, preferred_entity, target_entity, is_cross_entity):
    db_ent = _canonical_record_entity(db_ent)
    preferred_entity = _canonical_known_entity_name(preferred_entity)
    target_entity = _canonical_known_entity_name(target_entity)
    if not is_cross_entity:
        if db_ent == preferred_entity: return 0.20
        if db_ent == "Shared" or preferred_entity in ("Both", "Shared"): return 0.10
        return 0.0
    if db_ent == "Shared": return 0.15
    if db_ent == target_entity: return 0.15
    if db_ent in ("Rosm", "Eva"): return 0.10
    return 0.0


def _collect_memory_records(params, memory_state, encoder, reranker, current_user="Guest"):
    """Broad memory retrieval with safe keyword evidence.

    v13 separation of concerns:
      - Routing does NOT use profile/domain keyword bags to decide tools.
      - Inside MemorySearch, keywords may still be used as role-aware retrieval
        evidence after the tool has already been selected.

    Retrieval uses original query text + entity/pronoun injection + FAISS/BM25 +
    small field/metadata evidence + safe keyword evidence + CrossEncoder rerank +
    optional forced-choice Memory Judge.
    """
    if not memory_state or encoder is None or reranker is None:
        return {
            "target_entity": "",
            "keywords": [],
            "search_query": "",
            "records": [],
            "top1_score": 0.0,
            "error": "Memory module unavailable or DB empty.",
        }

    original_query = _param_to_text(params.get("query", ""))
    raw_keywords = _param_to_text(params.get("keywords", ""))
    target_entity = _canonical_target_entity(params.get("target_entity", "Both"), current_user=current_user)

    # Compatibility fallback only: never use keyword bags when a real query exists.
    if not original_query and raw_keywords:
        original_query = raw_keywords

    target_entity = _infer_memory_target_from_text(
        original_query,
        default_target=target_entity,
        current_user=current_user,
    )

    if not original_query:
        return {
            "target_entity": target_entity,
            "keywords": [],
            "search_query": "",
            "records": [],
            "top1_score": 0.0,
            "error": "No valid memory query was provided.",
        }

    # Build role-aware keywords for MemorySearch-internal retrieval only.
    # This does not affect current-turn tool routing.
    structured_keywords = _build_structured_keywords_from_query(
        query=original_query,
        target_entity=target_entity,
        current_user=current_user,
        raw_keywords=raw_keywords,
    )
    keywords = structured_keywords.get("display") or []
    entity_query = _entity_for_query_text(original_query, target_entity=target_entity, current_user=current_user)
    search_query = _inject_entity_pronouns(entity_query, target_entity)
    query_tokens = _tokenize_memory_query(search_query, memory_state["stopwords"])
    is_cross_entity = _is_cross_entity_query(search_query, current_user=current_user)
    preferred_entity = "Shared" if is_cross_entity else target_entity

    total_docs = len(memory_state["db_records"])
    if total_docs <= 0:
        return {
            "target_entity": target_entity,
            "keywords": keywords,
            "search_query": search_query,
            "records": [],
            "top1_score": 0.0,
            "error": None,
        }

    emb = encoder.encode([search_query], normalize_embeddings=True)
    faiss_distances, faiss_indices = memory_state["index"].search(np.array(emb, dtype="float32"), total_docs)

    faiss_scores_raw = np.zeros(total_docs, dtype=np.float32)
    for score, idx in zip(faiss_distances[0], faiss_indices[0]):
        if idx != -1:
            faiss_scores_raw[int(idx)] = float(score)

    faiss_norm = _normalize_scores(faiss_scores_raw)
    bm25_raw = memory_state["bm25"].get_scores(query_tokens)
    bm25_norm = _normalize_scores(bm25_raw)
    detected_fields = _detect_memory_fields(search_query)

    candidates = []
    for idx in range(total_docs):
        record = memory_state["db_records"][idx]
        meta = record.get("meta", {}) or {}
        db_ent = _canonical_record_entity(meta.get("entity", "Unknown"))
        f_score = float(faiss_norm[idx])
        b_score = float(bm25_norm[idx])
        entity_bonus = _compute_entity_bonus(
            db_ent=db_ent,
            preferred_entity=preferred_entity,
            target_entity=target_entity,
            is_cross_entity=is_cross_entity,
        )
        field_bonus = _field_match_bonus(record, search_query)
        metadata_bonus, metadata_reasons = _metadata_match_bonus(
            record,
            search_query,
            target_entity=target_entity,
            detected_fields=detected_fields,
        )
        keyword_bonus, keyword_reasons, keyword_debug = _safe_keyword_evidence_score(
            record,
            structured_keywords,
        )
        if (
            f_score < 0.1
            and b_score < 0.1
            and field_bonus <= 0.0
            and metadata_bonus <= 0.0
            and keyword_bonus <= 0.0
        ):
            continue

        final_score = (
            FAISS_WEIGHT * f_score
            + BM25_WEIGHT * b_score
            + entity_bonus
            + field_bonus
            + metadata_bonus
            + keyword_bonus
        )
        candidates.append({
            "content": _record_display_content(record),
            "rerank_text": _record_rerank_text(record),
            "vector_text": record.get("vector_text", ""),
            "entity": db_ent,
            "category": meta.get("category", "Lore"),
            "topic": meta.get("topic", ""),
            "score": final_score,
            "faiss_raw": float(faiss_scores_raw[idx]),
            "faiss_norm": f_score,
            "bm25_raw": float(bm25_raw[idx]),
            "bm25_norm": b_score,
            "entity_bonus": entity_bonus,
            "field_bonus": field_bonus,
            "metadata_bonus": metadata_bonus,
            "metadata_reasons": metadata_reasons,
            "keyword_bonus": keyword_bonus,
            "keyword_reasons": keyword_reasons,
            "keyword_debug": keyword_debug,
        })

    if not candidates:
        return {
            "target_entity": target_entity,
            "keywords": keywords,
            "search_query": search_query,
            "records": [],
            "top1_score": 0.0,
            "error": None,
        }

    candidates.sort(key=lambda x: x["score"], reverse=True)
    top_k_candidates = candidates[:max(RERANK_TOP_K, MEMORY_JUDGE_TOP_K)]
    cross_inp = [[search_query, c["rerank_text"]] for c in top_k_candidates]
    rerank_scores = reranker.predict(cross_inp)
    scored_results = sorted(zip(rerank_scores, top_k_candidates), key=lambda x: x[0], reverse=True)
    top1_score = float(scored_results[0][0])

    final_records = []
    for r_score, item in scored_results[:_retrieval_record_limit()]:
        r_score = float(r_score)
        if final_records:
            below_cutoff = r_score < (top1_score - RERANK_CUTOFF)
            strong_field_match = float(item.get("field_bonus", 0.0)) >= 0.18
            strong_metadata_match = bool(item.get("metadata_reasons")) and float(item.get("metadata_bonus", 0.0)) >= 0.45
            strong_keyword_match = float(item.get("keyword_bonus", 0.0)) >= 0.20
            if below_cutoff and not strong_field_match and not strong_metadata_match and not strong_keyword_match:
                break
        final_records.append({
            "content": item["content"],
            "entity": item["entity"],
            "category": item["category"],
            "topic": item.get("topic", ""),
            "rerank_score": r_score,
            "low_confidence": r_score < LOW_CONFIDENCE_THRESHOLD,
            "source_query": search_query,
            "source_original_query": original_query,
            "source_keywords": keywords,
            "field_bonus": item.get("field_bonus", 0.0),
            "entity_bonus": item.get("entity_bonus", 0.0),
            "metadata_bonus": item.get("metadata_bonus", 0.0),
            "metadata_reasons": item.get("metadata_reasons", []),
            "keyword_bonus": item.get("keyword_bonus", 0.0),
            "keyword_reasons": item.get("keyword_reasons", []),
            "keyword_debug": item.get("keyword_debug", {}),
        })

    return {
        "target_entity": target_entity,
        "keywords": keywords,
        "search_query": search_query,
        "records": final_records,
        "top1_score": top1_score,
        "error": None,
    }


# ============ Slot-level memory evidence extraction ============
def _detect_requested_slot_fields(text, encoder=None):
    """Return the list of `MEMORY_SLOT_FIELDS` keys plausibly requested
    by the user's query.

    Slot detection is two-stage:

    1. **Subject classification** (eva_subject_classifier): is the query
       about a Person, or a NonPerson (pet/place/object/media)? Layer 1
       regex covers ~90% of cases at zero cost; Layer 2 embedding
       nearest-neighbor optionally refines edge cases when `encoder`
       is provided.

    2. **Per-slot applicability** (`SLOT_APPLICABLE_SUBJECTS`): each
       slot declares which subject classes it applies to. Person-only
       slots (full_name / birthday / age / toy) are skipped when the
       subject classifier returns NonPerson.

    Replaces the pre-2026-05-11 `full_name_blocked` negative-list
    pattern (which grew per noun-category — places in P1.7.2,
    pets in P5.3) with a positive subject check that scales without
    additional regex per category.

    Args:
        text: raw user query.
        encoder: optional callable for Layer 2 embedding classification.
            Pass `agent.encoder.encode` (lambda-wrapped) when available;
            None falls back to Layer 1 only.
    """
    q_norm = _normalize_match_text(text or "")
    if not q_norm:
        return []

    from eva_subject_classifier import is_person_subject
    person_query = is_person_subject(text, encoder=encoder)

    slots = []
    for slot, aliases in MEMORY_SLOT_FIELDS.items():
        applicable = SLOT_APPLICABLE_SUBJECTS.get(slot, set())
        # If the slot declares a Person-only restriction and the query
        # isn't about a person, suppress the slot. Slots without an
        # entry in SLOT_APPLICABLE_SUBJECTS fire unconditionally
        # (back-compat for any future non-person slot).
        if applicable == {"Person"} and not person_query:
            continue
        if any(_phrase_matches_text(alias, q_norm) for alias in [slot, *aliases]):
            slots.append(slot)
    return list(dict.fromkeys(slots))


def _target_name_patterns(target_entity):
    target = _canonical_known_entity_name(target_entity or "Both")
    if target == "Rosm":
        return ["rosm", "master", "creator", "eva's creator"]
    if target == "Eva":
        return ["eva", "maid"]
    return [target.lower()] if target and target not in ("Both", "Shared") else []


def _record_can_support_target_slot(record, target_entity, slot):
    target = _canonical_known_entity_name(target_entity or "Both")
    rec_ent = _canonical_record_entity(record.get("entity", ""))
    content_norm = _normalize_match_text(record.get("content", ""))
    if target in ("Both", "Shared", ""):
        return True
    if rec_ent == target:
        return True
    if rec_ent == "Shared":
        return any(_phrase_matches_text(p, content_norm) for p in _target_name_patterns(target))
    return False


def _clean_extracted_value(value):
    value = (value or "").strip()
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"\s*(?:,|\.|;|!|\?)\s*$", "", value)
    value = re.split(r"\s+but\s+|\s+and\s+|\s+though\s+", value, maxsplit=1, flags=re.I)[0].strip()
    return value




def _attach_slot_evidence_to_collection(collected, requested_slots=None, target_entity=None):
    out = dict(collected or {})
    target = _canonical_known_entity_name(target_entity or out.get("target_entity") or "Both")
    slots = list(dict.fromkeys([s for s in (requested_slots or []) if s in MEMORY_SLOT_FIELDS]))
    out["requested_slots"] = slots
    out["slot_evidence"] = {}
    out["missing_slots"] = []
    if not slots:
        return out
    records = list(out.get("records") or [])
    slot_evidence = {}
    supporting_contents = set()
    # Lazy import: eva_slots imports leaf helpers from THIS module at top
    # level, so we cannot reciprocate at module top without a cycle. The
    # import is hoisted out of the inner loop for perf — it's already
    # cached in sys.modules after the first call.
    from eva_slots import _extract_slot_value_from_record
    for slot in slots:
        for rec in records:
            value = _extract_slot_value_from_record(rec, target, slot)
            if value:
                slot_evidence[slot] = {
                    "value": value,
                    "source": rec.get("content", ""),
                    "entity": target,
                    "record_entity": rec.get("entity", ""),
                    "topic": rec.get("topic", ""),
                }
                supporting_contents.add((rec.get("content") or "").strip())
                break
    missing = [s for s in slots if s not in slot_evidence]
    out["slot_evidence"] = slot_evidence
    out["missing_slots"] = missing
    out["slot_supporting_contents"] = list(supporting_contents)
    if supporting_contents:
        out["records"] = [r for r in records if (r.get("content") or "").strip() in supporting_contents]
    else:
        target_records = []
        for r in records:
            ent = _canonical_record_entity(r.get("entity", ""))
            if target in ("Both", "Shared") or ent == target:
                target_records.append(r)
        out["records"] = target_records[:3]
    out["slot_exact_count"] = len(slot_evidence)
    out["slot_missing_count"] = len(missing)
    if slot_evidence:
        out["exact_answer_found"] = True
        out["judge_exact_count"] = max(int(out.get("judge_exact_count") or 0), len(slot_evidence))
    return out


def _format_slot_evidence_block(collected):
    slots = collected.get("requested_slots") or []
    if not slots:
        return ""
    evidence = collected.get("slot_evidence") or {}
    target = collected.get("target_entity") or "Both"
    lines = [f"[SLOT EVIDENCE for {target}]"]
    for slot in slots:
        item = evidence.get(slot)
        if item:
            lines.append(f"- {slot}: FOUND = {item.get('value', '')}")
            src = item.get("source") or ""
            if src:
                lines.append(f"  Source: {src}")
        else:
            lines.append(f"- {slot}: MISSING")
    return "\n".join(lines)


def _format_memory_records_block(collected):
    target_entity = collected.get("target_entity") or "Both"
    keywords = collected.get("keywords") or []
    records = collected.get("records") or []
    top1_score = float(collected.get("top1_score") or 0.0)
    error = collected.get("error")
    if error == "No valid memory query was provided.":
        return (f"### [MEMORY MODULE DATA for '{target_entity}'] ###\n"
                f"No valid memory query was provided.\n\n"
                f"[STRICT]: Ask for a specific memory fact or say you don't have enough information.")
    if error:
        return error
    if not records:
        # When lore-corpus retrieval is empty, a saved note may still match
        # (the user-told fact lives only in Notes/, not in the main corpus).
        # Render a notes-only section before falling through to the
        # "no relevant records" message.
        notes = collected.get("notes") or []
        if notes:
            pieces = [
                f"### [MEMORY MODULE DATA for '{target_entity}'] ###",
                "(no lore-corpus records matched — saved notes below)",
                "",
                ">>> SAVED NOTES (pass record_id to ForgetMemory to delete) <<<",
            ]
            for i, rec in enumerate(notes, start=1):
                note_tag = f"[Note #{rec.get('note_id', '?')}]"
                topic_tag = f" [Topic: {rec.get('topic')}]" if rec.get("topic") else ""
                pieces.append(
                    f"  Note {i} {note_tag} [Subject: {rec.get('entity', '?')}]"
                    f"{topic_tag}: {rec.get('content', '')}"
                )
                # R-1: notes-only 路径同样渲染 slot_values（与主路径对齐）
                slot_vals = rec.get("slot_values") or {}
                if slot_vals:
                    for slot_name, slot_val in slot_vals.items():
                        pieces.append(f"    - {slot_name}: {slot_val}")
                # R-7: event 字段也在 notes-only 路径渲染
                event = rec.get("event") or {}
                if event:
                    event_parts = []
                    if event.get("date"): event_parts.append(f"date={event['date']}")
                    if event.get("time"): event_parts.append(f"time={event['time']}")
                    if event.get("type"): event_parts.append(f"type={event['type']}")
                    if event.get("participants"):
                        event_parts.append(f"participants={','.join(event['participants'])}")
                    if event_parts:
                        pieces.append(f"    [event] {' | '.join(event_parts)}")
            pieces.append(">>> END SAVED NOTES <<<")
            return "\n".join(pieces)

        judge_line = ""
        if collected.get("judge_applied"):
            judge_line = ("\n\n[MEMORY JUDGE RESULT]: The forced-choice judge found no EXACT or RELATED "
                          "records among the broad-retrieval candidates. Treat this as no usable memory evidence.")
        return (f"### [MEMORY MODULE DATA for '{target_entity}'] ###\n"
                f"No relevant records found.{judge_line}\n\n"
                f"[STRICT]: If you don't have a record of this fact, say so. Do not invent.")
    # P1.0: noise reduction in the records block injected into the model.
    # Removed surfaces:
    #   - per-record retrieval_debug (field_bonus / metadata / keyword / judge_scores)
    #   - per-record matched_by_query
    #   - [KEYWORDS USED - MEMORYSEARCH INTERNAL] tail
    #   - [LOW-CONFIDENCE QUERY] explanation paragraph (kept as a short tag instead)
    #   - long [STRICT] anti-transfer footer (subsumed by the wrapper Rules line)
    # Slot evidence FOUND values are promoted to the very top with a high-salience
    # marker so they outrank the surrounding RELATED records in the model's
    # attention. RELATED record list is capped: 1 supporting record when a slot
    # is FOUND, up to 3 records when nothing is FOUND.
    slot_block = _format_slot_evidence_block(collected)
    has_slot_found = bool(collected.get("slot_evidence"))
    record_cap = 1 if has_slot_found else 3
    body_pieces = []
    if slot_block:
        body_pieces.append(">>> ANSWER VALUE (use this as the fact) <<<")
        body_pieces.append(slot_block)
        body_pieces.append(">>> END ANSWER VALUE <<<")
        if records:
            body_pieces.append("")
            body_pieces.append("Supporting record (background only):")
    for i, rec in enumerate(records[:record_cap], start=1):
        confidence_tag = " [low-confidence]" if rec.get("low_confidence") else ""
        judge_tag = f" [Judge: {rec.get('judge_label')}]" if rec.get("judge_label") else ""
        topic_tag = f" [Topic: {rec.get('topic')}]" if rec.get("topic") else ""
        body_pieces.append(
            f"Record {i} [{rec['category']}] [Subject: {rec['entity']}]"
            f"{topic_tag}{confidence_tag}{judge_tag}: {rec['content']}"
        )

    # Render saved user notes in a dedicated section below the lore
    # records. Always shown when present (no cap), so the model reliably
    # sees what the user just asked it to remember even if lore retrieval
    # scored higher on this turn.
    notes = collected.get("notes") or []
    if notes:
        body_pieces.append("")
        body_pieces.append(">>> SAVED NOTES (pass record_id to ForgetMemory to delete) <<<")
        for i, rec in enumerate(notes, start=1):
            note_tag = f"[Note #{rec.get('note_id', '?')}]"
            topic_tag = f" [Topic: {rec.get('topic')}]" if rec.get("topic") else ""
            body_pieces.append(
                f"  Note {i} {note_tag} [Subject: {rec.get('entity', '?')}]"
                f"{topic_tag}: {rec.get('content', '')}"
            )
            # R-1: 当 note 带 slot_values（模型 RememberThis 时填了结构化字段），
            # 在 SAVED NOTES 区域之下逐条渲染。这样 inference 阶段模型能直接
            # 读到 "pet_name: Peach"，不需要从散文里再解析。
            slot_vals = rec.get("slot_values") or {}
            if slot_vals:
                for slot_name, slot_val in slot_vals.items():
                    body_pieces.append(f"    - {slot_name}: {slot_val}")
            # R-7: event 字段同样渲染。结构化日程信息直接展示给模型，
            # 不依赖在散文里识别 "next Monday (2026-05-18)" 这种内嵌注释。
            event = rec.get("event") or {}
            if event:
                event_parts = []
                if event.get("date"): event_parts.append(f"date={event['date']}")
                if event.get("time"): event_parts.append(f"time={event['time']}")
                if event.get("type"): event_parts.append(f"type={event['type']}")
                if event.get("participants"):
                    event_parts.append(f"participants={','.join(event['participants'])}")
                if event_parts:
                    body_pieces.append(f"    [event] {' | '.join(event_parts)}")
        body_pieces.append(">>> END SAVED NOTES <<<")

    body = "\n".join(body_pieces)

    # 2026-05-13: when user-saved notes are present in this packet, the
    # LOW-CONFIDENCE / NO ELABORATION / MEMORY JUDGE warnings apply ONLY
    # to the lore records (records the runtime retrieved from the curated
    # corpus) — NOT to SAVED NOTES (facts the user explicitly stored via
    # RememberThis, which are authoritative). Without this scope guard,
    # Eva sees "Master wants to buy a birthday gift" in a SAVED NOTE,
    # then reads "you must say no memory" from the RELATED-only judge,
    # and falsely denies the user-stored fact (Turn 3 of 2026-05-13 trace).
    has_user_notes = bool(collected.get("notes"))
    _NOTES_SCOPE_SUFFIX = (
        " NOTE: this applies to the lore records section only. The "
        "`>>> SAVED NOTES <<<` block below contains facts the user explicitly "
        "stored — those are AUTHORITATIVE and should be cited verbatim when "
        "they answer the user's question."
    ) if has_user_notes else ""

    confidence_warning = ""
    if top1_score < HIGH_CONFIDENCE_BAR and not has_slot_found:
        confidence_warning = (
            "\n\n[LOW-CONFIDENCE EVIDENCE ONLY]: The lore records above are weak matches, not direct answers. "
            "If the user's specific question is not literally answered by the text above, you MUST say "
            "\"I don't have a specific memory about that\" (in Eva's persona) instead of inventing details "
            "such as places, actions, or possessions."
            "\n[NO ELABORATION RULE]: You may ONLY state facts that are literally written in the records "
            "above. Do NOT invent specific details such as: how someone looked, what they wore, what they "
            "said, exact actions (twirling, tripping, laughing), mood, weather, time, room descriptions, "
            "or any descriptive scene that is not in the text. If the user asks for details that are not "
            "in the records, say \"I remember it happened, but the specifics aren't recorded\" (in Eva's persona)."
            + _NOTES_SCOPE_SUFFIX
        )

    judge_note = ""
    if collected.get("judge_applied"):
        exact_n = int(collected.get("judge_exact_count") or 0)
        related_n = int(collected.get("judge_related_count") or 0)
        if exact_n == 0 and related_n > 0 and not has_slot_found:
            judge_note = (
                "\n\n[MEMORY JUDGE]: only RELATED lore records, no EXACT match. "
                "If the user asked for a precise slot or a specific past event and it is not literally "
                "stated above, say it is not recorded — do not paraphrase RELATED records into a direct answer."
                + _NOTES_SCOPE_SUFFIX
            )

    slot_missing_note = ""
    if collected.get("requested_slots"):
        missing = collected.get("missing_slots") or []
        if missing and not has_slot_found:
            slot_missing_note = "\n\n[SLOT COVERAGE]: missing slot(s): " + ", ".join(missing) + "."
    return (f"### [MEMORY MODULE DATA for '{target_entity}'] ###\n"
            f"{body}{confidence_warning}{judge_note}{slot_missing_note}")


# Marker the formatter writes around the user-notes section. Used by
# downstream slot-warning logic to detect "saved notes surfaced this
# turn" without re-parsing the records list.
_NOTES_BLOCK_MARKER = ">>> SAVED NOTES"


def memory_block_has_notes(obs):
    """True when a MemorySearch observation contains the saved-notes section.

    Used by the dispatch sites to suppress the [MISSING SLOTS] warning
    when saved notes may already answer the question. Production slot
    extraction only scans `[SLOT EVIDENCE for X]` blocks (lore-corpus
    schema), so a note like "Master adopted a cat named Peach" can't
    satisfy a slot detection on `full_name` even though it literally
    contains the answer the user wants. Without this suppression the
    model gets a "[MISSING SLOTS]: full_name — answer must say these
    are not recorded" directive that contradicts the note sitting right
    above it, and over-hedges.
    """
    return _NOTES_BLOCK_MARKER in (obs or "")


# Minimum mpnet cosine for a saved note to surface in the records block.
# Below this is "barely related" — showing it adds noise on every unrelated
# query. 0.25 is a balance: a query in the same language that touches the
# same noun phrases will easily clear it, while orthogonal queries won't.
_NOTES_MIN_COSINE = 0.25


def _attach_user_notes(collected, notes_store, query, top_k=5,
                       min_cosine=_NOTES_MIN_COSINE):
    """Attach runtime user-notes to a collected pool — in a SEPARATE
    bucket, not merged with lore-corpus records.

    Why separate: lore-corpus records carry CrossEncoder rerank scores
    (logit-style, can exceed 3.0); notes carry mpnet cosine ([0, 1]).
    The two are not comparable, so merging + re-sorting buries the note
    below lore records when lore rerank is high. Live retrieval then
    never surfaces "what the user just told me to remember", which
    defeats the whole point.

    The `_format_memory_records_block` formatter renders notes in a
    dedicated `>>> SAVED NOTES <<<` section, always shown when
    `collected['notes']` is non-empty.

    Args:
        min_cosine: filter threshold. Notes below this cosine are
            considered too unrelated to the query to surface (e.g.,
            user saved a cat note last week and is now asking about weather).
    """
    if not notes_store or not query:
        return collected
    try:
        results = notes_store.search(query, top_k=top_k)
    except Exception as e:
        print(f"[WARN] notes_store.search failed: {e}")
        return collected
    if not results:
        return collected

    notes = []
    for r in results:
        cosine = float(r.get("score", 0.0))
        if cosine < min_cosine:
            continue
        meta = r.get("meta", {}) or {}
        notes.append({
            "content": r.get("content", ""),
            "entity": meta.get("entity", "Shared"),
            "category": meta.get("category", "Lore"),
            "topic": meta.get("topic", ""),
            "rerank_score": cosine,        # for ordering within the bucket
            "low_confidence": cosine < 0.4,
            "source_query": query,
            "source_original_query": query,
            "source_keywords": [],
            "note_id": r.get("note_id"),
            # R-1: 透传 meta.slot_values 到 note dict，便于 _format_memory_records_block
            # 在 SAVED NOTES 区域渲染结构化 slot。
            "slot_values": meta.get("slot_values") or {},
            # R-7: 透传 meta.event 到 note dict（date / time / type / participants
            # / expires_at），渲染时和 slot_values 并列。
            "event": meta.get("event") or {},
        })

    if not notes:
        return collected

    notes.sort(key=lambda x: float(x.get("rerank_score", 0.0)), reverse=True)
    collected["notes"] = notes
    return collected


def _merge_memory_collections(collections):
    if not collections:
        return {"target_entity": "Both", "keywords": [], "records": [], "top1_score": 0.0,
                "search_queries": [], "error": None}
    target_entity = ""
    for c in collections:
        te = c.get("target_entity")
        if te:
            target_entity = te
            break
    merged_keywords = []
    protected_contents = set()
    seen_kw = set()
    for c in collections:
        for kw in c.get("keywords") or []:
            k = _normalize_match_text(kw)
            if k and k not in seen_kw:
                seen_kw.add(k)
                merged_keywords.append(str(kw).strip())
    if KEEP_TOP1_PER_SUBQUERY:
        for c in collections:
            records = c.get("records") or []
            for rec in records[:PROTECTED_RECORDS_PER_SUBQUERY]:
                content = (rec.get("content") or "").strip()
                if content:
                    protected_contents.add(content)
    record_by_content = {}
    for c in collections:
        for rec in c.get("records") or []:
            content = (rec.get("content") or "").strip()
            if not content: continue
            rec_copy = dict(rec)
            sq = rec_copy.get("source_query")
            rec_copy["source_queries"] = [sq] if sq else []
            existing = record_by_content.get(content)
            if existing is None:
                record_by_content[content] = rec_copy
            else:
                old_sources = existing.get("source_queries", [])
                new_sources = rec_copy.get("source_queries", [])
                merged_sources = list(dict.fromkeys(old_sources + new_sources))
                existing["source_queries"] = merged_sources
                if float(rec_copy.get("rerank_score", -1e9)) > float(existing.get("rerank_score", -1e9)):
                    rec_copy["source_queries"] = merged_sources
                    record_by_content[content] = rec_copy
    merged_records = sorted(record_by_content.values(),
                            key=lambda x: float(x.get("rerank_score", -1e9)), reverse=True)
    if merged_records:
        global_top1 = float(merged_records[0].get("rerank_score", 0.0))
        filtered = []
        for r in merged_records:
            content = (r.get("content") or "").strip()
            score = float(r.get("rerank_score", -1e9))
            keep_by_score = score >= (global_top1 - RERANK_CUTOFF)
            keep_by_protection = content in protected_contents
            keep_by_metadata = _record_has_strong_metadata_evidence(r)
            if keep_by_score or keep_by_protection or keep_by_metadata:
                r["low_confidence"] = bool(score < LOW_CONFIDENCE_THRESHOLD and not keep_by_metadata)
                r["protected_subquery_top1"] = keep_by_protection
                filtered.append(r)
            if len(filtered) >= _retrieval_record_limit():
                break
        merged_records = filtered
        top1 = global_top1
    else:
        top1 = float(max((c.get("top1_score", 0.0) for c in collections), default=0.0))
    errors = [c.get("error") for c in collections if c.get("error")]
    error = errors[0] if errors and len(errors) == len(collections) else None
    search_queries = [c.get("search_query", "") for c in collections if c.get("search_query")]
    return {
        "target_entity": target_entity or "Both",
        "keywords": merged_keywords,
        "records": merged_records,
        "top1_score": top1,
        "search_queries": search_queries,
        "error": error,
    }


def run_memory_search(params, memory_state, encoder, reranker, current_user="Guest", judge_fn=None):
    """MemorySearch tool execution.

    v5 keeps the tool signature compatible but removes the old generated-keyword
    expansion path. Each query variant is collected as-is, then merged, then
    optionally judged by the forced-choice EXACT / RELATED / WRONG filter.
    """
    if not memory_state or encoder is None or reranker is None:
        return "Memory module unavailable or DB empty."

    raw_query = _param_to_text(params.get("query", ""))
    raw_keywords = _param_to_text(params.get("keywords", ""))
    target_entity = _canonical_target_entity(params.get("target_entity", "Both"), current_user=current_user)

    # Compatibility fallback only: query should normally be non-empty.
    effective_query = raw_query or raw_keywords
    target_entity = _infer_memory_target_from_text(
        effective_query,
        default_target=target_entity,
        current_user=current_user,
    )

    variants = _build_memory_query_variants_for_tool_call(
        query=effective_query,
        target_entity=target_entity,
        current_user=current_user,
    )
    # by the current-turn router, but they can strengthen/diagnose MemorySearch.
    display_keywords_for_tool = ", ".join(
        _build_display_keywords_from_query(
            effective_query,
            target_entity=target_entity,
            current_user=current_user,
            limit=16,
        )
    )

    notes_store = memory_state.get("notes_store") if memory_state else None

    if not variants:
        collected = _collect_memory_records(
            params={"query": effective_query, "target_entity": target_entity, "keywords": display_keywords_for_tool},
            memory_state=memory_state,
            encoder=encoder,
            reranker=reranker,
            current_user=current_user,
        )
        if judge_fn is not None:
            collected = judge_fn(collected, effective_query)
        requested_slots = _detect_requested_slot_fields(effective_query, encoder=encoder)
        collected = _attach_slot_evidence_to_collection(collected, requested_slots, target_entity)
        if notes_store:
            collected = _attach_user_notes(collected, notes_store, effective_query)
        return _format_memory_records_block(collected)

    collections = []
    for q in variants:
        collected = _collect_memory_records(
            params={"query": q, "target_entity": target_entity, "keywords": display_keywords_for_tool},
            memory_state=memory_state,
            encoder=encoder,
            reranker=reranker,
            current_user=current_user,
        )
        collections.append(collected)

    merged = _merge_memory_collections(collections)
    if judge_fn is not None:
        merged = judge_fn(merged, effective_query)
    requested_slots = _detect_requested_slot_fields(effective_query)
    merged = _attach_slot_evidence_to_collection(merged, requested_slots, target_entity)
    if notes_store:
        merged = _attach_user_notes(merged, notes_store, effective_query)
    return _format_memory_records_block(merged)


__all__ = [
    'ENTITY_ALIASES',
    'USER_ENTITY_ALIASES',
    '_all_memory_field_terms',
    '_attach_slot_evidence_to_collection',
    '_attach_user_notes',
    'memory_block_has_notes',
    '_build_display_keywords_from_query',
    '_build_memory_query_variants_for_tool_call',
    '_build_structured_keywords_from_query',
    '_canonical_known_entity_name',
    '_canonical_record_entity',
    '_canonical_target_entity',
    '_clean_extracted_value',
    '_collect_memory_records',
    '_collection_has_strong_metadata_evidence',
    '_compute_entity_bonus',
    '_contains_memory_field_terms',
    '_detect_memory_fields',
    '_detect_record_fields',
    '_detect_requested_slot_fields',
    '_entity_for_query_text',
    '_field_aliases',
    '_field_match_bonus',
    '_field_terms_for_keywords',
    '_format_memory_records_block',
    '_format_slot_evidence_block',
    '_infer_memory_target_from_text',
    '_inject_entity_pronouns',
    '_is_cross_entity_query',
    '_list_to_text',
    '_match_token_set',
    '_matched_keyword_terms',
    '_merge_memory_collections',
    '_metadata_match_bonus',
    '_metadata_text',
    '_metadata_values',
    '_normalize_match_text',
    '_normalize_scores',
    '_param_to_text',
    '_phrase_matches_text',
    '_query_token_stopwords',
    '_record_can_support_target_slot',
    '_record_display_content',
    '_record_has_strong_metadata_evidence',
    '_record_keyword_evidence_text',
    '_record_rerank_text',
    '_record_search_text',
    '_retrieval_record_limit',
    '_safe_keyword_evidence_score',
    '_split_compound_question',
    '_stem_token',
    '_stem_tokens',
    '_target_name_patterns',
    '_term_matches_record_text',
    '_tokenize_memory_query',
    '_truncate_for_judge',
    'load_memory_resources',
    'run_memory_search',
]
