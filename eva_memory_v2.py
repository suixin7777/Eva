"""
eva_memory_v2.py — Topic-based memory module for Eva (P2 redesign).

Replaces the multi-layer hard-guard + score-only PRE MEMORY PROBE that lived
inside ChatAgent in P1. Public surface is small and intentional:

    MemoryModule.decide(user_text, ...) -> {"action": "skip"|"probe", ...}
    MemoryModule.probe(user_text, target, matched_topics, ...) -> retrieval bundle
    MemoryModule.format_packet(probe_result, user_text, ...) -> prompt packet

Design rules (per user spec):
1. PRE MEMORY PROBE is driven by `topic_keywords.json`, not by FAISS rerank
   floors. If the user query mentions a topic alias, we probe; else we skip
   memory unless slot/identity hard-guard says probe.
2. Probe is dual-path (Q3 = scheme C):
     path A: pull all DB records whose meta.topic ∈ matched_topics, filtered by
             entity. Auto-promote to EXACT — they already passed human curation.
     path B: standard FAISS+BM25+rerank from legacy. Fills semantic gaps.
   Merge dedupes by content; topic-direct hits keep their EXACT promotion.
3. format_packet adds a [STRICT MATCH RULE] block when the user query implies
   a relational predicate ("with me", "together") that the records do not
   literally support — the model is then told to answer precisely
   ("I performed for you, not exactly with you") rather than paraphrase.
4. topic_keywords.json is human-editable; you can add topics or aliases
   without touching code.

Imported primitives (from eva_core.py):
- run_memory_search()        — full FAISS+BM25+rerank pipeline
- _canonical_known_entity_name / _canonical_target_entity / _canonical_record_entity
- _normalize_match_text / _phrase_matches_text
- _format_memory_records_block / _attach_slot_evidence_to_collection
- _detect_requested_slot_fields / _is_explicit_memory_search_request_helpers
"""

import os
import re
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from eva_core import (
    _canonical_known_entity_name,
    _canonical_target_entity,
    _canonical_record_entity,
    _normalize_match_text,
    _phrase_matches_text,
    _format_memory_records_block,
    _attach_slot_evidence_to_collection,
    _detect_requested_slot_fields,
    _infer_memory_target_from_text,
    _collect_memory_records,
    _merge_memory_collections,
    _build_memory_query_variants_for_tool_call,
    _build_display_keywords_from_query,
)
from eva_config import (
    TOPIC_KEYWORDS_PATH,
    WEAK_RELATED_TOP1_BAR,
)


# ============================================================
# Topic dictionary loader
# ============================================================
class TopicDictionary:
    """Loads and queries topic_keywords.json.

    Hot-reload friendly: call .reload() to pick up edits without
    restarting the kernel (useful in Colab cells).
    """

    def __init__(self, path: Optional[str] = None):
        self.path = path or TOPIC_KEYWORDS_PATH
        self.topics: Dict[str, List[str]] = {}
        self.subject_hints: Dict[str, Optional[str]] = {}
        self._compiled: Dict[str, List[re.Pattern]] = {}
        self.reload()

    def reload(self) -> None:
        if not os.path.exists(self.path):
            print(f"[TopicDict] WARN: {self.path} not found. Topic probe will match nothing.")
            self.topics = {}
            self.subject_hints = {}
            self._compiled = {}
            return
        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.topics = dict(data.get("topics", {}) or {})
        self.subject_hints = dict(data.get("subject_hints", {}) or {})
        self._compile_patterns()
        print(f"[TopicDict] Loaded {len(self.topics)} topics from {self.path}")

    def _compile_patterns(self) -> None:
        compiled: Dict[str, List[re.Pattern]] = {}
        for topic, aliases in self.topics.items():
            pats = []
            for alias in aliases or []:
                a = (alias or "").strip()
                if not a:
                    continue
                # Word-boundary match. Aliases may contain spaces; we treat
                # multi-word aliases as substring with leading/trailing
                # word-boundary anchors so "free time" matches but
                # "interest" does not match "interesting".
                escaped = re.escape(a.lower())
                # Allow optional whitespace collapse (\s+) inside alias.
                escaped = escaped.replace(r"\ ", r"\s+")
                pats.append(re.compile(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", re.I))
            compiled[topic] = pats
        self._compiled = compiled

    def match(self, user_text: str) -> List[str]:
        """Return all topic labels whose aliases appear in user_text."""
        if not isinstance(user_text, str) or not user_text.strip():
            return []
        q = user_text.lower()
        matched = []
        for topic, pats in self._compiled.items():
            if any(p.search(q) for p in pats):
                matched.append(topic)
        return matched

    def subject_hint(self, topic: str) -> Optional[str]:
        v = self.subject_hints.get(topic)
        if not v:
            return None
        return _canonical_known_entity_name(v)


# ============================================================
# Hard-guard regex for slot/identity questions
# Replaces the three-layer guard from P1 with a single explicit rule.
# ============================================================
_HARD_GUARD_REGEX = re.compile(
    r"(?:"
    # Identity / name
    r"\bwhat(?:'s|\s+is)\s+(?:your|my|his|her|their)\s+(?:full\s+)?name\b"
    r"|\bdo\s+you\s+(?:know|remember)\s+(?:my|his|her|their)\s+name\b"
    r"|\bwho\s+(?:are|is)\s+(?:you|i|he|she|they)\b"
    # Birthday / date
    r"|\bwhen\s+(?:is|was)\s+(?:your|my|his|her|their)\s+birthday\b"
    r"|\bhow\s+(?:many|much)\s+days?\s+(?:until|till|before)\b"
    r"|\bdays?\s+until\s+(?:your|my|his|her|their)\s+birthday\b"
    # Possession / preference (toy/pet/favorite)
    r"|\bdo\s+you\s+have\s+(?:a|any)\s+(?:toy|pet|hobby|favorite|favourite)\b"
    r"|\bwhat(?:'s|\s+is)\s+your\s+favorite\s+(?:toy|game|food|color|colour|book|movie|song)\b"
    r"|\bwhat\s+(?:toy|pet|hobby)\s+do\s+you\s+have\b"
    # Memory / recall
    r"|\bdo\s+you\s+remember\s+\w+"
    r"|\b(?:check|search|look\s+up)\s+(?:your|the)\s+memory\b"
    r")",
    re.I,
)


def _hard_guard_must_probe(user_text: str) -> Tuple[bool, str]:
    """Return (True, reason) if the question is a slot/identity/possession
    type that MUST probe memory regardless of other heuristics.
    """
    if not isinstance(user_text, str) or not user_text.strip():
        return (False, "")
    if _HARD_GUARD_REGEX.search(user_text):
        return (True, "hard_guard_slot_identity_question")
    return (False, "")


# ============================================================
# Strict semantic-match rules (Q5 = strict honest answer)
# ============================================================
_RELATIONAL_PREDICATES = [
    # (regex on user_text, complementary forms in records that DO NOT match)
    (re.compile(r"\b(with\s+me|with\s+you|together|with\s+us)\b", re.I),
     [r"\bin\s+front\s+of\b", r"\bfor\s+(?:rosm|me|you)\b", r"\bto\s+(?:rosm|me|you)\b"]),
    (re.compile(r"\bwe\s+(?:went|visited|ate|had|saw|watched|played)\b", re.I),
     [r"\b(?:eva|rosm)\s+(?:went|visited|ate|had)\b"]),
]


def _detect_strict_match_caveat(user_text: str, observation_text: str) -> str:
    """If the user asks a relational/joint predicate ('with me', 'we did X')
    but the records only support a one-sided form ('in front of', 'for'),
    return a STRICT MATCH note for the prompt so the model answers
    precisely instead of paraphrasing.
    """
    if not user_text or not observation_text:
        return ""
    ut = user_text.lower()
    obs = observation_text.lower()
    for user_pat, record_alt_pats in _RELATIONAL_PREDICATES:
        if not user_pat.search(ut):
            continue
        # Check if the observation contains any one-sided alt (which would be
        # narrower than what the user asked).
        for alt in record_alt_pats:
            if re.search(alt, obs):
                return (
                    "[STRICT MATCH RULE]: The user asked about a JOINT/SHARED action "
                    "('with me' / 'together' / 'we did X'), but the memory records "
                    "only describe a one-sided form (e.g. 'in front of', 'for'). "
                    "You MUST distinguish: answer precisely (e.g. 'I performed for "
                    "you, not exactly with you') instead of paraphrasing the "
                    "one-sided form as 'with you / together'. Do NOT invent a "
                    "joint version of the event."
                )
    return ""


# ============================================================
# Path A' (Step 1.5) — meta-keyword direct match gate
#
# Eva's memory_meta.json populates a `keywords` field on every record
# with proper nouns and concrete entity terms (e.g. ["Apex Legends",
# "Curiosity"], ["NieR Automata", "Yoko Taro"]). PRE PROBE was
# leaving this curated entity index unused — only meta.topic was
# consulted. Path A' fixes the named-entity under-injection case
# (TODO 1 mirror failure: "what do you think of NieR Automata?")
# by direct verbatim match against meta.keywords.
#
# This gate decides which keywords are eligible for matching. The
# field mixes proper nouns ("Apex Legends", "Real Name") with
# generic descriptor tags ("Date", "Color", "Curiosity", "Cheerful")
# that would cause false positives if matched literally — e.g. "what
# is the date today?" should not pull the Birthday record via the
# "Date" keyword.
#
# Conservative Step 1.5 rule: ONLY multi-word keywords are eligible.
# - Multi-word entries are nearly always concrete entities and safe
#   to verbatim-match.
# - Single-word proper nouns ("Battlefield", "JRPGs", "Rosmarinus")
#   remain reachable via topic_keywords.json alias backstop.
# - Single-word descriptors ("Date", "Curiosity", "Cheerful") are
#   correctly excluded.
#
# This rule can be relaxed later (e.g. add explicit allowlist for
# acronyms) once regression measures the actual under-match rate.
# ============================================================
def _is_meta_keyword_specific(keyword) -> bool:
    """Return True iff `keyword` is eligible for Path A' verbatim match.

    Step 1.5 gate: requires the keyword to contain at least one
    whitespace character (i.e. multi-word). See block-level comment
    above for rationale.
    """
    if not isinstance(keyword, str):
        return False
    return bool(re.search(r"\s", keyword.strip()))


# ============================================================
# IntentClassifier — pluggable layer for topic-relevance decisions
#
# Step 1 of TODO 4: extract the "given topic-keyword candidates,
# decide which ones are actually relevant to the user's intent"
# decision into a pluggable interface so Step 2/3 can layer an LLM
# judge on top without rewriting MemoryModule.decide.
#
# The current keyword path acts as the trivial baseline:
# every candidate the regex matched is considered relevant. This
# preserves today's behaviour exactly.
#
# The interface intentionally takes `candidate_topics`, not raw
# user_text alone — keyword matching has already done the cheap
# scan, and the classifier's job is to confirm/filter, not to
# search the full topic universe. This matches the locked design
# in TODO.md (situation (a) only).
# ============================================================
@dataclass
class IntentResult:
    """Output of an IntentClassifier.

    relevant_topics:
        Subset of the candidate topics that the classifier judged
        actually relevant to the user query. Empty list = "skip"
        (caller falls through to the no-topic branch).

    skip_reason:
        Optional reason string used when relevant_topics is empty
        and the caller wants a more specific skip explanation than
        the default "no_topic_match_and_no_hard_guard". For the
        keyword classifier this stays None (skip will only happen
        when candidate_topics itself was empty, which is reported
        upstream).

    confidence:
        Calibration value, [0.0, 1.0]. For the keyword classifier
        this is always 1.0 when topics are returned (the regex
        matched, by definition). LLM classifiers will populate
        this from their per-topic verdict.

    source:
        Free-form tag for log/debug. Suggested values:
        "keyword", "llm", "layered:<inner>", "fallback".
    """
    relevant_topics: List[str]
    skip_reason: Optional[str] = None
    confidence: float = 1.0
    source: str = "keyword"


class IntentClassifier(ABC):
    """Abstract base for topic-relevance classifiers used by PRE PROBE.

    Implementations decide which of the keyword-matched candidate
    topics are actually relevant to the user's intent. They MUST
    NOT decide whether to probe at all — that's MemoryModule.decide's
    job, which composes IntentClassifier with hard guards and the
    Path A' gate.

    Implementations MUST be deterministic for a given (user_text,
    candidate_topics, current_user). LLM-backed classifiers achieve
    this with temperature=0 and per-turn caching (see Step 2/3).
    """

    @abstractmethod
    def classify(
        self,
        user_text: str,
        candidate_topics: List[str],
        current_user: str = "Guest",
    ) -> IntentResult:
        """Return which candidate topics are relevant.

        Args:
            user_text: the raw current-turn user message (not normalized).
            candidate_topics: topics the keyword regex matched. May be
                empty — implementations should handle this gracefully
                (return empty IntentResult) rather than treating an
                empty input as "search the whole universe", which is
                the explicit non-choice in the TODO 4 design.
            current_user: speaker name, for entity disambiguation.
                Most classifiers won't need it, but it's part of the
                interface so future implementations don't break.
        """
        raise NotImplementedError


class KeywordIntentClassifier(IntentClassifier):
    """Trivial baseline: every keyword-matched topic is relevant.

    This is the behaviour that has been in place since the topic
    dictionary was introduced. Extracting it as a classifier
    implementation is a no-op refactor that lets Step 2/3 layer
    real intent classification on top without touching the
    MemoryModule.decide control flow.
    """

    def classify(
        self,
        user_text: str,
        candidate_topics: List[str],
        current_user: str = "Guest",
    ) -> IntentResult:
        return IntentResult(
            relevant_topics=list(candidate_topics or []),
            skip_reason=None,
            confidence=1.0,
            source="keyword",
        )


# ============================================================
# LLMIntentClassifier (TODO 4 Step 3)
# ============================================================
# Asks the LLM judge to filter candidate topics down to those
# actually relevant to the user's intent. The contract is
# subset-only: the judge returns a subset of `candidate_topics`,
# never new topics — that's the explicit non-choice in the TODO 4
# design (situation (a) only).
#
# Failure handling — three cases for the judge return value:
#   - non-empty list  -> use it as relevant_topics
#   - empty list      -> caller should skip; encoded with
#                        skip_reason='llm_rejected_all_candidates'
#                        and source='llm'
#   - None (errored / over budget / disabled) -> caller's safest
#                        recovery is "fall back to keyword behaviour",
#                        i.e. trust the input candidates. We surface
#                        this by returning the input candidates with
#                        source='llm:fallback'.
#
# This makes LLMIntentClassifier composable in a Layered chain: if
# the judge is silent, the layer is a no-op. The Layered classifier
# below relies on this contract to avoid double-rejecting on judge
# failure.
# ============================================================
class LLMIntentClassifier(IntentClassifier):
    """LLM-judge backed topic-relevance filter.

    Constructor:
        judge_fn: callable (query, candidates) -> Optional[List[str]].
            The Optional[List[str]] subset is the eva_intent_judge
            contract (see judge_topic_subset). The classifier wraps
            this into IntentResult.

    Why the indirection through judge_fn instead of importing
    judge_topic_subset directly: the per-turn JudgeState lives on
    ChatAgent, not on MemoryModule. The wiring layer in
    eva_inference_P2 closes over self._llm_judge_state to produce
    the callable. This keeps MemoryModule unaware of who owns the
    state and avoids a hard dependency on eva_core / ChatAgent.
    """

    def __init__(self, judge_fn):
        self._judge_fn = judge_fn

    def classify(
        self,
        user_text: str,
        candidate_topics: List[str],
        current_user: str = "Guest",
    ) -> IntentResult:
        # Empty input -> empty output, no judge call.
        if not candidate_topics:
            return IntentResult(
                relevant_topics=[],
                skip_reason=None,
                confidence=0.0,
                source="llm:empty_input",
            )

        # Pass current_user as `speaker` to the judge_fn so the LLM
        # can disambiguate first-person queries: 'what do I like'
        # from Rosm probes Rosm's stored profile (KEEP candidates);
        # the same query from a Guest should reject candidates
        # (Eva has no Guest profile). The two-arg fallback below
        # preserves compatibility with judge_fn implementations that
        # haven't been updated yet — they just won't get speaker
        # info, identical to the pre-2026-05-06 behaviour.
        try:
            verdict = self._judge_fn(user_text, list(candidate_topics),
                                     current_user)
        except TypeError:
            # Old-style judge_fn(query, candidates) — call without speaker.
            verdict = self._judge_fn(user_text, list(candidate_topics))

        # Judge silent (None) -> degrade to input candidates so the
        # Layered chain treats this layer as a no-op. This is the
        # critical asymmetric-fallback property: a failed judge can
        # never make the system MORE restrictive than baseline.
        if verdict is None:
            return IntentResult(
                relevant_topics=list(candidate_topics),
                skip_reason=None,
                confidence=0.0,
                source="llm:fallback",
            )

        # Confident verdict (possibly empty list).
        # Whitelist-filter again as a defense-in-depth — even though
        # judge_topic_subset already does this, an external judge_fn
        # implementation might not.
        cand_set = set(candidate_topics)
        relevant = [t for t in verdict if t in cand_set]

        if not relevant:
            return IntentResult(
                relevant_topics=[],
                skip_reason="llm_rejected_all_candidates",
                confidence=1.0,
                source="llm",
            )
        return IntentResult(
            relevant_topics=relevant,
            skip_reason=None,
            confidence=1.0,
            source="llm",
        )


class LayeredIntentClassifier(IntentClassifier):
    """Compose multiple classifiers into a sequential filter.

    Each subsequent layer takes the previous layer's relevant_topics
    as its input candidates. The first layer sees the original
    keyword-matched candidates. Order matters: cheaper/coarser
    layers first, expensive/precise layers last.

    Stop conditions (after any layer):
      - relevant_topics is empty -> stop, return that result
        (the earliest skip_reason wins).
      - skip_reason is set       -> stop, return that result.

    The recommended composition for TODO 4 is:
        LayeredIntentClassifier(
            KeywordIntentClassifier(),       # confirms/passes through
            LLMIntentClassifier(judge_fn),   # filters paraphrases
        )

    KeywordIntentClassifier is effectively the identity here (it
    always returns its input). It's included for two reasons:
      1. Explicit composition order is documentation.
      2. If a future cheap classifier (e.g. embedding-based, Tier 2
         from TODO 4) is added, it slots in cleanly between Keyword
         and LLM without changing the wiring.

    Source attribution: the final IntentResult.source is a colon-
    delimited stack of layer sources, e.g. 'layered:keyword>llm'.
    This makes log lines self-describing about which layer produced
    the verdict.
    """

    def __init__(self, *layers: IntentClassifier):
        if not layers:
            raise ValueError("LayeredIntentClassifier requires ≥1 layer")
        self.layers = layers

    def classify(
        self,
        user_text: str,
        candidate_topics: List[str],
        current_user: str = "Guest",
    ) -> IntentResult:
        topics = list(candidate_topics or [])
        sources: List[str] = []
        confidence = 1.0
        skip_reason: Optional[str] = None
        last_source = ""

        for layer in self.layers:
            res = layer.classify(user_text, topics, current_user)
            last_source = res.source or "?"
            sources.append(last_source)

            # Update accumulators. Confidence is the min across
            # layers — uncertainty in any layer makes the final
            # verdict at most that confident.
            confidence = min(confidence, res.confidence)
            if res.skip_reason and not skip_reason:
                skip_reason = res.skip_reason

            topics = list(res.relevant_topics or [])

            # Short-circuit on empty result. Skip_reason already
            # captured above (or None if the layer didn't supply one).
            if not topics:
                break

        # Build composite source: 'layered:keyword>llm' (etc.). For a
        # single-layer chain we collapse to the inner source verbatim
        # so existing reason-string format ('topic_match[keyword]')
        # is preserved.
        if len(sources) == 1:
            composite = sources[0]
        else:
            composite = "layered:" + ">".join(sources)

        return IntentResult(
            relevant_topics=topics,
            skip_reason=skip_reason,
            confidence=confidence,
            source=composite,
        )


# ============================================================
# MemoryModule — public API
# ============================================================
class MemoryModule:
    """Topic-based PRE MEMORY PROBE + dual-path retrieval + packet builder.

    Constructor takes the raw memory state dict produced by
    eva_tools.load_memory_resources(). Pass None to operate in
    no-DB Mock mode (decide always returns 'skip').
    """

    def __init__(
        self,
        memory_state: Optional[dict],
        encoder=None,
        reranker=None,
        topic_dict: Optional[TopicDictionary] = None,
        current_user: str = "Guest",
        intent_classifier: Optional[IntentClassifier] = None,
    ):
        self.memory_state = memory_state or {}
        self.encoder = encoder
        self.reranker = reranker
        self.topic_dict = topic_dict or TopicDictionary()
        self.current_user = current_user
        # Step 1: intent classifier is pluggable. Default = keyword
        # baseline (every regex match is relevant), preserving today's
        # behaviour exactly. Step 2/3 will add LLMIntentClassifier and
        # LayeredIntentClassifier; eva_inference_P2 will wire them.
        self.intent_classifier: IntentClassifier = (
            intent_classifier or KeywordIntentClassifier()
        )
        self._available = bool(memory_state and encoder is not None and reranker is not None)
        if not self._available:
            print("[MemoryModule] DB / encoder / reranker not all available — running in skip-only mode.")

    # ----------- topic match (cheap, no model) -----------
    def match_topics(self, user_text: str) -> List[str]:
        return self.topic_dict.match(user_text or "")

    # ----------- decide (PRE MEMORY PROBE entry) -----------
    def decide(
        self,
        user_text: str,
        explicit_memory_request: bool = False,
    ) -> Dict[str, Any]:
        """Decide whether the current turn should probe memory.

        Returns a dict:
          {
            "action": "skip" | "probe",
            "matched_topics": [...],
            "target_entity": "Eva|Rosm|Both|Shared",
            "reason": str,
          }
        """
        text = (user_text or "").strip()
        result: Dict[str, Any] = {
            "action": "skip",
            "matched_topics": [],
            "target_entity": "Both",
            "reason": "no_signal",
        }
        if not text:
            return result

        if not self._available:
            result["reason"] = "memory_module_unavailable"
            return result

        # 1. Explicit memory check requests always probe.
        #    Same fix as hard-guard: also run topic matching so that path A
        #    (topic-direct) can pull curated records.
        if explicit_memory_request:
            matched = self.match_topics(text)
            target = None
            if matched:
                hints = {self.topic_dict.subject_hint(t) for t in matched}
                hints.discard(None)
                if len(hints) == 1:
                    target = next(iter(hints))
            if target is None:
                target = _infer_memory_target_from_text(
                    text, default_target="Both",
                    current_user=self.current_user)
            reason = "explicit_memory_request"
            if matched:
                reason = f"explicit_memory_request+topic_match:{','.join(matched[:3])}"
            result.update(action="probe", reason=reason,
                          matched_topics=matched,
                          target_entity=target)
            return result
            return result

        # 2. Hard guard: slot/identity questions must probe.
        #    NOTE: even when hard-guard fires, we still run topic matching
        #    so that path A (topic-direct) can pull curated records — without
        #    this the probe falls back to FAISS-only and curated topic
        #    records never make it into the packet (e.g. "when is your
        #    birthday?" hits hard-guard but matched_topics stays empty,
        #    Birthday curated record never gets path-A injection).
        must, why = _hard_guard_must_probe(text)
        matched = self.match_topics(text)
        if must:
            # Subject hint from topics if all converge, else infer from text.
            target = None
            if matched:
                hints = {self.topic_dict.subject_hint(t) for t in matched}
                hints.discard(None)
                if len(hints) == 1:
                    target = next(iter(hints))
            if target is None:
                target = _infer_memory_target_from_text(
                    text, default_target="Both",
                    current_user=self.current_user)
            reason = why
            if matched:
                reason = f"{why}+topic_match:{','.join(matched[:3])}"
            result.update(action="probe", reason=reason,
                          matched_topics=matched,
                          target_entity=target)
            return result

        # 3. NEW Path A' (Step 1.5): meta-keyword direct match.
        #    Symmetric counterpart to Path A topic-direct match. If any
        #    record's meta.keywords contains a multi-word phrase that
        #    appears verbatim in the query, that's a strong "this is
        #    about something we have a curated record for" signal even
        #    if no topic_keywords.json alias fired.
        #
        #    Concrete failure this fixes (TODO 1 mirror failure):
        #      Query: "what do you think of NieR Automata?"
        #      Future record: keywords=["NieR Automata", ...]
        #      Today: no topic alias hit -> skip, record never injected.
        #      With Path A': verbatim match on "NieR Automata" -> probe.
        #
        #    Path A' precedes plain topic_match (branch 4) so even when
        #    BOTH fire, the reason field reflects the more specific
        #    keyword signal. Subject hint logic mirrors hard_guard /
        #    topic_match: prefer convergent topic hint, else text infer.
        target_for_pathA = None
        if matched:
            hints = {self.topic_dict.subject_hint(t) for t in matched}
            hints.discard(None)
            if len(hints) == 1:
                target_for_pathA = next(iter(hints))
        if target_for_pathA is None:
            target_for_pathA = _infer_memory_target_from_text(
                text, default_target="Both",
                current_user=self.current_user)
        meta_kw_records = self._fetch_by_meta_keywords(text, target_for_pathA)
        if meta_kw_records:
            # Build a compact reason string showing the matched keywords
            # (deduped, first-3 cap) for log/debug visibility.
            seen_kws, kw_list = set(), []
            for r in meta_kw_records:
                for kw in r.get("source_keywords", []):
                    if kw and kw not in seen_kws:
                        seen_kws.add(kw)
                        kw_list.append(kw)
            reason = f"meta_keyword_match:{','.join(kw_list[:3])}"
            if matched:
                reason = f"{reason}+topic_match:{','.join(matched[:3])}"
            result.update(
                action="probe",
                matched_topics=matched,
                target_entity=target_for_pathA,
                reason=reason,
            )
            return result

        # 4. Topic-keyword match (Step 1: routed through IntentClassifier).
        #
        # The keyword regex above produced `matched` candidates. We hand
        # those to the configured IntentClassifier to confirm/filter
        # relevance. The default KeywordIntentClassifier returns every
        # candidate as relevant (preserving today's behaviour); LLM
        # classifiers added in Step 2/3 may filter the set or return
        # empty (=> skip).
        #
        # Hard guards above (explicit_memory_request, hard_guard_must_probe,
        # Path A' meta-keyword match) are NOT routed through the classifier
        # — they reflect direct user intent or curated metadata signals
        # that override topic-relevance judgement.
        if matched:
            intent = self.intent_classifier.classify(
                user_text=text,
                candidate_topics=matched,
                current_user=self.current_user,
            )
            relevant = intent.relevant_topics or []
            if relevant:
                # subject hint: prefer convergent topic hint over the
                # RELEVANT subset (not the raw matched set), so the LLM
                # classifier filtering down to the right topic also
                # tightens the entity inference.
                hints = {self.topic_dict.subject_hint(t) for t in relevant}
                hints.discard(None)
                if len(hints) == 1:
                    target = next(iter(hints))
                else:
                    target = _infer_memory_target_from_text(
                        text, default_target="Both",
                        current_user=self.current_user)
                # Reason string reflects the classifier source for
                # debugging. KeywordIntentClassifier => "topic_match"
                # (unchanged today); LLM => "topic_match[llm]" etc.
                src = (intent.source or "keyword").lower()
                tag = "" if src == "keyword" else f"[{src}]"
                result.update(
                    action="probe",
                    matched_topics=relevant,
                    target_entity=target,
                    reason=f"topic_match{tag}:{','.join(relevant[:3])}",
                )
                return result
            # Classifier rejected every candidate: skip with a
            # specific reason for log/debug. Falls through to branch 5
            # if classifier provided no skip_reason.
            if intent.skip_reason:
                result["reason"] = intent.skip_reason
                return result
            # No reason but empty result: fall through to default skip.

        # 5. No topic, no hard-guard → skip. FAISS-only fishing is unreliable
        #    (root cause of the dance-turn miss in P1 testing).
        result["reason"] = "no_topic_match_and_no_hard_guard"
        return result

    # ----------- probe (dual-path) -----------
    def probe(
        self,
        user_text: str,
        target_entity: str = "Both",
        matched_topics: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run dual-path retrieval (topic-direct + FAISS) and merge.

        Returns:
          {
            "records": [...],          # merged, sorted (topic-direct first then FAISS)
            "exact_count": int,
            "related_count": int,
            "top1_score": float,
            "topic_hit_count": int,
            "target_entity": str,
            "search_query": str,
            "keywords": [...],
            "error": Optional[str],
          }
        """
        matched_topics = matched_topics or []
        if not self._available:
            return self._empty_probe(target_entity, error="memory_module_unavailable")

        # Path A: topic-direct
        topic_records = self._fetch_by_topics(matched_topics, target_entity)

        # Path A' (Step 1.5): meta-keyword direct match. Runs
        # unconditionally; cost is one O(N records * avg keywords)
        # scan, sub-millisecond at Eva's scale (~91 records). Hits
        # are EXACT-promoted curated truth — same priority as Path A.
        meta_keyword_records = self._fetch_by_meta_keywords(
            user_text, target_entity
        )
        # Merge curated paths (A + A') with content-dedup before they
        # meet FAISS; if a record qualifies for both, the topic-direct
        # version wins (it's listed first). Both carry topic_direct=True
        # so downstream exact_count / topic_hit_count treat them the
        # same; meta_keyword_direct=True remains as a debug marker.
        curated_records = self._merge_records(topic_records, meta_keyword_records)

        # Path B: standard FAISS+BM25+rerank — call the legacy collector
        # functions directly so we get a dict bundle (records list +
        # top1_score), not the prompt-string output of run_memory_search.
        target_canon = _canonical_target_entity(target_entity, current_user=self.current_user)
        target_canon = _infer_memory_target_from_text(
            user_text, default_target=target_canon, current_user=self.current_user
        )
        variants = _build_memory_query_variants_for_tool_call(
            query=user_text,
            target_entity=target_canon,
            current_user=self.current_user,
        )
        display_keywords = ", ".join(
            _build_display_keywords_from_query(
                user_text,
                target_entity=target_canon,
                current_user=self.current_user,
                limit=16,
            )
        )
        if not variants:
            faiss_bundle = _collect_memory_records(
                params={"query": user_text, "target_entity": target_canon, "keywords": display_keywords},
                memory_state=self.memory_state,
                encoder=self.encoder,
                reranker=self.reranker,
                current_user=self.current_user,
            )
        else:
            collections = []
            for q in variants:
                collections.append(_collect_memory_records(
                    params={"query": q, "target_entity": target_canon, "keywords": display_keywords},
                    memory_state=self.memory_state,
                    encoder=self.encoder,
                    reranker=self.reranker,
                    current_user=self.current_user,
                ))
            faiss_bundle = _merge_memory_collections(collections)

        # faiss_bundle is now a dict: {"records": [...], "top1_score": ...}
        if not isinstance(faiss_bundle, dict):
            faiss_bundle = {"records": [], "top1_score": 0.0,
                            "target_entity": target_canon, "search_query": user_text,
                            "keywords": [], "error": "collector_returned_non_dict"}
        faiss_records = faiss_bundle.get("records", []) or []

        # Final merge: curated (A + A') first, FAISS appended where not deduped.
        merged = self._merge_records(curated_records, faiss_records)

        exact_count = sum(1 for r in merged if r.get("judge_label") == "EXACT" or r.get("topic_direct"))
        related_count = sum(1 for r in merged if r.get("judge_label") == "RELATED" and not r.get("topic_direct"))
        top1 = float(faiss_bundle.get("top1_score", 0.0))

        return {
            "records": merged,
            "exact_count": exact_count,
            "related_count": related_count,
            "top1_score": top1,
            # topic_hit_count counts BOTH Path A (topic-direct) and Path A'
            # (meta-keyword direct) hits — they're equally curated. The
            # de-duped curated pool length is the right value here.
            "topic_hit_count": len(curated_records),
            "meta_keyword_hit_count": len(meta_keyword_records),
            "target_entity": faiss_bundle.get("target_entity", target_entity),
            "search_query": faiss_bundle.get("search_query", user_text),
            "keywords": faiss_bundle.get("keywords", []),
            "error": faiss_bundle.get("error"),
        }

    # ----------- format packet -----------
    def format_packet(
        self,
        probe_result: Dict[str, Any],
        user_text: str,
        inject_reason: str = "",
    ) -> Dict[str, str]:
        """Build the [Active Memory Evidence] prompt packet.

        Returns a dict with the cleaned observation string + the full packet.
        """
        target_entity = probe_result.get("target_entity", "Both")
        records = probe_result.get("records", []) or []

        # Build a 'collected' shape that legacy formatters expect.
        collected = {
            "target_entity": target_entity,
            "keywords": probe_result.get("keywords", []),
            "search_query": probe_result.get("search_query", user_text),
            "records": records,
            "top1_score": probe_result.get("top1_score", 0.0),
            "error": probe_result.get("error"),
        }

        requested_slots = _detect_requested_slot_fields(
            user_text, encoder=self.encoder,
        ) or []
        if requested_slots:
            collected = _attach_slot_evidence_to_collection(
                collected, requested_slots=requested_slots, target_entity=target_entity
            )

        observation = _format_memory_records_block(collected)

        # Strict-match caveat (Q5).
        strict_note = _detect_strict_match_caveat(user_text, observation)
        if strict_note:
            observation = observation.rstrip() + "\n\n" + strict_note + "\n"

        slot_line = ""
        if requested_slots:
            slot_line = f"\nRequested slots: {', '.join(requested_slots)}"

        rules = (
            "Rules: Use [SLOT EVIDENCE] / [MEMORY JUDGE RESULT] above as ground truth for this turn. "
            "If a fact is NOT literally present, retract honestly instead of paraphrasing."
        )

        packet = (
            f"[Active Memory Evidence]\n"
            f"User: {user_text}\n"
            f"Target: {target_entity}{slot_line}\n\n"
            f"{observation}\n\n"
            f"{rules}\n"
            f"[/Active Memory Evidence]"
        )
        return {
            "observation": observation,
            "packet": packet,
            "requested_slots": requested_slots,
            "strict_match_applied": bool(strict_note),
            # R-1.1 (2026-05-13)：把 _attach_slot_evidence_to_collection
            # 算出的 slot_evidence dict 透传给 caller。让
            # eva_inference_P2._new_refresh_active_memory 能把 PRE PROBE 阶段
            # 抽出的 slot value 写进 TurnEvidenceLedger 时带正确 slot 标签——
            # 否则 verifier 的 _exact_memory_evidence_for(subject, slot="toy")
            # 看不到这条 evidence，误报 unsupported_exact_toy_claim
            # （2026-05-13 实跑 Turn 5/6 复盘）。
            "slot_evidence": collected.get("slot_evidence", {}),
        }

    # ----------- inject decision (post-probe) -----------
    def should_inject(self, probe_result: Dict[str, Any]) -> Tuple[bool, str]:
        """Decide whether to inject the packet into the system prompt.

        Returns (inject, reason).
        Rules:
          - exact_count > 0  -> inject (high confidence)
          - topic_hit_count > 0 -> inject (curated topic hits override score floor)
          - related_count > 0 AND top1_score >= WEAK_RELATED_TOP1_BAR -> inject
          - else -> skip injection (model answers from persona, won't fabricate
            from junk RELATED records).  This is the P1.8.2 weak-related gate.
        """
        if probe_result.get("error"):
            return False, f"error:{probe_result.get('error')}"
        if probe_result.get("exact_count", 0) > 0:
            return True, "exact_evidence"
        if probe_result.get("topic_hit_count", 0) > 0:
            return True, "topic_direct_hit"
        related = probe_result.get("related_count", 0)
        top1 = float(probe_result.get("top1_score", 0.0))
        if related > 0 and top1 >= WEAK_RELATED_TOP1_BAR:
            return True, "related_above_top1_bar"
        if related > 0:
            return False, f"weak_related_top1_below_bar({top1:.2f}<{WEAK_RELATED_TOP1_BAR})"
        return False, "no_evidence"

    # ============================================================
    # Internals
    # ============================================================
    def _fetch_by_topics(self, topics: List[str], target_entity: str) -> List[dict]:
        """Path A: pull all DB records whose meta.topic ∈ topics, filtered by
        entity. These records are auto-promoted to EXACT (curated truth).
        """
        if not topics or not self.memory_state.get("db_records"):
            return []
        target_canon = _canonical_target_entity(target_entity, current_user=self.current_user)
        topics_lc = {t.lower().strip() for t in topics}
        out = []
        for idx, rec in enumerate(self.memory_state["db_records"]):
            # Defensive: skip records whose shape is unexpected so a single bad
            # record cannot crash the whole probe.
            if not isinstance(rec, dict):
                continue
            meta = rec.get("meta") or {}
            if not isinstance(meta, dict):
                continue
            rec_topic = (meta.get("topic") or "").lower().strip()
            if not rec_topic or rec_topic not in topics_lc:
                continue
            db_ent = _canonical_record_entity(meta.get("entity", "Unknown"))
            # Entity filter: Both/Shared accepts everything; specific entity
            # accepts that entity + Shared.
            if target_canon in ("Both", ""):
                pass
            elif target_canon == "Shared":
                if db_ent not in ("Shared", "Both"):
                    continue
            else:
                if db_ent not in (target_canon, "Shared", "Both"):
                    continue
            out.append({
                "content": rec.get("content", "") or "",
                "vector_text": rec.get("vector_text", "") or "",
                "entity": db_ent,
                "category": meta.get("category", "Lore"),
                "topic": meta.get("topic", "") or "",
                "rerank_score": 99.0,            # virtual top score
                "low_confidence": False,
                "judge_label": "EXACT",          # auto-promote
                "topic_direct": True,            # marker for merge / packet
                "source_query": "topic_direct",
                "source_original_query": "topic_direct",
                "source_keywords": list(topics),
                "field_bonus": 0.0,
                "entity_bonus": 0.0,
                "metadata_bonus": 0.0,
                "metadata_reasons": [f"topic={meta.get('topic','')}"],
                "keyword_bonus": 0.0,
                "keyword_reasons": [],
                "keyword_debug": {},
            })
        return out

    def _fetch_by_meta_keywords(
        self, user_text: str, target_entity: str
    ) -> List[dict]:
        """Path A' (Step 1.5): pull DB records whose meta.keywords contain
        a phrase that appears verbatim in the user query.

        Symmetric counterpart to _fetch_by_topics:
            Path A  matches on meta.topic   (curated topic label)
            Path A' matches on meta.keywords (curated entity terms)

        Hits are auto-promoted to EXACT (topic_direct=True) so downstream
        merge / packet / inject logic treats them identically to
        topic-direct hits. An additional `meta_keyword_direct=True`
        marker is set for log/debug visibility.

        Eligibility gate: only multi-word keywords trigger matching
        (see _is_meta_keyword_specific). This excludes generic
        descriptor tags ("Date", "Color") while keeping concrete
        entities ("Apex Legends", "NieR Automata", "Real Name").
        """
        if not user_text or not self.memory_state.get("db_records"):
            return []
        text_norm = _normalize_match_text(user_text)
        if not text_norm:
            return []
        target_canon = _canonical_target_entity(
            target_entity, current_user=self.current_user
        )
        out = []
        for rec in self.memory_state["db_records"]:
            # Defensive: skip malformed records so a single bad entry
            # cannot crash the probe (mirrors _fetch_by_topics).
            if not isinstance(rec, dict):
                continue
            meta = rec.get("meta") or {}
            if not isinstance(meta, dict):
                continue
            keywords = meta.get("keywords") or []
            if not isinstance(keywords, list):
                continue
            # Find which eligible keywords appear verbatim in the query.
            hit_kws = []
            for kw in keywords:
                if not _is_meta_keyword_specific(kw):
                    continue
                if _phrase_matches_text(kw, text_norm):
                    hit_kws.append(kw.strip())
            if not hit_kws:
                continue
            # Entity filter: same shape as _fetch_by_topics. Both/Shared
            # accepts everything; specific entity accepts that entity +
            # Shared. Tightening or relaxing this is independent of the
            # keyword match itself.
            db_ent = _canonical_record_entity(meta.get("entity", "Unknown"))
            if target_canon in ("Both", ""):
                pass
            elif target_canon == "Shared":
                if db_ent not in ("Shared", "Both"):
                    continue
            else:
                if db_ent not in (target_canon, "Shared", "Both"):
                    continue
            out.append({
                "content": rec.get("content", "") or "",
                "vector_text": rec.get("vector_text", "") or "",
                "entity": db_ent,
                "category": meta.get("category", "Lore"),
                "topic": meta.get("topic", "") or "",
                "rerank_score": 99.0,            # virtual top score
                "low_confidence": False,
                "judge_label": "EXACT",          # auto-promote (curated)
                "topic_direct": True,            # treated as Path-A peer
                "meta_keyword_direct": True,     # marker for logs/debug
                "source_query": "meta_keyword_direct",
                "source_original_query": "meta_keyword_direct",
                "source_keywords": list(hit_kws),
                "field_bonus": 0.0,
                "entity_bonus": 0.0,
                "metadata_bonus": 0.0,
                "metadata_reasons": [f"meta_keyword={kw}" for kw in hit_kws],
                "keyword_bonus": 0.0,
                "keyword_reasons": [],
                "keyword_debug": {},
            })
        return out

    def _merge_records(self, topic_records: List[dict], faiss_records: List[dict]) -> List[dict]:
        """Topic-direct first; FAISS records that aren't already covered (by
        normalised content match) appended after.
        """
        seen_content = set()
        out = []
        for r in topic_records:
            if not isinstance(r, dict):
                continue
            key = _normalize_match_text(r.get("content", "") or "")
            if key and key not in seen_content:
                seen_content.add(key)
                out.append(r)
        for r in faiss_records:
            if not isinstance(r, dict):
                continue
            key = _normalize_match_text(r.get("content", "") or "")
            if key and key not in seen_content:
                seen_content.add(key)
                out.append(r)
        return out

    def _empty_probe(self, target_entity: str, error: Optional[str] = None) -> Dict[str, Any]:
        return {
            "records": [],
            "exact_count": 0,
            "related_count": 0,
            "top1_score": 0.0,
            "topic_hit_count": 0,
            "meta_keyword_hit_count": 0,
            "target_entity": target_entity,
            "search_query": "",
            "keywords": [],
            "error": error,
        }


__all__ = [
    "TopicDictionary",
    "MemoryModule",
    "IntentClassifier",
    "IntentResult",
    "KeywordIntentClassifier",
    "LLMIntentClassifier",
    "LayeredIntentClassifier",
]
