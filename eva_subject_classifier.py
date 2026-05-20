"""eva_subject_classifier.py — Person vs NonPerson query classification.

Used by `_detect_requested_slot_fields` (eva_memory_legacy) to gate
person-only memory slots (full_name, birthday, age, toy) so they only
fire when the query is plausibly about a person (Eva, Rosm, Master,
the user). Replaces the historical whack-a-mole `full_name_blocked`
negative-list pattern with a positive subject check.

Architectural rationale:

  - `MEMORY_SLOT_FIELDS["full_name"]` has the alias "name" (and
    "called", "identity") to catch natural phrasings like "what's
    your name". This is over-broad — any sentence with the word
    "name" triggers full_name slot detection, including queries
    about places ("name of the museum"), pets ("the cat's name"),
    or media ("what's that song called").
  - Pre-2026-05-11 fix was a per-noun-category negative list. Each
    new noun category required adding another regex clause —
    classic whack-a-mole.
  - This module flips the polarity: ask "is this about a person?"
    (positive subject classifier) and gate the slot per
    SLOT_APPLICABLE_SUBJECTS table (eva_config). Adding a new noun
    category never requires touching slot code.

Three layers:

  1. **Layer 1 (regex, always-on)**: positive Person pattern
     (possessive pronoun + name with no NonPerson noun) and
     negative NonPerson pattern (pet/place/object/media noun near
     "name"). Catches ~90% of cases at zero cost.

  2. **Layer 2 (embedding, optional)**: prototype-based nearest
     neighbor over mpnet embeddings. Only called when Layer 1
     gives no signal AND an encoder is provided. Prototypes are
     encoded once and cached at module load.

  3. **Default**: fail-open to Person — preserves the legacy
     behavior on ambiguous queries. Better to over-detect a slot
     and let downstream "missing slot" warnings fire honestly than
     to silently drop slot extraction.
"""
from __future__ import annotations

import re
from typing import Callable, Optional

import numpy as np


# ============================================================
# Layer 1 — regex
# ============================================================

# NonPerson nouns that, when adjacent to "name", indicate the query is
# about a non-person entity. Bilingual; pet/place/media combined.
_NONPERSON_NOUNS = (
    # Pets / animals
    r"cat|kitten|dog|puppy|pet|hamster|bird|fish|rabbit|tabby|"
    r"parrot|turtle|snake|lizard|horse|pony|cow|sheep|goat|"
    r"猫|狗|宠物|兔子|仓鼠|鸟|"
    # Places / venues
    r"place|places|venue|venues|location|locations|spot|spots|"
    r"museum|aquarium|gallery|park|store|shop|restaurant|cafe|caf\xe9|"
    r"hotel|building|street|city|town|country|"
    # Media / objects
    r"game|games|movie|movies|book|books|song|songs|album|albums|"
    r"show|shows|painting|sculpture|"
    # Food / drink
    r"food|drink|dish|recipe|cuisine|cocktail|wine"
)

# NonPerson STRICT signal: English noun adjacent to "name" / pronoun "its".
# Catches "name of cat" / "cat's name" / "its name" / "name of the museum".
_NONPERSON_NEAR_NAME_RE = re.compile(
    rf"\bname\s+of\s+(?:the|that|those|these|this|a|an|my|your|his|her)?\s*"
    rf"(?:{_NONPERSON_NOUNS})\b"
    rf"|\b(?:{_NONPERSON_NOUNS})\s*'?s?\s+name\b"
    rf"|\b(?:its|it'?s)\s+name\b",
    re.IGNORECASE,
)

# Strong Person signal: English possessive pronoun (or known person alias)
# + "name" with no intervening other noun. "your name" / "my full name"
# / "Eva's name".
_PERSON_POSSESSIVE_NAME_RE = re.compile(
    r"\b(your|my|his|her|the\s+(?:user|guest|maid|creator)|eva|rosm|master)"
    r"(?:'s)?\s+(?:full\s+|real\s+|legal\s+|complete\s+)?name\b",
    re.IGNORECASE,
)

# NonPerson LOOSE signal (bilingual fallback). Fires when the query
# contains both a NonPerson noun AND a name-related keyword anywhere,
# regardless of adjacency. Catches:
#   - "did the rabbit have a name?"          (English, non-adjacent)
#   - "我家那只猫的名字是什么"                  (Chinese)
#   - "狗叫什么"                                (Chinese)
# Order matters in `is_person_subject`: this loose check runs AFTER the
# strict Person check, so "your name vs cat's claws" still routes to
# Person via the possessive pattern.
_NONPERSON_NOUN_RE = re.compile(rf"\b(?:{_NONPERSON_NOUNS})\b", re.IGNORECASE)
_NONPERSON_NOUN_ZH_RE = re.compile(r"(?:猫|狗|宠物|兔子|仓鼠|鸟)")
_NAME_KEYWORD_EN_RE = re.compile(r"\b(name|named|called)\b", re.IGNORECASE)
_NAME_KEYWORD_ZH_RE = re.compile(r"名字|叫")


def _has_nonperson_noun(q: str) -> bool:
    return bool(_NONPERSON_NOUN_RE.search(q) or _NONPERSON_NOUN_ZH_RE.search(q))


def _has_name_keyword(q: str) -> bool:
    return bool(_NAME_KEYWORD_EN_RE.search(q) or _NAME_KEYWORD_ZH_RE.search(q))


# ============================================================
# Layer 2 — embedding nearest-neighbor (optional)
# ============================================================

# Prototype queries for each class. Designed to span common phrasings
# (English + Chinese) that a regex layer alone might not catch.
_PERSON_PROTOTYPES = [
    "What is your full name?",
    "What's my real name?",
    "Tell me your name.",
    "What's my birthday?",
    "When is your birthday?",
    "How old is Eva?",
    "How old are you?",
    "What's your favorite toy?",
    "Who are you, Master?",
    "你叫什么名字？",
    "你的全名是什么？",
    "我的生日是哪天？",
    "你多大了？",
]

_NONPERSON_PROTOTYPES = [
    "What was the cat's name?",
    "What's the dog called?",
    "Name of that museum?",
    "What's that restaurant called?",
    "What was the song's name?",
    "Name of the book we read?",
    "Which movie did we watch?",
    "What's that bird called?",
    "What's the dish called?",
    "我家那只猫叫什么？",
    "那家餐厅叫什么名字？",
    "那首歌叫什么？",
    "我们去的那个博物馆叫什么？",
]


class _EmbeddingClassifier:
    """Lazy-encoded prototype embeddings for Person vs NonPerson.

    Initialized on first call with a valid encoder. After init, the
    same prototype embeddings are reused across calls (single global
    instance is enough since prototypes are constant).
    """

    def __init__(self):
        self._person_emb: Optional[np.ndarray] = None
        self._nonperson_emb: Optional[np.ndarray] = None

    def _maybe_init(self, encoder: Callable[[list], np.ndarray]) -> bool:
        if self._person_emb is not None:
            return True
        try:
            person = np.asarray(encoder(_PERSON_PROTOTYPES), dtype="float32")
            nonperson = np.asarray(encoder(_NONPERSON_PROTOTYPES), dtype="float32")
            if person.ndim != 2 or nonperson.ndim != 2:
                return False
            self._person_emb = person
            self._nonperson_emb = nonperson
            return True
        except Exception as e:
            # Encoder failure is non-fatal — caller falls back to Layer 1 default.
            print(f"[subject_classifier] embedding init failed: {e}")
            return False

    def classify(
        self,
        query: str,
        encoder: Callable[[list], np.ndarray],
        margin: float = 0.05,
    ) -> Optional[bool]:
        """Return True (Person) / False (NonPerson) / None (ambiguous).

        margin: minimum cosine gap between Person-max and NonPerson-max
        prototypes for a confident verdict. Below this gap the query
        falls back to the caller's default.
        """
        if not self._maybe_init(encoder):
            return None
        try:
            q_emb = np.asarray(encoder([query]), dtype="float32")
        except Exception:
            return None
        if q_emb.ndim != 2 or q_emb.shape[0] != 1:
            return None
        person_max = float((q_emb @ self._person_emb.T).max())
        nonperson_max = float((q_emb @ self._nonperson_emb.T).max())
        if abs(person_max - nonperson_max) < margin:
            return None
        return person_max > nonperson_max


# Module-level singleton — prototypes are constant, no per-agent state.
_EMBED = _EmbeddingClassifier()


def _normalize_encoder(
    encoder,
) -> Optional[Callable[[list], np.ndarray]]:
    """Normalize the encoder argument to a callable signature.

    Accepts either a plain callable `(list[str]) -> ndarray` or a
    SentenceTransformer-like object (has `.encode(texts, normalize_embeddings=True)`).
    Returns a callable, or None if the argument is unusable.
    """
    if encoder is None:
        return None
    enc_method = getattr(encoder, "encode", None)
    if callable(enc_method):
        return lambda texts: np.asarray(
            enc_method(texts, normalize_embeddings=True),
            dtype="float32",
        )
    if callable(encoder):
        return encoder
    return None


# ============================================================
# Public API
# ============================================================

def is_person_subject(
    query: str,
    encoder=None,
) -> bool:
    """Decide whether `query` is most likely about a person.

    Used to gate person-only memory slots in
    `_detect_requested_slot_fields`.

    Args:
        query: user query text.
        encoder: optional. May be either:
            (a) a callable `list[str] -> (n, dim) float32 normalized`
            (b) a SentenceTransformer-like object with `.encode(...)` method.
            When provided, enables Layer 2 (embedding nearest-neighbor)
            for queries the regex layer cannot decisively classify.

    Returns:
        True  — query is likely about a person; person-only slots may fire.
        False — query is about a non-person entity; person-only slots
                must be suppressed.
        On ambiguous input, returns True (preserves legacy slot-firing
        behavior; downstream "missing slot" warnings then fire honestly).
    """
    if not isinstance(query, str) or not query.strip():
        return True

    # Layer 1a: strict NonPerson (English adjacency).
    if _NONPERSON_NEAR_NAME_RE.search(query):
        return False
    # Layer 1b: strict Person (possessive + name).
    if _PERSON_POSSESSIVE_NAME_RE.search(query):
        return True
    # Layer 1c: loose NonPerson — query has both a NonPerson noun AND a
    # name keyword anywhere (bilingual). Catches "did the rabbit have a
    # name?" and Chinese forms like "猫的名字" / "狗叫什么".
    if _has_nonperson_noun(query) and _has_name_keyword(query):
        return False

    # Layer 2: embedding nearest-neighbor (only if encoder available).
    enc_fn = _normalize_encoder(encoder)
    if enc_fn is not None:
        verdict = _EMBED.classify(query, enc_fn)
        if verdict is not None:
            return verdict

    # Default: assume Person — preserves legacy behavior on ambiguous queries.
    return True


__all__ = ["is_person_subject"]
