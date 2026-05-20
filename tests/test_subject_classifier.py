"""Offline unit tests for eva_subject_classifier and slot detection
gated by it.

Two layers:

  - Layer 1 (regex): tested without an encoder — pure pattern matching.
  - Layer 2 (embedding): tested with a fake encoder so we can drive
    deterministic prototype hits without loading mpnet.

Plus a golden-query fixture covering the cross-product of subject
class × slot field, asserting the expected slot set after the new
subject-aware filter.

Run from project root:

    D:/Anaconda/envs/py310/python.exe tests/test_subject_classifier.py
"""
import hashlib
import os
import sys
import types
import unittest

import numpy as np


# ------------------------------------------------------------
# Path setup + torch stub (eva_config imports torch for cudnn flags)
# ------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

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

from eva_subject_classifier import is_person_subject  # noqa: E402
from eva_slots import extract_memory_slots  # noqa: E402
from eva_memory_legacy import _detect_requested_slot_fields  # noqa: E402


# ============================================================
# Layer 1 — pure regex (no encoder)
# ============================================================

class TestRegexLayerNonPerson(unittest.TestCase):
    """NonPerson noun adjacent to 'name' / 'its name' suppresses Person."""

    def test_name_of_cat(self):
        self.assertFalse(is_person_subject("what was the name of that cat?"))

    def test_cats_name(self):
        self.assertFalse(is_person_subject("what's the cat's name?"))

    def test_dogs_name(self):
        self.assertFalse(is_person_subject("name of my dog?"))

    def test_pet_name(self):
        self.assertFalse(is_person_subject("the pet's name was funny"))

    def test_its_name(self):
        self.assertFalse(is_person_subject(
            "Did I tell you about my new dog? What was its name?"
        ))

    def test_name_of_museum(self):
        self.assertFalse(is_person_subject(
            "what was the name of that museum we visited?"
        ))

    def test_restaurant_name(self):
        self.assertFalse(is_person_subject("the restaurant name?"))

    def test_song_name(self):
        # Layer 1c (loose NonPerson) catches "song" + "called" presence.
        self.assertFalse(is_person_subject("what was the song called?"))
        # Strict adjacency form via "name of":
        self.assertFalse(is_person_subject("what's the name of that song?"))

    def test_name_of_movie(self):
        self.assertFalse(is_person_subject("name of the movie we watched?"))

    def test_book_name(self):
        self.assertFalse(is_person_subject("the book name?"))

    def test_chinese_cat_name(self):
        self.assertFalse(is_person_subject("我家那只猫的名字是什么？"))

    def test_chinese_dog_name(self):
        self.assertFalse(is_person_subject("狗的名字叫什么"))


class TestRegexLayerPerson(unittest.TestCase):
    """Person-possessive 'name' patterns explicitly assert Person."""

    def test_your_full_name(self):
        self.assertTrue(is_person_subject("what's your full name?"))

    def test_your_name(self):
        self.assertTrue(is_person_subject("what is your name"))

    def test_my_real_name(self):
        self.assertTrue(is_person_subject("do you remember my real name?"))

    def test_eva_name(self):
        self.assertTrue(is_person_subject("what's Eva's name?"))

    def test_master_name(self):
        self.assertTrue(is_person_subject("Master's name is what?"))

    def test_his_name(self):
        self.assertTrue(is_person_subject("his name?"))

    def test_her_name(self):
        self.assertTrue(is_person_subject("her name was Sarah"))


class TestRegexLayerDefaults(unittest.TestCase):
    """No clear signal → default to Person (preserve legacy behavior)."""

    def test_empty_query_defaults_person(self):
        self.assertTrue(is_person_subject(""))
        self.assertTrue(is_person_subject(None))

    def test_no_name_keyword_defaults_person(self):
        # No "name", no NonPerson noun near name.
        self.assertTrue(is_person_subject("how are you today"))
        self.assertTrue(is_person_subject("what's your favorite color"))

    def test_topic_words_alone_default_person(self):
        # Just a topic word with no explicit pet/place noun nearby.
        self.assertTrue(is_person_subject("tell me about birthdays"))


# ============================================================
# Layer 2 — embedding nearest-neighbor (fake encoder)
# ============================================================

DIM = 32


def _hash_to_vec(text: str, dim: int = DIM) -> np.ndarray:
    h = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(h[:8], "big") & 0x7FFFFFFF
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype("float32")
    n = np.linalg.norm(v)
    return v / (n if n > 0 else 1.0)


def fake_encoder(texts):
    """Per-text hash → unit vector (no semantic correlation)."""
    return np.stack([_hash_to_vec(t) for t in texts]).astype("float32")


def biased_encoder_factory(person_words, nonperson_words):
    """Encoder that gives high cosine to the side whose keyword appears
    in the input. Used to simulate Layer 2 verdicts deterministically.
    """
    person_anchor = _hash_to_vec("__PERSON_ANCHOR__")
    nonperson_anchor = _hash_to_vec("__NONPERSON_ANCHOR__")

    def _enc(texts):
        out = []
        for t in texts:
            t_lower = t.lower()
            has_p = any(w in t_lower for w in person_words)
            has_np = any(w in t_lower for w in nonperson_words)
            if has_p and not has_np:
                base = person_anchor
            elif has_np and not has_p:
                base = nonperson_anchor
            else:
                base = _hash_to_vec(t)
            jitter = 0.05 * _hash_to_vec(t)
            v = base + jitter
            v = v / (np.linalg.norm(v) or 1.0)
            out.append(v)
        return np.stack(out).astype("float32")

    return _enc


class TestEmbeddingLayer(unittest.TestCase):
    """Layer 2 (embedding nearest-neighbor) only fires when Layer 1
    is undecided. Tests use a fake encoder; production behavior with
    real mpnet is a separate concern validated via Colab fixture.
    """

    def test_layer2_skipped_when_layer1_decisive(self):
        # "cat's name" is decisively NonPerson by Layer 1a regex.
        # Pass a deliberately broken encoder — should never be called.
        def assert_not_called(_texts):
            raise AssertionError("Layer 2 should not be reached when Layer 1 decides")
        self.assertFalse(
            is_person_subject("what's the cat's name?", encoder=assert_not_called)
        )
        # Person side:
        self.assertTrue(
            is_person_subject("what's your full name?", encoder=assert_not_called)
        )

    def test_encoder_failure_is_not_fatal(self):
        # Encoder throws on first call → falls back to default Person.
        def bad_encoder(texts):
            raise RuntimeError("oops")
        self.assertTrue(is_person_subject(
            "totally ambiguous query", encoder=bad_encoder,
        ))

    def test_sentence_transformer_like_encoder_accepted(self):
        # Object with .encode(...) method should be accepted and
        # normalized through _normalize_encoder. Build a stub that
        # pretends to be a SentenceTransformer.
        class _StubST:
            def encode(self, texts, normalize_embeddings=True):
                # Return random unit vectors — Layer 2 verdict will be
                # ambiguous (margin too small), so default Person fires.
                return fake_encoder(list(texts))

        # Query has no Layer 1 signal — falls through to Layer 2 → ambiguous → default Person.
        self.assertTrue(is_person_subject(
            "totally unrelated query about technology", encoder=_StubST(),
        ))


# ============================================================
# Slot detection — both detectors gated through subject classifier
# ============================================================

class TestSlotDetectionExtractMemorySlots(unittest.TestCase):
    """eva_slots.extract_memory_slots — used by ChatAgent dispatch
    (eva_core MemorySearch + verifier-injected repair path)."""

    def test_pet_name_does_not_request_full_name(self):
        slots = extract_memory_slots("what was the cat's name?")
        self.assertNotIn("full_name", slots)

    def test_its_name_does_not_request_full_name(self):
        slots = extract_memory_slots(
            "Did I tell you about my new dog? What was its name?"
        )
        self.assertNotIn("full_name", slots)

    def test_museum_name_does_not_request_full_name(self):
        slots = extract_memory_slots("name of the museum we visited?")
        self.assertNotIn("full_name", slots)

    def test_song_name_does_not_request_full_name(self):
        slots = extract_memory_slots("name of that song?")
        self.assertNotIn("full_name", slots)

    def test_chinese_cat_name_does_not_request_full_name(self):
        slots = extract_memory_slots("我家那只猫的名字是什么？")
        self.assertNotIn("full_name", slots)

    def test_your_name_still_requests_full_name(self):
        slots = extract_memory_slots("what's your full name?")
        self.assertIn("full_name", slots)

    def test_my_real_name_still_requests_full_name(self):
        slots = extract_memory_slots("do you remember my real name?")
        self.assertIn("full_name", slots)

    # --- non-name slots also gated ---
    def test_birthday_for_person_query_fires(self):
        slots = extract_memory_slots("when is your birthday?")
        self.assertIn("birthday", slots)

    def test_birthday_for_nonperson_does_not_fire(self):
        # Query mentions cat AND birthday — subject classifier sees pet
        # in query, so person-only slots (including birthday) suppress.
        # NOTE: this is the right thing semantically — Eva's birthday
        # is a Person fact; "cat birthday" usage would be wrong slot.
        slots = extract_memory_slots(
            "what's the cat's name and birthday?"
        )
        # Subject = NonPerson (cat) → person-only slots all suppressed.
        self.assertNotIn("full_name", slots)
        self.assertNotIn("birthday", slots)

    def test_age_for_person_query_fires(self):
        # Note: MEMORY_SLOT_FIELDS["age"] aliases are ["age", "years old"].
        # A query needs to literally contain one of those substrings.
        # "how old are you?" doesn't (no "age" word), so we use a
        # phrasing that does.
        slots = extract_memory_slots("what's your age?")
        self.assertIn("age", slots)

    def test_toy_for_person_query_fires(self):
        slots = extract_memory_slots("what's your favorite toy?")
        self.assertIn("toy", slots)


class TestSlotDetectionDetectRequestedSlotFields(unittest.TestCase):
    """eva_memory_legacy._detect_requested_slot_fields — used by
    run_memory_search slot-evidence attachment."""

    def test_pet_name_does_not_request_full_name(self):
        slots = _detect_requested_slot_fields("what was the cat's name?")
        self.assertNotIn("full_name", slots)

    def test_museum_name_does_not_request_full_name(self):
        slots = _detect_requested_slot_fields(
            "name of the museum we visited?"
        )
        self.assertNotIn("full_name", slots)

    def test_your_full_name_still_requests_full_name(self):
        slots = _detect_requested_slot_fields("what's your full name?")
        self.assertIn("full_name", slots)


# ============================================================
# Golden query fixture — cross-product subject × slot
# ============================================================

class TestGoldenQueries(unittest.TestCase):
    """30+ golden queries with expected (subject_class, slots) tuples.

    Acts as a regression fence. Failing tests here mean a slot/subject
    contract regression — investigate before merging.
    """

    GOLDEN = [
        # ---- Person + slot fires ----
        ("what's your full name?",                     True,  {"full_name"}),
        ("what is your real name?",                    True,  {"full_name"}),
        ("do you remember my real name?",              True,  {"full_name"}),
        ("Eva's name?",                                True,  {"full_name"}),
        ("Master's name?",                             True,  {"full_name"}),
        ("when is your birthday?",                     True,  {"birthday"}),
        ("what's my birth date?",                      True,  {"birthday"}),
        ("what's your age?",                           True,  {"age"}),
        ("tell me Eva's age",                          True,  {"age"}),
        ("what's your favorite toy?",                  True,  {"toy"}),
        ("tell me about your favorite plushie",        True,  {"toy"}),

        # ---- NonPerson queries (Pet / Place / Object) — slot SUPPRESSED ----
        ("what was the name of that cat?",             False, set()),
        ("the cat's name?",                            False, set()),
        ("what's the dog's name?",                     False, set()),
        ("name of my pet?",                            False, set()),
        ("did the rabbit have a name?",                False, set()),
        ("what was its name?",                         False, set()),
        ("the bird's name?",                           False, set()),
        ("name of that museum?",                       False, set()),
        ("the restaurant name?",                       False, set()),
        ("name of the song?",                          False, set()),
        ("what's the book's name?",                    False, set()),
        ("name of the movie we watched?",              False, set()),
        ("name of that aquarium?",                     False, set()),
        ("我家那只猫的名字是什么？",                     False, set()),
        ("狗的名字是什么",                              False, set()),

        # ---- Subject ambiguous, no slot keyword → no slot regardless ----
        ("how are you today?",                         True,  set()),
        ("did we go to the park last week?",           True,  set()),
        ("what did we do last summer?",                True,  set()),
    ]

    def test_golden_subject_classes(self):
        for query, expected_person, _ in self.GOLDEN:
            actual = is_person_subject(query)
            self.assertEqual(
                actual, expected_person,
                f"subject mismatch for: {query!r} "
                f"(expected person={expected_person}, got {actual})",
            )

    def test_golden_slot_extraction_via_extract_memory_slots(self):
        for query, _, expected_slots in self.GOLDEN:
            actual = set(extract_memory_slots(query))
            self.assertEqual(
                actual, expected_slots,
                f"slot mismatch for: {query!r} "
                f"(expected={expected_slots}, got={actual})",
            )

    def test_golden_slot_extraction_via_detect_requested(self):
        # Both detectors must agree.
        for query, _, expected_slots in self.GOLDEN:
            actual = set(_detect_requested_slot_fields(query))
            self.assertEqual(
                actual, expected_slots,
                f"slot mismatch (legacy detector) for: {query!r} "
                f"(expected={expected_slots}, got={actual})",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
