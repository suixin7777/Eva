"""
add_memory.py — interactive memory adder powered by DeepSeek.

Instead of editing rewrite_memory.py to add new records by hand, just
describe the memory in plain English; the script asks DeepSeek to
turn it into a structured record (entity / category / topic / vector_text
/ content / keywords) that matches the existing corpus style, shows you
the proposal, and on confirmation appends it to 8.memory_optimized.jsonl.

Usage:
    # Single add, interactive confirm
    python add_memory.py "Eva tried baking cookies last weekend and burnt them"

    # Read from stdin (paste a longer description)
    cat memory.txt | python add_memory.py

    # Batch (one memory per line in the file)
    python add_memory.py --batch new_memories.txt

    # Skip confirmation (use only when LLM output you trust)
    python add_memory.py "..." --auto

    # Append + rebuild FAISS index immediately
    python add_memory.py "..." --rebuild

    # Manual fallback when DeepSeek is unavailable
    python add_memory.py --manual

Requirements:
    DEEPSEEK_API_KEY must be set in env (same key Eva uses elsewhere).
    Run from any cwd — script resolves paths relative to its own location.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
JSONL_PATH = SCRIPT_DIR / "8.memory_optimized.jsonl"
TOPIC_KW_PATH = PROJECT_ROOT / "topic_keywords.json"
MEMORY_PY = SCRIPT_DIR / "Memory.py"


# ============================================================
# DeepSeek prompt
# ============================================================
def build_system_prompt(canonical_topics):
    topics_str = ", ".join(sorted(canonical_topics))
    return f"""You are a memory-recording assistant for an AI maid named Eva.
Eva is tsundere, witty, mischievous; her Master is Rosm — gentle, thoughtful, a little shy.

Given the user's plain-English description of a memory, output ONE strict JSON
record that fits Eva's memory database.

Output JSON schema (strict — no extra keys, no comments, no prose):
{{
  "entity": "Eva" | "Rosm" | "Shared",
  "category": "Lore" | "Event",
  "topic": "<one of the canonical topics below>",
  "vector_text": "<1-3 sentences, clean natural language for embedding retrieval>",
  "content": "<2-4 sentences with concrete scene, sensory detail, and Eva's tsundere or Rosm's gentle voice>",
  "keywords": ["<3-6 short keyword strings>"],
  "secondary_topics": ["<optional canonical topic if cross-cutting>"],
  "participants": ["Eva"] | ["Rosm"] | ["Eva","Rosm"]
}}

Field guides:
- entity:
    "Eva"   — facts/preferences/lore about Eva alone
    "Rosm"  — facts about Master alone
    "Shared" — anything involving both, including Events that happened together
- category:
    "Lore"  — durable trait/preference/habit (default)
    "Event" — a specific past occurrence with a time and outcome
- topic: MUST be one of: {topics_str}
- vector_text: NO metadata tags like [Topic:...] or [Entity:...]. Plain prose,
  ideally including synonyms so varied user phrasings still match
  (e.g. mention both "toy" and "plushie" if relevant).
- content: rich and specific. Add place names, sensory details (color, sound,
  smell, texture), an emotional reaction. Use Eva's tsundere voice or Rosm's
  gentle voice. AVOID generic "she felt happy"; SHOW the moment.
- keywords: short noun-phrases pulled from the content (good for FAISS hits).
- secondary_topics: only if the memory genuinely spans two canonical topics.

If the input is ambiguous about who or what category, pick the most reasonable
default and note it in the content. If user input is empty/garbage, return a
record with entity=Shared, category=Lore, topic=Personality, and content
explaining the input was unclear.

Output JSON ONLY. No markdown fences, no preamble.
"""


# ============================================================
# LLM call (uses Eva's existing DeepSeek wrapper)
# ============================================================
def call_llm(system_prompt, user_text, debug=False):
    """Call DeepSeek via the project's existing call_deepseek_judge."""
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    try:
        from eva_tools_runtime import call_deepseek_judge
    except ImportError as e:
        sys.exit(f"ERROR: cannot import eva_tools_runtime: {e}")
    if not os.environ.get("DEEPSEEK_API_KEY", "").startswith("sk-"):
        sys.exit("ERROR: DEEPSEEK_API_KEY not set or invalid")

    # call_deepseek_judge takes a system instruction + a user payload and
    # returns the parsed JSON response (or {"ok": None, "error": ...} on
    # failure). It already requests JSON-mode response_format from DeepSeek.
    result = call_deepseek_judge(system_prompt, user_text,
                                 debug=debug, timeout=20.0)
    if not isinstance(result, dict):
        sys.exit(f"ERROR: LLM returned non-dict: {result!r}")
    if "error" in result and "vector_text" not in result:
        sys.exit(f"ERROR: LLM call failed: {result.get('error')}")
    return result


# ============================================================
# Validation
# ============================================================
ALLOWED_ENTITY = {"Eva", "Rosm", "Shared"}
ALLOWED_CATEGORY = {"Lore", "Event"}


def validate(rec, canonical_topics):
    """Return list of warnings (empty = clean record).

    Accepts either the normalized {vector_text, content, meta:{...}} shape
    OR the flat LLM-output shape — both happen in the pipeline.
    """
    warnings = []
    # Pull entity/category/topic from meta if normalized, else top-level.
    meta = rec.get("meta") if isinstance(rec.get("meta"), dict) else rec
    e = meta.get("entity")
    if e not in ALLOWED_ENTITY:
        warnings.append(f"entity={e!r} not in {ALLOWED_ENTITY}")
    c = meta.get("category")
    if c not in ALLOWED_CATEGORY:
        warnings.append(f"category={c!r} not in {ALLOWED_CATEGORY}")
    t = meta.get("topic")
    if t not in canonical_topics:
        warnings.append(f"topic={t!r} not in canonical list "
                        f"(record will fall back to FAISS-only retrieval)")
    kws = meta.get("keywords", [])
    if not isinstance(kws, list) or not all(isinstance(k, str) for k in kws):
        warnings.append("keywords must be list[str]")
    vt = rec.get("vector_text", "")
    if not vt or not isinstance(vt, str):
        warnings.append("vector_text empty or not a string")
    elif "[Topic:" in vt or "[Entity:" in vt:
        warnings.append("vector_text contains [Topic:] or [Entity:] tag — "
                        "should be plain prose")
    content = rec.get("content", "")
    if not content or not isinstance(content, str):
        warnings.append("content empty or not a string")
    elif len(content) < 50:
        warnings.append(f"content too short ({len(content)} chars) — "
                        f"event records need scene+emotion detail")
    return warnings


# ============================================================
# Build canonical record dict (matches the exact shape of existing data)
# ============================================================
def normalize_record(raw):
    """Map LLM output to the canonical {vector_text, content, meta} shape
    used in 8.memory_optimized.jsonl.

    The LLM returns flat fields; we re-nest the meta-only ones."""
    entity = raw.get("entity", "Shared")
    meta = {
        "entity": entity,
        "category": raw.get("category", "Lore"),
        "topic": raw.get("topic", "Personality"),
        "participants": raw.get("participants") or (
            [entity] if entity in ("Eva", "Rosm") else ["Eva", "Rosm"]),
        "keywords": raw.get("keywords") or [],
    }
    sec = raw.get("secondary_topics")
    if sec:
        meta["secondary_topics"] = sec
    return {
        "vector_text": raw.get("vector_text", "").strip(),
        "content": raw.get("content", "").strip(),
        "meta": meta,
    }


# ============================================================
# Append + rebuild
# ============================================================
def append_record(rec):
    with open(JSONL_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def rebuild_index():
    print("\n[rebuild] running Memory.py...")
    r = subprocess.run(
        [sys.executable, str(MEMORY_PY), "--no-test"],
        cwd=SCRIPT_DIR, capture_output=False,
    )
    if r.returncode == 0:
        print("[rebuild] done.")
    else:
        print(f"[rebuild] FAILED (returncode={r.returncode})")


# ============================================================
# UI helpers
# ============================================================
def print_record(rec, prefix="  "):
    print(f"{prefix}entity:    {rec['meta']['entity']}")
    print(f"{prefix}category:  {rec['meta']['category']}")
    print(f"{prefix}topic:     {rec['meta']['topic']}")
    if rec['meta'].get('secondary_topics'):
        print(f"{prefix}secondary: {rec['meta']['secondary_topics']}")
    print(f"{prefix}keywords:  {rec['meta']['keywords']}")
    print(f"{prefix}vector:    {rec['vector_text']}")
    print(f"{prefix}content:")
    for line in rec['content'].splitlines() or [rec['content']]:
        print(f"{prefix}  {line}")


def confirm_loop(rec, raw_input_text, system_prompt, canonical_topics, auto):
    """Show record, accept regenerate or cancel."""
    while True:
        print("\n" + "=" * 60)
        print("Proposed memory record:")
        print("=" * 60)
        print_record(rec)

        warnings = validate(rec, canonical_topics)
        if warnings:
            print("\n  ⚠  warnings:")
            for w in warnings:
                print(f"    - {w}")

        if auto:
            return rec

        choice = input(
            "\n[a]ccept / [r]egenerate / [f]eedback+regen / [c]ancel: "
        ).strip().lower()
        if choice in ("a", ""):
            return rec
        if choice == "c":
            return None
        if choice == "f":
            fb = input("Feedback for the LLM (e.g. 'make this Rosm not Shared'): ").strip()
            if not fb:
                continue
            modified_input = f"{raw_input_text}\n\nFeedback for revision: {fb}"
            raw = call_llm(system_prompt, modified_input)
            rec = normalize_record(raw)
            continue
        if choice == "r":
            raw = call_llm(system_prompt, raw_input_text)
            rec = normalize_record(raw)
            continue
        print("  (unknown choice — try a/r/f/c)")


# ============================================================
# Manual fallback (no LLM)
# ============================================================
def manual_input():
    print("Manual mode — fill in fields directly. Empty line skips optional fields.")
    entity = input("entity (Eva/Rosm/Shared): ").strip() or "Shared"
    category = input("category (Lore/Event) [Lore]: ").strip() or "Lore"
    topic = input("topic: ").strip()
    vector_text = input("vector_text (1-3 sentences): ").strip()
    print("content (multi-line; finish with empty line):")
    content_lines = []
    while True:
        ln = input()
        if not ln:
            break
        content_lines.append(ln)
    content = " ".join(content_lines).strip()
    kws_raw = input("keywords (comma-separated): ").strip()
    keywords = [k.strip() for k in kws_raw.split(",") if k.strip()]
    return {
        "entity": entity, "category": category, "topic": topic,
        "vector_text": vector_text, "content": content, "keywords": keywords,
    }


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text", nargs="?",
                    help="Memory description (or omit and pipe via stdin)")
    ap.add_argument("--batch", type=Path,
                    help="Process one memory per line from a text file")
    ap.add_argument("--auto", action="store_true",
                    help="Auto-accept LLM output without confirmation")
    ap.add_argument("--rebuild", action="store_true",
                    help="Rebuild FAISS index after appending")
    ap.add_argument("--manual", action="store_true",
                    help="Skip LLM; manually enter each field")
    ap.add_argument("--debug", action="store_true",
                    help="Print LLM raw response for debugging")
    args = ap.parse_args()

    # Resolve input source
    if args.batch:
        if not args.batch.exists():
            sys.exit(f"ERROR: batch file {args.batch} not found")
        with open(args.batch, encoding="utf-8") as f:
            inputs = [ln.strip() for ln in f if ln.strip()]
    elif args.manual:
        inputs = ["__MANUAL__"]
    elif args.text:
        inputs = [args.text]
    else:
        # stdin
        text = sys.stdin.read().strip()
        if not text:
            ap.print_help()
            sys.exit("\nERROR: no input. Pass text as arg or pipe via stdin.")
        inputs = [text]

    canonical_topics = set()
    if TOPIC_KW_PATH.exists():
        with open(TOPIC_KW_PATH, encoding="utf-8") as f:
            canonical_topics = set(json.load(f).get("topics", {}).keys())
        canonical_topics.discard("_comment")  # safety
    else:
        print(f"WARN: {TOPIC_KW_PATH} not found — topic validation skipped")

    system_prompt = build_system_prompt(canonical_topics)

    appended_count = 0
    for raw_input_text in inputs:
        if raw_input_text == "__MANUAL__":
            raw = manual_input()
        else:
            print(f"\n>>> input: {raw_input_text}")
            raw = call_llm(system_prompt, raw_input_text, debug=args.debug)

        rec = normalize_record(raw)
        final = confirm_loop(rec, raw_input_text, system_prompt,
                              canonical_topics, args.auto)
        if final is None:
            print("[skipped]")
            continue

        append_record(final)
        appended_count += 1
        print(f"[appended] record #{appended_count} → {JSONL_PATH.name}")

    print(f"\nDone. {appended_count} record(s) added.")

    if appended_count > 0 and args.rebuild:
        rebuild_index()
    elif appended_count > 0:
        print(f"Run `python {MEMORY_PY.name}` to rebuild the FAISS index "
              f"when you're done adding records.")


if __name__ == "__main__":
    main()
