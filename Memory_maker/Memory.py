"""
Memory.py — build FAISS index + content/meta JSON from the JSONL source.

Reads `8.memory_optimized.jsonl` (one record per line: vector_text +
content + meta), computes embeddings via sentence-transformers MPNet,
and writes:

  Memory/memory.index         — FAISS IndexFlatIP
  Memory/memory_content.json  — list[str] of `content` field
  Memory/memory_meta.json     — list[dict] of `meta` field

Run from this directory:
    python Memory.py

After building, copy or symlink `Memory/` into the project's root
`Memory/` (eva_memory_legacy expects `./Memory/memory.index` etc.).

Rewritten 2026-05-08 — removed brittle "Who am I" version-check
(was tied to a specific older corpus); added robust path handling
and CLI flag for output dir.
"""
import json
import os
import sys
import argparse
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = SCRIPT_DIR / "8.memory_optimized.jsonl"
# Output goes to the project-root Memory/ directly (sibling of
# Memory_maker/), so eva_memory_legacy can read it without a copy step.
DEFAULT_OUTDIR = SCRIPT_DIR.parent / "Memory"
DEFAULT_MODEL = "sentence-transformers/all-mpnet-base-v2"


def load_records(data_path: Path):
    """Load JSONL records and split into vector_texts / contents / metas."""
    if not data_path.exists():
        sys.exit(f"ERROR: data file not found: {data_path}")

    vtexts, contents, metas = [], [], []
    skipped = 0
    with open(data_path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  [skip] line {ln}: malformed JSON ({e})")
                skipped += 1
                continue
            v = item.get("vector_text", "")
            c = item.get("content", "")
            m = item.get("meta", {}) or {}
            if not (v and c):
                print(f"  [skip] line {ln}: missing vector_text or content")
                skipped += 1
                continue
            vtexts.append(v)
            contents.append(c)
            metas.append(m)

    print(f"Loaded {len(vtexts)} records from {data_path.name} "
          f"({skipped} skipped)")
    return vtexts, contents, metas


def build_index(vector_texts, model_name: str):
    """Encode vector_texts and build a FAISS IndexFlatIP."""
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    print(f"Encoding {len(vector_texts)} vectors...")
    embeddings = model.encode(
        vector_texts, normalize_embeddings=True, show_progress_bar=True
    )

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype("float32"))
    print(f"FAISS index built — {index.ntotal} vectors, dim={dim}")
    return model, index, embeddings


def save_artifacts(outdir: Path, index, contents, metas):
    """Persist index + content + meta to outdir."""
    outdir.mkdir(parents=True, exist_ok=True)
    index_path = outdir / "memory.index"
    content_path = outdir / "memory_content.json"
    meta_path = outdir / "memory_meta.json"

    faiss.write_index(index, str(index_path))
    with open(content_path, "w", encoding="utf-8") as f:
        json.dump(contents, f, ensure_ascii=False)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metas, f, ensure_ascii=False)

    print(f"Wrote:\n  {index_path}\n  {content_path}\n  {meta_path}")


def smoke_test(model, index, contents, metas, query: str,
               target_entity=None, top_k=5):
    """Quick retrieval check — encode query, retrieve top_k, print results."""
    q_emb = model.encode([query], normalize_embeddings=True)
    D, I = index.search(q_emb.astype("float32"), top_k)
    print(f"\n--- query: {query!r} (target_entity={target_entity}) ---")
    for rank, (score, idx) in enumerate(zip(D[0], I[0]), 1):
        if idx == -1:
            continue
        meta = metas[idx]
        ent = meta.get("entity", "?")
        if target_entity and target_entity.lower() not in ent.lower():
            continue
        topic = meta.get("topic", "?")
        snippet = contents[idx][:90].replace("\n", " ")
        print(f"  #{rank} score={score:.3f} [{ent}/{topic}] {snippet}...")


def main():
    p = argparse.ArgumentParser(description="Build Eva memory FAISS index.")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA,
                   help=f"Input JSONL (default: {DEFAULT_DATA.name})")
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR,
                   help=f"Output dir (default: {DEFAULT_OUTDIR.name}/)")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--no-test", action="store_true",
                   help="Skip the smoke-test queries at the end")
    args = p.parse_args()

    vtexts, contents, metas = load_records(args.data)
    if not vtexts:
        sys.exit("ERROR: no records loaded")

    model, index, _ = build_index(vtexts, args.model)
    save_artifacts(args.outdir, index, contents, metas)

    if not args.no_test:
        # Smoke tests targeting cases that previously failed in shadow runs.
        smoke_test(model, index, contents, metas,
                   "do you have a toy?", target_entity="Eva")
        smoke_test(model, index, contents, metas,
                   "how about a toy?", target_entity="Eva")
        smoke_test(model, index, contents, metas,
                   "your favorite plushie", target_entity="Eva")
        smoke_test(model, index, contents, metas,
                   "what kinds of gift do you want?", target_entity=None)
        smoke_test(model, index, contents, metas,
                   "did we visit a museum together?", target_entity="Shared")


if __name__ == "__main__":
    main()
