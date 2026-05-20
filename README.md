# Eva — Multi-turn LLM Dialogue System

An experimental multi-turn dialogue agent built on **Qwen3.5-VL-9B + custom LoRA**, with a three-stage architecture (Advisor / Local / Verifier) designed for stable, low-latency, character-consistent interaction.

Production deployment is a Discord bot with per-user state isolation, running on a single GPU pod.

## Architecture

```mermaid
flowchart TB
    U([User message])
    OUT([Reply])

    U --> ADV[Advisor<br/>remote DeepSeek · 1 call/turn]

    subgraph Core["Inference Core"]
        direction TB
        LM[Local Model<br/>Qwen3.5-VL-9B + LoRA]
        V{Verifier<br/>regex + DeepSeek<br/>1 call/turn}
        RG[Regenerate Guard]
        LM --> V
        V -->|fail| RG
        RG -.retry hint.-> LM
    end

    subgraph Mem["Memory + Tools"]
        direction TB
        LORE[(Lore Memory<br/>FAISS + jsonl)]
        T[Tool Runtime<br/>RememberThis · ForgetMemory<br/>MemorySearch · WebSearch · Vision]
        NOTES[(Notes Store<br/>FAISS + jsonl<br/>per-user)]
        T <-.R/W.-> NOTES
    end

    ADV --> LM
    V -->|pass| OUT
    LORE -.read.-> LM
    LM <-->|tool calls| T
```

Three layers handle each turn:

1. **Advisor** (remote DeepSeek) — analyzes user intent, selects tools, emits per-turn guidance for the local model. Replaces the previous "model self-classifies in prompt" approach.
2. **Local Model** (Qwen3.5-VL-9B + LoRA) — generates the response using advisor context + tools + memory + notes.
3. **Verifier** (remote DeepSeek + regex) — semantic + format check. On failure, **Regenerate Guard** retries once with verifier feedback injected as a hint. Retry budget is capped per turn.

Side channels:
- **Lore Memory** — hand-curated character bible (FAISS + jsonl)
- **Notes Store** — runtime additions via `RememberThis` / `ForgetMemory`, per-user, tombstone-based delete with optional compaction

## Technical Highlights

### 1. Advisor refactor — 3× latency reduction, 2.7× compound-query correctness

The original design had the local model self-classify intent and pick tools inside the prompt. This produced unstable plans and required ~6–9 DeepSeek judge calls per turn just to verify the chain. The Advisor refactor (May 2026) replaced this with a single remote DeepSeek call up front that emits intent + `suggested_calls`, and a single DeepSeek call at the end for semantic verification.

| Metric | Before | After |
|---|---|---|
| DeepSeek calls per turn | 6–9 | **2** |
| Median turn latency | 12–18 s | **6–9 s** |
| Compound query correctness | ~30% | **~80%** |
| Verifier kill rate | every 3–5 turns | **~0** |
| Dead/patch code | ~1500 lines | ~500 lines remaining (scheduled removal) |

Full writeup: [`docs/REFACTOR_COMPLETE_2026-05-14.md`](docs/REFACTOR_COMPLETE_2026-05-14.md).

### 2. Two-layer verifier + Regenerate Guard

Verifier runs regex (format / structural) checks first, then semantic LLM checks gated on what the Advisor asked the model to do. On failure, `RegenerateGuard` retries the local model once with the failure reason injected as a side-channel hint. Hard-guard reasons reduced from ~9 mixed regex/LLM checks to 4 clean ones (1 format + 3 LLM judges). The verifier is also the backstop for capabilities the model fails to invoke (see Notes module below).

See [`eva_verifier_logic.py`](eva_verifier_logic.py), [`eva_verifier_semantic.py`](eva_verifier_semantic.py), [`eva_regenerate_guard.py`](eva_regenerate_guard.py).

### 3. User Notes module — capability without SFT

`RememberThis` / `ForgetMemory` are exposed as model-callable tools; **verifier-injection backstops** the cases where the model fails to call them on a clear user request. Each user gets isolated `Notes/` storage (FAISS + jsonl + audit log). 90 unit tests, no regression in 3 adjacent suites. The capability lands without any model fine-tuning — it's purely architecture + verifier policy.

See [`docs/USER_NOTES_MODULE.md`](docs/USER_NOTES_MODULE.md), [`Memory_maker/notes_runtime.py`](Memory_maker/notes_runtime.py).

### 4. Slot Subject Classifier — killing whack-a-mole

Earlier slot-detection patches kept growing negative-list regexes ("don't fire `full_name` if the question is about a pet"). The Slot Subject Classifier replaced this with `(slot_field, subject_class)` 2-tuple judgment. Person-only slots (`full_name` / `birthday` / `age` / `toy`) are gated through `is_person_subject()` — three layers: regex-strict / regex-loose / embedding NN, bilingual. 42 new unit tests + a 28-query golden fixture. The old stop-gap regexes are gone.

See [`eva_subject_classifier.py`](eva_subject_classifier.py), [`docs/SLOT_SUBJECT_CLASSIFIER_PLAN.md`](docs/SLOT_SUBJECT_CLASSIFIER_PLAN.md).

### 5. Discord deployment with multi-user session isolation

9B model doesn't support concurrent batching, so all Discord requests serialize through an `asyncio.Lock`. Per-Discord-user state lives in swap-in/out snapshots so users never bleed into each other's context. The pattern is covered by [`tests/test_session_isolation.py`](tests/test_session_isolation.py). `supervisord` manages the bot process with auto-restart.

See [`eva_discord.py`](eva_discord.py), [`eva_discord_sessions.py`](eva_discord_sessions.py).

## Tech Stack

| Layer | Choice |
|---|---|
| Base model | Qwen3.5-VL-9B + custom LoRA |
| Remote LLM | DeepSeek (Advisor + Verifier + Expert) |
| Embeddings + Index | FAISS (HNSW), in-memory |
| External tools | Tavily (WebSearch), DashScope qwen-vl-plus (Vision) |
| Frontends | Discord (production), Jupyter / Colab (development) |
| Deployment | RunPod (Ubuntu 22.04 + CUDA 12.4), supervisord |
| Process management | `asyncio.Lock` serializes inference, per-user session snapshots |

## How to Run

### Prerequisites
- Python 3.10
- ~50 GB disk (for merged model weights)
- For inference: 1× GPU with ≥ 24 GB VRAM (tested on RTX A6000 48 GB)
- API keys: DeepSeek, Tavily, DashScope; Discord bot token if running Discord mode

### Setup

```bash
git clone https://github.com/suixin7777/Eva.git
cd Eva
pip install -r requirements.txt
```

### Configure

Create a `.env` (already in `.gitignore`):

```dotenv
DEEPSEEK_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
VISION_API_KEY=sk-...           # DashScope qwen-vl-plus
DISCORD_TOKEN=...               # only for Discord mode
EVA_MODEL_PATH=/path/to/Qwen3.5-VL-9B-Merged
```

You will need to **supply your own base model and LoRA weights**. The author's trained weights are not redistributed. To use your own LoRA, point `EVA_MODEL_PATH` to a directory containing the merged Qwen3.5-VL-9B + LoRA weights produced by [`merge_lora.py`](merge_lora.py).

### Run

```bash
# Colab / Jupyter interactive (no Discord)
python eva_chat_colab.py

# Discord bot
python eva_discord.py

# Tests (offline; no API calls)
python -m pytest tests/
```

## Deployment (Overview)

Production setup runs on a single RunPod pod (RTX A6000 48 GB, Ubuntu 22.04, CUDA 12.4):

- **Inference**: single instance — 9B model doesn't support concurrent batching, so all Discord requests serialize through `asyncio.Lock`.
- **Process**: `supervisord` supervises `eva_discord.py`; auto-restart on crash; logs to `/workspace/eva_logs/`.
- **Multi-user**: each Discord `user_id` holds an isolated session snapshot in [`eva_discord_sessions.py`](eva_discord_sessions.py); swap-in/out is covered by [`tests/test_session_isolation.py`](tests/test_session_isolation.py).
- **Secrets**: all keys loaded from `.env`; never committed to git.
- **Model**: trained via LoRA on Qwen3.5-VL-9B, merged with [`merge_lora.py`](merge_lora.py) into a single inference-ready directory.

The personal deployment guide (with exact pod-creation steps, vendor SKUs, dashboard screenshots) is kept private. The above plus the source files (`eva_discord*.py`, `merge_lora.py`, `supervisord.conf`, `start.sh`) are enough to reproduce the architecture.

## Project Layout

```
eva_*.py                    core inference + verifier + memory + tools (24 files)
Advisor/                    remote DeepSeek Advisor module
Memory_maker/               lore-corpus build tools + Notes runtime store
  SAMPLE_LORE.jsonl         3 sample lore records (schema reference)
docs/                       architecture docs + design records
tests/                      24 test files, all offline
generate/                   (gitignored) data generation + training pipeline
Memory/                     (gitignored) lore corpus FAISS index + json
Notes/                      (gitignored, runtime) per-user notes stores
```

The **lore corpus** (full `Memory/`, `Memory_maker/*.jsonl`) is the character bible for Eva and her companion. It's kept private as creative IP; only [`Memory_maker/SAMPLE_LORE.jsonl`](Memory_maker/SAMPLE_LORE.jsonl) (3 records) is committed as a schema reference. To run Eva with personality, you'll need to author your own lore corpus following the schema.

## Status

Active personal project. Not soliciting contributions — feel free to fork.

## Author

[@suixin7777](https://github.com/suixin7777)
