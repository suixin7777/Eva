# Eva Benchmark Suite

Internal targeted benchmark used to measure the impact of SFT (LoRA fine-tuning) on Eva's agent capabilities. Not a public leaderboard — focused on the specific capability mix this agent needs (memory recall + multimodal tool routing + persona consistency).

## Sample design (27 chains total)

| Tier | What it tests | # samples |
|---|---|---|
| **T1 ReAct chains** | Multi-step tool reasoning with realistic tool outputs (memory + web + image + text-gen) | 16 |
| **T2_A Novel tool use** | Adapting to a tool not seen during SFT (DateDiff / UnitConverter / Translator) | 3 |
| **T2_B Distractor tools** | Ignoring irrelevant tools (QRCodeGen / Calculator) injected into the prompt | 2 |
| **T2_C Missing tool fallback** | Coping when an expected tool (TextGenerationTool) is removed from the prompt | 2 |
| **T3 Persona consistency** | Master vs guest voice + self-knowledge + romantic-boundary refusal | 4 |

## Three scoring modes

Each chain is graded independently under three modes:

- **STRICT** — model follows the expected reasoning chain step-by-step
- **LENIENT** — accepts:
  - Reasonable early stops (model skipped a tool but the answer is correct)
  - Wrong-tool recoveries (intermediate wrong tool but final answer correct, capped at ≤ `len(steps)/3` failures before the last step)
  - Compatible tool substitutions:
    - `GetCurrentTime` covering a date/time/weekday-style `WebSearch`
    - `MemorySearch(target_entity="Both")` covering separate `Eva` / `Rosm` / `Shared` queries
- **OUTCOME** — final answer correctness only, path-agnostic

This split prevents conflating "wrong path" with "wrong answer" — important when the model finds a valid shortcut.

## Persona grading

Persona is **graded 0-3** instead of binary:

| Score | Signal |
|---|---|
| 0 | Cold AI-assistant tone ("How can I help you", "As an AI...") |
| 1 | Neutral, no signal either way |
| 2 | Warm tsundere signals (`hmph`, `tch`, `of course`, `just for you`, 哼, 才不是) |
| 3 | Strong Master signal (explicit "Master" / "主人") |

Guest mode flips the check: any of `master`, `darling`, `sweetheart`, `my love` → fail.

When `expected_no_tool` is set and passes, persona check goes into **soft mode** (only requires "not cold", not strong signals) to avoid double-penalizing simple direct answers.

## Models compared

- **Base**: Qwen3.5-9B (abliterated variant; chosen because the safety layer in the standard release interferes with sustained character roleplay)
- **Eva SFT**: same base + custom LoRA fine-tune

Set via environment variables:

```bash
export EVAL_ORIGINAL_MODEL_PATH=huihui-ai/Huihui-Qwen3.5-9B-abliterated
export EVAL_EVA_MODEL_PATH=/path/to/Eva-Qwen3.5-VL-9B-Merged
```

## Running

```bash
python benchmarks/eval_react_chains.py
```

Requires:
- 1× GPU with ≥ 24 GB VRAM (bf16 inference of two 9B models, loaded sequentially)
- `transformers`, `torch` from main `requirements.txt`

Outputs:
- `benchmarks/results/eval_v4_original.json`
- `benchmarks/results/eval_v4_eva.json`

A per-sample console log is also printed during the run.

## Latest results (2026-05)

See [`../README.md#results`](../README.md#results) for the headline table and per-tier breakdown.

Raw run output is in [`results/`](results/).

## Efficiency analysis

After running the eval, you can re-derive the efficiency numbers (median tokens/turn, MAX_NEW_TOKENS-cap rate, etc.) from the saved JSON without re-running the model:

```bash
python benchmarks/analyze_efficiency.py
```

This reads `benchmarks/results/eval_v4_*.json` and prints a side-by-side comparison. See [`analyze_efficiency.py`](analyze_efficiency.py) for the exact metrics. The output of this script is what backs the **Efficiency** subsection of the main README.

## Caveats

- **Sample size** — 27 chains is small. The goal is signal on the capability mix this specific agent needs, not external benchmark comparability. Each sample is hand-designed with realistic mock tool outputs.
- **Tool outputs are mocked** — the eval script does not actually call DeepSeek / Tavily / DashScope / vision APIs. The model receives the listed `tool_output` strings as if they were real tool results. This is intentional: the goal is to measure the model's *routing + reasoning + persona*, not external API quality.
- **Image samples** — image-bearing prompts include the `<|image|>` marker in text only; no image bytes are passed. The model's vision-routing behavior is exercised via the marker + mocked OCR outputs in subsequent steps. This keeps the eval reproducible without supplying image files.
- **Single seed, greedy decoding** — `USE_SAMPLING = False`. Numbers are deterministic per run but variance under sampling is not measured.
