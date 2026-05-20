# Benchmark Results

Raw JSON output from `benchmarks/eval_react_chains.py`.

| File | Description |
|---|---|
| `eval_v4_original.json` | Base model: Qwen3.5-9B (abliterated) |
| `eval_v4_eva.json` | Eva SFT: Qwen3.5-9B + custom LoRA |

Each file contains:

```json
{
  "model_name": "...",
  "config": { /* eval flags at run time */ },
  "results": [
    {
      "sample_id": "...",
      "tier": "...",
      "strict_pass": bool,
      "lenient_pass": bool,
      "outcome_pass": bool,
      "step_results": [ /* per-step parse + check details */ ],
      ...
    },
    ...
  ]
}
```

Headline numbers and per-tier breakdown are summarized in [`../../README.md#results`](../../README.md#results).
