"""
Analyze inference efficiency from eval JSON outputs.

Reads benchmarks/results/eval_v4_original.json and eval_v4_eva.json,
computes per-model:
    - total turns generated
    - total / avg characters in raw model output
    - share of turns that hit MAX_NEW_TOKENS (proxy: raw_response very long
      AND no <|end_react|> emitted)
    - share of chains with format_or_parse_error

Prints a comparison table. Numbers feed README.md Results > Efficiency.

Run:
    python benchmarks/analyze_efficiency.py
"""

import json
import os
import sys
from typing import Dict, Any

DEFAULT_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# MAX_NEW_TOKENS in eval_react_chains.py defaults to 768. With a typical
# tokenizer ratio of ~3.5 chars/token, 768 tokens ~= 2700 chars. A raw
# response >= 2500 chars is essentially at the cap. This catches the case
# where the model emits <|end_react|> as multi-token text (so generation
# never eos-stops) and just keeps going to the token budget.
NEAR_CAP_THRESHOLD = 2500


def analyze_run(path: str, label: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data["results"]

    total_chains = len(results)
    total_turns = 0
    total_cleaned_chars = 0
    chains_with_format_error = 0
    near_cap_turns = 0
    raw_lens = []

    for r in results:
        chain_format_err = False
        for s in r["step_results"]:
            raw = s.get("raw_response", "") or ""
            cleaned = s.get("cleaned_response", "") or ""

            # Skip virtual merged-memory entries (no real generation)
            if not raw:
                continue

            raw_lens.append(len(raw))
            total_cleaned_chars += len(cleaned)

            if s.get("error_type") == "format_or_parse_error":
                chain_format_err = True

            if len(raw) >= NEAR_CAP_THRESHOLD:
                near_cap_turns += 1

        if chain_format_err:
            chains_with_format_error += 1

        total_turns += r["generated_turn_count"]

    raw_lens.sort()
    n = len(raw_lens) if raw_lens else 1
    total_raw_chars = sum(raw_lens)

    def pct(p: float) -> int:
        if not raw_lens:
            return 0
        idx = min(int(n * p), n - 1)
        return raw_lens[idx]

    avg_turns_per_chain = total_turns / total_chains if total_chains else 0
    avg_raw_chars_per_turn = total_raw_chars / total_turns if total_turns else 0
    avg_cleaned_chars_per_turn = total_cleaned_chars / total_turns if total_turns else 0
    # Rough English-ish token estimate. Real ratio varies by tokenizer.
    est_tokens_per_turn = avg_raw_chars_per_turn / 3.5

    return {
        "label": label,
        "chains": total_chains,
        "turns": total_turns,
        "avg_turns_per_chain": avg_turns_per_chain,
        "raw_chars": total_raw_chars,
        "cleaned_chars": total_cleaned_chars,
        "avg_raw_chars_per_turn": avg_raw_chars_per_turn,
        "avg_cleaned_chars_per_turn": avg_cleaned_chars_per_turn,
        "est_tokens_per_turn": est_tokens_per_turn,
        "format_err_chains": chains_with_format_error,
        "near_cap_turns": near_cap_turns,
        "raw_min": raw_lens[0] if raw_lens else 0,
        "raw_p25": pct(0.25),
        "raw_median": pct(0.50),
        "raw_p75": pct(0.75),
        "raw_max": raw_lens[-1] if raw_lens else 0,
    }


def print_run(stats: Dict[str, Any]) -> None:
    print(f"=== {stats['label']} ===")
    print(f"  chains                       : {stats['chains']}")
    print(f"  total turns                  : {stats['turns']}")
    print(f"  avg turns / chain            : {stats['avg_turns_per_chain']:.2f}")
    print(f"  total raw chars              : {stats['raw_chars']:,}")
    print(f"  total cleaned chars          : {stats['cleaned_chars']:,}")
    print(f"  avg raw chars / turn         : {stats['avg_raw_chars_per_turn']:.0f}")
    print(f"  est. tokens / turn           : {stats['est_tokens_per_turn']:.0f}  (chars / 3.5)")
    print(f"  raw chars distribution       :")
    print(f"      min / p25 / median / p75 / max =")
    print(f"      {stats['raw_min']} / {stats['raw_p25']} / {stats['raw_median']} / "
          f"{stats['raw_p75']} / {stats['raw_max']}")
    print(f"  near-cap turns (>= {NEAR_CAP_THRESHOLD} chars): "
          f"{stats['near_cap_turns']} / {stats['turns']}  "
          f"({stats['near_cap_turns']/max(stats['turns'],1)*100:.1f}%)")
    print(f"  chains w/ format error       : {stats['format_err_chains']} / {stats['chains']}")


def print_comparison(base: Dict[str, Any], eva: Dict[str, Any]) -> None:
    print("=" * 60)
    print("Comparison (Base vs Eva SFT)")
    print("=" * 60)

    def ratio(a: float, b: float) -> str:
        if b == 0:
            return "inf" if a > 0 else "n/a"
        return f"{a / b:.2f}x"

    print(f"  total raw chars        : Base {base['raw_chars']:,}   "
          f"Eva {eva['raw_chars']:,}   "
          f"-> {ratio(base['raw_chars'], eva['raw_chars'])} more text from base")
    print(f"  median raw chars/turn  : Base {base['raw_median']}        "
          f"Eva {eva['raw_median']}         "
          f"-> {ratio(base['raw_median'], eva['raw_median'])}")
    print(f"  avg raw chars / turn   : Base {base['avg_raw_chars_per_turn']:.0f}       "
          f"Eva {eva['avg_raw_chars_per_turn']:.0f}      "
          f"-> {ratio(base['avg_raw_chars_per_turn'], eva['avg_raw_chars_per_turn'])} more per turn")
    print(f"  est. tokens / turn     : Base {base['est_tokens_per_turn']:.0f}        "
          f"Eva {eva['est_tokens_per_turn']:.0f}        "
          f"-> ~{ratio(base['est_tokens_per_turn'], eva['est_tokens_per_turn'])}")
    print(f"  turns / chain          : Base {base['avg_turns_per_chain']:.2f}       "
          f"Eva {eva['avg_turns_per_chain']:.2f}")
    print(f"  near-cap turns (>={NEAR_CAP_THRESHOLD}c): "
          f"Base {base['near_cap_turns']} / {base['turns']} "
          f"({base['near_cap_turns']/max(base['turns'],1)*100:.0f}%)   "
          f"Eva {eva['near_cap_turns']} / {eva['turns']} "
          f"({eva['near_cap_turns']/max(eva['turns'],1)*100:.0f}%)")
    print(f"  format-error chains    : Base {base['format_err_chains']} / {base['chains']}      "
          f"Eva {eva['format_err_chains']} / {eva['chains']}")

    print()
    print("Notes on the latency estimate:")
    print("  - Wall-clock time wasn't logged in the eval JSON. Same model")
    print("    architecture + same GPU + batch=1 -> inference latency is")
    print("    approximately linear in tokens generated, so the per-turn")
    print("    token ratio is a reasonable proxy for the wall-clock ratio.")
    print("  - End-to-end per-chain latency factor ~ (turns/chain) * (tokens/turn)")
    base_chain_tokens = base["avg_turns_per_chain"] * base["est_tokens_per_turn"]
    eva_chain_tokens = eva["avg_turns_per_chain"] * eva["est_tokens_per_turn"]
    print(f"    Base : {base['avg_turns_per_chain']:.2f} * {base['est_tokens_per_turn']:.0f}"
          f" ~= {base_chain_tokens:.0f} tokens / chain")
    print(f"    Eva  : {eva['avg_turns_per_chain']:.2f} * {eva['est_tokens_per_turn']:.0f}"
          f" ~= {eva_chain_tokens:.0f} tokens / chain")
    print(f"    Eva is ~{ratio(base_chain_tokens, eva_chain_tokens)} less work per chain.")


def main() -> int:
    results_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RESULTS_DIR
    base_path = os.path.join(results_dir, "eval_v4_original.json")
    eva_path = os.path.join(results_dir, "eval_v4_eva.json")

    missing = [p for p in (base_path, eva_path) if not os.path.exists(p)]
    if missing:
        print("Missing JSON files. Run benchmarks/eval_react_chains.py first.")
        for p in missing:
            print(f"  - not found: {p}")
        return 1

    base = analyze_run(base_path, "Original Model (base)")
    print_run(base)
    print()
    eva = analyze_run(eva_path, "Eva SFT")
    print_run(eva)
    print()
    print_comparison(base, eva)
    return 0


if __name__ == "__main__":
    sys.exit(main())
