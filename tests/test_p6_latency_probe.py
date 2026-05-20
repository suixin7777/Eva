"""P6.3 production latency probe.

Standalone script to measure DeepSeek call latency for the pronoun
resolver from any environment — no model loading, no agent build.
Run this in your TARGET production region to verify P95 latency
falls within v4 thresholds before final P6.3 rollout.

Usage (from project root):
    export DEEPSEEK_API_KEY=sk-...
    python tests/test_p6_latency_probe.py

Or with custom sample count:
    python tests/test_p6_latency_probe.py --n 50

Output:
    Per-call latency + P50/P95/max summary + verdict against thresholds.

Why this script exists:
    Shadow tests run on Colab measured P95 ≈ 15s, but that's
    Colab(US/EU) → DeepSeek(CN) cross-ocean. Production is presumably
    closer to DeepSeek's servers and should see P95 < 1500ms.

    This probe isolates the LLM call from everything else (no model
    inference, no chat session, no verifier path) so the number is
    a clean lower bound on resolver latency in production.

v4 thresholds (revised 2026-05-08 after n=50 stable baseline showed
DeepSeek server time + variance puts realistic P95 at ~4500ms even
in-region; original 800ms / 4000ms targets were unreachable):

    cn_native (server in China, direct DeepSeek API):
        P50 ≤ 3000ms, P95 ≤ 5000ms
    cn_proxy (server in China via proxy):
        P50 ≤ 3500ms, P95 ≤ 5500ms
    cross_ocean (anywhere outside CN):
        not gated — for sanity-check only
"""
import argparse
import json
import os
import sys
import time
from statistics import mean, median

# ------------------------------------------------------------
# Path setup — file lives in tests/, imports eva_* from parent.
# ------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ============================================================
# Sample queries — representative of what hits the resolver in
# production. Mix of short follow-ups, varied antecedent types.
# Each entry is (query, mock_history_assistant_text). The script
# constructs the recent_turns dict shape the resolver expects.
# ============================================================
SAMPLES = [
    ("really? Check it",        "I have a music box on my shelf."),
    ("check it",                "Your favorite toy is a cuddly bunny."),
    ("hold on, check it",       "Yes, your birthday is November 25th."),
    ("do it again",             "Here's a joke about cats."),
    ("really? do that",         "I love ballet dancing."),
    ("can you check it?",       "I have a special collection of records."),
    ("sorry, check that",       "The photo is on the table."),
    ("look at them",            "These are my favorite games: Apex, Battlefield."),
    ("tell me about it",        "Yesterday we went to the museum."),
    ("what is it?",             "I gave you a chocolate cake last birthday."),
]


def build_payload(query, history_assistant):
    """Construct the JSON payload exactly as resolve_pronoun does."""
    recent_turns = [
        {"user": "", "assistant": history_assistant[:400].strip()},
    ]
    return json.dumps(
        {"query": query, "recent_turns": recent_turns},
        ensure_ascii=False,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=30,
                        help="number of probe calls (default: 30)")
    parser.add_argument("--region", choices=["cn_native", "cn_proxy",
                                             "cross_ocean", "auto"],
                        default="auto",
                        help="region for verdict thresholds (default: auto-detect)")
    args = parser.parse_args()

    # ---- Pre-flight ----
    if not os.environ.get("DEEPSEEK_API_KEY", "").startswith("sk-"):
        print("ERROR: DEEPSEEK_API_KEY not set or invalid", file=sys.stderr)
        sys.exit(1)

    # Import lazily so missing deps surface clearly
    try:
        from eva_pronoun_resolver import PROMPT_PRONOUN_RESOLVER
        from eva_tools_runtime import call_deepseek_judge
    except ImportError as e:
        print(f"ERROR: cannot import from eva_*: {e}", file=sys.stderr)
        print("  Run this script from the Eva project directory.", file=sys.stderr)
        sys.exit(1)

    # ---- Run probes ----
    print(f"=== P6.3 latency probe ===")
    print(f"  samples: {len(SAMPLES)} queries × {args.n // len(SAMPLES) + 1} cycles "
          f"(~{args.n} total calls)")
    print(f"  region:  {args.region}")
    print()

    latencies = []
    failures = 0
    cycles = (args.n + len(SAMPLES) - 1) // len(SAMPLES)

    call_idx = 0
    for cycle in range(cycles):
        for query, history in SAMPLES:
            if call_idx >= args.n:
                break
            call_idx += 1
            payload = build_payload(query, history)

            t0 = time.perf_counter()
            try:
                result = call_deepseek_judge(
                    PROMPT_PRONOUN_RESOLVER, payload,
                    debug=False, timeout=15.0,
                )
                ms = (time.perf_counter() - t0) * 1000.0
                ok = isinstance(result, dict) and "needs_resolution" in result
                if not ok:
                    failures += 1
                    err = result.get("error") if isinstance(result, dict) else "?"
                    print(f"  [{call_idx:3d}] FAIL  err={err!r}  q={query!r}")
                else:
                    latencies.append(ms)
                    print(f"  [{call_idx:3d}] {ms:7.1f}ms  q={query!r}")
            except Exception as e:
                ms = (time.perf_counter() - t0) * 1000.0
                failures += 1
                print(f"  [{call_idx:3d}] EXC   {ms:7.1f}ms  err={e!r}  q={query!r}")

    # ---- Summary ----
    print()
    print(f"=== summary ===")
    print(f"  total calls:    {call_idx}")
    print(f"  successes:      {len(latencies)}  ({len(latencies)/call_idx:.1%})")
    print(f"  failures:       {failures}")

    if not latencies:
        print(f"  no successful calls — check API key / network")
        sys.exit(2)

    lats = sorted(latencies)
    def pct(p):
        k = max(0, min(len(lats) - 1, int(round(p * (len(lats) - 1)))))
        return lats[k]

    p50, p95, lat_max = pct(0.50), pct(0.95), lats[-1]
    print(f"  P50:            {p50:.1f}ms")
    print(f"  P95:            {p95:.1f}ms")
    print(f"  max:            {lat_max:.1f}ms")
    print(f"  mean:           {mean(lats):.1f}ms")

    # ---- Auto-detect region if not specified ----
    # Heuristic based on 2026-05-08 baseline: cn_native sees ~2500ms P50,
    # cross_ocean sees ~2800ms+. Rough P95 cuts: <3500 cn, <5000 cn-proxy,
    # else cross_ocean.
    if args.region == "auto":
        if p95 < 3500:
            detected = "cn_native"
        elif p95 < 5500:
            detected = "cn_proxy"
        else:
            detected = "cross_ocean"
        print(f"  region (auto):  {detected}  (P95={p95:.0f}ms)")
        region = detected
    else:
        region = args.region

    # ---- Verdict ----
    print()
    print(f"=== v4 verdict (region={region}) ===")
    fails = []
    if region == "cn_native":
        if p50 > 3000:
            fails.append(f"P50 {p50:.0f}ms > 3000ms (cn_native threshold)")
        if p95 > 5000:
            fails.append(f"P95 {p95:.0f}ms > 5000ms (cn_native threshold)")
    elif region == "cn_proxy":
        if p50 > 3500:
            fails.append(f"P50 {p50:.0f}ms > 3500ms (cn_proxy threshold)")
        if p95 > 5500:
            fails.append(f"P95 {p95:.0f}ms > 5500ms (cn_proxy threshold)")
    elif region == "cross_ocean":
        print("  (cross_ocean — not gated; latency expected high. Run this on")
        print("   a server in/near DeepSeek's region for production decision.)")

    if region != "cross_ocean":
        if not fails:
            print(f"  PASS — latency is within {region} threshold")
        else:
            print(f"  FAIL:")
            for f in fails:
                print(f"    • {f}")
            sys.exit(3)


if __name__ == "__main__":
    main()
