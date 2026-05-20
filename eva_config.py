"""
eva_config.py — Centralised tunables for Eva inference (P2 refactor).

This module owns ALL global constants:
- API keys / model paths
- ReAct/ChatML tokens & special-token glue
- Sampling presets for phase-1 (planning) and phase-2 (answer)
- Retrieval thresholds (FAISS / BM25 / rerank cutoffs)
- Memory-judge thresholds (EXACT/RELATED tiering)
- Verifier flags

It is import-safe in any order: nothing here triggers model load or HTTP.
"""

import os
import re
import torch
from pathlib import Path


# ============================================================
# 0a. Lightweight .env loader (no python-dotenv dependency).
# 2026-05-13: 把项目根 / generate/ 下的 .env 自动注入 os.environ，
# 让 migrate_slot_values.py 等开发脚本不用 export 就能拿到 DEEPSEEK_API_KEY。
# 设计原则：
#   - 已在 os.environ 里的值不被覆盖（shell export 优先于 .env）
#   - 文件不存在 / 格式错静默跳过——dev 友好，不影响生产
#   - 只接受 KEY=VALUE 行，# 开头视为注释，引号自动剥
# ============================================================
def _load_dotenv():
    here = Path(__file__).resolve().parent
    candidates = [
        here / ".env",
        here / "generate" / ".env",
        Path.cwd() / ".env",
    ]
    for p in candidates:
        if not p.exists():
            continue
        try:
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                # shell-export 优先：os.environ 已有值不覆盖
                if k and v and k not in os.environ:
                    os.environ[k] = v
        except Exception:
            # .env loader 是辅助功能，任何错误都不该破坏 import 链
            pass


_load_dotenv()

# ============================================================
# 0b. 时间/时区接口
#
# 唯一入口在 eva_time.get_current_time(tz=None)。这里 re-export，让
# eva_core 透过 `from eva_config import *` 仍能拿到 local_now（向后
# 兼容别名）。
# ============================================================
from eva_time import EVA_TIMEZONE, get_current_time
local_now = get_current_time   # 向后兼容别名


# ============================================================
# 0. Backend stability (Blackwell sm_120 friendliness)
# ============================================================
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# ============================================================
# 1. External services + paths
#
# API keys are read STRICTLY from environment variables. The fallback
# default for the 3 secret keys is "" (empty string), not a hardcoded
# key. Reasons:
#   - Hardcoded keys leak into git history even after rotation.
#   - Silent acceptance of an expired hardcoded key produces
#     hard-to-debug 401 failures deep in the call stack (the lesson
#     from the 2026-05-06 vision-tool incident: a stale fallback
#     key let four consecutive regression runs "PASS" while every
#     vision call silently 401'd).
#
# Non-secret values (BASE_URL, MODEL, file paths) keep their defaults
# so a fresh checkout still boots without a 6-line export incantation.
#
# Set the keys in Colab via:
#   import os
#   os.environ["DEEPSEEK_API_KEY"] = "sk-..."
#   os.environ["TAVILY_API_KEY"]   = "tvly-..."
#   os.environ["VISION_API_KEY"]   = "sk-..."
# ============================================================
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")
VISION_API_KEY = os.environ.get("VISION_API_KEY", "")
VISION_BASE_URL = os.environ.get(
    "VISION_BASE_URL",
    "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
)
VISION_MODEL = os.environ.get("VISION_MODEL", "qwen-vl-plus")

MODEL_PATH = os.environ.get("EVA_MODEL_PATH", "/content/Eva-Qwen3.5-VL-9B-Merged")
MEMORY_INDEX_PATH = os.environ.get("EVA_MEMORY_INDEX", "Memory/memory.index")
MEMORY_CONTENT_PATH = os.environ.get("EVA_MEMORY_CONTENT", "Memory/memory.jsonl")
TOPIC_KEYWORDS_PATH = os.environ.get(
    "EVA_TOPIC_KEYWORDS", "topic_keywords.json"
)


# ============================================================
# Boot-time sanity check — fail fast if a key is unset.
#
# Each of the 3 keys is required by a distinct subsystem:
#   DEEPSEEK_API_KEY -> verifier judges (Plan B + PRE PROBE) +
#                       call_deepseek_expert
#   TAVILY_API_KEY   -> WebSearch tool
#   VISION_API_KEY   -> AskRemoteVision tool
#
# We DO NOT raise on missing keys at import time — Eva should still
# boot for offline tests, slot-only conversations, and
# memory-only debugging. Instead we print a single warning per
# missing key so the operator knows which subsystem is degraded.
# ============================================================
def _warn_missing_api_keys():
    missing = []
    if not DEEPSEEK_API_KEY:
        missing.append(("DEEPSEEK_API_KEY",
                        "verifier judges + DeepSeek expert"))
    if not TAVILY_API_KEY:
        missing.append(("TAVILY_API_KEY", "WebSearch tool"))
    if not VISION_API_KEY:
        missing.append(("VISION_API_KEY", "AskRemoteVision tool"))
    for name, subsystem in missing:
        print(f"[eva_config] WARNING: {name} not set in env — "
              f"{subsystem} will fail at call time.")


_warn_missing_api_keys()

# ============================================================
# 2. Runtime caps
# ============================================================
USE_SEARCH = True
MAX_STEPS = 8
MAX_NEW_TOKENS_TURN = 2048
LOCAL_PIXELS = 1048576

# ============================================================
# 3. ReAct / ChatML tokens
# These MUST match SFT vocabulary. Do not change.
# ============================================================
REACT = {
    "tool_code": "<|tool_code|>",
    "tool_output": "<|tool_output|>",
    "answer": "<|answer|>",
    "end": "<|end_react|>",
}
THINK_START = "<think>"
THINK_END = "</think>"
EOT = "<|im_end|>"

# Force <think> prefix at Phase-1 decode start (2026-05-08).
# Greedy decoder occasionally skipped <think> for short queries
# (e.g. "for example?", "what kinds of gift do you want?") and
# went straight to <|answer|>. Empirically this correlates with
# higher hallucination rate (e.g. memory-bound answers fabricating
# unrecorded specifics). Prefixing <think> hard-forces the model
# to start inside a think block, guaranteeing a self-reflection
# step before answering. Cost: ~200-500ms per turn for thought
# generation. Set False to revert to model's native choice.
FORCE_THINK_PREFIX = True
SPECIAL_TOKENS = {"additional_special_tokens": list(REACT.values())}
# Legacy alias (eva_core.py historically called this SPECIAL).
SPECIAL = SPECIAL_TOKENS
TAG_RE = re.compile(
    r"(<think>|</think>|<\|tool_code\|>|<\|tool_output\|>|<\|answer\|>)"
)
IMAGE_BLOCK = "<|vision_start|><|image_pad|><|vision_end|>"

# ============================================================
# 4. Phase-2 sampling presets
# ============================================================
PHASE2_SAMPLING_PRESETS = {
    # 2026-05-15: direct mode 调低 temp/rep_pen 减少 sampling collapse。
    # 旧值 (0.82, 0.90, 1.08) 触发 orphan markdown 截断（模型打开 ** 后
    # 找不到能逃过 rep_penalty 的 continuation 就早停）。新值 (0.78, 0.90,
    # 1.06) 在保留 persona variance 的前提下，给"合理重复"留出概率空间。
    "direct":       {"temperature": 0.78, "top_p": 0.90, "repetition_penalty": 1.06},
    "after_tool":   {"temperature": 0.60, "top_p": 0.85, "repetition_penalty": 1.06},
    "after_memory": {"temperature": 0.35, "top_p": 0.80, "repetition_penalty": 1.05},
}
PHASE2_MODE_TO_PRESET = {
    "direct": "direct",
    "after_tool": "after_tool",
    "after_memory": "after_memory",
    # Low-confidence memory should still be evidence-grounded => after_memory.
    "low_confidence": "after_memory",
}


def get_phase2_sampling_config(mode: str):
    return PHASE2_SAMPLING_PRESETS[PHASE2_MODE_TO_PRESET.get(mode, "direct")]


# ============================================================
# 5. Memory retrieval thresholds
# ============================================================
RERANK_TOP_K = 15
FINAL_TOP_K = 8
RERANK_CUTOFF = 1.5
KEEP_TOP1_PER_SUBQUERY = True
PROTECTED_RECORDS_PER_SUBQUERY = 1

LOW_CONFIDENCE_THRESHOLD = 4.0
HIGH_CONFIDENCE_BAR = 5.0
WEAK_RELATED_TOP1_BAR = 2.0  # P1.8.2: skip injection when no exact + top1 below this

FAISS_WEIGHT = 0.6
BM25_WEIGHT = 0.4

# ============================================================
# 6. Memory judge (EXACT / RELATED labelling)
# ============================================================
MEMORY_JUDGE_TOP_K = 20
MEMORY_JUDGE_KEEP_TOP_K = FINAL_TOP_K
MEMORY_JUDGE_DEBUG = True
EXACT_RERANK_DELTA_TIER = 1.0
MIN_EXACT_RERANK_ABSOLUTE = 4.0

# ============================================================
# 7. Safe keyword evidence inside MemorySearch
# ============================================================
ENABLE_SAFE_KEYWORD_EVIDENCE = True
MEMORY_ATTRIBUTE_FIELDS = {"favorite"}
SAFE_KEYWORD_REQUIRED_HIT_BONUS = 0.14
SAFE_KEYWORD_OPTIONAL_HIT_BONUS = 0.05
SAFE_KEYWORD_MAX_BONUS = 0.44
SAFE_KEYWORD_GATE_FAIL_PENALTY = -0.16
SAFE_KEYWORD_MAX_TERMS_PER_GROUP = 24

# ============================================================
# 8. Verifier
# ============================================================
ENABLE_ANSWER_VERIFIER = True
VERIFIER_DEBUG = True

# 2026-05-08: visual styling for the STEP-5 trace rewrite block.
# When verifier rejects a phase-1 answer and the controller rewrites
# the history into tool-call shape, the prior THOUGHT/ANSWER lines
# are still scrolled up on screen — operators have to mentally tag
# them as superseded. The new rendering prints a loud divider with
# explicit "above is superseded" wording plus [REWRITTEN] prefixes
# on the new thought + tool_code, so the rewrite boundary is visible
# at a glance.
#
# Style options:
#   "ansi"  — strikethrough+dim ANSI codes for the supersede notice.
#             Most modern terminals (VS Code, modern Windows Terminal,
#             Linux/macOS) render this. Some legacy Windows consoles
#             (cmd.exe pre-Windows-10, conhost without VT mode) print
#             the raw escape codes — flip to "ascii" there.
#   "ascii" — plain ASCII divider with === bars. Always renders.
#
# Default "ansi": new terminals are the dominant case in 2026.
TRACE_REWRITE_STYLE = "ansi"
HARD_VERIFIER_REASONS = {
    # TODO 11-arch (2026-05-07): SUPERSEDED by REASON_POLICY in
    # eva_verifier_logic.py. The verifier no longer reads this set —
    # severity is derived per-reason from REASON_POLICY (default
    # fail-safe is hard). This set remains here only as a back-compat
    # surface in case external code (offline tests, integrations) still
    # imports it. Don't extend it; add new reasons to REASON_POLICY.
    "missing_web_evidence_for_external_or_current_request",
    "missing_memorysearch_for_explicit_memory_check",
    "missing_date_calculation_evidence",
    "tool_call_leaked_in_answer",
    "eva_self_birthday_pronoun_mismatch",
    "date_math_target_date_mismatch",
}

# ============================================================
# 8d. P0 — Semantic-class reason gating (TODO refactor v2)
#
# The 3 "semantic" verifier reasons below judge meaning (pronoun
# referent, perspective, fact match) using local regex over the
# answer string ALONE — no conversation history, no LLM. They are
# the dominant source of false positives in the current verifier
# (see Turn 3 death-loop incident).
#
# These flags allow demoting them to log-only soft warnings while
# the v2 SemanticVerifier (LLM + history) is built out. Set
# ENABLE_SEMANTIC_HARD_FAIL = False to immediately stop the false-
# positive cascade without removing detection: reasons are still
# logged so we can collect samples for the v2 prompt.
# ============================================================
ENABLE_SEMANTIC_HARD_FAIL = False  # P0: regex-based semantic checks DEMOTED

# Per-reason override. None = follow ENABLE_SEMANTIC_HARD_FAIL.
# True/False = explicit override for that reason.
SEMANTIC_REASON_HARD_OVERRIDES = {
    "eva_self_birthday_pronoun_mismatch": None,
    "textgen_perspective_mismatch": None,
    "toy_value_conflicts_with_exact_memory": None,
}

# Maximum regenerate attempts PER reason PER turn. 0 = never
# regenerate (rely on phase-2 first sample). Defends against the
# "same regex judges same answer" infinite-loop class.
MAX_REGENERATE_ATTEMPTS_PER_REASON = 1

# ============================================================
# 8e. P1 — RegenerateGuard total budget + SemanticVerifier shadow
#
# RegenerateGuard generalises the legacy `last_regenerate_reason`
# single-slot guard in eva_core.py into a per-turn ledger keyed by
# dominant reason. Two limits compose:
#   - per-reason limit (MAX_REGENERATE_ATTEMPTS_PER_REASON above):
#     defends against the "same regex flags same answer" loop.
#   - total per-turn limit (below): defends against the
#     reason-A -> regenerate -> reason-B -> regenerate -> reason-A
#     hop-around loop where each individual reason stays within its
#     per-reason budget but the turn never converges.
# When the total budget is exhausted the verifier dispatcher releases
# the most recent phase-2 answer (NOT a canned apology) — same
# fail-open posture as the P0 semantic-reason path.
# ============================================================
MAX_REGENERATE_ATTEMPTS_PER_TURN = 2

# ============================================================
# 8f. P1 — SemanticVerifier (LLM-based, history-aware)
#
# Replaces the regex-based semantic reason detection (currently
# DEMOTED to soft via ENABLE_SEMANTIC_HARD_FAIL=False). The new
# verifier sends the last N (user, assistant) pairs + the current
# answer to DeepSeek and asks for a structured verdict. To avoid
# breaking production traffic on day one it ships in SHADOW mode:
# the verdict is logged & cached but NEVER influences dispatch.
#
# Promotion path (handled in subsequent phases — not auto):
#   shadow -> warn-only (logs surface as soft warnings)
#         -> hard (replaces regex semantic checks entirely)
#
# Disabling: set ENABLE_SEMANTIC_VERIFIER = False to no-op the
# whole module; nothing else needs to change.
# ============================================================
ENABLE_SEMANTIC_VERIFIER = True
# P3: graduate the semantic verifier from shadow to warn. Warn mode
# logs `[WARN] semantic verifier flagged ...` to stdout but does NOT
# block or replace the answer — same dispatch as shadow. Promotion to
# 'hard' requires the adversarial test set to pass first.
SEMANTIC_VERIFIER_MODE = "warn"  # "shadow" | "warn" | "hard"

# ============================================================
# 8d. P3 final: Legacy semantic regex switch
#
# Three semantic reasons (eva_self_birthday_pronoun_mismatch,
# toy_value_conflicts_with_exact_memory, textgen_perspective_mismatch)
# are fully owned by SemanticVerifier (eva_verifier_semantic.py) since
# P3. The old regex paths in eva_verifier_logic.verify_final_answer
# are kept behind this flag so we can re-enable them as a fallback
# if SemanticVerifier ever regresses (DeepSeek outage, prompt drift).
#
# Default False = SemanticVerifier owns these checks alone (CPU saved,
# no double-firing). Flip True only as an emergency rollback; pair
# with SEMANTIC_VERIFIER_MODE = "warn" or "shadow" to avoid double-
# firing on the same answer.
#
# NOTE: unsupported_exact_toy_claim is NOT under this flag — it remains
# a structural check (toy_animals named without backing evidence,
# routed via inject_tool to MemorySearch) that SemanticVerifier does
# not own. Only the conflicts-with-exact and pronoun/perspective
# checks are gated.
# ============================================================
ENABLE_LEGACY_SEMANTIC_REGEX = False
SEMANTIC_VERIFIER_DEBUG = True
SEMANTIC_VERIFIER_HISTORY_TURNS = 4
# Per-turn budget shared with Plan-B / PRE-PROBE judges already capped
# by LLM_JUDGE_MAX_CALLS_PER_TURN. The semantic verifier additionally
# caps its own calls below to avoid eating the global budget on a
# single turn that produces multiple answers (regenerate path).
SEMANTIC_VERIFIER_MAX_CALLS_PER_TURN = 4
# Cache key is sha1(answer + last_user_text + history_digest); same
# input within a turn returns cached verdict instead of re-paying the
# DeepSeek round-trip.
SEMANTIC_VERIFIER_CACHE_ENABLED = True

# Per-call timeout for the SemanticVerifier judge call. Independent
# from LLM_JUDGE_TIMEOUT_SECONDS (8s) which sits on short PUBLIC_FACT
# / EXPLICIT_MEMORY classifiers — the semantic verifier sends a
# longer payload (history + answer + evidence) and consistently
# tripped the 8s cap during the Turn-3 live trace. 15s gives DeepSeek
# enough headroom for the JSON-mode response without making the
# overall turn perceptibly slower; on hard timeout the verifier
# fail-opens to 'pass' so the user-facing latency cap is still
# bounded.
SEMANTIC_VERIFIER_TIMEOUT_SECONDS = 15

# P3: confidence floor for promoting an LLM-judge `fail` verdict to a
# verifier hard_fail. Below this threshold, fails are demoted to warn.
# Set conservatively — the prompt itself already requires >=0.80 for
# fail, so this is a second backstop against borderline calls leaking
# into the regenerate path.
SEMANTIC_VERIFIER_FAIL_CONFIDENCE = 0.80
# Subset of issue types we treat as escalation-eligible. Anything else
# (e.g. perspective_slip alone) stays warn even in hard mode — these
# two are the only types tied to a clean "regenerate fixes it" repair.
SEMANTIC_VERIFIER_HARD_ISSUE_TYPES = (
    "pronoun_referent_mismatch",
    "internal_self_contradiction",
    "fact_conflict_with_evidence",
)

# ============================================================
# 8b. LLM judge for verifier semantic classification (Plan B)
#
# When enabled, three of the verifier's "is this query a XYZ kind of
# request?" classifiers consult an external LLM (DeepSeek) AFTER the
# regex check returns False. This catches synonym/paraphrase variants
# that the regex misses (e.g. "who made the music" not matching the
# 'composed' verb list). The regex layer stays in front so:
#   - The cheap path remains cheap (no LLM call when regex matches).
#   - LLM judge is purely additive — it can only flip False -> True,
#     never True -> False. This means enabling the judge cannot make
#     the verifier MORE permissive than today, only less.
#
# Failure mode: if DeepSeek returns garbage / errors / times out, the
# call falls back to the regex result silently. Verifier behaviour
# degrades to current state, never below.
# ============================================================
ENABLE_LLM_VERIFIER_JUDGE = True
LLM_JUDGE_DEBUG = True
# Soft per-turn limit — if a single turn would issue more judge calls
# than this (e.g. verifier loops), we stop calling the judge and fall
# back to regex for the remainder of the turn.
LLM_JUDGE_MAX_CALLS_PER_TURN = 6
# Per-call sampling — judge must be deterministic enough that the
# same query gets the same verdict; very low temperature.
LLM_JUDGE_TEMPERATURE = 0.0
# Soft timeout in seconds. Below this we trust the answer; over it we
# fall back to regex. (Tavily ~30s timeout in tools_runtime is too lax
# for a per-turn auditor.)
LLM_JUDGE_TIMEOUT_SECONDS = 8

# ============================================================
# 8c. LLM judge for PRE PROBE topic-relevance subset (TODO 4 Step 3)
#
# When enabled, MemoryModule.decide consults DeepSeek AFTER the
# topic_keywords.json regex returns ≥1 candidate topics, asking it
# to filter the candidate set down to actually-relevant topics.
# The LLM acts as the second tier of LayeredIntentClassifier and
# only flips relevant -> NOT-relevant (subset filter); it cannot
# add new topics that the keyword layer didn't see.
#
# Shares the per-turn budget pool (LLM_JUDGE_MAX_CALLS_PER_TURN)
# and per-turn cache with the Plan B verifier classifiers, since
# both run in the same JudgeState owned by ChatAgent.
#
# Failure mode: if the judge returns None / errors / times out,
# LLMIntentClassifier falls back to the input candidates unchanged
# (degraded -> same as keyword-only baseline). PRE PROBE behaviour
# can never become MORE restrictive than today as a result of judge
# failure.
# ============================================================
ENABLE_LLM_PRE_PROBE_JUDGE = True

# ============================================================
# 8d. Pronoun resolver (P6, replaces _PRONOUN_FOLLOWUP_PATTERNS)
#
# Replaces the regex-based pronoun-followup detection in
# eva_verifier_logic.build_required_memory_params with an LLM-driven
# resolver. The LLM answers two questions in one call: "is this a
# pronoun follow-up?" AND "what is the antecedent?"; the regex layer
# cannot do the second question and required two heuristic helpers
# (_is_pronoun_followup + _extract_topical_nouns_from_recent_turns)
# wired in series — each compounding the other's errors.
#
# Budget pool is INDEPENDENT from LLM_JUDGE_MAX_CALLS_PER_TURN. The
# resolver tracks its own counter on JudgeState (pronoun_call_count)
# so PRE PROBE / Plan B exhausting the global pool cannot starve the
# resolver — and vice versa.
#
# Failure mode: when the LLM is unavailable, MODE="llm_first" falls
# back to the regex path during P6.0–P6.3 migration. P6.4 deletes
# regex entirely and the failure mode becomes "skip resolution"
# (i.e. degrade to pre-P5 behaviour for that turn — acceptable since
# P5 is an optimisation, not a correctness requirement).
# ============================================================
ENABLE_PRONOUN_RESOLVER = True
# Mode: "llm_first" | "regex_only" | "off".
# - "llm_first" (P6.3+ default, current): LLM main path; regex only
#                triggers as fallback on LLM failure. Promoted from
#                "regex_only" at P6.3 cutover (2026-05-07) after v4
#                shadow verdict — see docs/P6_pronoun_resolver_refactor_v4.md
#                § 六 for the audit trail (50 obs, effective_quality=0.86,
#                llm_rescue_rate=0.76, true_disagree_rate=0.12, all
#                v4 thresholds met).
# - "regex_only": migration-period fallback; preserves pre-P6 behaviour.
#                 Kept until P6.4, then removed.
# - "off": disable resolver completely; build_required_memory_params
#          falls through to its original cleaned-query branch.
PRONOUN_RESOLVER_MODE = "llm_first"
# Verdicts below this confidence are treated as needs_resolution=False.
# Tuned during P6.2 shadow phase based on LLM calibration.
PRONOUN_RESOLVER_MIN_CONFIDENCE = 0.60
# When True, PronounResolver prints LLM raw response for prompt tuning.
# Independent from LLM_JUDGE_DEBUG so resolver-specific noise can be
# enabled without flooding traces from other judges.
PRONOUN_RESOLVER_DEBUG = False
# Stage 1 cheap-gate word cap. Queries longer than this skip the
# resolver entirely (return needs_resolution=False, source="skip").
# Matches the spirit of the existing _is_pronoun_followup heuristic
# (≤6 words) with a small margin for LLM context (e.g. "really? can
# you check it for me"). Bumping this above 10 lets long sentences
# in and wastes LLM budget.
PRONOUN_RESOLVER_MAX_WORDS = 8
# Independent budget pool — does NOT consume LLM_JUDGE_MAX_CALLS_PER_TURN.
# Resolver tracks state.pronoun_call_count separately so its
# availability is not coupled to PRE PROBE / Plan B usage in the same
# turn.
PRONOUN_RESOLVER_MAX_CALLS_PER_TURN = 2
# P6.2 shadow mode. Orthogonal to PRONOUN_RESOLVER_MODE — when both
# this flag and MODE="llm_first" are set, the resolver runs BOTH the
# LLM and regex paths but adopts the REGEX verdict (preserving P6.1
# behaviour). The LLM verdict is logged via a [PRONOUN-SHADOW] trace
# line so operators can compare the two over a sample window before
# committing to P6.3 (regex → LLM main).
#
# Has no effect when MODE != "llm_first" — there's nothing to compare
# against in regex_only / off modes.
#
# Cost: every shadow turn consumes one LLM call from
# PRONOUN_RESOLVER_MAX_CALLS_PER_TURN. Disable when shadow analysis is
# done.
PRONOUN_RESOLVER_SHADOW = False

# ============================================================
# 8z. Remote Advisor (2026-05-13)
#
# The Advisor is a remote-LLM (DeepSeek) preview hop that runs ONCE per turn
# right after PRE PROBE. It looks at the user input + recent history +
# EvaState + recent_notes + retrieved memory, and outputs a short
# natural-language hint that gets injected into Eva's system prompt suffix
# for THIS TURN ONLY (not persisted to history).
#
# Why this exists:
#   Eva (Qwen3.5-VL 9B, post-SFT) is good at language + persona but weak at
#   decomposing compound user input into N tool calls, choosing the right
#   tool, and tracking note record_ids. The advisor offloads that decision
#   work to a stronger remote model. Eva still follows in-context
#   instructions reliably (empirically validated by existing tool
#   recognition), so a few advice lines at the top of the prompt is enough
#   to materially shift behaviour without retraining.
#
# Failure mode:
#   Any error / timeout → silent fallback, no advice injected, Eva runs as
#   before. Verifier + NotesStore safety invariants still catch disasters.
#
# Flags:
#   ENABLE_ADVISOR — master toggle. Set False to bypass entirely (zero
#                    network calls, zero added latency).
#   ADVISOR_MODEL  — DeepSeek model name; "-flash" is the fastest tier
#                    (~700ms typical) and good enough for short advice.
#   ADVISOR_TIMEOUT_SECONDS — hard cap; over this, no advice this turn.
#   ADVISOR_MAX_CALLS_PER_TURN — defends against regenerate loops
#                    re-querying advisor with same context (cached, but
#                    cap is the second backstop).
#   ADVISOR_HISTORY_TURNS — how many recent (user,assistant) pairs to feed
#                    advisor as context.
#   ADVISOR_DEBUG  — print raw response to stdout.
# ============================================================
ENABLE_ADVISOR = os.environ.get("EVA_ADVISOR_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
ADVISOR_MODEL = os.environ.get("EVA_ADVISOR_MODEL", "deepseek-v4-flash")
ADVISOR_TIMEOUT_SECONDS = float(os.environ.get("EVA_ADVISOR_TIMEOUT", "30.0"))
ADVISOR_MAX_CALLS_PER_TURN = int(os.environ.get("EVA_ADVISOR_MAX_CALLS", "2"))
ADVISOR_HISTORY_TURNS = int(os.environ.get("EVA_ADVISOR_HISTORY_TURNS", "3"))
ADVISOR_DEBUG = os.environ.get("EVA_ADVISOR_DEBUG", "false").lower() in {"1", "true", "yes", "on"}

# ============================================================
# TextGenerationTool model（2026-05-16）
#
# 跟 ADVISOR_MODEL 分开配置，因为它们的负载剖面不同：
#   - Advisor：每轮都跑，对延迟敏感 → flash 更合适
#   - TextGen：用户显式让 Eva "写一段诗 / 翻译 / 长代码"才调，对质量敏感
#     → pro 更合适（多花几秒换更好的产出）
#
# 通过 EVA_TEXTGEN_MODEL 环境变量覆盖。默认 deepseek-v4-pro。
# ============================================================
TEXTGEN_MODEL = os.environ.get("EVA_TEXTGEN_MODEL", "deepseek-v4-pro")

# Advisor-first refactor (2026-05-13): fallback behaviour when advisor is
# unavailable. Three modes, picked at runtime via env or attr-mutation:
#   "judges"  — run the pre-refactor local LLM intent classifier flow
#               (EXPLICIT_MEMORY / PRE_PROBE_RELEVANCE / PUBLIC_FACT
#               judges). Safest, but slowest in advisor-down scenarios.
#               Default.
#   "chat"    — treat advisor failure as "chat intent, no memory needed".
#               Fastest, may miss memory-bound queries entirely.
#   "strict"  — skip memory probe outright, let Eva run with whatever
#               context she has. For latency-critical benchmarking.
EVA_ADVISOR_FALLBACK_MODE = os.environ.get(
    "EVA_ADVISOR_FALLBACK_MODE", "judges",
).lower()
if EVA_ADVISOR_FALLBACK_MODE not in {"judges", "chat", "strict"}:
    EVA_ADVISOR_FALLBACK_MODE = "judges"


# ------------------------------------------------------------
# User notes module (RememberThis / ForgetMemory)
# ------------------------------------------------------------
# Eva's runtime-mutable user-notes store at NOTES_DIR/. Parallel to the
# hand-curated lore corpus at Memory/. The model writes here via
# RememberThis and tombstones here via ForgetMemory. Surfaces in
# MemorySearch output under a dedicated `>>> SAVED NOTES <<<` section.
#
# Default ON: production users get this baseline capability without any
# build_agent kwarg. Flip ENABLE_USER_NOTES=False to fully disable
# (skips store init, prompt appendix, verifier reasons, dispatch).
ENABLE_USER_NOTES = True
NOTES_DIR = "Notes"

# Hard per-turn caps on the user-notes mutators (FLOOR; advisor can raise).
# Independent budget pool — does NOT consume LLM_JUDGE_MAX_CALLS_PER_TURN
# (these are local I/O, not LLM calls).
#
# 2026-05-13 Advisor-aware: this value is now a FLOOR, not a ceiling. The
# actual effective cap each turn is `max(this_value, count of this tool in
# advisor.suggested_calls)`. So a compound input ("buy X AND finish Y")
# where advisor lists RememberThis × 2 raises the cap to 2 automatically.
# The floor protects against model thrashing when advisor is unavailable.
#
# Default 1: covers the most common single-fact case without
# burning store quota. Raise to 2 if you want a higher floor even
# without advisor; not needed in normal advisor-on operation.
REMEMBER_TOOL_MAX_CALLS_PER_TURN = 1
FORGET_TOOL_MAX_CALLS_PER_TURN = 1

# ------------------------------------------------------------
# P6.3 cutover audit trail (applied 2026-05-07)
#
# Switched MODE from "regex_only" to "llm_first" based on v4 shadow
# verdict. See docs/P6_pronoun_resolver_refactor_v4.md § 六 for the full
# 50-observation breakdown. Headline numbers:
#
#     effective_quality   = 0.86  (≥ 0.80 v4 threshold)
#     llm_rescue_rate     = 0.76  (≥ 0.20 — regex was missing 76% of
#                                  pronoun follow-ups, validating the
#                                  P6 redesign)
#     true_disagree_rate  = 0.12  (≤ 0.15)
#     llm_availability    = 0.98  (≥ 0.95)
#
# 6/6 true_disagree audit: 0 cases of LLM-clearly-wrong; 3/6 are
# substring-matcher false negatives (LLM correct, just different
# presentation); 3/6 are low-confidence (≤ 0.7) cases that
# PRONOUN_RESOLVER_MIN_CONFIDENCE=0.6 demotes anyway.
#
# Latency (P50=4901ms, P95=15619ms) was measured on Colab→DeepSeek
# (cross-ocean) and is NOT representative of production. Production
# region must run a separate 30-50 obs latency check before final
# rollout (cn_native ≤ 1200ms, cn_proxy ≤ 2000ms).
#
# Rollback path if P6.3 misbehaves in production:
#   a. Set PRONOUN_RESOLVER_SHADOW = True   → re-enter P6.2 audit mode
#                                              (regex decides, LLM logs)
#   b. Set PRONOUN_RESOLVER_MODE  = "regex_only" → revert to P6.1
#   c. Set PRONOUN_RESOLVER_MODE  = "off"   → bypass resolver entirely
#                                              (degrade to pre-P5)
#
# P6.4 (next phase, ≥ 30 days after P6.3 stable + LLM success ≥ 98%):
# apply docs/P6_4_deletion_patch.md to remove the 4 legacy regex symbols
# (_PRONOUN_FOLLOWUP_PATTERNS, _FOLLOWUP_NOUN_STOPWORDS,
#  _is_pronoun_followup, _extract_topical_nouns_from_recent_turns)
# and the "regex_only" mode option.
# ------------------------------------------------------------

# ============================================================
# 9. Memory schema aliases (slot vs domain)
# ============================================================
MEMORY_SLOT_FIELDS = {
    "birthday": ["birthday", "birth date", "date of birth", "born"],
    "full_name": [
        "full name", "real name", "legal name", "complete name",
        "name", "called", "identity",
    ],
    "age": ["age", "years old"],
    "toy": [
        "toy", "favorite toy", "childhood toy", "stuffed toy",
        "plush", "plushie",
    ],
}

# Per-slot applicable subject classes. Used by
# `_detect_requested_slot_fields` to gate slot detection: a slot whose
# subject class is "Person" only fires when the query is plausibly
# about a person (Eva / Rosm / Master / the user). Replaces the
# pre-2026-05-11 whack-a-mole `full_name_blocked` negative-list pattern
# with a positive subject check. See eva_subject_classifier and
# docs/SLOT_SUBJECT_CLASSIFIER_PLAN.md.
#
# All current slots are Person-only:
#   - full_name: human full name (Rosm = Rosmarinus, Eva = Eva Louisa)
#   - birthday:  human birthday (Eva July 7, etc.)
#   - age:       human age
#   - toy:       Eva's favorite toy
#
# Empty set or absent key → slot fires for any query (back-compat).
SLOT_APPLICABLE_SUBJECTS = {
    "full_name": {"Person"},
    "birthday":  {"Person"},
    "age":       {"Person"},
    "toy":       {"Person"},
}

MEMORY_DOMAIN_FIELDS = {
    "favorite": [
        "favorite", "favourite", "likes", "like",
        "preference", "prefer", "love", "enjoy",
    ],
    "interests": [
        "interest", "interests", "hobby", "hobbies",
        "free time", "spare time", "activity", "activities",
        "what do you do when free", "what will you do when you are free",
        "what do you do in your free time", "likes to do",
    ],
    "gaming": [
        "game", "games", "video game", "video games", "gaming",
        "play game", "play games", "playing game", "playing games",
    ],
    "art": ["sketch", "sketching", "doodle", "doodling", "draw", "drawing"],
    "dance": ["ballet", "dance", "dancing"],
    "project": [
        "project", "dataset", "training", "model",
        "code", "backend", "extension",
    ],
}

MEMORY_FIELD_SYNONYMS = {**MEMORY_SLOT_FIELDS, **MEMORY_DOMAIN_FIELDS}

# ============================================================
# 10. Route LM judge (forced-choice classifier on local 9B)
# ============================================================
ENABLE_ROUTE_LM_JUDGE = True
ROUTE_LM_CHOICES = ("MEMORY_LOOKUP", "WEB_SEARCH", "TIME_LOOKUP", "DIRECT")
ROUTE_LM_DEBUG = False

# ============================================================
# 11. Active-memory injection caps & debug
# ============================================================
ACTIVE_MEMORY_MAX_CHARS = 3500
ACTIVE_MEMORY_MULTI_QUERY_MAX = 6
ACTIVE_MEMORY_RECENT_CONTEXT_THRESHOLD = 4
ACTIVE_MEMORY_DEBUG_PRINT_INJECTION = True

# Whether weak ([Judge: RELATED]/below confidence) records are still injected
# in low-confidence mode. False = drop them and answer cautiously.
INJECT_LOW_CONFIDENCE_MEMORY_RECORDS = False

# ============================================================
# 12. Pre-memory probe pattern tables
# These are *current-turn intent hints*. They never decide injection alone;
# they feed into _should_skip_memory_probe_obvious / route judge.
# ============================================================
ACTIVE_MEMORY_NO_TRIGGER_PATTERNS = [
    r"^\s*translate",
    r"^\s*rewrite",
    r"^\s*summarize",
    r"^\s*format",
    r"^\s*explain",
    r"^\s*what is\s+\d+\s*[\+\-\*/]\s*\d+",
    r"^\s*\d+\s*[\+\-\*/]\s*\d+",
    r"capital of france",
    r"^\s*(hi+|hello+|hey+|yo+|hiya|howdy|sup)\b[\s,!.~?]*([a-z]{1,10})?[\s,!.~?]*$",
    r"^\s*(meow+|nya+|woof+|hmm+|huh|ok+|okay|kk|yeah?|yep|nope|no|yes|y|n)[\s,!.~?]*$",
    r"^\s*(thanks|thank you|ty|thx|tysm)\b[\s,!.~?]*$",
    r"^\s*(bye+|goodbye|gn|good\s+night|gm|good\s+morning|cya|see\s+ya)\b[\s,!.~?]*\w*[\s,!.~?]*$",
    r"^\s*(lol|lmao|haha+|hehe+|hahaha+)[\s,!.~?]*$",
    r"^\s*(eva|rosm|master|creator)[\s,!.~?]*$",
]

PERSONAL_QUESTION_MARKERS = [
    r"\b(remember|remembered|recall|recalled|recollect|forget|forgot|forgotten)\b",
    r"\bdo\s+you\s+remember\b",
    r"\bcheck\s+(your\s+)?memory\b",
    r"\b(my|your|our|shared)\s+(memory|memories|history|records?)\b",
    r"\b(before|earlier|previously)\b",
    r"\blast\s+(time|week|month|year|night|day)\b",
    r"\bwhat\s+did\s+(i|we|you)\s+(say|tell|mention|discuss|decide|promise)\b",
    r"\b(did|have)\s+(i|we|you)\s+.*\b(before|previously|last\s+time)\b",
    r"\bwhen\s+(we|i|you)\s+(said|discussed|decided|promised|met|talked)\b",
    r"\b(my|your|rosm'?s|eva'?s|master'?s|creator'?s)\s+(birthday|birth\s*date|date\s+of\s+birth|full\s+name|real\s+name|legal\s+name|complete\s+name|age|favorite|favourite|preference|hobby|hobbies|interest|interests|free\s*time|spare\s*time|game|games|gaming|family|sibling|parent|partner|friend|project|habit)\b",
    r"\bwhat\s+(?:will|do)\s+you\s+do\s+(?:when\s+you\s+are\s+free|in\s+your\s+free\s+time)\b",
    r"\bdo\s+you\s+(?:like|love|enjoy|play)\s+(?:video\s*)?games?\b",
    r"\b(our|shared)\s+(memory|memories|history|conversation|event|plan|promise|decision)\b",
]

PUBLIC_FACT_RELATION_PATTERNS = [
    r"\b(where|which|what)\b.*\b(come\s+from|comes\s+from|from|belong[s]?\s+to|source|origin)\b",
    r"\b(come\s+from|comes\s+from|belong[s]?\s+to|source|origin|official)\b",
    r"\b(does|did|is|are|was|were|has|have)\b.+\b(exist|official|same\s+name|called|titled|belong[s]?\s+to)\b",
    r"\b(released?|published|created|composed|developed|made|announced|written\s+by|produced\s+by)\b",
    r"\b(current|latest|today|now|recent|price|version|schedule|release\s+date)\b",
]

PUBLIC_FACT_ENTITY_HINT_PATTERNS = [
    r"\b(song|track|album|ost|theme|soundtrack|music)\b",
    r"\b(game|anime|movie|film|book|paper|article|dataset)\b",
    r"\b(library|package|framework|model|api|company|product|service)\b",
    r"\b(artist|author|composer|developer|publisher|studio|label)\b",
    r"[\u4e00-\u9fff]{2,}",
]

PUBLIC_FACT_CURRENT_PATTERNS = [
    r"\b(current|latest|today|now|recent|price|version|schedule|release\s+date)\b",
]

MEMORY_VERIFICATION_PATTERNS = [
    r"\bcheck\s+(your\s+)?memory\b",
    r"\b(my|your|our|shared)\s+(memory|memories|records?)\b",
    r"\bremember\b",
    r"\brecall\b",
    r"\b(with|in|from|using|via)\s+(your\s+|my\s+|our\s+|the\s+)?(memory|memories|records?)\b",
    r"\b(memory|memories|records?)\s+(check|search|lookup|recall)\b",
    r"\b(search|seach|lookup|look\s+up|check|verify|confirm)\b.*\b(memory|memories|record|records|lore|database|db)\b",
    r"\b(memory|memories|record|records|lore|database|db)\b.*\b(search|seach|lookup|look\s+up|check|verify|confirm)\b",
    r"\b(search|seach|lookup|look\s+up)\b.*\b(topic|field)\b.*\b(interest|interests|hobby|hobbies|favorite|favourite|gaming)\b",
]

SUBJECTIVE_QUERY_PATTERNS = [
    r"\bwhat\b.*\bdo\s+you\s+(like|love|prefer|want|enjoy|hate|dislike|recommend|think\s+of)\b",
    r"\bwhich\b.*\bdo\s+you\s+(like|love|prefer|want|enjoy|hate|dislike|recommend)\b",
    r"\b(do|does|did|don'?t|doesn'?t|didn'?t)\s+you\s+(like|love|hate|enjoy|prefer|want|miss|trust|believe|fear|liked|loved|enjoyed|preferred|wanted)\b",
    r"\bwhat(?:\s+(?:is|are)|'?s|'re)\s+your\s+(favorite|favourite|preferred|top)\b",
    r"\byour\s+(favorite|favourite|preferred|top)\s+\w+\s+(is|are)\b",
    r"\bare\s+you\s+(happy|sad|tired|bored|excited|angry|jealous|scared|lonely|fine|okay|ok|alright|comfortable|alive|real|conscious|sentient)\b",
    r"\bdo\s+you\s+(love|hate|trust|miss|fear|care\s+about)\s+(me|us|him|her|them)\b",
    r"\bhow\s+do\s+you\s+(feel|think)\s+about\b",
    r"\bwhat\s+do\s+you\s+(think|feel)\s+about\b",
    r"\bcan\s+you\s+(teach|sing|dance|draw|cook|play|read|write|tell|make|recommend|suggest|imagine|pretend|act|role.?play)\b",
    r"\bwill\s+you\s+(teach|sing|dance|draw|cook|play|read|write|tell|make|recommend|suggest|imagine|pretend|act|role.?play)\b",
    r"\b(write|compose|make\s+up|invent|create|generate|draft)\s+(me\s+)?(a\s+|an\s+|the\s+)?(poem|story|song|joke|haiku|limerick|essay|recipe|letter|message|caption)\b",
    r"\btell\s+(me\s+)?(a\s+|an\s+|the\s+)?(joke|story|poem|fact|secret)\b",
    r"\bif\s+you\s+(were|could|had|would)\b",
    r"\bwould\s+you\s+(ever|rather|like|want|prefer)\b",
    r"\bsuppose\s+you\b",
    r"\bimagine\s+(you|if)\b",
    r"\bpretend\s+(to\s+be|you\s+are|you'?re)\b",
    r"\bact\s+(as|like|the\s+role\s+of)\b",
    r"\brole.?play\s+as\b",
    r"\bwhat\s+do\s+you\s+think\b",
    r"\bwhat'?s\s+your\s+opinion\b",
    r"\bin\s+your\s+opinion\b",
    r"\bwhat\s+should\s+i\b",
    r"\bgive\s+me\s+(your\s+)?(advice|opinion|recommendation|thoughts)\b",
    r"\byou\s+(don'?t|do\s+not|didn'?t|did\s+not|never)\s+(answer|say|tell|reply|respond)\b",
]
