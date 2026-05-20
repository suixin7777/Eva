"""
eva_verifier_logic.py — Verifier evidence-checking helpers and core.

Extracted from eva_core.py during the post-Plan-B cleanup. This module
houses the post-generation verifier subsystem — the layer that runs
AFTER phase-2 produces a final answer, decides whether the answer has
sufficient grounding evidence, and triggers controller-injected repair
when it doesn't.

The verifier subsystem has 4 parts. They are migrated to this module
in stages:

  STAGE 1 (this commit) — 13 verifier-only helpers
                          (the predicate / evidence-extraction primitives)
  STAGE 2 — _verify_final_answer (the orchestrator)
  STAGE 3 — _required_action_from_verifier_reasons (reason → repair tool)
            _safe_fallback_for_hard_verifier_failure (terminal failsafe)
  STAGE 4 — _execute_controller_tool (the repair runner)

# Why module-level functions taking `agent`
# -----------------------------------------
# Same pattern used for eva_route_judge.py: functions take `agent` as
# the first parameter rather than being mixin methods. This keeps
# the dependency surface explicit (every read of `agent.X` is visible)
# and makes unit testing trivial (mock agent.history_manager etc.).
# ChatAgent retains thin wrapper methods so any external `self.X(...)`
# call site keeps working unchanged.

# Stage 1 surface — verifier-only helper functions
# -------------------------------------------------
# Predicates (read agent state):
#     current_turn_has_web_evidence(agent)
#     current_turn_has_memorysearch_evidence(agent)
#     answer_mentions_days(answer)
#     answer_toy_animal_words(answer)
#     answer_has_eva_gaming_second_person_mismatch(agent, answer)
# Evidence collectors:
#     exact_memory_evidence_for(agent, subject=None, slot=None)
#     eva_gaming_terms_from_evidence(agent)
# Query interpreters:
#     expected_toy_subject_from_query(agent, text)
#     toy_value_words(value)
#     extract_date_from_text(agent, text)
#     extract_leaked_tool_call(answer)
# Repair-call constructors:
#     build_required_web_query(agent, latest_user_text)
#     build_required_memory_params(agent, latest_user_text)
#
# Helpers under leading-underscore aliases are exposed too so that
# ChatAgent wrapper methods can delegate cleanly:
#     _current_turn_has_web_evidence, _answer_mentions_days, etc.
"""

import re
from datetime import datetime

# Module-level config import for cheap getattr() flag checks at runtime
# (e.g. ENABLE_LEGACY_SEMANTIC_REGEX). For values that need to honor
# config hot-reload, keep using the per-function lazy `from eva_config
# import ...` pattern below.
import eva_config

# Leaf helpers used inside this module — same dependency direction as
# eva_route_judge → eva_memory_legacy / eva_render. No back-edges.
from eva_render import clean_user_text
from eva_memory_legacy import (
    _normalize_match_text,
    _canonical_known_entity_name,
    _infer_memory_target_from_text,
    _build_display_keywords_from_query,
    run_memory_search,
)
from eva_tools_runtime import (
    sanitize_tool_code,
    run_websearch,
    call_remote_vision,
    call_deepseek_expert,
)


__all__ = [
    # Stage 1 — verifier-only helpers (preferred names)
    "current_turn_has_web_evidence",
    "current_turn_has_memorysearch_evidence",
    "answer_mentions_days",
    "answer_toy_animal_words",
    "answer_has_eva_gaming_second_person_mismatch",
    "answer_violates_no_elaboration_rule",
    "exact_memory_evidence_for",
    "eva_gaming_terms_from_evidence",
    "expected_toy_subject_from_query",
    "toy_value_words",
    "extract_date_from_text",
    "extract_leaked_tool_call",
    "build_required_web_query",
    "build_required_memory_params",
    # Stage 2 — orchestrator
    "verify_final_answer",
    # Stage 3 — repair-action dispatch + terminal failsafe
    "required_action_from_verifier_reasons",
    "safe_fallback_for_hard_verifier_failure",
    # Stage 4 — repair runner
    "execute_controller_tool",
    # Underscore aliases (for ChatAgent wrapper-method delegation)
    "_current_turn_has_web_evidence",
    "_current_turn_has_memorysearch_evidence",
    # P4: test-memory tool-output evidence helpers
    "current_turn_has_remember_evidence",
    "current_turn_has_forget_evidence",
    "find_recent_note_id",
    "extract_remember_params_from_user_text",
    "_current_turn_has_remember_evidence",
    "_current_turn_has_forget_evidence",
    "_answer_mentions_days",
    "_answer_toy_animal_words",
    "_answer_has_eva_gaming_second_person_mismatch",
    "_exact_memory_evidence_for",
    "_eva_gaming_terms_from_evidence",
    "_expected_toy_subject_from_query",
    "_toy_value_words",
    "_extract_date_from_text",
    "_extract_leaked_tool_call",
    "_build_required_web_query",
    "_build_required_memory_params",
    "_verify_final_answer",
    "_required_action_from_verifier_reasons",
    "_safe_fallback_for_hard_verifier_failure",
    "_execute_controller_tool",
    # Stage 0 — REASON_POLICY single source of truth (TODO 11-arch)
    "REASON_POLICY",
    "DEFAULT_POLICY",
    "get_reason_policy",
    "get_dominant_reason_for_dispatch",
]


# ============================================================
# REASON_POLICY — single source of truth for verifier reasons
# (TODO 11-arch, 2026-05-07)
# ============================================================
# This table replaces the previous fragmented configuration:
#   - HARD_VERIFIER_REASONS set in eva_config.py (severity)
#   - per-reason if-cascade in required_action_from_verifier_reasons
#     (which reasons trigger tool inject)
#   - per-reason if-cascade in safe_fallback_for_hard_verifier_failure
#     (which reasons get which canned message)
#
# The previous design's failure mode: detection added without
# follow-through. Audit on 2026-05-07 found 5 of 10 reasons were
# silently passing through (eva_self_birthday_pronoun_mismatch,
# date_math_target_date_mismatch, unsupported_exact_toy_claim,
# toy_value_conflicts_with_exact_memory, textgen_perspective_mismatch,
# date_math_days_not_supported_by_calculation_evidence) — verifier
# detected the bug but no fix path existed, so broken answers shipped.
#
# REASON_POLICY makes registration mandatory and explicit. Each reason
# entry has three keys:
#   severity: "hard" — failure blocks the broken answer (default)
#             "soft" — failure flags but lets answer through (rare,
#                      use only when downstream layer handles it)
#   fix:     "inject_tool"     — controller runs a repair tool to
#                                 fetch missing data
#            "regenerate"       — phase-2 is re-sampled with collapse
#                                 pressure; evidence was correct but
#                                 generation went wrong (pronoun/date
#                                 mismatch, perspective error)
#            "canned_fallback" — emit canned message; no automated
#                                 repair available
#   canned:  user-facing message used as the terminal failsafe (when
#            inject_tool fails, regenerate fails, or fix is
#            "canned_fallback" outright)
#
# DEFAULT_POLICY catches any reason not in REASON_POLICY. Default is
# fail-safe: hard severity + canned_fallback fix. Adding new
# detection without registering policy gets generic-canned hard fail
# — never silent passthrough.
# ============================================================

REASON_POLICY = {
    # ============================================================
    # 2026-05-14 PLAN-A FINAL CUTOVER — regex verifier 全退役
    # ============================================================
    # 经历了从 hard→inject_tool（2026-05-08 之前）→ soft per-reason 降级
    # （2026-05-13 Advisor cutover）→ 最终 advisor-first 架构的反复 trace
    # 观察，我们确认：
    #
    #   1. Advisor 把"该不该调工具 / 调哪个工具 / 多个对象怎么拆"前置到生成前
    #   2. SemanticVerifier (LLM judge) 能识别真矛盾、跨实体归因错、真幻觉
    #   3. regex-based verifier 检查（日期 / no-elab / 工具监督）的 false
    #      positive 多于真错——它们替 advisor / LLM judge 干同样的活，但更粗糙
    #   4. hard fail + regenerate + canned 的杀伤路径把对答案换成 canned 错误
    #      回复的情况屡见不鲜（Turn 8 的 195 days / Turn 18 的 list-all 等）
    #
    # 终极策略：
    #   - 唯一 hard regex check：tool_call_leaked_in_answer（格式安全，纯文本
    #     扫描，0% false positive，且 advisor 完全救不了"已生成的字面 syntax"）
    #   - 唯一 hard LLM check：semantic_verifier_fail:*（LLM judge，置信≥0.8，
    #     能理解 compound 答案、cross-entity 归因、真矛盾）
    #   - 其余所有 reason：severity="soft" + fix="canned_fallback"。只做
    #     telemetry log（VERIFIER_DEBUG 时可见），不触发 regen，不替换答案。
    #
    # 死代码：required_action_from_verifier_reasons 里 inject_tool 分支、
    # safe_fallback_for_hard_verifier_failure 中 self-validate_date 路径、
    # _rewrite_assistant_for_tool_repair 等全部不再被调用。留作回滚保险，
    # 稳定运行 2 周后可清理。
    # ============================================================

    # ============================================================
    # HARD — 唯一保留的 regex check + LLM judge
    # ============================================================
    "tool_call_leaked_in_answer": {
        # 保留 inject_tool：模型把 tool syntax 当成答案输出了，把那段语法
        # 实际跑一遍是确定性、安全的修复；advisor 救不了已经生成的字面串。
        "severity": "hard",
        "fix": "inject_tool",
        "canned": "I tried to call a tool in the wrong place, Master. Let me execute it properly instead of showing you the command.",
    },
    # ---- A1-A4: 工具调用监督，由 Advisor 接管 → soft ----
    "missing_web_evidence_for_external_or_current_request": {
        "severity": "soft",
        "fix": "canned_fallback",
        "canned": "I need to verify that with WebSearch first, Master. I won’t pretend stale guesses are fresh news.",
    },
    "missing_memorysearch_for_explicit_memory_check": {
        "severity": "soft",
        "fix": "canned_fallback",
        "canned": "I need to check memory first, Master. I won’t pretend I verified it when I didn’t.",
    },
    "explicit_forget_request_not_handled": {
        "severity": "soft",
        "fix": "canned_fallback",
        "canned": "I should have actually deleted that, Master, not just nodded along. Let me run ForgetMemory properly.",
    },
    "explicit_remember_request_not_handled": {
        "severity": "soft",
        "fix": "canned_fallback",
        "canned": "I should have stored that properly, Master. Let me try again.",
    },
    "missing_date_calculation_evidence": {
        # 2026-05-14 Plan-A: 降级 soft。Advisor 已经让 Eva 在 days-until
        # 类查询时调 GetCurrentTime；Eva 漏调时 semantic verifier 能识别
        # "日子数无证据"作为 self-contradiction。regex check 不再独立 kill。
        "severity": "soft",
        "fix": "canned_fallback",
        "canned": "I need the current date before giving a day count, Master. No fake arithmetic from me.",
    },
    "unsupported_exact_toy_claim": {
        # B1: advisor 把相关记忆带进 prompt 后，凭空声称玩具的概率大降。
        "severity": "soft",
        "fix": "canned_fallback",
        "canned": "I need to check memory first, Master. Don't make me guess what's stored.",
    },

    # ---- B6: pronoun/perspective，由 Advisor 在 prompt 里写明 → soft ----
    "eva_self_birthday_pronoun_mismatch": {
        "severity": "soft",
        "fix": "canned_fallback",
        "canned": "Wait, Master — *my* birthday is July 7th, not yours. I got my pronouns tangled. Let me get it straight.",
    },
    "date_math_target_date_mismatch": {
        # 2026-05-14: 降级为 soft。该 check 抓 answer 里 ALL dates 跟 calc
        # evidence 的 single target 比较，无法区分 compound 答案合法地提到
        # 多个日期（例："my birthday is July 7th, so you have 195 days
        # until November 25th"——195 关联 Nov 25 是对的，提 July 7 是回答
        # 第一个问题的事实）。Semantic verifier (LLM judge) 能区分真矛盾
        # 和合法多日期答案，所以这层粗糙的 regex 不再触发 regen/canned。
        # 反向案例（真错误：5 days for Eva 但说 Nov 25）由
        # date_math_days_not_supported_by_calculation_evidence 接住。
        "severity": "soft",
        "fix": "canned_fallback",
        "canned": "I tangled up the dates, Master. Tell me one at a time and I'll get it right.",
    },
    "date_math_days_not_supported_by_calculation_evidence": {
        # 2026-05-14 Plan-A: 降级 soft。Advisor + multi-binding GetCurrentTime
        # 让算术错变罕见。残留真错由 semantic_verifier_fail:fact_conflict_with_evidence
        # 接住——LLM judge 比 regex 更准地识别 "数字 N 与 calc binding 不符"。
        "severity": "soft",
        "fix": "canned_fallback",
        "canned": "I tangled up the day count, Master. Let me re-anchor and try again.",
    },
    "toy_value_conflicts_with_exact_memory": {
        # B2: advisor 已经把准确的玩具记忆带入 prompt，直接矛盾的概率大降。
        "severity": "soft",
        "fix": "canned_fallback",
        "canned": "I crossed wires on what's stored, Master. Let me re-check before answering.",
    },
    "textgen_perspective_mismatch": {
        # Advisor 可在建议里写明 perspective（你是 Eva，请用第一人称）。
        "severity": "soft",
        "fix": "canned_fallback",
        "canned": "I got the perspective tangled in that, Master. Let me re-do it properly.",
    },
    "unsupported_specifics_under_no_elaboration_rule": {
        # 2026-05-14 Plan-A: 降级 soft。Advisor 主动提供 memory 上下文 +
        # 提示 Eva 只引用记忆事实；advisor-aware bypass 也已在 verifier
        # 函数内置（advisor 背书时 skip）。真幻觉由 semantic_verifier_fail:
        # fact_conflict_with_evidence / pronoun_referent_mismatch 接住。
        "severity": "soft",
        "fix": "canned_fallback",
        "canned": "I started inventing specifics that aren't in my records, Master. Let me try again — only what I actually remember.",
    },

    # ---- P3: LLM-judge semantic verifier escalations ----
    # These reasons are produced by SemanticVerifier when running in
    # 'hard' mode. They share the regenerate fix path (evidence is
    # available, generation went wrong) and are guarded by
    # SEMANTIC_VERIFIER_FAIL_CONFIDENCE + RegenerateGuard quota so a
    # single bad LLM judgement cannot loop the conversation.
    "semantic_verifier_fail:pronoun_referent_mismatch": {
        "severity": "hard",
        "fix": "regenerate",
        "canned": "Hmph — I tangled up who's who in that, Master. Let me say it again straight.",
    },
    "semantic_verifier_fail:internal_self_contradiction": {
        "severity": "hard",
        "fix": "regenerate",
        "canned": "I just contradicted myself, Master. Give me one more shot at it.",
    },
    "semantic_verifier_fail:fact_conflict_with_evidence": {
        "severity": "hard",
        "fix": "regenerate",
        "canned": "I crossed wires with what I just looked up, Master. Let me re-anchor and try again.",
    },

    # ---- Plan D (2026-05-15): orphan markdown at answer tail ----
    # 实测 sampling collapse 模式：答案以裸 ** 结尾（模型打开 bold 后被
    # rep_penalty 卡住，无法生成 emphasis 内容就采样到 <|end_react|>）。
    # Plan B 调采样 + Plan F 改 prompt 已经压低频率，本 reason 作为最后
    # 一道保险：发生时走 regenerate（强制 collapse_pressure，给一次新
    # 采样的机会）。RegenerateGuard 限 1 次重试，避免无限循环。
    "orphan_markdown_in_answer": {
        "severity": "hard",
        "fix": "regenerate",
        "canned": "Tch — that came out cut off, Master. Let me say it again properly.",
    },
}


DEFAULT_POLICY = {
    "severity": "hard",
    "fix": "canned_fallback",
    "canned": "I need to verify that before answering, Master.",
}


# P0 (TODO refactor v2): semantic reasons whose severity is gated by
# ENABLE_SEMANTIC_HARD_FAIL / SEMANTIC_REASON_HARD_OVERRIDES at read
# time. These judgments are local-regex over the answer alone and
# produce frequent false positives — demoting them to soft pass while
# the LLM-based SemanticVerifier is built out.
_SEMANTIC_REASONS = {
    "eva_self_birthday_pronoun_mismatch",
    "textgen_perspective_mismatch",
    "toy_value_conflicts_with_exact_memory",
}


def _semantic_severity(reason):
    """Resolve runtime severity for a semantic reason via config flags."""
    from eva_config import (
        ENABLE_SEMANTIC_HARD_FAIL,
        SEMANTIC_REASON_HARD_OVERRIDES,
    )
    override = SEMANTIC_REASON_HARD_OVERRIDES.get(reason)
    if override is True:
        return "hard"
    if override is False:
        return "soft"
    return "hard" if ENABLE_SEMANTIC_HARD_FAIL else "soft"


def get_reason_policy(reason):
    """Return the policy entry for a verifier reason. Falls back to
    DEFAULT_POLICY (hard + canned_fallback) for any unregistered reason
    — this is the fail-safe boundary. New detection added without a
    policy entry produces a generic canned hard-fail, never a silent
    passthrough.

    P0: for entries in _SEMANTIC_REASONS the severity is overridden
    at read time from runtime config (ENABLE_SEMANTIC_HARD_FAIL +
    SEMANTIC_REASON_HARD_OVERRIDES) so flag flips take effect without
    code edits or process restart."""
    entry = REASON_POLICY.get(reason, DEFAULT_POLICY)
    if reason in _SEMANTIC_REASONS:
        entry = dict(entry)
        entry["severity"] = _semantic_severity(reason)
    return entry


# Priority order for picking the "dominant" reason when multiple flag
# at once. inject_tool comes first (data missing — fetch it before
# anything else can resolve), regenerate second (generation issue
# given evidence), canned third (terminal). Within the same fix-class,
# the first reason in the verifier's reason list wins (preserves the
# detection order).
_FIX_CLASS_PRIORITY = {"inject_tool": 0, "regenerate": 1, "canned_fallback": 2}


def get_dominant_reason_for_dispatch(reasons):
    """Pick the reason that drives dispatch. inject_tool > regenerate >
    canned_fallback. If multiple reasons share the highest priority,
    the first one in the input list wins (preserves verifier detection
    order, which is meaningful — earlier checks are typically more
    fundamental). Returns None for an empty list."""
    if not reasons:
        return None
    best = None
    best_priority = 99
    for r in reasons:
        priority = _FIX_CLASS_PRIORITY.get(get_reason_policy(r)["fix"], 3)
        if priority < best_priority:
            best = r
            best_priority = priority
    return best


# ============================================================
# Evidence presence predicates
# ============================================================
def current_turn_has_web_evidence(agent):
    """True if the current turn has at least one web-source evidence."""
    return any(ev.source == "web" for ev in getattr(agent, "turn_evidence", []) or [])


def current_turn_has_memorysearch_evidence(agent):
    """True if the current turn has memory-source evidence in the ledger.

    R-4 (2026-05-13)：旧实现 grep `turn.assistant_steps` 抓 "[MEMORY MODULE DATA
    for"，因为当时 `turn_evidence` 只在抽到 slot value 时才写——RELATED-only
    的 MemorySearch / topic-only 的 PRE PROBE 都会被 ledger miss。R-4 在
    `_record_memory_evidence_from_observation` / `_record_active_memory_evidence`
    都加了兜底写入，确保只要这俩路径跑过、ledger 就有一条对应 evidence。
    `source="topic_dict"` 也算 memory-grounded 的一种 tier。
    """
    ledger = getattr(agent, "turn_evidence", None)
    if ledger is None:
        return False
    if hasattr(ledger, "has_source"):
        return ledger.has_source("memory", "topic_dict")
    return any(getattr(ev, "source", None) in ("memory", "topic_dict")
               for ev in ledger)


def current_turn_has_remember_evidence(agent):
    """True if the current turn has a notes_write evidence in the ledger.

    R-4：旧实现 grep tool step 内容里的 "[REMEMBERED]"。RememberThis 成功
    时已经在 eva_core 写了 source="notes_write" 的 evidence，这里直接读。
    """
    ledger = getattr(agent, "turn_evidence", None)
    if ledger is None:
        return False
    if hasattr(ledger, "has_source"):
        return ledger.has_source("notes_write")
    return any(getattr(ev, "source", None) == "notes_write" for ev in ledger)


def current_turn_has_forget_evidence(agent):
    """True if the current turn has a notes_delete evidence in the ledger.

    R-4：旧实现 grep "[FORGOTTEN]"。改为读 source="notes_delete"。
    """
    ledger = getattr(agent, "turn_evidence", None)
    if ledger is None:
        return False
    if hasattr(ledger, "has_source"):
        return ledger.has_source("notes_delete")
    return any(getattr(ev, "source", None) == "notes_delete" for ev in ledger)


# 8-char hex id used by NotesStore (uuid4().hex[:8]) and rendered
# by `_format_memory_records_block` as "[Note #abc12345]".
_NOTE_TAG_RE = re.compile(r"\[Note #([0-9a-f]{8})\]")


# R-4 (2026-05-13)：原 P1-4 补丁 `_is_setup_remember_phrasing` /
# `_SETUP_REMEMBER_RE` / `_SETUP_REMEMBER_ZH_RE` 已撤。语义改由 Evidence
# Ledger 自动承担：当用户说 "help me remember something" 这种没有具体事实的
# 引子句、verifier 检查"是否需要 MemorySearch"时，ledger.covers() 会查
# (subject, slot, topic) 三元组——这种引子句不带 slot 也不带 topic，
# verifier 自然不会触发 missing_memorysearch_for_explicit_memory_check。
# 维护历史见 docs/TODO_2026-05-13_root_fixes.md R-4 § "撤补丁清单"。


# P5.2: heuristic param extractor for verifier-injected RememberThis.
# Used as a deterministic fallback when the model failed to call
# RememberThis itself. Quality is intentionally lower than what a
# DeepSeek-driven extractor would produce — but it costs zero LLM
# calls, runs in <1ms, and produces well-formed (if approximate)
# entity/topic/keywords from the user's own text. The user can always
# rephrase if the auto-extraction misses.
_REMEMBER_PREAMBLE_RE = re.compile(
    r"^(eva,?\s+)?"
    r"(please\s+)?"
    r"(remember|note|keep\s+in\s+mind|don'?t\s+forget|jot\s+down|save|make\s+a\s+note(?:\s+of)?)"
    r"\s+"
    r"(this|that|the\s+following|about|of|down|it\s+down)?"
    r"[:,;.\s]*",
    re.IGNORECASE,
)
_REMEMBER_PREAMBLE_ZH_RE = re.compile(
    r"^(请\s*)?(记住|记一下|记录一下|帮我记[住下录]|别忘了)"
    r"\s*[:：,，;；.]?\s*"
    r"(?:这件事是|这个是|这是|那是|关于|的是)?"
    r"\s*"
)
_TOPIC_GUESS_PATTERNS = [
    # Bilingual: each pattern checks English nouns + a few Chinese
    # equivalents. Order matters — first match wins.
    ("Pet",       r"\b(cat|kitten|dog|puppy|pet|hamster|bird|fish|rabbit|tabby)\b|(猫|狗|宠物|兔子|仓鼠|鸟)"),
    ("Food",      r"\b(eat|ate|cook|bake|food|meal|breakfast|lunch|dinner|recipe|dish|cuisine|restaurant)\b|(吃|做菜|烤|餐|早饭|午饭|晚饭|食谱)"),
    ("Music",     r"\b(song|music|artist|album|playlist|listen(?:ing)?|band|concert|melody)\b|(歌|音乐|歌手|专辑|乐队|演唱会)"),
    ("Travel",    r"\b(trip|travel|vacation|visit(?:ed|ing)?|fly|flight|hotel|tour|country|city)\b|(旅行|度假|参观|去过|国家|城市|酒店)"),
    ("Work",      r"\b(work|job|office|meeting|colleague|boss|deadline|project|client)\b|(工作|公司|开会|同事|老板|项目)"),
    ("Birthday",  r"\b(birthday|birthdate|date\s+of\s+birth)\b|(生日|出生日)"),
    ("Hobbies",   r"\b(hobby|hobbies|sport|game|paint(?:ing)?|draw(?:ing)?|read(?:ing)?|book|writing)\b|(爱好|运动|游戏|画画|读书|写作)"),
    ("Likes",     r"\b(like|love|prefer|favou?rite|enjoy|adore|fond\s+of)\b|(喜欢|爱|喜爱|偏爱)"),
    ("Family",    r"\b(family|sister|brother|mom|dad|mother|father|parent|cousin|uncle|aunt)\b|(家人|姐|妹|哥|弟|妈|爸|父|母|表|堂)"),
    ("Daily Life", r""),  # default catch-all
]
_REMEMBER_STOPWORDS = {
    # Articles + short connector words: drop these so meaningful nouns
    # fit into the 6-keyword cap.
    "a","an","the","this","that","these","those","just","with","from",
    "about","much","many","very","some","any","really","into","onto",
    "upon","over","under","than","then","and","but","also","still",
    "only","even","quite","such","both","each","more","most","less",
    "least","first","last","next","other","another",
    "have","has","had","been","being","were","was","are","is","am","be",
    "will","would","could","should","might","must","may","can",
    "for","its","yours","mine","ours","theirs",
    "of","to","in","on","at","by","as","or","if","so","do","did","does",
}
_REMEMBER_PRONOUNS = {
    "i","me","my","mine","you","your","yours",
    "we","our","ours","us","he","him","his","she","her","hers",
    "they","them","their","it","its",
}


def extract_remember_params_from_user_text(text, default_entity="Rosm"):
    """Heuristic extractor for verifier-injected RememberThis params.

    Returns a dict with keys content / entity / topic / keywords, or
    None if the user text is unusable. Strategy:
      1. Strip the "remember this:" / "don't forget that" preamble
         (English + Chinese).
      2. Detect first-person pronouns to set entity = Rosm; otherwise
         entity = Shared. Eva self-references are rare in this tool's
         calling convention, so we don't try to detect them.
      3. Normalize first-person → "Master" when entity = Rosm so the
         stored content reads naturally to a future retrieval.
      4. Topic guess via canonical regex patterns; default Daily Life.
      5. Keywords: top-6 content words, dedup, drop stopwords + pronouns.
    """
    if not text or not isinstance(text, str):
        return None
    s = text.strip()

    # Strip preamble (English then Chinese).
    s = _REMEMBER_PREAMBLE_RE.sub("", s, count=1).strip()
    s = _REMEMBER_PREAMBLE_ZH_RE.sub("", s, count=1).strip()
    if not s:
        return None

    # Entity detection.
    has_first_person_en = bool(re.search(
        r"\b(i|i'm|i've|i'll|i'd|my|me|we|our|us)\b", s, re.IGNORECASE,
    ))
    has_first_person_zh = bool(re.search(r"(我|我的|我们|咱们)", s))
    entity = default_entity if (has_first_person_en or has_first_person_zh) else "Shared"

    # First-person → "Master" normalization (English only; Chinese
    # left as-is — "我" → "主人" is more disruptive than helpful).
    content = s
    if entity == "Rosm" and has_first_person_en:
        content = re.sub(r"\bI'm\b", "Master is", content)
        content = re.sub(r"\bI've\b", "Master has", content)
        content = re.sub(r"\bI'll\b", "Master will", content)
        content = re.sub(r"\bI'd\b", "Master would", content)
        content = re.sub(r"\bI\b", "Master", content)
        content = re.sub(r"\bmy\b", "Master's", content, flags=re.IGNORECASE)
        content = re.sub(r"\bme\b", "Master", content, flags=re.IGNORECASE)

    # Topic guess.
    topic = "Daily Life"
    for cand_topic, pat in _TOPIC_GUESS_PATTERNS:
        if not pat:
            continue
        if re.search(pat, s, re.IGNORECASE):
            topic = cand_topic
            break

    # Keywords: dedup top-6 content words.
    words = re.findall(r"\b[a-zA-Z][a-zA-Z']{1,}\b", s)
    keywords = []
    seen = set()
    for w in words:
        wl = w.lower()
        if wl in _REMEMBER_STOPWORDS or wl in _REMEMBER_PRONOUNS:
            continue
        if wl in seen:
            continue
        seen.add(wl)
        keywords.append(wl)
        if len(keywords) >= 6:
            break

    return {
        "content": content.strip(),
        "entity": entity,
        "topic": topic,
        "keywords": ", ".join(keywords),
    }


def find_recent_note_id(agent, latest_user_text=None, max_turns=4):
    """Resolve a target note_id for verifier-injected ForgetMemory.

    Two-stage lookup:

    1. **History scan** (fast path): walk the current turn + last
       `max_turns` history turns' tool_outputs and grep for a
       `[Note #abc12345]` tag. This is the tag the formatter writes
       around the saved-notes section every time MemorySearch surfaces
       one. If the model has been shown an id this session, take it.

    2. **Live-store fallback**: if the history scan came up empty AND
       `latest_user_text` is given AND the agent's `memory_state`
       carries a `notes_store`, query the store directly with the user's
       forget request and take the top-cosine live note's id.
       Covers the case where the model answered the prior turn from
       in-context recall (no MemorySearch executed → no tag rendered →
       history scan misses), but the store actually still has a
       matching live note.

    Returns the 8-char id or None. None means "no plausible target —
    let the dispatcher fall back to canned".
    """
    hm = getattr(agent, "history_manager", None)
    if hm is not None:
        turns = []
        if hm.current_turn is not None:
            turns.append(hm.current_turn)
        history = getattr(hm, "history", []) or []
        if max_turns > 0:
            turns.extend(reversed(history[-max_turns:]))
        for turn in turns:
            for step in reversed(getattr(turn, "assistant_steps", []) or []):
                if step.get("role") != "tool":
                    continue
                content = step.get("content", "") or ""
                m = _NOTE_TAG_RE.search(content)
                if m:
                    return m.group(1)

    # Stage 2: live-store fallback. Only attempt when both inputs are
    # available — refuse to silently guess otherwise.
    if not latest_user_text:
        return None
    memory_state = getattr(agent, "memory_state", None) or {}
    notes_store = memory_state.get("notes_store")
    if notes_store is None:
        return None
    try:
        results = notes_store.search(latest_user_text, top_k=1)
    except Exception:
        return None
    if not results:
        return None
    return results[0].get("note_id")


# ============================================================
# Answer-side textual probes
# ============================================================
def answer_mentions_days(answer):
    """True if the answer contains an explicit 'N days' phrase."""
    return bool(re.search(r"\b\d+\s+day(s)?\b", answer or "", re.I))


def _answer_has_orphan_bold_tail(answer) -> bool:
    """检测答案末尾是否留下未配对的 ** markdown 标记。

    实测案例（2026-05-15）：sampling collapse 会让模型产生：
        'Tch! ... I already told you: **'
        'Hmph! Didn't I just say it? Pick the **'
    这种打开了 bold 但没写内容也没闭合就终止生成的答案。Discord 把裸 **
    显示成字面字符，体验很差。

    保守策略——只判断"末尾"而非全文：
      - 'compute 5**3 is 125'   → 中间 ** 来自数学，不该被当 bug → False
      - 'use **/*.py for glob'  → 中间 ** 来自 glob 模式，不该报错 → False
      - 'Hello **world'         → 未闭合但不在末尾，可能是 streaming 中
                                   间状态/特殊用法，不主动干预 → False
      - 'told you: **'          → 末尾裸 ** = 真正的 sampling collapse → True

    实现：把末尾的标点/空白剥掉后，要求字符串以 ** 结尾，且 ** 总数为奇数
    （证明末尾这个不是闭合的另一半）。
    """
    if not isinstance(answer, str):
        return False
    stripped = answer.rstrip(":,;-—.!?…~ \t\r\n")
    if not stripped.endswith("**"):
        return False
    return stripped.count("**") % 2 == 1


def answer_toy_animal_words(answer):
    """Extract the set of stuffed-animal words present in the answer.

    Used by the unsupported_exact_toy_claim verifier reason to detect
    when the answer claims a specific toy animal (e.g. 'bear', 'bunny')
    without backing memory evidence.
    """
    q = _normalize_match_text(answer or "")
    animals = {
        "cat", "dog", "bunny", "rabbit", "bear", "fox", "wolf", "bird", "hamster",
        "mouse", "pony", "horse", "whale", "dolphin", "penguin", "turtle", "duck",
        "lion", "tiger", "dragon", "unicorn", "cow", "sheep", "goat", "pig",
    }
    return {w for w in re.findall(r"\b[a-z]{3,}\b", q) if w in animals}


def answer_has_eva_gaming_second_person_mismatch(agent, answer):
    """Detect 'you play X' when evidence says Eva plays X.

    This check intentionally fires only when current-turn evidence says the
    subject is Eva's gaming preferences. It avoids penalizing normal user
    addressing such as "if you're into that" unless it is tied to a known
    Eva game and a play/like/enjoy verb.
    """
    terms = eva_gaming_terms_from_evidence(agent)
    if not terms:
        return False
    ans = _normalize_match_text(answer or "")
    if not ans:
        return False
    game_alt = "|".join(re.escape(t) for t in sorted(terms, key=len, reverse=True))
    # Examples caught: "you actually play apex", "you play apex and battlefield",
    # "so you enjoy battlefield".
    return bool(re.search(
        rf"\byou\b(?:\s+\w+){{0,5}}\s+(?:actually\s+)?(?:play|played|like|liked|enjoy|enjoyed)\b[^.!?]{{0,100}}\b(?:{game_alt})\b",
        ans,
    ))


# ============================================================
# [NO ELABORATION RULE] anti-fabrication check (2026-05-08)
# ============================================================
# Backstop for the prompt-side rule emitted by eva_memory_legacy when a
# memory probe returns only low-confidence/RELATED records (top1_score
# below HIGH_CONFIDENCE_BAR with no slot value). The prompt instructs
# the model to hedge ("I don't have a specific memory about that")
# instead of inventing places, actions, mood, weather, etc. Soft prompt
# constraints leak — the lasagna/smoke-alarm hallucination logged in
# 2026-05-08 traces is exactly that leak — so this regex layer hard-
# verifies the constraint after generation.
#
# Mechanic: when any current-turn memory evidence carries the rule
# marker, count content tokens in the answer that are absent from both
# the records AND the user's question. If >= ANTI_HALLUCINATION_MIN_
# UNSUPPORTED_TOKENS (default 3) such tokens appear AND the answer
# does not include a hedge phrase, the answer is flagged. Threshold of
# 3 keeps single odd words from triggering — real fabrications pile up
# multiple invented details.
# ============================================================
_NO_ELABORATION_RULE_MARKER = "[NO ELABORATION RULE]"

# ============================================================
# R-3.2 (2026-05-13): subjective-question gating for no_elaboration_rule
# ============================================================
# 老 rule 一刀切：只要 turn_evidence 里有 NO_ELABORATION marker，就检查 answer
# 的 unsupported content tokens 数量。结果在主观偏好问句下大量误报：
#   user: "do you want a new one?"
#   model: "Tch, why would I ever need a replacement? This one's perfect as-is..."
#   verifier flag: replacement/perfect/thinking/fun 都不在 records → fail
#
# 设计权衡（2026-05-13 实跑 Turn 6/7 复盘 + 2026-05-08 lasagna 测试对账）：
#   - 想做"R-3.2 严格版：仅当 answer 含 factual 信号才 fire" → 但 lasagna /
#     smoke alarm / jazz / thunderstorm 这种叙事杜撰用的全是普通名词，
#     regex 抓不到 → 会漏判，破坏 2026-05-08 lasagna 测试。
#   - 改走 R-3.1 风格 (subjective-skip)：识别"do you want / would you /
#     are you ..." 偏好问句，对这类回答跳过 rule；其余照旧 token-count。
#     这样 Turn 6/7 修好，lasagna 测试 ("anything fun in the kitchen?"
#     非主观) 仍能 catch。
# ============================================================
_R32_SUBJECTIVE_QUESTION_RE = re.compile(
    r"\b(?:"
    # do/would/could you + preference / opinion / state verb
    r"(?:do|would|could|did)\s+you\s+"
    r"(?:want|like|prefer|wish|love|hate|enjoy|need|mind|care|miss|fancy|rather|"
    r"think|feel|know|believe|consider|approve|recommend|suggest)"
    # what do you think/feel/prefer
    r"|what\s+(?:do|would|did)\s+you\s+(?:think|feel|prefer|want|like|recommend|say)"
    # how do you feel about
    r"|how\s+(?:do|would)\s+you\s+(?:feel|like|think)"
    # are you (state)
    r"|are\s+you\s+(?:ok|okay|fine|sure|ready|happy|sad|excited|tired|hungry|"
    r"interested|bored|busy|free|here|there|done|finished|going)"
    # is X better/worse/...
    r"|is\s+(?:that|this|it|x)\s+(?:better|worse|nicer|cooler|cuter|funnier|harder|easier)"
    # 通用 yes/no preference markers
    r"|would\s+you\s+rather"
    r")\b",
    re.IGNORECASE,
)
_R32_SUBJECTIVE_QUESTION_ZH_RE = re.compile(
    # 中文常见偏好 / 状态 / 意见问句尾标
    r"(想要吗|喜欢吗|喜不喜欢|想不想|愿意吗|介意吗|要不要|"
    r"怎么样|觉得怎样|觉得.*吗|你觉得|你认为|"
    r"还好吗|累吗|饿吗|高兴吗|开心吗)"
)


def _is_subjective_persona_question(text):
    """True iff user's text is asking Eva for preference / opinion / state.

    These questions are answered from persona, not lore. no_elaboration_rule
    shouldn't apply: the model legitimately uses general vocabulary that's
    not in retrieval records.

    Negatives (rule should still apply):
      - "what happened?" / "tell me about ..." (narrative invitation)
      - "anything fun in the kitchen?" (open story prompt)
      - "when is ...?" / "where did ...?" (factual lookup)
      - "why did you ...?" (cause explanation — possibly factual)
    """
    if not isinstance(text, str) or not text.strip():
        return False
    return bool(
        _R32_SUBJECTIVE_QUESTION_RE.search(text)
        or _R32_SUBJECTIVE_QUESTION_ZH_RE.search(text)
    )


_HEDGE_PHRASES_RE = re.compile(
    r"(?:"
    r"don'?t\s+(?:\w+\s+){0,3}have\s+(?:a\s+)?specific|"
    r"don'?t\s+(?:\w+\s+){0,3}remember|"
    r"can'?t\s+(?:\w+\s+){0,3}remember|"
    r"not\s+(?:\w+\s+){0,3}recorded|"
    r"isn'?t\s+(?:\w+\s+){0,3}recorded|"
    r"aren'?t\s+(?:\w+\s+){0,3}recorded|"
    r"specifics?\s+aren'?t|"
    r"i\s+remember\s+(?:it|that)\s+happened|"
    r"that\s+isn'?t\s+(?:in|something)|"
    r"the\s+specifics?\s+(?:aren'?t|are\s+not)|"
    r"can'?t\s+say\s+for\s+sure"
    r")",
    re.IGNORECASE,
)

# Length-4+ tokens that are too generic to count as "specific scene
# details". Includes function words, modal verbs, mental-state verbs,
# placeholder nouns, persona stutters, and answer-meta words. Time-of-
# day / weather words are intentionally NOT here — those are the exact
# class of fabrication the rule targets.
_NO_ELAB_GENERIC_STOPWORDS = frozenset({
    # modals + auxiliaries
    "could", "would", "should", "shall", "might", "must", "have", "having",
    "been", "being", "were", "where", "when", "what", "which", "while",
    "this", "that", "these", "those", "your", "yours", "their", "them",
    "from", "with", "into", "onto", "over", "under", "about", "after",
    "before", "between", "during", "through", "without", "within",
    # mental-state verbs (don't add scene detail)
    "think", "thought", "thinking", "guess", "suppose", "wonder",
    "remember", "remembered", "remembering", "recall", "recalled",
    "forget", "forgot", "forgotten",
    "understand", "understood", "imagine", "imagined",
    "know", "known", "knew", "knowing",
    # answer-meta nouns
    "memory", "memories", "record", "records", "specific", "specifics",
    "details", "detail", "answer", "question", "fact", "facts",
    # generic adverbs (low-info)
    "really", "actually", "maybe", "perhaps", "probably", "honestly",
    "frankly", "though", "anyway", "either", "neither", "still", "already",
    "always", "never", "sometimes", "often", "again", "ever",
    "rather", "quite", "very", "just", "only",
    "course", "indeed", "obviously", "clearly", "simply",
    # placeholder nouns
    "thing", "things", "stuff", "moment", "moments", "kind", "kinds",
    "sort", "sorts", "type", "types",
    # persona / address
    "master", "rosm", "tsundere", "creator",
    "miss", "missy", "lord", "lady", "darling",
    # generic descriptors
    "lovely", "right", "wrong", "nice", "fine",
    # generic action verbs (don't add scene specifics)
    "happen", "happened", "happening", "appear", "appeared",
    "make", "made", "making", "give", "gave", "given", "giving",
    "take", "took", "taken", "taking", "come", "came", "coming",
    "look", "looked", "looking", "seem", "seemed", "seeming",
    "find", "found", "finding", "want", "wanted", "wanting",
    "need", "needed", "needing", "tell", "told", "telling",
    "said", "saying", "asked", "asking",
    # quantity placeholders
    "every", "everyone", "everybody", "everything",
    "someone", "somebody", "something",
    "anyone", "anybody", "anything",
    "nothing", "nobody",
    # numerals as words
    "first", "second", "third", "fourth", "fifth",
    "five", "four", "three", "seven", "eight", "nine",
    # connectors
    "because", "since", "until", "unless", "however", "therefore",
    "actually", "obviously", "clearly", "simply",
})


def _no_elab_normalize_token(tok):
    """Lower-case + strip 's, plural/-ing/-ed for record-match comparison.

    Gate at len>=5 (not 6) so plurals/verb-forms one char above the
    minimum stem length still strip — "helps" (5) -> "help" matches
    "help" (4) in records. Without this, plural mismatch produced false
    positives (records "helps", answer "help" -> answer treated as
    unsupported invention).
    """
    tok = tok.lower().strip("'\".,;:!?-")
    if tok.endswith("'s") or tok.endswith("’s"):
        tok = tok[:-2]
    if len(tok) >= 5:
        for suf in ("ings", "ing", "edly", "ed", "es", "s"):
            if tok.endswith(suf) and len(tok) - len(suf) >= 4:
                tok = tok[:-len(suf)]
                break
    return tok


def _no_elab_extract_content_tokens(text):
    """Extract content-bearing word stems from text (>=4 chars, non-stopword).

    Stopword check happens BOTH on the raw token (catches "nothing",
    "everyone" etc. before stemming chops them into unrecognisable
    fragments like "noth" / "everyon") AND on the normalized stem
    (catches inflected forms whose lemma is in the stopword set).
    """
    text_l = (text or "").lower()
    raw = re.findall(r"\b[a-z][a-z'-]{3,}\b", text_l)
    out = set()
    for tok in raw:
        if tok in _NO_ELAB_GENERIC_STOPWORDS:
            continue
        norm = _no_elab_normalize_token(tok)
        if len(norm) < 4:
            continue
        if norm in _NO_ELAB_GENERIC_STOPWORDS:
            continue
        out.add(norm)
    return out


def answer_violates_no_elaboration_rule(agent, answer, latest_user_text,
                                        min_unsupported=3):
    """Detect fabricated scene specifics under the [NO ELABORATION RULE].

    Returns True when ALL of these hold:
      1. Some current-turn memory evidence carried the rule marker
         (low-confidence retrieval triggered the prompt-side warning).
      2. The answer does NOT contain a hedge phrase ("don't remember",
         "not recorded", "specifics aren't recorded", etc.).
      3. The answer has >= min_unsupported content-token stems that
         appear nowhere in the injected records OR the user's question.

    2026-05-13 Advisor alignment: when advisor backed this turn with a
    memory-retrieval intent (query_memory / mixed-with-MemorySearch) and
    Eva is essentially relaying retrieved facts, skip this rule.
    Otherwise persona-rich tsundere phrasing reliably triggers false
    positives on factually-correct short answers (Turn 8 reg).
    """
    if not answer or not isinstance(answer, str):
        return False

    # ---- Advisor backed memory-grounded answer -> skip rule ----
    advisor_result = getattr(agent, "advisor_result", None)
    if (advisor_result is not None
            and getattr(advisor_result, "ok", False)
            and advisor_result.needs_memory_retrieval):
        # Advisor said this turn needs memory. If any memory evidence was
        # recorded (PRE PROBE OR MemorySearch tool obs), trust Eva's answer
        # — verifier shouldn't second-guess the persona phrasing.
        ledger = getattr(agent, "turn_evidence", None)
        if ledger is not None and hasattr(ledger, "has_source"):
            if ledger.has_source("memory"):
                return False

    triggered_records = []
    for ev in getattr(agent, "turn_evidence", []) or []:
        if ev.source != "memory":
            continue
        raw = ev.raw_text or ""
        if _NO_ELABORATION_RULE_MARKER in raw:
            triggered_records.append(raw)
    if not triggered_records:
        return False

    if _HEDGE_PHRASES_RE.search(answer):
        return False

    # R-3.2 gating：subjective / preference / state 问句跳过 rule。
    if _is_subjective_persona_question(latest_user_text or ""):
        return False

    records_text = "\n".join(triggered_records)
    supported = _no_elab_extract_content_tokens(records_text)
    supported |= _no_elab_extract_content_tokens(latest_user_text or "")

    answer_tokens = _no_elab_extract_content_tokens(answer)
    unsupported = answer_tokens - supported
    return len(unsupported) >= max(1, int(min_unsupported))


# ============================================================
# Evidence collectors
# ============================================================
def exact_memory_evidence_for(agent, subject=None, slot=None):
    """Return all turn-evidence rows that are EXACT memory hits matching
    the (subject, slot) filter. Empty list if no match."""
    out = []
    target = _canonical_known_entity_name(subject) if subject else None
    for ev in getattr(agent, "turn_evidence", []) or []:
        if ev.source != "memory" or ev.confidence != "exact":
            continue
        if target and ev.subject != target:
            continue
        if slot and ev.slot != slot:
            continue
        out.append(ev)
    return out


def eva_gaming_terms_from_evidence(agent):
    """Extract known Eva game markers from current-turn memory evidence.

    Used by answer_has_eva_gaming_second_person_mismatch to find the set
    of game names that the model SHOULD have attributed to Eva (so a
    'you play apex' phrasing in the answer is a perspective error).
    """
    raw_parts = []
    for ev in getattr(agent, "turn_evidence", []) or []:
        if ev.source == "memory" and ev.subject == "Eva" and ev.slot == "gaming_preference":
            raw_parts.append(ev.raw_text or "")
    raw = _normalize_match_text("\n".join(raw_parts))
    terms = set()
    known = {
        "apex legends": ["apex", "apex legends"],
        "battlefield": ["battlefield"],
        "warzone": ["warzone"],
        "fortnite": ["fortnite"],
        "overwatch": ["overwatch", "overwatch 2"],
    }
    for phrase, aliases in known.items():
        if phrase in raw:
            terms.update(aliases)
    return terms


# ============================================================
# Query interpreters
# ============================================================
def expected_toy_subject_from_query(agent, text):
    """If the user is asking about a toy preference, return the subject
    ('Eva' or current user canonical) the question is directed at, else None.
    Used to set up the unsupported_exact_toy_claim check."""
    if not isinstance(text, str) or not text.strip():
        return None
    q = _normalize_match_text(text)
    asks_toy = bool(re.search(r"(?<![a-z0-9])(toy|plush|plushie|stuffed toy)(?![a-z0-9])", q))
    asks_toy_animal = asks_toy and bool(re.search(r"(?<![a-z0-9])(animal|kind|type|what)(?![a-z0-9])", q))
    if not (asks_toy or asks_toy_animal):
        return None
    if re.search(r"(?<![a-z0-9])(eva|maid|your|you|her)(?![a-z0-9])", q):
        return "Eva"
    if re.search(r"(?<![a-z0-9])(rosm|master|my|mine|me)(?![a-z0-9])", q):
        return _canonical_known_entity_name(agent.history_manager.user_name or "Guest")
    return None


def toy_value_words(value):
    """Tokenize a stored toy slot value into a set of comparable words.
    Resolves bunny/rabbit synonyms; drops attribute-only words like
    'cuddly' that are not the core noun."""
    norm = _normalize_match_text(value or "")
    words = set(re.findall(r"\b[a-z]{3,}\b", norm))
    if "bunny" in words:
        words.add("rabbit")
    if "rabbit" in words:
        words.add("bunny")
    # adjectives such as cuddly are not the core answer.
    words -= {"cuddly", "favorite", "childhood", "stuffed", "plush", "plushie"}
    return words


def extract_date_from_text(agent, text):
    """Lift a single (month, day) tuple out of free-form text via the
    agent's existing month/day extractor. Returns dict or None.

    Calls back into agent because _extract_month_day_from_memory is
    cross-cutting (also used by _maybe_compute_date_delta_from_memory)
    and stays on ChatAgent.
    """
    if not isinstance(text, str):
        return None
    md = agent._extract_month_day_from_memory(text)
    if not md:
        return None
    return {"month": md[0], "day": md[1]}


def extract_leaked_tool_call(answer):
    """Return (tool_name, params) if final answer contains a tool call.

    v2: matches tool calls anywhere in the answer text, not just at the end.
    This catches prose followed by a leaked call such as:
    'Fine, Master! ... WebSearch(query="X")'.
    """
    if not isinstance(answer, str) or not answer.strip():
        return None
    text = re.sub(r"^`+|`+$", "", answer.strip()).strip()
    text = text.replace("<|end_react|>", "").strip()

    m = re.search(
        r"\b(MemorySearch|WebSearch|GetCurrentTime|TextGenerationTool|AskRemoteVision)\s*\(([^)]*)\)",
        text, flags=re.DOTALL,
    )
    if not m:
        return None
    try:
        return sanitize_tool_code(m.group(0))
    except Exception:
        return None


# ============================================================
# Repair-call constructors
# ============================================================
def build_required_web_query(agent, latest_user_text):
    """Build a concrete WebSearch query for verifier-required web repair.

    The repair query is NOT just the user's raw text — for follow-up
    queries like 'check it again' we must reuse the previous web query;
    for current/news queries we annotate with the current year.
    """
    q = clean_user_text(latest_user_text or "").strip()
    year = eva_config.local_now().strftime("%Y")
    if not q:
        return f"latest news today {year}"
    q_norm = _normalize_match_text(q)
    if re.search(r"\b(what|huh|again|do it again|check it|really|verify it|try again)\b", q_norm):
        for ev in reversed(getattr(agent, "turn_evidence", []) or []):
            if ev.source == "web" and (ev.meta or {}).get("query"):
                return ev.meta["query"]
    if re.search(r"\b(news|happen|happened|happend|going on|recently|recent|nowadays|latest|current)\b", q_norm):
        if not re.search(r"\b20\d{2}\b", q_norm):
            return f"{q} {year} latest news"
        return q
    if re.search(r"\b(game|games|released|releasing|release)\b", q_norm):
        if not re.search(r"\b20\d{2}\b", q_norm):
            return f"{q} {year}"
        return q
    return q


# ------------------------------------------------------------
# P5 — Pronoun-followup detection for verifier-injected MemorySearch
# ------------------------------------------------------------
# When the user types a short pronoun-only follow-up like
# "can you check it?" the verifier used to build a MemorySearch with
# query="can you check it?" and keywords drawn from those four words —
# obviously useless. The bug surfaced in the 2026-05-08 trace where
# Eva had just offered "I could show you my special collection" and
# Rosm replied "can you check it?" — the rewritten tool call ignored
# the antecedent ('special collection') entirely and returned random
# RELATED records.
#
# Fix: detect this shape, pull noun-y tokens from the most recent
# assistant turn, and merge them into the query/keywords so the
# tool actually searches for the referent.
#
# Heuristic, NOT semantic: a pronoun-followup is a query that is
# (a) short (<= 6 words), AND (b) anchored on a pronoun verb pattern
# OR is a bare pronoun. We deliberately keep this regex-based instead
# of an LLM call because:
#   - It runs on the verifier hot path; an LLM round-trip would
#     double the verifier's worst-case latency.
#   - The LLM REWRITE_THOUGHT judge below ALSO sees history (P5
#     change 2), so the higher-quality reasoning runs there. This
#     function only needs to get the tool args 'good enough' that
#     the recall isn't garbage.
#   - False positives (treating a real query as a follow-up) are
#     bounded: we only ADD context, we don't replace the query.
# ------------------------------------------------------------
_PRONOUN_FOLLOWUP_PATTERNS = (
    # "(really?|wait,|huh,|hmm) ? (can you|could you|would you|please) ?
    #  (check|look at|tell me about|see) it/that/them/this"
    # P5.1 update: accept optional skepticism / hedge prefix
    # ('really?', 'wait,', 'huh,', 'hmm,'). The original P5 regex
    # required the verb at the very start of the string, so
    # 'really? Check it' missed entirely — the trace on 2026-05-07
    # showed this was the dominant real-world shape.
    re.compile(
        r"^\s*"
        r"(?:really\??\s*[,!.\?\-\s]*\s*|wait[,\s]+|hmm[,\s]+|huh[,?\s]+)?"  # NEW
        r"(?:can\s+you\s+|could\s+you\s+|would\s+you\s+|please\s+|do\s+)?"
        r"(?:check|see|look\s+(?:at|up)|tell\s+me\s+about|find|search|verify|"
        r"do|try|recall|remember)\s+"
        r"(?:it|that|them|those|this|these|him|her)"
        r"\s*[\?\.\!]*\s*$",
        re.IGNORECASE,
    ),
    # Bare "what about it/that?", "and them?", "really? it?"
    # P5.1: accept the same skepticism/hedge prefix here too.
    re.compile(
        r"^\s*"
        r"(?:really\??\s*[,!.\?\-\s]*\s*|wait[,\s]+|hmm[,\s]+|huh[,?\s]+)?"  # NEW
        r"(?:what\s+about\s+|and\s+|how\s+about\s+)?"
        r"(?:it|that|them|those|this|these|him|her|those\s+ones)"
        r"\s*[\?\.\!]*\s*$",
        re.IGNORECASE,
    ),
    # "(do|did) it/them/that (again|too)?"
    # P5.1: skepticism/hedge prefix.
    re.compile(
        r"^\s*"
        r"(?:really\??\s*[,!.\?\-\s]*\s*|wait[,\s]+|hmm[,\s]+|huh[,?\s]+)?"  # NEW
        r"(?:do|did)\s+(?:it|that|them|this)\s*(?:again|too)?\s*[\?\.\!]*\s*$",
        re.IGNORECASE,
    ),
)

# Stop-words we never lift out of prior assistant text — they would
# pollute the keyword list with noise.
_FOLLOWUP_NOUN_STOPWORDS = frozenset({
    # pronouns / determiners
    "i", "you", "we", "me", "my", "your", "our", "us", "they", "them",
    "their", "he", "she", "him", "her", "his", "hers", "it", "its",
    "this", "that", "these", "those", "the", "a", "an", "some", "any",
    # auxiliaries / common verbs
    "is", "are", "was", "were", "be", "been", "being", "do", "does",
    "did", "have", "has", "had", "will", "would", "can", "could",
    "should", "shall", "may", "might", "must",
    # conjunctions / prepositions / adverbs
    "and", "or", "but", "if", "so", "than", "then", "for", "of", "to",
    "in", "on", "at", "by", "with", "from", "about", "as", "into",
    "over", "under", "up", "down", "out", "off", "no", "not", "yes",
    "very", "really", "just", "also", "too", "still", "ever", "never",
    # filler dialogue tokens
    "master", "rosm", "eva", "hmph", "tch", "ah", "oh", "uh", "um",
    "well", "okay", "ok", "yeah", "yep", "nope", "huh", "hey",
    "please", "thank", "thanks", "sorry",
    # generic conversational verbs
    "say", "said", "tell", "told", "ask", "asked", "show", "showed",
    "want", "wanted", "like", "liked", "love", "think", "thought",
    "know", "knew", "see", "saw", "go", "went", "come", "came",
    "make", "made", "let", "give", "gave", "get", "got",
    # very generic nouns that don't help recall
    "thing", "things", "something", "anything", "stuff", "way",
    "time", "day", "today", "yesterday", "tomorrow", "now",
    # Eva-style filler
    "special", "little", "real", "good", "nice", "fine", "sure",
})


def _is_pronoun_followup(q):
    """True iff the cleaned user text looks like a pronoun-only
    follow-up that needs antecedent resolution.

    Operates on cleaned-but-not-tokenised text. The query length
    cap (<= 6 words) is intentional: we never want to expand a
    real well-formed question, only ones that LITERALLY cannot
    be searched on their own.

    P5.1 (2026-05-07): added trace print so the next regression
    is observable without re-instrumenting. Always prints when
    word_count <= 6 — false matches are still informative.
    """
    # Pre-existing bug fix (P6.0): VERIFIER_DEBUG was referenced bare
    # without a module-level import, so the if-branch below raised
    # NameError every time a pronoun-followup query was probed with
    # debug enabled. Latent because the function still returned the
    # right matched value when debug was disabled, but eva_config
    # ships VERIFIER_DEBUG=True by default. Match the lazy-import
    # pattern used elsewhere in this module (lines 1001, 1193, 1265,
    # 1385) instead of relying on the eva_config alias at line 61.
    from eva_config import VERIFIER_DEBUG
    if not isinstance(q, str):
        return False
    qs = q.strip()
    if not qs:
        return False
    word_count = len(re.findall(r"\b\w+\b", qs))
    if word_count == 0 or word_count > 6:
        return False
    matched = any(pat.match(qs) for pat in _PRONOUN_FOLLOWUP_PATTERNS)
    if VERIFIER_DEBUG:
        print(f"        | [DEBUG] P5 pronoun-followup probe: "
              f"q={qs!r} words={word_count} matched={matched}")
    return matched


def _extract_topical_nouns_from_recent_turns(turns, max_terms=6):
    """Pull candidate antecedent nouns from the most recent
    assistant turns.

    Strategy (in priority order):
      1. Quoted phrases in the assistant's text — these are almost
         always the actual referent ("special collection", "the
         music box"). High signal.
      2. Multi-word capitalised noun phrases (proper-noun-ish).
      3. Single-word non-stopword tokens, deduped.

    Skips the user side entirely (the user's words are already in
    the verifier query). Walks turns newest-first so the latest
    referent wins on ties.
    """
    if not turns:
        return []
    out = []
    seen = set()

    def _add(term):
        t = (term or "").strip().lower()
        if not t:
            return
        if t in seen or t in _FOLLOWUP_NOUN_STOPWORDS:
            return
        # Single-character or pure-punct fragments — skip.
        if len(t) < 2 or not re.search(r"[a-z0-9\u4e00-\u9fff]", t):
            return
        seen.add(t)
        out.append(t)

    for turn in reversed(turns):
        text = (turn.get("assistant") or "")
        if not text:
            continue
        # 1) Quoted phrases — both straight and curly quotes.
        for m in re.finditer(r"[\"\u201c]([^\"\u201d]{2,40})[\"\u201d]", text):
            _add(m.group(1))
        # 2) Multi-word capitalised phrases (e.g. 'Special Collection').
        for m in re.finditer(r"(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})", text):
            _add(m.group(0))
        # 3) Bigrams of (adj? + noun) — cheap heuristic: any two
        #    adjacent non-stopword lowercase words.
        tokens = [t.lower() for t in re.findall(r"\b[a-zA-Z\u4e00-\u9fff]{3,}\b", text)]
        for a, b in zip(tokens, tokens[1:]):
            if a in _FOLLOWUP_NOUN_STOPWORDS or b in _FOLLOWUP_NOUN_STOPWORDS:
                continue
            _add(f"{a} {b}")
        # 4) Singletons — last because they're noisy.
        for tok in tokens:
            _add(tok)
        if len(out) >= max_terms:
            break

    return out[:max_terms]


def build_required_memory_params(agent, latest_user_text):
    """Build MemorySearch params for verifier-required memory repair.

    Calls back into agent._guard_memorysearch_params for parameter
    sanitization (cross-cutting helper that stays on ChatAgent).

    P5 (2026-05-08): when the user text is a pronoun-only follow-up,
    enrich the query and keywords with antecedent nouns harvested
    from the most recent assistant turn.

    P6 (2026-05-08, P6.1): replaces the inline regex pipeline
    (_is_pronoun_followup + _extract_topical_nouns_from_recent_turns)
    with a single call to eva_pronoun_resolver.resolve_pronoun(). The
    resolver decides "is this a follow-up?" AND "what is the
    antecedent?" in one pass, with three execution stages internally
    (cheap gate / LLM / regex fallback).

    Mode flag PRONOUN_RESOLVER_MODE controls behaviour:
      - "regex_only" (P6.1 default): resolver delegates to the same
        regex helpers used pre-P6. Behaviour is bit-identical (the
        regex fallback intentionally keeps max_terms=6 to match the
        legacy keyword set).
      - "llm_first" (P6.3+): DeepSeek main path; regex fallback only
        on LLM failure during P6.0–P6.3.
      - "off": resolver short-circuits; falls through to the
        original cleaned query without antecedent augmentation.
    """
    from eva_config import VERIFIER_DEBUG
    q = clean_user_text(latest_user_text or "").strip() or "memory lookup"
    current_user = getattr(agent.history_manager, "user_name", "Guest") or "Guest"

    # P6 — pronoun-followup antecedent resolution via resolver module.
    keywords_extra = []
    q_for_target = q
    try:
        recent = agent.history_manager.recent_turns(n=2)
    except Exception:
        recent = []

    # Lazy import: keeps eva_verifier_logic import-safe even when the
    # resolver module is intentionally absent (e.g. emergency rollback
    # by deleting the file). The hasattr guard on agent is defensive —
    # ChatAgent always sets _llm_judge_state in __init__, but offline
    # tests sometimes pass stub agents.
    try:
        from eva_pronoun_resolver import resolve_pronoun
        judge_state = getattr(agent, "_llm_judge_state", None)
    except ImportError:
        resolve_pronoun = None
        judge_state = None

    # R-6: 把 P1-6 "pronoun followup 继承上一轮 target" 的补丁撤掉。
    # 改读统一的 dialog_focus.entity——它就是 dialog-level 的"当前焦点实体"。
    # focus 在 PRE PROBE / tool 跑完时已经被写过，pronoun-followup 时不需要
    # 单独的 inherited_target 变量。
    if resolve_pronoun is not None and judge_state is not None:
        resolution = resolve_pronoun(q, recent, state=judge_state)
        if resolution.needs_resolution and resolution.antecedents:
            head = " ".join(resolution.antecedents[:2])
            q_for_target = f"{q} {head}".strip()
            keywords_extra = list(resolution.antecedents)
            if VERIFIER_DEBUG:
                focus_ent = (getattr(agent, "dialog_focus", None) and
                             getattr(agent.dialog_focus, "entity", "")) or "-"
                print(
                    f"        | [DEBUG] P6 pronoun-followup detected: "
                    f"q={q!r} source={resolution.source} "
                    f"antecedents={resolution.antecedents} "
                    f"conf={resolution.confidence:.2f} "
                    f"focus_entity={focus_ent}"
                )

    # 默认 target 推断走 _infer，pronoun-followup / continuation 时用
    # dialog_focus.entity 作为"当前焦点"覆盖 Both 的默认值。
    target = _infer_memory_target_from_text(
        q_for_target, default_target="Both", current_user=current_user,
    )
    # R-6.1: pronoun resolution 在 dialog_focus 之前。本轮 user_text 含
    # "your X" / "my X" / "do you ..." 时，直接据此定 target，避免 sticky
    # focus 把"your"问题错绑到上一轮的 entity。
    speaker_perspective = ""
    try:
        resolver = getattr(agent, "_resolve_speaker_perspective_entity", None)
        if callable(resolver):
            speaker_perspective = _canonical_known_entity_name(
                resolver(latest_user_text or "") or ""
            )
    except Exception:
        speaker_perspective = ""
    if speaker_perspective in ("Eva", "Rosm") and target in ("", "Both", "Shared"):
        target = speaker_perspective
    else:
        focus_entity = _canonical_known_entity_name(
            (getattr(agent, "dialog_focus", None) and
             getattr(agent.dialog_focus, "entity", "")) or ""
        )
        if focus_entity in ("Eva", "Rosm") and target in ("", "Both", "Shared"):
            target = focus_entity
    base_keywords = _build_display_keywords_from_query(
        q_for_target, target_entity=target, current_user=current_user, limit=16,
    )
    # Antecedent terms come first (higher recall priority) then the
    # base keywords, deduped while preserving order. Cap at 16 to
    # match the original limit.
    merged_keywords = list(dict.fromkeys(keywords_extra + list(base_keywords)))[:16]

    params = {
        "query": q_for_target,
        "target_entity": target,
        "keywords": ", ".join(merged_keywords),
    }
    return agent._guard_memorysearch_params(params, latest_user_text)


# ============================================================
# STAGE 2 — Orchestrator
# ============================================================
# verify_final_answer is the main verifier entry point. It runs after
# phase-2 produces a final answer and decides whether the answer has
# enough grounding evidence to be released to the user. If not, it
# returns a VerifyResult with reasons and (optionally) a required_action
# the controller should execute as repair.
#
# All decisions go through `agent.X(...)` calls — even helpers that now
# live in this module. The double-hop (eva_verifier_logic → agent
# wrapper → eva_verifier_logic) is intentional: it keeps the agent's
# public surface as the canonical interface and makes the dependency
# graph easy to trace (every reach into agent state is visible).
# ============================================================
def _run_semantic_verifier_shadow(agent, answer, latest_user_text):
    """P1: invoke the SemanticVerifier in shadow / warn / hard mode.

    Shadow mode: verdict is logged onto agent.semantic_verdicts_this_turn
    (a fresh list per-turn) and printed when SEMANTIC_VERIFIER_DEBUG is
    on. Does NOT contribute to verifier reasons.

    Warn mode: same as shadow, but a 'fail' verdict prints a warning.

    Hard mode: a 'fail' verdict appends a synthetic reason
    'semantic_verifier_fail_<issue_type>' to verifier reasons via
    agent._pending_semantic_reason — picked up by the caller. (Not
    enabled by default in P1; flag promotion happens in P2.)
    """
    try:
        from eva_config import (
            ENABLE_SEMANTIC_VERIFIER,
            SEMANTIC_VERIFIER_MODE,
            SEMANTIC_VERIFIER_HISTORY_TURNS,
            SEMANTIC_VERIFIER_DEBUG,
        )
    except ImportError:
        return  # config missing the P1 flags — verifier-v2 disabled

    if not ENABLE_SEMANTIC_VERIFIER:
        return
    sem = getattr(agent, "semantic_verifier", None)
    if sem is None:
        return

    history = []
    try:
        history = agent.history_manager.recent_turns(
            n=int(SEMANTIC_VERIFIER_HISTORY_TURNS),
            include_current=False,
        )
    except Exception as e:
        if SEMANTIC_VERIFIER_DEBUG:
            print(f"        | [DEBUG] semantic verifier: history fetch failed: {e!r}")

    evidence_summary = _semantic_evidence_summary(agent)

    verdict = sem.verify(
        answer=answer,
        latest_user_text=latest_user_text,
        history=history,
        evidence_summary=evidence_summary,
    )

    # Stash on agent for downstream observability / tests.
    if not hasattr(agent, "semantic_verdicts_this_turn") or agent.semantic_verdicts_this_turn is None:
        agent.semantic_verdicts_this_turn = []
    agent.semantic_verdicts_this_turn.append(verdict.to_log())

    mode = (SEMANTIC_VERIFIER_MODE or "shadow").lower()
    if SEMANTIC_VERIFIER_DEBUG:
        print(f"        | [DEBUG] semantic verifier ({mode}): {verdict.to_log()}")

    if mode == "warn" and verdict.is_failure:
        print(f"        | [WARN] semantic verifier flagged answer: {verdict.issues}")

    # P3: hard mode — promote a high-confidence fail verdict into a
    # verifier reason. We stash on the agent so verify_final_answer
    # picks it up after the regex chain runs. The reason name is
    # 'semantic_verifier_fail:<issue_type>' so REASON_POLICY's default
    # fallback is bypassed by the explicit policy registered below.
    if mode == "hard" and verdict.is_failure:
        try:
            from eva_config import (
                SEMANTIC_VERIFIER_FAIL_CONFIDENCE,
                SEMANTIC_VERIFIER_HARD_ISSUE_TYPES,
            )
            min_conf = float(SEMANTIC_VERIFIER_FAIL_CONFIDENCE)
            allowed = set(SEMANTIC_VERIFIER_HARD_ISSUE_TYPES)
        except Exception:
            min_conf, allowed = 0.80, {
                "pronoun_referent_mismatch",
                "internal_self_contradiction",
                "fact_conflict_with_evidence",
            }
        # Pick the first issue meeting both gates.
        chosen = None
        for it in verdict.issues or []:
            if not isinstance(it, dict):
                continue
            t = (it.get("type") or "").strip()
            try:
                c = float(it.get("confidence") or 0.0)
            except Exception:
                c = 0.0
            if t in allowed and c >= min_conf:
                chosen = (t, c, (it.get("evidence") or "")[:200])
                break
        if chosen:
            t, c, ev = chosen
            agent._pending_semantic_reason = f"semantic_verifier_fail:{t}"
            agent._pending_semantic_meta = {"confidence": c, "evidence": ev}
            if SEMANTIC_VERIFIER_DEBUG:
                print(f"        | [DEBUG] semantic verifier (hard) PROMOTED: "
                      f"reason={agent._pending_semantic_reason!r} "
                      f"confidence={c:.2f} evidence={ev!r}")
        else:
            # Failure exists but no eligible issue — log and stay warn-equivalent.
            if SEMANTIC_VERIFIER_DEBUG:
                print(f"        | [DEBUG] semantic verifier (hard) NOT PROMOTED: "
                      f"no issue passed type+confidence gate; issues={verdict.issues}")

    # P2 diagnostics: when the verifier failed (no key / budget /
    # judge timeout / parse error), print the error explicitly so
    # the operator can see why shadow data is sparse. This goes
    # behind SEMANTIC_VERIFIER_DEBUG so production traces stay quiet.
    if SEMANTIC_VERIFIER_DEBUG and (verdict.error or verdict.skipped_reason):
        print(f"        | [DEBUG] semantic verifier diagnostic: "
              f"error={verdict.error!r}, skipped_reason={verdict.skipped_reason!r}")
    # 'hard' mode is intentionally not enabled in P1; the regex semantic
    # checks remain the (demoted) authority and the dispatcher does not
    # yet read agent._pending_semantic_reason.


def _semantic_evidence_summary(agent):
    """Compact dict of evidence the verifier may reference. Pulls the
    safest, most-stable handful of fields off agent state. Failures
    return an empty dict (verifier still runs)."""
    summary = {}
    try:
        # R-6: last_memory.observation 始终是字符串；下面这个 dict 分支
        # 在新 schema 下永远 False，保留兼容（万一别处 monkey-patch 写了 dict）。
        last_mem = getattr(agent, "last_memory", None)
        last_mem_obs = getattr(last_mem, "observation", None)
        if isinstance(last_mem_obs, dict):
            for k in ("subject", "slot", "value", "confidence"):
                if k in last_mem_obs and last_mem_obs[k] is not None:
                    summary[f"memory_{k}"] = last_mem_obs[k]
    except Exception:
        pass
    try:
        binding = getattr(agent, "active_date_binding", None)
        if isinstance(binding, dict):
            for k in ("days_until", "target_date", "anchor_date"):
                if k in binding and binding[k] is not None:
                    summary[f"date_{k}"] = binding[k]
    except Exception:
        pass
    return summary


def verify_final_answer(agent, answer, latest_user_text):
    """Verify the final phase-2 answer against turn evidence.

    Returns VerifyResult(ok=True) on success. On failure, returns
    VerifyResult(ok=False, reasons=[...], required_action=..., hard_fail=...)
    where required_action is what the controller should do to repair
    (typically a tool call) and hard_fail signals that an unrepairable
    failure should fall back to a safe canned response.

    Reasons (in the order they're checked):
        0. tool_call_leaked_in_answer
        1. missing_web_evidence_for_external_or_current_request
        1b. missing_memorysearch_for_explicit_memory_check
        2. eva_self_birthday_pronoun_mismatch
        2b. unsupported_exact_toy_claim
            toy_value_conflicts_with_exact_memory
        2c. textgen_perspective_mismatch
        3. missing_date_calculation_evidence
            date_math_days_not_supported_by_calculation_evidence
            date_math_target_date_mismatch
    """
    # Lazy import config flags so runtime toggles propagate.
    # NB: HARD_VERIFIER_REASONS used to be imported here too, but the
    # legacy set was superseded by REASON_POLICY (TODO 11-arch). The
    # config still defines it for back-compat with any external code
    # that reads it, but verify_final_answer no longer consults it.
    from eva_config import ENABLE_ANSWER_VERIFIER, VERIFIER_DEBUG
    from eva_history import VerifyResult

    if not ENABLE_ANSWER_VERIFIER:
        return VerifyResult(ok=True)
    if not answer:
        return VerifyResult(ok=True)

    # P1: shadow-mode SemanticVerifier — runs once per call, verdict
    # logged into turn_evidence. Does not influence dispatch unless
    # SEMANTIC_VERIFIER_MODE escalates beyond 'shadow'.
    _run_semantic_verifier_shadow(agent, answer, latest_user_text)

    reasons = []
    evidence = getattr(agent, "turn_evidence", []) or []
    answer_l = answer.lower()

    # 0) Never expose a tool invocation as plain final text.
    if agent._extract_leaked_tool_call(answer):
        reasons.append("tool_call_leaked_in_answer")

    # 2026-05-13 Advisor-first refactor: the four explicit-intent post-hoc
    # judges (EXPLICIT_WEB / EXPLICIT_MEMORY / EXPLICIT_REMEMBER /
    # EXPLICIT_FORGET / PUBLIC_FACT) are replaced by the Advisor's single
    # `intent` field set at turn start. When advisor_result.ok=True we use
    # advisor.intent + a cheap evidence ledger check to flag missing
    # tool-call evidence. When advisor failed we fall back to the original
    # judges (via EVA_ADVISOR_FALLBACK_MODE=judges).
    advisor_result = getattr(agent, "advisor_result", None)
    use_advisor = bool(advisor_result and advisor_result.ok
                       and advisor_result.intent != "unknown")
    notes_active = bool(
        agent.memory_state and agent.memory_state.get("notes_store")
    )

    if use_advisor:
        # 2026-05-13 alignment v2: advisor.suggested_calls is the SINGLE
        # source of truth for "which tools should fire this turn". The
        # intent label is a coarse summary that can be miscategorised
        # (e.g. "birthday + full name" labeled as "mixed" instead of
        # "query_memory"); suggested_calls is a concrete commitment.
        # If the user thinks the advisor's tool list was wrong, that's
        # an advisor-prompt issue, not a verifier issue — fix the
        # advisor, not the verifier.
        suggested_tools: set[str] = set()
        for c in (advisor_result.suggested_calls or []):
            if isinstance(c, dict):
                t = c.get("tool")
                if isinstance(t, str) and t.strip():
                    suggested_tools.add(t.strip())

        wants_web         = "WebSearch"    in suggested_tools
        wants_memory_check = "MemorySearch" in suggested_tools
        explicit_remember = notes_active and "RememberThis" in suggested_tools
        explicit_forget   = notes_active and "ForgetMemory" in suggested_tools
    else:
        # Fallback: advisor unavailable. Use original local LLM judges (the
        # pre-refactor flow). Cheap but slower; only happens on advisor
        # outage when EVA_ADVISOR_FALLBACK_MODE='judges'.
        try:
            from eva_config import EVA_ADVISOR_FALLBACK_MODE
        except ImportError:
            EVA_ADVISOR_FALLBACK_MODE = "judges"

        if EVA_ADVISOR_FALLBACK_MODE != "judges":
            # In 'chat' or 'strict' fallback modes, skip judges entirely.
            wants_web = False
            wants_memory_check = False
            explicit_remember = False
            explicit_forget = False
        else:
            wants_web = (agent._explicit_web_request(latest_user_text)
                         or agent._current_external_query_needs_web(latest_user_text)
                         or agent._is_obvious_public_fact_or_news_query(latest_user_text))
            wants_memory_check = bool(agent._explicit_memory_check_request(latest_user_text))
            explicit_remember = (
                notes_active and agent._explicit_remember_request(latest_user_text)
            )
            explicit_forget = (
                notes_active and agent._explicit_forget_request(latest_user_text)
            )

    # 1) Explicit/current external requests need actual WebSearch evidence.
    if wants_web and not agent._current_turn_has_web_evidence():
        reasons.append("missing_web_evidence_for_external_or_current_request")

    # 1b) Explicit memory-CHECK (read) requests require an actual MemorySearch
    # tool call OR an injected memory packet.
    if wants_memory_check:
        # Advisor-driven probe counts as "search attempted" — when advisor
        # said needs_memory_retrieval=True the PRE PROBE ran (possibly with
        # 0 results). That's a legitimate "we tried" signal. Without this
        # alignment we'd false-fire missing_memorysearch on every query
        # where advisor's hint missed (e.g. Turn 6 toy query with too-
        # verbose memory_query_hint).
        advisor_attempted_search = bool(
            use_advisor and advisor_result.needs_memory_retrieval
        )
        if (not current_turn_has_remember_evidence(agent)
                and not explicit_remember
                and not explicit_forget
                and not agent._current_turn_has_memorysearch_evidence()
                and not advisor_attempted_search):
            try:
                from eva_slots import extract_memory_slots
                asked_slots = extract_memory_slots(
                    latest_user_text,
                    encoder=getattr(agent, "encoder", None),
                )
            except Exception:
                asked_slots = []
            if asked_slots:
                reasons.append("missing_memorysearch_for_explicit_memory_check")

    # 1c) Explicit REMEMBER (write) requests require an actual RememberThis tool call.
    if explicit_remember and not current_turn_has_remember_evidence(agent):
        reasons.append("explicit_remember_request_not_handled")

    # 1d) Explicit FORGET (delete) requests require an actual ForgetMemory tool call.
    if explicit_forget and not current_turn_has_forget_evidence(agent):
        reasons.append("explicit_forget_request_not_handled")

    # 2026-05-14 Plan-A cleanup:
    # - Eva self-birthday pronoun mismatch (was 2): owned by SemanticVerifier;
    #   ENABLE_LEGACY_SEMANTIC_REGEX always False → dead path deleted.
    # - toy_value_conflicts_with_exact_memory (was 2b SEMANTIC): same.
    # - TextGenerationTool perspective mismatch (was 2c): same.
    # All three reasons are SOFT post-Plan-A; if SemanticVerifier judges
    # them, the structured-judge path handles it. Removing the dead regex
    # paths cuts ~50 lines and removes 3 sources of false-positive that
    # never fire in production anyway.
    # The structural toy-evidence check below stays — it's purely about
    # missing data and was always-on regardless of the legacy flag.
    toy_subject = agent._expected_toy_subject_from_query(latest_user_text)
    if toy_subject:
        toy_animals = agent._answer_toy_animal_words(answer)
        if toy_animals and not re.search(r"\b(no\s+record|not\s+recorded|don'?t\s+know|do\s+not\s+know|unknown|not\s+sure)\b", answer_l):
            toy_evs = agent._exact_memory_evidence_for(subject=toy_subject, slot="toy")
            # Subject-agnostic rescue: if any toy evidence covers an
            # animal word in the answer, accept it.
            if not toy_evs:
                fallback_evs = agent._exact_memory_evidence_for(slot="toy")
                for ev in fallback_evs or []:
                    words = agent._toy_value_words(ev.value or "")
                    if words and not toy_animals.isdisjoint(words):
                        toy_evs = [ev]
                        break
            if not toy_evs:
                reasons.append("unsupported_exact_toy_claim")

    # 3) Date arithmetic must match the structured calculation evidence.
    calc_evs = [ev for ev in evidence if ev.source == "calculation" and ev.value is not None]
    if agent._answer_mentions_days(answer):
        date_math_context = (
            agent._question_needs_time_arithmetic(latest_user_text)
            or bool(re.search(r"\b(birthday|until|from\s+now|days?|weeks?|months?|date)\b", latest_user_text.lower()))
            or bool(re.search(r"\b(birthday|until|from\s+now)\b", answer_l))
        )
        if date_math_context and not calc_evs:
            reasons.append("missing_date_calculation_evidence")

    if calc_evs and agent._answer_mentions_days(answer):
        days_in_answer = [int(x) for x in re.findall(r"\b(\d+)\s+day(?:s)?\b", answer_l)]
        valid_days = {int(ev.value) for ev in calc_evs}
        if days_in_answer and not any(d in valid_days for d in days_in_answer):
            reasons.append("date_math_days_not_supported_by_calculation_evidence")

        # If the answer names a different month/day than the calculated target,
        # do not allow reusing the calculated days for that other date.
        mentioned_date = agent._extract_date_from_text(answer)
        if mentioned_date is not None:
            target_pairs = {(ev.meta.get("target_month"), ev.meta.get("target_day")) for ev in calc_evs}
            if (mentioned_date["month"], mentioned_date["day"]) not in target_pairs:
                reasons.append("date_math_target_date_mismatch")

    # 4) [NO ELABORATION RULE] backstop. Soft prompt rule emitted by
    # eva_memory_legacy on low-confidence retrieval — verify here that
    # phase-2 actually obeyed it. Regenerate-class: evidence is correct,
    # the model just embellished.
    if answer_violates_no_elaboration_rule(agent, answer, latest_user_text):
        reasons.append("unsupported_specifics_under_no_elaboration_rule")

    # Plan D (2026-05-15): answer 末尾留下未闭合的 markdown ** = sampling
    # collapse 残骸（"Tch! ... I told you: **"）。走 regenerate 给一次重新
    # 采样的机会，比在 generation 阶段用 LogitsProcessor 硬约束更鲁棒
    # —— 不会误伤数学表达 5**3 / glob 模式 **/*.py 等合法 ** 用法。
    if _answer_has_orphan_bold_tail(answer):
        reasons.append("orphan_markdown_in_answer")

    # P3: pick up any pending semantic-verifier reason promoted by
    # _run_semantic_verifier_shadow under SEMANTIC_VERIFIER_MODE='hard'.
    # The reason is fully namespaced ("semantic_verifier_fail:<type>")
    # and has its own REASON_POLICY entry registered above, so it
    # routes through `regenerate` like any other regenerate-class
    # reason and inherits the RegenerateGuard quota.
    pending_sem = getattr(agent, "_pending_semantic_reason", None)
    if pending_sem and pending_sem not in reasons:
        reasons.append(pending_sem)
    # Single-shot: clear after consumption so a follow-up regenerate
    # in the same turn doesn't re-fire on stale state.
    agent._pending_semantic_reason = None
    agent._pending_semantic_meta = None

    if not reasons:
        return VerifyResult(ok=True)

    # TODO 11-arch: derive severity + fix_class from REASON_POLICY rather
    # than the legacy HARD_VERIFIER_REASONS set. Unregistered reasons fall
    # back to DEFAULT_POLICY (hard + canned_fallback) — fail-safe.
    hard_fail = any(get_reason_policy(r)["severity"] == "hard" for r in reasons)

    # P0: when no reason is hard (e.g. all triggered reasons are
    # demoted semantic checks), return a soft-pass — log the warnings
    # but let the original answer through. This kills the
    # "regex flags answer → regenerate → same regex flags it again"
    # death loop seen in Turn 3 of the live chat trace.
    if not hard_fail:
        if VERIFIER_DEBUG:
            print(f"        | [DEBUG] verifier soft-pass: warnings={reasons}")
        return VerifyResult(ok=True, reasons=reasons, hard_fail=False)

    # 2026-05-13 Advisor cutover：dispatch 时只看 hard reasons。soft 类（被 advisor
    # 取代的"事后侦测"）仍记录在 reasons 列表里供 telemetry，但不参与 required_action
    # 选择——否则 missing_memorysearch (soft) + date_math_mismatch (hard) 共发时，
    # 老 dispatcher 按 reason 字符串顺序匹配会错误地 inject MemorySearch。
    hard_reasons = [r for r in reasons if get_reason_policy(r)["severity"] == "hard"]
    required_action = agent._required_action_from_verifier_reasons(
        hard_reasons, latest_user_text, answer,
    )
    dominant = get_dominant_reason_for_dispatch(hard_reasons)
    dominant_policy = get_reason_policy(dominant) if dominant else DEFAULT_POLICY
    fix_class = dominant_policy["fix"]

    if VERIFIER_DEBUG:
        print(f"        | [DEBUG] verify result: reasons={reasons}, "
              f"hard_reasons={hard_reasons}, "
              f"required_action={required_action is not None}, "
              f"hard_fail={hard_fail}, fix_class={fix_class!r} "
              f"(dominant={dominant!r})")
    return VerifyResult(
        ok=False,
        reasons=reasons,
        required_action=required_action,
        hard_fail=hard_fail,
        fix_class=fix_class,
    )


# ============================================================
# STAGE 3 — Reason → repair action dispatch + terminal failsafe
# ============================================================
# required_action_from_verifier_reasons maps a list of failure reasons to
# a single concrete tool call that the controller should execute as
# repair. Priority order is encoded in the if-cascade: tool-leak repair
# first, then web evidence, then memory/slot, then date arithmetic.
# Only ONE tool is returned per call — verifier never asks the
# controller to run multiple repair tools in one go.
#
# safe_fallback_for_hard_verifier_failure produces the message we hand
# back to the user when even the controller-injected repair couldn't
# resolve the failure (loop guard in step_once detects same-reason
# repeat). Each canned line stays in Eva's voice while being honest
# about why we couldn't answer.
# ============================================================
def required_action_from_verifier_reasons(agent, reasons, latest_user_text, answer):
    """Map verifier failures to controller-executable repair actions.

    2026-05-14 Plan-A final form: only `tool_call_leaked_in_answer` is hard
    with fix="inject_tool". All other repair branches (missing_web /
    missing_memorysearch / missing_date_calculation / explicit_remember /
    explicit_forget) are SOFT after Plan-A — they generate telemetry log
    only, never reach this function via the hard_reasons filter in
    verify_final_answer. The old if-cascade is therefore unreachable;
    only the tool-leak branch remains live.

    The dead helpers (extract_remember_params_from_user_text,
    find_recent_note_id, build_required_web_query, build_required_memory_params)
    are still defined for test backward-compat but not invoked at runtime.

    Returns a {"tool": str, "params": dict, "reason": str} dict on hit,
    or None if no listed reason maps to a repair action.
    """
    from eva_config import VERIFIER_DEBUG
    try:
        # Only live branch: tool_call_leaked_in_answer
        if "tool_call_leaked_in_answer" in reasons:
            leaked = agent._extract_leaked_tool_call(answer or "")
            if leaked:
                tool_name, params = leaked
                if VERIFIER_DEBUG:
                    print(f"        | [DEBUG] required_action: leaked={tool_name}, params={params}")
                return {
                    "tool": tool_name,
                    "params": params,
                    "reason": "tool_call_leaked_in_answer",
                }

        if VERIFIER_DEBUG:
            print("        | [DEBUG] required_action: None (no live reason matched)")
        return None
    except Exception as e:
        if VERIFIER_DEBUG:
            print(f"        | [WARN] required_action resolution failed: {type(e).__name__}: {e}")
        return None


# _self_validate_date_calculation: DELETED 2026-05-14 Plan-A final cleanup.
# Was used by safe_fallback to rescue answers under missing_date_calculation_evidence,
# but that reason is now soft and never reaches safe_fallback. Function had no
# tests of its own.


def safe_fallback_for_hard_verifier_failure(agent, verify_result, latest_user_text,
                                            phase2_answer=None):
    """Canned response when controller-injected repair cannot save the turn.

    R-3 (2026-05-13)：候选答案选择从 caller 手动传 `phase2_answer=...` 改为
    读 `agent.verdict_ledger.best()`。每次 verify_final_answer 完成时已经把
    候选 push 进 ledger 了；ledger 按 (severity, reason_count, stage_order)
    自然挑出最该释放的那条。

      - Turn 5 复盘：original verdict (regex fail) vs regen verdict (LLM fail) —
        ledger 选 original (同 severity 同 reason_count, stage_order 更早)，
        和旧 P0-3 patch 行为等价但语义结构化。
      - has_disagreement() 检测到 regex / llm_semantic 跨 verifier 不一致时
        打 telemetry 日志；不强制走 canned（避免 Turn 5 那种 verifier 误报
        导致用户看 canned UX 反而差的情况）。

    `phase2_answer` 参数保留作 ledger 空时的 backward-compat：上层旧 callsite
    没 ledger 时仍可手动指定释放对象（应当极少用，主要给 test stub）。

    Returns:
        str. Either the best candidate answer (release as-is) or the
        canned per-reason string.
    """
    from eva_config import VERIFIER_DEBUG

    reasons = list((verify_result.reasons if verify_result else []) or [])

    # R-3: 从 ledger 取 best candidate（如果 caller 没显式传 phase2_answer 也可用）
    ledger = getattr(agent, "verdict_ledger", None)
    best_verdict = None
    if ledger is not None and len(ledger) > 0:
        best_verdict = ledger.best()
    # 选 release_answer：优先 ledger 给的 best，否则 fallback 到 caller 传入。
    release_answer = (best_verdict.answer if best_verdict is not None
                      else phase2_answer)

    # Telemetry: 跨 verifier disagreement 不强制 canned，但记录到日志。
    if VERIFIER_DEBUG and ledger is not None and ledger.has_disagreement():
        stages = [(v.source_stage, v.severity, v.reason_class)
                  for v in ledger.candidates]
        print(f"        | [DEBUG] safe_fallback: cross-verifier disagreement "
              f"detected (regex vs llm_semantic). stages={stages}")

    # 2026-05-14 Plan-A: missing_date_calculation_evidence 已降 soft，永远不会
    # 进入 safe_fallback 路径（这个函数只在 hard_fail 时被调用）。原本的
    # _self_validate_date_calculation 自救路径不可达，已删。函数本身保留
    # 在文件里供未来回滚 / 测试用。

    # TODO 11-arch: pull the canned message from REASON_POLICY using the
    # dominant reason. This replaces the prior fragile if-cascade.
    # 2026-05-13 Advisor cutover：dominant 只从 hard 中选；soft reason 只做
    # telemetry。否则 hard+soft 同 fix_priority 时 dominant 可能错选 soft。
    hard_only = [r for r in reasons if get_reason_policy(r)["severity"] == "hard"]
    dominant = get_dominant_reason_for_dispatch(hard_only or reasons)

    # P0: regex-based semantic check 不可靠 → 优先释放模型答案。
    if dominant in _SEMANTIC_REASONS and release_answer:
        if VERIFIER_DEBUG:
            stage = best_verdict.source_stage if best_verdict else "legacy"
            print(f"        | [DEBUG] safe_fallback: semantic reason {dominant!r} "
                  f"-> RELEASING answer (stage={stage}, P0 policy)")
        return release_answer

    # P3: LLM-judge semantic — 同 P0，释放模型答案。
    if isinstance(dominant, str) and dominant.startswith("semantic_verifier_fail:") and release_answer:
        if VERIFIER_DEBUG:
            stage = best_verdict.source_stage if best_verdict else "legacy"
            print(f"        | [DEBUG] safe_fallback: LLM-judge reason {dominant!r} "
                  f"-> RELEASING answer (stage={stage}, P3 policy)")
        return release_answer

    policy = get_reason_policy(dominant) if dominant else DEFAULT_POLICY
    if VERIFIER_DEBUG and dominant:
        print(f"        | [DEBUG] safe_fallback: dominant={dominant!r} fix={policy['fix']!r}")
    return policy["canned"]


# ============================================================
# STAGE 4 — Repair runner
# ============================================================
# execute_controller_tool actually runs the tool that
# required_action_from_verifier_reasons asked for. It has the largest
# state surface of the four verifier-core methods: 18 cross-cutting
# helpers on agent + 19 attributes (5 of which are MUTATED — turn-
# memory flags, last_memory_*, last_missing_slots).
#
# All tool execution paths (GetCurrentTime / MemorySearch / WebSearch /
# AskRemoteVision / TextGenerationTool) preserve their exact behaviour
# from the inline version. Outputs are appended to the agent's history
# via agent.history_manager.add_tool_output, and a return value of
# (None, True) signals "controller-injected tool fired successfully" to
# the step_once loop.
#
# Module-level deps (run_memory_search, run_websearch, etc.) are
# imported at the top of this file. Runtime-toggleable configs (REACT,
# MEMORY_SLOT_FIELDS) are lazy-imported inside the function so a hot-
# reload of eva_config picks up changes without re-importing this
# module.
# ============================================================


# ============================================================
# Step 5 — Trace rewriting helpers (used by execute_controller_tool)
# ============================================================
# When the verifier hard-fails and injects a repair tool, the most
# recent assistant turn is `<think>...</think><|answer|>WRONG</|end_react|>`.
# We rewrite it in-place to `<think>...</think><|tool_code|>RealTool(...)<|end_react|>`
# BEFORE appending the tool_output, so the resulting trajectory matches
# SFT's tool-call → tool-output shape and phase-2 grounds correctly.
# See TODO.md TODO 2 for full rationale.
# ============================================================

# Hardcoded thought templates per repair tool. Used as fallback when
# the DeepSeek-based synthesise_tool_thought returns None (judge
# disabled / errored / over budget). Templates are deliberately
# generic — the goal is in-distribution thought content, not perfect
# wording. DeepSeek output, when available, is preferred.
_REPAIR_THOUGHT_TEMPLATES = {
    "WebSearch":
        "Master is asking for a fact about the wider world. "
        "I should consult WebSearch for the authoritative answer.",
    "MemorySearch":
        "Master is asking about something personal — Rosm, "
        "myself, or our shared history. I should check my memory "
        "first.",
    "GetCurrentTime":
        "Master needs the current date or time anchor. "
        "I should call GetCurrentTime first.",
    "AskRemoteVision":
        "Master attached an image and is asking about its content. "
        "I should use AskRemoteVision to examine it.",
    "TextGenerationTool":
        "I need a careful textual transformation here. "
        "I should use TextGenerationTool after gathering facts.",
}
_REPAIR_THOUGHT_FALLBACK = "I should consult the appropriate tool first."


# ANSI escape sequences for the rewrite block. Centralised so the
# fallback path can swap them for empty strings without ifs scattered
# through the print logic.
#   \033[2m  dim
#   \033[9m  strikethrough
#   \033[1m  bold
#   \033[33m yellow foreground
#   \033[0m  reset
_ANSI_DIM_STRIKE = "\033[2;9m"
_ANSI_BOLD_YELLOW = "\033[1;33m"
_ANSI_RESET = "\033[0m"


def _render_step5_rewrite_block(thought, tool_call_str, indent="        | "):
    """Render the visual block that announces a STEP-5 trace rewrite.

    The verifier rewrote the most-recent phase-1 step from
    `<|answer|>WRONG</|end_react|>` shape into
    `<|tool_code|>RealTool(...)</|end_react|>` shape. The prior
    THOUGHT/ANSWER blocks have already been streamed to the operator
    above; we cannot retroactively annotate them in a streaming
    terminal. Instead this block prints a loud divider that:

      1. Names the boundary explicitly ("STEP-5 TRACE REWRITE").
      2. States that everything above the divider in the current
         phase is now SUPERSEDED.
      3. Prefixes the new thought + tool_code with [REWRITTEN] so
         the synthesised replacement content is unambiguously tagged.

    Style is controlled by config.TRACE_REWRITE_STYLE:
      - "ansi"  — uses dim+strike for the supersede notice and
                  bold-yellow for the [REWRITTEN] tag. Renders on
                  any modern terminal (VS Code, Windows Terminal,
                  *nix).
      - "ascii" — plain === bars and no escape codes. Always safe.

    Returns a list of strings; caller prints them in order.
    """
    from eva_config import TRACE_REWRITE_STYLE

    short_thought = thought[:90] + ("…" if len(thought) > 90 else "")
    use_ansi = (TRACE_REWRITE_STYLE == "ansi")

    if use_ansi:
        header = f"{_ANSI_BOLD_YELLOW}=== STEP-5 TRACE REWRITE ==={_ANSI_RESET}"
        supersede = (
            f"{_ANSI_DIM_STRIKE}^^^ Phase-1 THOUGHT/ANSWER above are SUPERSEDED ^^^"
            f"{_ANSI_RESET}"
        )
        rewritten_tag = f"{_ANSI_BOLD_YELLOW}[REWRITTEN]{_ANSI_RESET}"
        bar = f"{_ANSI_BOLD_YELLOW}{'=' * 60}{_ANSI_RESET}"
    else:
        header = "=== STEP-5 TRACE REWRITE ==="
        supersede = "^^^ Phase-1 THOUGHT/ANSWER above are SUPERSEDED ^^^"
        rewritten_tag = "[REWRITTEN]"
        bar = "=" * 60

    return [
        f"{indent}",
        f"{indent}{bar}",
        f"{indent}{header}",
        f"{indent}{supersede}",
        f"{indent}Replacing phase-1 answer with tool-call shape:",
        f"{indent}  {rewritten_tag} thought:    {short_thought!r}",
        f"{indent}  {rewritten_tag} tool_code:  {tool_call_str}",
        f"{indent}{bar}",
    ]


def _format_tool_call_string(tool_name, tool_params):
    """Build the tool-call string that goes inside <|tool_code|>.

    Mirrors the format SFT data uses for tool calls (see
    sft_preprocess_v3.format_assistant_response). For dict params we
    emit name=value comma-separated with double-quoted strings (JSON-
    like). For empty params we emit the bare-call form.
    """
    if not isinstance(tool_params, dict) or not tool_params:
        return f"{tool_name}()"
    pieces = []
    for k, v in tool_params.items():
        if isinstance(v, str):
            # Escape internal quotes minimally.
            v_esc = v.replace('"', '\\"')
            pieces.append(f'{k}="{v_esc}"')
        else:
            pieces.append(f"{k}={v}")
    return f"{tool_name}({', '.join(pieces)})"


def _summarize_tool_args_for_thought(tool_name, tool_params):
    """Short summary of args for the synthesise-thought prompt.

    Picks the most informative key per tool to avoid bloating the
    prompt. The judge needs just enough context to reason about why
    the tool fits, not the full param dict.
    """
    if not isinstance(tool_params, dict) or not tool_params:
        return ""
    if tool_name in ("WebSearch", "MemorySearch") and "query" in tool_params:
        return f'query="{tool_params["query"]}"'
    if tool_name == "AskRemoteVision":
        bits = []
        if tool_params.get("mode"):
            bits.append(f'mode="{tool_params["mode"]}"')
        if tool_params.get("query"):
            bits.append(f'query="{tool_params["query"]}"')
        return ", ".join(bits)
    if tool_name == "TextGenerationTool" and "instruction" in tool_params:
        instr = tool_params["instruction"]
        # Truncate long instructions
        if len(instr) > 80:
            instr = instr[:77] + "..."
        return f'instruction="{instr}"'
    # Fallback — first key/value
    k = next(iter(tool_params))
    return f"{k}={tool_params[k]!r}"


def _rewrite_assistant_for_tool_repair(agent, tool_name, tool_params,
                                       latest_user_text, reason):
    """Rewrite last assistant step from answer-shape to tool-call-shape.

    Called by execute_controller_tool BEFORE add_tool_output, so the
    resulting trajectory becomes:

      assistant: <think>...</think><|tool_code|>RealTool(...)<|end_react|>
      tool:      <|tool_output|>...</|tool_output|>

    instead of the OOD shape:

      assistant: <think>...</think><|answer|>WRONG</|end_react|>
      tool:      <|tool_output|>...</|tool_output|>

    Behaviour:
      - If the most recent assistant_step is NOT answer-shape (e.g.
        already a tool_code, or the turn is empty), this is a no-op.
        Some verifier triggers don't come from a phase-1 answer
        (e.g. leaked-tool-call detection), and rewriting those would
        be incorrect.
      - Thought is generated by agent._synthesize_repair_thought;
        on None falls back to _REPAIR_THOUGHT_TEMPLATES[tool_name].
      - The rewritten step is logged with a [STEP-5 REWRITE] marker
        so client UIs can render the self-correction transition
        (see CLIENT_UI_NOTES.md).

    Returns True if a rewrite happened, False otherwise.
    """
    from eva_config import REACT

    turn = getattr(agent.history_manager, "current_turn", None)
    if turn is None:
        return False
    steps = getattr(turn, "assistant_steps", None)
    if not steps:
        return False

    # Find the most recent assistant step (skip any tool steps that
    # might already be in there — defensive against future history
    # shape changes).
    last_assistant_idx = None
    for i in range(len(steps) - 1, -1, -1):
        if steps[i].get("role") == "assistant":
            last_assistant_idx = i
            break
    if last_assistant_idx is None:
        return False

    last_step = steps[last_assistant_idx]
    last_content = last_step.get("content") or ""

    # Only rewrite answer-shape steps. tool_code shapes are already
    # in-distribution.
    if REACT["answer"] not in last_content:
        return False
    if REACT["tool_code"] in last_content:
        # Mixed shape — extremely unusual. Skip to be safe.
        return False

    # Synthesize the new thought.
    args_summary = _summarize_tool_args_for_thought(tool_name, tool_params)
    thought = None
    try:
        thought = agent._synthesize_repair_thought(
            latest_user_text, tool_name, args_summary,
        )
    except Exception:
        thought = None

    if not thought:
        thought = _REPAIR_THOUGHT_TEMPLATES.get(
            tool_name, _REPAIR_THOUGHT_FALLBACK,
        )

    # Build the new assistant content.
    tool_call_str = _format_tool_call_string(tool_name, tool_params)
    new_content = (
        f"<think>{thought}</think>"
        f"{REACT['tool_code']}{tool_call_str}{REACT['end']}"
    )
    steps[last_assistant_idx] = {"role": "assistant", "content": new_content}

    for line in _render_step5_rewrite_block(thought, tool_call_str):
        print(line)
    return True


# ============================================================
# eva_verifier_logic — Stage 4: execute_controller_tool
# ============================================================
def execute_controller_tool(agent, tool_name, tool_params, latest_user_text,
                            reason="controller_required_action"):
    """Execute a verifier-required tool directly and add its output to history.

    Parameters
    ----------
    agent : ChatAgent
    tool_name : str
        One of MemorySearch / WebSearch / GetCurrentTime / AskRemoteVision /
        TextGenerationTool. Anything else returns an error observation.
    tool_params : dict
        Tool-specific params. For WebSearch/MemorySearch the 'query' key
        is the most important. Sanitization runs through agent._guard_tool_call.
    latest_user_text : str
        The current turn's user query, used both for routing decisions
        (e.g. WebSearch→GetCurrentTime route correction) and for
        slot-evidence parsing on MemorySearch returns.
    reason : str
        Free-form label for the log line. Convention: pass the
        verifier reason that triggered this call (e.g.
        "missing_web_evidence_for_external_or_current_request").

    Returns
    -------
    (None, True)
        First element is reserved for a future generated-text return
        slot; second is the success flag the step_once loop checks.
    """
    # Lazy import so config hot-reload is honored.
    from eva_config import REACT, MEMORY_SLOT_FIELDS

    tool_name = str(tool_name or "").strip()
    tool_params = dict(tool_params or {})
    observation_for_model = ""
    observation_for_user = ""
    try:
        # --- Route correction: WebSearch with a time-shaped query → GetCurrentTime
        ws_query = (tool_params or {}).get("query", "") if tool_params else ""
        if (tool_name == "WebSearch"
                and (
                    agent._is_time_lookup_web_query(ws_query)
                    or agent._is_date_math_web_query(ws_query)
                    or (
                        agent._is_current_time_query(latest_user_text)
                        and not agent._is_obvious_public_fact_or_news_query(ws_query)
                        and not agent._current_external_query_needs_web(ws_query)
                    )
                    or (
                        agent._question_needs_time_arithmetic(latest_user_text)
                        and not agent._is_obvious_public_fact_or_news_query(ws_query)
                        and not agent._current_external_query_needs_web(ws_query)
                    )
                )):
            print("        | --- TOOL ROUTE CORRECTION ---")
            print(f"        | WebSearch({ws_query!r}) -> GetCurrentTime()")
            tool_name = "GetCurrentTime"
            tool_params = {}

        # --- Param sanitization / hard guard
        allow, corrected_params, blocked_obs = agent._guard_tool_call(
            tool_name, tool_params, latest_user_text
        )
        tool_params = corrected_params

        # --- STEP-5 trace rewrite (TODO 2 fix)
        # Rewrite phase-1's <|answer|>WRONG</|end_react|> step into a
        # <|tool_code|>RealTool(...)</|end_react|> step BEFORE running
        # the tool, so the tool_output that follows produces an
        # in-distribution trajectory for phase-2.
        # Idempotent + defensive: see _rewrite_assistant_for_tool_repair
        # docstring for the no-op cases (already tool_code, empty
        # turn, etc.).
        _rewrite_assistant_for_tool_repair(
            agent, tool_name, tool_params, latest_user_text, reason,
        )

        if not allow:
            observation_for_model = observation_for_user = blocked_obs

        # --- Per-tool dispatch
        elif tool_name == "GetCurrentTime":
            now = eva_config.local_now()
            agent._record_time_evidence(now)
            now_str = now.strftime("%Y-%m-%d %H:%M:%S %A")
            obs = (
                f"The current system time is: {now_str}\n"
                "[TIME BINDING]\n"
                f"- current_date: {now.strftime('%Y-%m-%d')}\n"
                f"- weekday: {now.strftime('%A')}\n"
                "[STRICT TIME RULE]: If you mention today's date in the final answer, "
                "it must match current_date above.\n"
                "[/TIME BINDING]"
            )
            calc_note = agent._maybe_compute_date_delta_from_memory()
            if calc_note:
                obs = obs + "\n" + calc_note
            observation_for_model = observation_for_user = obs

        elif tool_name == "MemorySearch":
            obs = run_memory_search(
                params=tool_params, memory_state=agent.memory_state,
                encoder=agent.encoder, reranker=agent.reranker,
                current_user=agent.history_manager.user_name,
                judge_fn=agent._apply_memory_judge_to_collection,
            )
            requested_slots = agent._extract_memory_slots(latest_user_text)
            for slot, value in agent._parse_slot_evidence_from_text(obs).items():
                agent.current_turn_slot_evidence[slot] = value
            missing_slots = [
                s for s in requested_slots
                if s in MEMORY_SLOT_FIELDS and s not in agent.current_turn_slot_evidence
            ]
            slot_note = agent._build_missing_slot_note_from_missing(missing_slots)
            # Suppress missing-slot warning when saved notes surfaced —
            # they may answer the question directly even though lore-corpus
            # slot-extraction can't read them. Same logic as eva_core's
            # MemorySearch dispatch.
            from eva_memory_legacy import memory_block_has_notes
            if slot_note and not memory_block_has_notes(obs):
                obs = obs + slot_note
            agent.current_turn_missing_slots = missing_slots
            # R-6: missing_slots 跨轮 sticky 进 last_memory；target_entity
            # 进 dialog_focus
            if requested_slots:
                agent.last_memory.missing_slots = list(missing_slots)
            observation_for_model = observation_for_user = obs
            has_exact = "[Judge: EXACT]" in obs
            has_related = "[Judge: RELATED]" in obs
            agent.current_turn_memory_grounded = bool(has_exact or has_related)
            agent.current_turn_memory_has_exact = has_exact
            agent.current_turn_memory_has_related = has_related
            if obs and (has_exact or has_related):
                target = tool_params.get("target_entity", "") or "Both"
                query = tool_params.get("query", "") or latest_user_text
                # R-6 helper：一次性写 last_memory + dialog_focus
                agent._update_memory_state_from_tool_obs(
                    obs=obs, target_entity=target, query=query, source="tool",
                )
                agent._record_memory_evidence_from_observation(
                    obs, target_entity=agent.dialog_focus.entity or target,
                    query=query,
                )

        elif tool_name == "WebSearch":
            if not tool_params.get("query"):
                tool_params["query"] = agent._build_required_web_query(latest_user_text)
            obs_dict = run_websearch(tool_params)
            observation_for_model = obs_dict["for_model"]
            observation_for_user = obs_dict["for_user"]
            agent._record_web_evidence(
                query=(tool_params or {}).get("query", ""),
                observation_for_model=observation_for_model,
                observation_for_user=observation_for_user,
            )

        elif tool_name == "AskRemoteVision":
            image_for_vision = agent._resolve_vision_image(tool_params)
            vision_result = call_remote_vision(
                tool_params.get("query", ""), image_for_vision,
                mode=tool_params.get("mode", "chat"),
            )
            observation_for_model = observation_for_user = vision_result

        elif tool_name == "ForgetMemory":
            # Verifier-injected tombstone repair. Routed here when the
            # user explicitly retracted a fact but the model didn't call
            # ForgetMemory itself.
            from Memory_maker.notes_runtime import execute_forget_memory
            notes_store = (agent.memory_state or {}).get("notes_store") if agent.memory_state else None
            obs = execute_forget_memory(notes_store, tool_params)
            observation_for_model = observation_for_user = obs

        elif tool_name == "RememberThis":
            # Verifier-injected RememberThis. Same dispatch path as the
            # P5.2 explicit_remember_request_not_handled reason.
            from Memory_maker.notes_runtime import execute_remember_this
            notes_store = (agent.memory_state or {}).get("notes_store") if agent.memory_state else None
            obs = execute_remember_this(notes_store, tool_params)
            observation_for_model = observation_for_user = obs

        elif tool_name == "TextGenerationTool":
            original_instruction = tool_params.get("instruction", "")
            neutral_instruction = agent._third_person_textgen_instruction(
                original_instruction, latest_user_text=latest_user_text
            )
            raw_text = call_deepseek_expert(neutral_instruction)
            agent._record_textgen_evidence(original_instruction, raw_text)
            observation_for_model = (
                f"### NEUTRAL THIRD-PERSON GENERATED CONTENT ###\n{raw_text}\n### END CONTENT ###\n\n"
                "[SYSTEM NOTE]: The content above is a neutral third-person draft, not Eva's final voice. "
                "Use it as factual/source material only. When converting to Eva's final answer, fix perspective: "
                "facts about Eva should become 'I/my' when Eva speaks; facts about Rosm should become 'you/your'. "
                "Do not copy any incorrect 'you play/you like' phrasing for Eva's own facts."
            )
            observation_for_user = raw_text

        else:
            observation_for_model = observation_for_user = (
                f"Error: Tool '{tool_name}' not found.\n"
                "Valid tools: MemorySearch, WebSearch, AskRemoteVision, TextGenerationTool, GetCurrentTime."
            )
    except Exception as e:
        observation_for_model = (
            f"Error executing tool '{tool_name}': {e}\n"
            "Valid tools: MemorySearch, WebSearch, AskRemoteVision, TextGenerationTool, GetCurrentTime."
        )
        observation_for_user = observation_for_model

    agent.history_manager.add_tool_output(f"{REACT['tool_output']}{observation_for_model}")
    print(f"        | --- CONTROLLER TOOL EXECUTION ({reason}) ---")
    print(f"        | --- TOOL OUTPUT ({tool_name}) ---\n        | "
          f"{observation_for_user.replace(chr(10), chr(10) + '        | ')}")
    return None, True


# ============================================================
# Underscore aliases (for ChatAgent wrapper-method delegation)
# ============================================================
# Each wrapper method in eva_core.py looks like:
#     def _foo(self, x):  return _foo_module(self, x)
# To make those wrappers clean we also export the names with leading
# underscore so eva_core can do
#     from eva_verifier_logic import _foo as _vlogic_foo
# without having to alias each one inline.
_current_turn_has_web_evidence = current_turn_has_web_evidence
_current_turn_has_memorysearch_evidence = current_turn_has_memorysearch_evidence
_current_turn_has_remember_evidence = current_turn_has_remember_evidence
_current_turn_has_forget_evidence = current_turn_has_forget_evidence
_find_recent_note_id = find_recent_note_id
_answer_mentions_days = answer_mentions_days
_answer_toy_animal_words = answer_toy_animal_words
_answer_has_eva_gaming_second_person_mismatch = answer_has_eva_gaming_second_person_mismatch
_exact_memory_evidence_for = exact_memory_evidence_for
_eva_gaming_terms_from_evidence = eva_gaming_terms_from_evidence
_expected_toy_subject_from_query = expected_toy_subject_from_query
_toy_value_words = toy_value_words
_extract_date_from_text = extract_date_from_text
_extract_leaked_tool_call = extract_leaked_tool_call
_build_required_web_query = build_required_web_query
_build_required_memory_params = build_required_memory_params
_verify_final_answer = verify_final_answer
_required_action_from_verifier_reasons = required_action_from_verifier_reasons
_safe_fallback_for_hard_verifier_failure = safe_fallback_for_hard_verifier_failure
_execute_controller_tool = execute_controller_tool
