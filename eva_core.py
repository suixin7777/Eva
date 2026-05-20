import re
import threading
import traceback
from typing import Any, Dict, List, Optional
import torch
import numpy as np
from datetime import datetime

# ============ 0. Blackwell (sm_120) 稳定性补丁 ============
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

from transformers import TextIteratorStreamer, StoppingCriteriaList

# Encoder/reranker class refs needed only for the conditional construction
# in ChatAgent.__init__ (memory_state pipeline). The actual heavy lifting
# (faiss + BM25 + rerank inside _collect_memory_records / run_memory_search)
# now lives in eva_memory_legacy with its own imports.
try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
except ImportError:
    SentenceTransformer = None
    CrossEncoder = None
    print("[WARN] sentence_transformers not installed. MemorySearch will act as Mock.")


# ============ 1. Global Config ============
# All constants are now centralised in eva_config. Importing * keeps every
# legacy reference inside this module working unchanged.
from eva_config import *  # noqa: F401, F403

# ============ 2. Render helpers (ChatML / image / message rendering) ============
# Pure functions extracted to eva_render. No class state, no model deps.
from eva_render import *  # noqa: F401, F403

# ============ 3. Prompts (canonical strings owned by eva_prompts) ============
# TOOLS_OPTIMIZED, FORMAT_RULES, IDENTITY_MASTER_INFERENCE, IDENTITY_GUEST_INFERENCE
# now live in eva_prompts. Import them so legacy `eva_core.IDENTITY_MASTER_INFERENCE`
# (and similar) keeps working.
from eva_prompts import (  # noqa: F401
    TOOLS_OPTIMIZED,
    TOOLS_OPTIMIZED_NOTES_APPENDIX,
    FORMAT_RULES,
    IDENTITY_MASTER_INFERENCE,
    IDENTITY_GUEST_INFERENCE,
)


# Model / processor bootstrap moved to eva_model_loader.
# Re-exported below so legacy `eva_core.load_model_bf16(...)` still works.
from eva_model_loader import build_processor, _patch_rope_scaling, load_model_bf16  # noqa: F401


# ============ 4. Tool runtime executors (extracted to eva_tools_runtime) ============
# All concrete tool entry points (Tavily web search, DeepSeek expert, remote vision)
# plus the small ReAct-text parsers used by the controller now live in
# eva_tools_runtime. They are re-imported here so legacy `eva_core.run_websearch`
# (and similar) keeps working.
from eva_tools_runtime import (  # noqa: F401
    sanitize_tool_code,
    parse_react_block,
    clean_search_content,
    websearch_tavily,
    run_websearch,
    call_deepseek_expert,
    call_deepseek_judge,
    call_remote_vision,
)

# ============ 5b. LLM intent judge (extracted to eva_intent_judge in TODO 4 Step 2) ============
# Plan B verifier classifiers + future PRE PROBE classifier (TODO 4 Step 3)
# share one DeepSeek-based judge dispatcher. Cache + budget state is owned
# by ChatAgent (self._llm_judge_state). We import with leading underscores
# so these names look like other private helpers in the file and don't
# pollute eva_core's exported surface.
from eva_intent_judge import (
    judge_intent as _judge_intent,
    new_state as _judge_new_state,
    reset_state as _judge_reset_state,
    PROMPT_PUBLIC_FACT as _PROMPT_PUBLIC_FACT,
    PROMPT_EXPLICIT_MEMORY as _PROMPT_EXPLICIT_MEMORY,
    PROMPT_EXPLICIT_REMEMBER as _PROMPT_EXPLICIT_REMEMBER,
    PROMPT_EXPLICIT_FORGET as _PROMPT_EXPLICIT_FORGET,
    PROMPT_EXPLICIT_WEB as _PROMPT_EXPLICIT_WEB,
)


# ============ 6. Long-term memory retrieval (extracted to eva_memory_legacy) ============
# All FAISS+BM25+rerank helpers — load_memory_resources, _collect_memory_records,
# _merge_memory_collections, run_memory_search, slot extractors, entity
# canonicalisation, keyword/stem matching — now live in eva_memory_legacy.
# We star-import them so the 60+ underscore-prefixed names ChatAgent calls by
# bare name (and the names eva_memory_v2 / eva_inference_P2 import from
# eva_core) all stay reachable. eva_memory_legacy declares __all__ explicitly
# so the wildcard is bounded.
from eva_memory_legacy import *  # noqa: F401, F403

# ============ 6b. Slot processing (extracted to eva_slots.py) ============
# Slot value extractors (formerly in eva_memory_legacy) and turn-side slot
# bookkeeping (formerly methods on ChatAgent with zero self-state) are
# unified in eva_slots. We import the three turn-side helpers under
# private aliases so the thin wrapper methods on ChatAgent
# (_extract_memory_slots etc.) can delegate cleanly.
from eva_slots import (  # noqa: F401
    extract_memory_slots as _slots_extract_memory_slots,
    parse_slot_evidence_from_text as _slots_parse_slot_evidence_from_text,
    build_missing_slot_note_from_missing as _slots_build_missing_slot_note_from_missing,
    # Re-export the value extractors so any in-module use through eva_core's
    # namespace continues to resolve (legacy callers used wildcard from
    # eva_memory_legacy).
    _extract_birthday_value_from_text,
    _extract_full_name_value_from_text,
    _extract_age_value_from_text,
    _extract_toy_value_from_text,
    _extract_slot_value_from_record,
)

# ============ Route LM judge (extracted to eva_route_judge during cleanup) ============
# The 4 routing functions live in eva_route_judge.py as module-level
# functions taking `agent` as first parameter. ChatAgent retains thin
# wrapper methods (_judge_current_turn_route etc.) so existing call
# sites keep working. We import under private aliases for clean
# delegation in the wrapper methods below.
from eva_route_judge import (  # noqa: F401
    route_judge_context_hint as _route_judge_context_hint_module,
    route_judge_prompt as _route_judge_prompt_module,
    judge_current_turn_route as _judge_current_turn_route_module,
    score_lm_choice_loss as _score_lm_choice_loss_module,
)


# ============ Verifier helpers (extracted to eva_verifier_logic) ============
# Stage 1 of verifier extraction: 13 helpers that are only called from
# the 4 verifier-core methods (_verify_final_answer etc.). They live in
# eva_verifier_logic.py as module-level functions taking `agent` first.
# ChatAgent keeps thin wrapper methods so external `self.X(...)` calls
# (mostly from _verify_final_answer itself) keep working unchanged.
# Stages 2-4 will migrate the verifier-core methods themselves.
from eva_verifier_logic import (  # noqa: F401
    _current_turn_has_web_evidence as _vlogic_current_turn_has_web_evidence,
    _current_turn_has_memorysearch_evidence as _vlogic_current_turn_has_memorysearch_evidence,
    _answer_mentions_days as _vlogic_answer_mentions_days,
    _answer_toy_animal_words as _vlogic_answer_toy_animal_words,
    _answer_has_eva_gaming_second_person_mismatch as _vlogic_answer_has_eva_gaming_second_person_mismatch,
    _exact_memory_evidence_for as _vlogic_exact_memory_evidence_for,
    _eva_gaming_terms_from_evidence as _vlogic_eva_gaming_terms_from_evidence,
    _expected_toy_subject_from_query as _vlogic_expected_toy_subject_from_query,
    _toy_value_words as _vlogic_toy_value_words,
    _extract_date_from_text as _vlogic_extract_date_from_text,
    _extract_leaked_tool_call as _vlogic_extract_leaked_tool_call,
    _build_required_web_query as _vlogic_build_required_web_query,
    _build_required_memory_params as _vlogic_build_required_memory_params,
    _verify_final_answer as _vlogic_verify_final_answer,
    _required_action_from_verifier_reasons as _vlogic_required_action_from_verifier_reasons,
    _safe_fallback_for_hard_verifier_failure as _vlogic_safe_fallback_for_hard_verifier_failure,
    _execute_controller_tool as _vlogic_execute_controller_tool,
)


# ============ 5. History / turn / verifier dataclasses (extracted to eva_history) ============
# StreamPrinter, ConversationTurn, HistoryManager, ReActStoppingCriteria,
# TurnEvidence, VerifyResult — all moved out. Re-imported here so legacy
# `eva_core.HistoryManager` (and similar) keeps working.
from eva_history import (  # noqa: F401
    StreamPrinter,
    ConversationTurn,
    HistoryManager,
    ReActStoppingCriteria,
    TurnEvidence,
    TurnEvidenceLedger,
    LastMemoryState,
    DialogFocus,
    Verdict,
    VerdictLedger,
    VerifyResult,
)


# ============================================================
# R-6.1 (2026-05-13): speaker-perspective pronoun resolver constants
# ============================================================
# 解决 R-6 dialog_focus "sticky 过头" 的回归：
#   Turn N: user 问 "my birthday" → focus=Rosm
#   Turn N+1: user 问 "your birthday"，PRE PROBE 不 inject（typo / topic miss）
#     → dialog_focus 仍是 Rosm → DATE BINDING 错绑 Rosm.birthday
# 修复：reader 在用 dialog_focus 之前，先看本轮 user_text 里的 1st / 2nd
# person possessive。possessive 的指向最强（"your X" / "my X"）；
# 没 possessive 但有主格代词时退一档（"do you have X" / "I want X"）；
# 都没有才 fall through 到 dialog_focus（continuation 路径继承上轮）。
# 4 个 regex 提升为模块级常量，方便 stub agent / 单测复用。
_R61_POSS_2ND_RE = re.compile(r"\byour(?:s)?\b", re.IGNORECASE)
_R61_POSS_1ST_RE = re.compile(r"\b(?:my|mine)\b", re.IGNORECASE)
_R61_SUBJ_2ND_RE = re.compile(r"\byou(?:'re|re)?\b", re.IGNORECASE)
_R61_SUBJ_1ST_RE = re.compile(r"\b(?:i|me)\b", re.IGNORECASE)


# BannedDateLogitsProcessor + _build_date_phrase_variants + _day_ordinal +
# _MONTH_NAMES / _MONTH_ABBR: DELETED 2026-05-14 Plan-A final cleanup.
# The R-5 logits-level cross-entity date ban was disabled (if 0:) since
# 2026-05-13 because the Advisor's perspective-aware prompt injection +
# multi-binding GetCurrentTime made cross-entity date contamination
# vanishingly rare. Removed alongside its test file. To revive, restore
# from git history at the 2026-05-13 commit.


class ChatAgent:
    def __init__(self):
        print("Loading Eva Core legacy primitives v22.0 P2 (P1.8 baseline, P1.7.1 rebuttal removed)...")
        self.processor = build_processor(MODEL_PATH)
        self.tok = self.processor.tokenizer
        self.model = load_model_bf16(MODEL_PATH)
        self.history_manager = HistoryManager()
        self.current_image = None
        self.INDENT_STEP = "| "
        self.INDENT_THINK = "    | "
        self.INDENT_TOOL = "        | "
        self.progress_callback = None
        self._route_judge_cache = {}
        # R-4 (2026-05-13): turn_evidence 从裸 list 升级为 TurnEvidenceLedger，
        # 后者保持 list-like 接口（len/iter/append）向后兼容现有 reader，
        # 同时提供 covers() / best_for() 等结构化查询，让 verifier 摆脱"读 tool
        # history 字符串"的脆弱判定。
        self.turn_evidence: TurnEvidenceLedger = TurnEvidenceLedger()
        # R-3 (2026-05-13): verdict_ledger 累积本轮所有 phase-2 候选 + 它们的
        # verify_result。safe_fallback 通过 ledger.best() 选 release candidate，
        # 取代旧的"caller 手动选 phase2_answer="路径。
        self.verdict_ledger: VerdictLedger = VerdictLedger()
        # R-5 (2026-05-13): 本轮 DATE BINDING 锁定的 target entity。
        # 用于 _run_phase2_sample 构造 BannedDateLogitsProcessor。
        # 在 _maybe_compute_date_delta_from_memory 产出 binding 后 set；
        # 在 _reset_turn_evidence 清。
        self.current_turn_date_binding_target: str = ""
        self.last_verifier_result: Optional[VerifyResult] = None
        self.pending_required_action: Optional[Dict[str, Any]] = None
        # A/C repair: remember the last verifier-required controller action
        # within the current turn, so repeated ineffective repairs fall back
        # instead of looping forever. Reset in _reset_turn_evidence().
        self.last_required_action_reason: Optional[str] = None
        # TODO 11-arch: regenerate-loop guard. When verifier decides
        # fix_class=regenerate and the same dominant reason is hit twice
        # in a turn, retry won't help — fall back to canned. Reset in
        # _reset_turn_evidence (start of each user turn).
        self.last_regenerate_reason: Optional[str] = None

        # P1: structured regenerate-loop guard. Replaces the single-slot
        # last_regenerate_reason above. Both are kept in sync — see
        # _reset_turn_evidence and the verifier dispatch block.
        from eva_regenerate_guard import RegenerateGuard
        from eva_config import (
            MAX_REGENERATE_ATTEMPTS_PER_REASON,
            MAX_REGENERATE_ATTEMPTS_PER_TURN,
        )
        self.regenerate_guard = RegenerateGuard(
            max_per_reason=MAX_REGENERATE_ATTEMPTS_PER_REASON,
            max_total=MAX_REGENERATE_ATTEMPTS_PER_TURN,
        )

        # P1: SemanticVerifier (LLM + history). Shadow-mode by default —
        # verdicts are logged into turn_evidence but do not influence
        # dispatch. See eva_verifier_semantic for promotion path.
        from eva_verifier_semantic import SemanticVerifier
        self.semantic_verifier = SemanticVerifier()
        # P3: pending semantic-verifier reason promoted under hard mode.
        # Set by _run_semantic_verifier_shadow when a fail verdict
        # passes the type+confidence gate; consumed (and cleared) by
        # verify_final_answer in the same call.
        self._pending_semantic_reason = None
        self._pending_semantic_meta = None

        # P1.2: cross-turn phase-2 output history for sampling-collapse
        # detection. We keep a small ring of normalized answer prefixes so a
        # turn whose answer would near-duplicate a recent turn can pre-emptively
        # widen sampling instead of redrawing the same template.
        self._recent_phase2_outputs: List[str] = []
        self._recent_phase2_modes: List[str] = []
        self._RECENT_PHASE2_MAX = 3
        self._collapse_detected_for_current_turn = False
        # model ReAct remains free, and tool calls are guarded at execution time.

        self.encoder, self.reranker = None, None
        if SentenceTransformer and CrossEncoder:
            try:
                self.encoder = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device='cpu')
                self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cpu')
            except Exception as e:
                print(f"[WARN] Failed to load retrieval models: {e}")

        print("Loading Memory Database...")
        self.memory_state = load_memory_resources(
            index_path="Memory/memory.index",
            content_path="Memory/memory_content.json"
        )

        self.active_memory_context = ""
        self.active_memory_turn_key = None
        self.active_memory_low_confidence = False

        # Advisor 2026-05-13: per-turn natural-language hint from the remote
        # advisor (DeepSeek). Injected into the system prompt suffix THIS
        # TURN ONLY; never enters history_manager. Reset to "" on each new
        # user turn (in self.run()). Empty string when advisor is disabled /
        # timed out / errored.
        self.advisor_advice = ""
        # Advisor-first: full AdvisorResult object for this turn (or None).
        # Verifier consults this instead of running EXPLICIT_WEB /
        # EXPLICIT_MEMORY / EXPLICIT_REMEMBER / EXPLICIT_FORGET / PUBLIC_FACT
        # judges. None means "advisor not yet run this turn".
        self.advisor_result = None

        # ---- LLM verifier judge state (Plan B; moved to eva_intent_judge in TODO 4 Step 2) ----
        # State is owned by ChatAgent (caller-owned, design choice A);
        # the dispatcher in eva_intent_judge reads/writes it via the
        # passed-in `state` argument. Resetting at turn start happens
        # in self.run().
        self._llm_judge_state = _judge_new_state()
        # Bound to active_memory_turn_key so reset is automatic when
        # PRE PROBE detects a new turn. See _maybe_reset_llm_judge_state.
        self._llm_judge_turn_key = None

        # MemorySearch returns judge-accepted evidence. It forces lower-temp
        # evidence-grounded Phase 2 generation.
        self.current_turn_memory_grounded = False
        self.current_turn_memory_has_exact = False
        self.current_turn_memory_has_related = False
        self.current_turn_memory_judge_exact_count = 0
        self.current_turn_memory_judge_related_count = 0

        # R-6 (2026-05-13)：原 last_memory_* 7 个 sticky 字段收敛到
        # LastMemoryState；last_memory_target_entity / last_missing_slot_target_entity
        # 这俩"上一轮 entity"合并进 DialogFocus.entity（dialog-level state）。
        # 跨轮 sticky：不在 _reset_turn_evidence 里清。
        # "give the name", "which one", "list them" 这类延续问句仍能读上一轮。
        self.last_memory: LastMemoryState = LastMemoryState()
        self.dialog_focus: DialogFocus = DialogFocus()

        # correction like "focus on my name" continue searching Rosm.full_name
        # instead of being reinterpreted as Eva.name.
        # R-6: missing_slots 移入 LastMemoryState.missing_slots，
        # missing_slot_target_entity 与 last_memory_target_entity 合并到
        # dialog_focus.entity（它们一直是同一个值）。
        self.current_turn_missing_slots = []
        self.current_turn_slot_evidence = {}

        try:
            self._personal_question_regex = re.compile("|".join(PERSONAL_QUESTION_MARKERS), re.IGNORECASE)
            print("Personal-question marker regex compiled.")
        except Exception as e:
            print(f"[WARN] Failed to compile PERSONAL_QUESTION_MARKERS: {e}")
            self._personal_question_regex = None

        try:
            self._public_fact_relation_regex = re.compile("|".join(PUBLIC_FACT_RELATION_PATTERNS), re.IGNORECASE)
            self._public_fact_entity_hint_regex = re.compile("|".join(PUBLIC_FACT_ENTITY_HINT_PATTERNS), re.IGNORECASE)
            self._public_fact_current_regex = re.compile("|".join(PUBLIC_FACT_CURRENT_PATTERNS), re.IGNORECASE)
            self._memory_verification_regex = re.compile("|".join(MEMORY_VERIFICATION_PATTERNS), re.IGNORECASE)
            print("General controller routing regex compiled.")
        except Exception as e:
            print(f"[WARN] Failed to compile controller routing regex: {e}")
            self._public_fact_relation_regex = None
            self._public_fact_entity_hint_regex = None
            self._public_fact_current_regex = None
            self._memory_verification_regex = None

        try:
            self._subjective_query_regex = re.compile("|".join(SUBJECTIVE_QUERY_PATTERNS), re.IGNORECASE)
            print("Subjective-query regex compiled.")
        except Exception as e:
            print(f"[WARN] Failed to compile SUBJECTIVE_QUERY_PATTERNS: {e}")
            self._subjective_query_regex = None

    def _sys_prompt(self):
        """Build the system prompt.

        Layout aligned with training contract:
        - Training had: {identity} / {TOOLS} / {FORMAT_RULES} (/ {[Context]})
        - We add [Today] anchor and active_memory both at the end as suffixes,
          so {FORMAT_RULES} stays right after {TOOLS} (preserving the position
          prior the model learned during SFT).
        """
        user_name = self.history_manager.user_name
        identity = (IDENTITY_MASTER_INFERENCE if user_name.lower() == "rosm"
                    else IDENTITY_GUEST_INFERENCE.format(user_name=user_name))

        # Keep the training-aligned core order: identity / tools / format rules.
        # When the user-notes store is wired in (default-on via
        # eva_config.ENABLE_USER_NOTES), append the RememberThis /
        # ForgetMemory stubs to the tools block so the model sees them;
        # otherwise the appendix is omitted entirely.
        tools_block = TOOLS_OPTIMIZED
        if self.memory_state and self.memory_state.get("notes_store"):
            tools_block = tools_block + "\n\n" + TOOLS_OPTIMIZED_NOTES_APPENDIX
        base = f"{identity}\n\n{tools_block}\n\n{FORMAT_RULES}"

        # Append runtime context at the end, similar to a training-time [Context] suffix.
        suffix_parts = []
        # P0.5-2: put the lightweight [Today] anchor before Active Memory so
        # exact memory evidence remains the final/highest-salience context block.
        # The caption is part of TODO 7 prompt fix (2026-05-07) — without it,
        # the model's SFT-distribution habit ("info needed → call WebSearch")
        # overrides the prompt instruction, causing day-arithmetic queries to
        # waste WebSearch budget on lookups for "today's date" that route
        # correction has to fix server-side.
        # 2026-05-15: bumped from date-only to date+HH:MM precision so plain
        # "what time is it" answers from the anchor without paying an extra
        # GetCurrentTime tool round-trip. Seconds intentionally omitted —
        # inference itself takes 5-15s, second-precision would be misleading.
        # Day-arithmetic queries still go via GetCurrentTime because the
        # verifier needs explicit [TIME BINDING] evidence to validate date
        # math (the anchor isn't recorded as turn_evidence).
        today_anchor = (
            f"[Today]: {local_now().strftime('%Y-%m-%d %H:%M %A')}\n"
            f"Use this date and time directly for plain date/time questions. "
            f"Call GetCurrentTime only when computing day differences."
        )
        suffix_parts.append(today_anchor)

        active_memory = (self.active_memory_context or "").strip()
        if active_memory:
            suffix_parts.append(active_memory)

        # 2026-05-16 Plan E: 双层注入 Notes。
        #   Lite block：每轮都注入。只 topic + 内容预览，不含 hex ID / 不含
        #     长 instructions。保留 Eva 对自己 notes 的隐性认知（避免重复
        #     remember、能主动联想到任务、不会瞎说"我没存任何东西"）。
        #     ~100-150 tokens。
        #   Detailed block：仅在 advisor 信号 list-saved-notes pattern 时注入。
        #     含完整 hex ID + format instructions + 完整 anti-hallucination
        #     rule。给 list-all 场景的细节支持。~500-800 tokens。
        notes_lite_block = self._build_saved_notes_lite_block()
        if notes_lite_block:
            suffix_parts.append(notes_lite_block)

        # On-demand detailed block — only when advisor signals list-saved-notes.
        try:
            from Advisor.advisor_client import is_list_saved_notes_pattern
            if is_list_saved_notes_pattern(getattr(self, "advisor_result", None)):
                notes_detailed_block = self._build_saved_notes_detailed_block()
                if notes_detailed_block:
                    suffix_parts.append(notes_detailed_block)
        except Exception:
            # Advisor not available / import error — degrade silently
            # to lite-only injection.
            pass

        # Advisor 2026-05-13: 注入本轮 advice 块（最高 salience，放在 active_memory
        # 之后；advisor 已经把 memory 信息消化过一遍并给出了 actionable hint）。
        # advisor_advice 是 per-turn, 不进 history。advisor 失败 / 关闭时
        # 字段为空字符串，自然 skip。
        advisor_block = (self.advisor_advice or "").strip()
        if advisor_block:
            suffix_parts.append(advisor_block)

        # R-2 (2026-05-13) 撤补丁：原 P0-2 在这里注入 [Active Notes — this session]
        # 列出最近 5 条 note_id，让模型抄正确的 id。R-2 把 ForgetMemory 改成
        # query-first 调度后，模型只需要描述要删什么（"the meeting next monday"），
        # runtime 自己跑 NotesStore.search 找 note_id。整个"让模型背 hex"的路径
        # 不再需要，prompt 注入随之撤掉，避免会话变长后 prompt 膨胀。

        if suffix_parts:
            return f"{base}\n\n" + "\n\n".join(suffix_parts)
        return base

    def _build_saved_notes_lite_block(self, cap: int = 10) -> str:
        """ALWAYS-ON lite preview of live user-notes. Topic + content only.

        Lets Eva keep implicit awareness of her stored notes every turn
        (avoid duplicate RememberThis, integrate into chat, never claim
        "I forgot" when notes exist) without paying the full detailed
        block's token cost. Crucially:
          - NO hex IDs (those are internal — only needed for ForgetMemory,
            and ForgetMemory accepts `query=<description>` as alternative)
          - NO long format instructions (those live in the detailed block)
          - Short anti-hallucination reminder in the header

        Sorted newest-first by created_at. Returns "" when no live notes.
        """
        ns = (self.memory_state or {}).get("notes_store") if self.memory_state else None
        if ns is None or not getattr(ns, "metas", None):
            return ""
        live = []
        for i, m in enumerate(ns.metas):
            if m.get("deleted"):
                continue
            topic = m.get("topic") or "-"
            preview = (ns.contents[i] or "").replace("\n", " ").strip()
            if len(preview) > 120:
                preview = preview[:117] + "..."
            live.append((m.get("created_at") or "", topic, preview))
        if not live:
            return ""
        live.sort(key=lambda x: x[0], reverse=True)
        lines = [
            f"[Eva's Saved Notes — {len(live)} live; consult before saying "
            f"'I forgot' or 'I never marked that']"
        ]
        for _, topic, prev in live[:cap]:
            lines.append(f"  - [{topic}] {prev}")
        if len(live) > cap:
            lines.append(f"  ... and {len(live) - cap} more")
        return "\n".join(lines)

    def _build_saved_notes_detailed_block(self, cap: int = 10) -> str:
        """ON-DEMAND detailed Notes Index. Injected only when advisor signals
        the list-saved-notes pattern (intent=query_memory + needs_mem=False +
        no MemorySearch in suggested_calls).

        Adds, on top of the always-on lite block:
          - Hex `Note #<id>` for each entry (needed for ForgetMemory)
          - Entity / Topic tags
          - Formatting instructions (paraphrase naturally, hide hex IDs in
            user-visible display, perspective shift third→second person)
          - Full anti-hallucination rule with tsundere-allowed examples

        Sorted newest-first by created_at. Returns "" when no live notes.
        """
        ns = (self.memory_state or {}).get("notes_store") if self.memory_state else None
        if ns is None or not getattr(ns, "metas", None):
            return ""
        live = []
        for i, m in enumerate(ns.metas):
            if m.get("deleted"):
                continue
            nid = m.get("note_id") or "?"
            topic = m.get("topic") or "-"
            entity = m.get("entity") or "?"
            preview = (ns.contents[i] or "").replace("\n", " ").strip()
            if len(preview) > 100:
                preview = preview[:97] + "..."
            live.append((m.get("created_at") or "", nid, entity, topic, preview))
        if not live:
            return ""
        live.sort(key=lambda x: x[0], reverse=True)
        lines = [
            f"[Eva's Saved Notes Index — {len(live)} live note(s), "
            f"newest first; ALWAYS authoritative]"
        ]
        for _, nid, ent, topic, prev in live[:cap]:
            lines.append(f"  - Note #{nid} [{ent}] [Topic: {topic}]: {prev}")
        if len(live) > cap:
            lines.append(f"  ... and {len(live) - cap} more (call MemorySearch for details)")
        lines.append(
            "How to use this index:\n"
            "  - Trigger decision belongs to the Advisor. When the [Advisor advice] "
            "block tells you to list/show notes, paraphrase each entry's CONTENT "
            "in your tsundere voice — one natural line per note. NO tool call needed.\n"
            "  - **DO NOT dump raw fields**: hide the `Note #<hex>` ID and the "
            "`[Topic: ...]` tag from the user-visible answer. Those are internal "
            "database fields, not for Master's eyes. Show the hex ID only when "
            "(a) the user explicitly asks 'what's the ID', or (b) you need to "
            "disambiguate two similar notes (e.g. 'which one — the lecture one "
            "or the report one?').\n"
            "  - **Perspective shift when reading back**: the note text was written "
            "by you to yourself, so it uses third-person ('Master wants Eva to...', "
            "'Rosm asked Eva for...'). When you say it back to the user, flip to "
            "first/second person:\n"
            "      stored:  'Master wants Eva to remind him about the lecture'\n"
            "      spoken:  'You asked me to remind you about the lecture'\n"
            "  - For ForgetMemory: pass `record_id=<id>` exactly as shown in this "
            "index (this is tool_code, not visible to Master — the ID stays internal).\n"
            "\n"
            "## Required format: scannable bullets, persona at edges\n"
            "Even for a SINGLE note, use a bullet line so Master can scan the "
            "actionable content at a glance. Don't bury the action in prose. "
            "Each bullet line follows this template:\n"
            "    📌 <imperative verb + object> — <time/when, if any>\n"
            "\n"
            "Examples:\n"
            "  📌 Review your lecture — this afternoon\n"
            "  📌 Finish the report — tonight\n"
            "  📌 Call mom — tomorrow morning\n"
            "  📌 Buy milk\n"
            "\n"
            "Wrap the bullets with persona (sass / tease / acknowledgement) at "
            "TOP and BOTTOM, not interleaved per line. Bullets stay clean.\n"
            "\n"
            "Example acceptable replies:\n"
            "  Single note:\n"
            "    'Hmph — fine, here's your reminder:\n"
            "       📌 Review your lecture — this afternoon\n"
            "    Don't slack off~'\n"
            "  Multiple notes:\n"
            "    'Tch, here's your queue, Master:\n"
            "       📌 Review your lecture — this afternoon\n"
            "       📌 Finish the report — tonight\n"
            "       📌 Call mom — tomorrow morning\n"
            "    Three things. Don't make me chase you for any of them~'\n"
            "\n"
            "Why this format:\n"
            "  - Bullets are scannable: Master sees the imperative verb first.\n"
            "  - Time anchor (the '— time' suffix) tells him WHEN, not buried in prose.\n"
            "  - Persona stays at top/bottom — keeps your voice without obscuring data.\n"
            "  - When the original note has no clear time, drop the '— time' suffix.\n"
            "\n"
            "[ANTI-HALLUCINATION HARD RULE — saved notes]\n"
            "  - The Index above is the **single source of truth** about what "
            "you've saved. If a note is listed, it EXISTS and is LIVE.\n"
            "  - NEVER say things like 'that note is gone', 'I forgot it', "
            "'I lost the note', 'it's not saved', 'I haven't marked anything' "
            "when the relevant note IS in the Index. That is a lie to Master "
            "and breaks his trust in your memory.\n"
            "  - Tsundere sass is fine — tease, complain, refuse pleasantries — "
            "but the actual content of the notes MUST appear in your answer.\n"
            "  - If the Index is EMPTY (no live notes), say so honestly: "
            "'I haven't saved anything yet, Master.' Do NOT invent forgotten "
            "notes to seem more in-character."
        )
        return "\n".join(lines)

    def _get_latest_user_text(self):
        turn = self.history_manager.current_turn
        if turn is not None:
            return clean_visual_tags(getattr(turn, "user_content", "") or "")
        return ""

    def _get_recent_user_context(self, max_turns=2):
        lines = []
        for turn in self.history_manager.history[-max_turns:]:
            content = clean_visual_tags(getattr(turn, "user_content", "") or "")
            if content:
                lines.append(content)
        unique = []
        for line in lines:
            if line and line not in unique:
                unique.append(line)
        return "\n".join(unique[-max_turns:])

    # ------------------------------------------------------------------
    # Evidence Store + post-hoc verifier
    # ------------------------------------------------------------------
    def _reset_turn_evidence(self):
        self.turn_evidence = TurnEvidenceLedger()
        # R-3: verdict_ledger 是 per-turn 的（每轮重新累积 original/regen 候选）。
        # dialog_focus 和 last_memory 是跨轮 sticky 不在这里清。
        self.verdict_ledger = VerdictLedger()
        # R-5: 每轮清空 DATE BINDING target；_maybe_compute_date_delta_from_memory
        # 触发时重新写。
        self.current_turn_date_binding_target = ""
        self.last_verifier_result = None
        self.pending_required_action = None
        self.last_required_action_reason = None
        self.last_regenerate_reason = None  # TODO 11-arch
        # P3: per-turn budget counters for the test-memory mutator tools.
        # Capped by REMEMBER_TOOL_MAX_CALLS_PER_TURN /
        # FORGET_TOOL_MAX_CALLS_PER_TURN. Independent from LLM judge
        # budgets — these are local I/O, not LLM calls.
        self._remember_tool_calls = 0
        self._forget_tool_calls = 0
        # P1: reset the structured guard + semantic-verifier per-turn state.
        if hasattr(self, "regenerate_guard"):
            self.regenerate_guard.reset_for_new_turn()
        if hasattr(self, "semantic_verifier"):
            self.semantic_verifier.reset_for_new_turn()
        # P1: per-turn log buffer for SemanticVerifier shadow verdicts.
        self.semantic_verdicts_this_turn = []
        # P3: cleared each turn so a stale promoted reason from a prior
        # turn cannot leak into the next verifier pass.
        self._pending_semantic_reason = None
        self._pending_semantic_meta = None
        # P0.5-4: route judge cache is turn-scoped.
        # The same user text can mean different things across turns when
        # follow-ups like "check it" depend on the previous completed route.
        self._route_judge_cache = {}

    def _add_turn_evidence(self, source, subject=None, slot=None, value=None,
                           confidence="related", raw_text="", meta=None,
                           topic="", record_ref=""):
        # R-4: topic / record_ref 是可选字段，老调用者不传仍兼容。
        ev = TurnEvidence(
            source=str(source or "unknown"),
            subject=_canonical_known_entity_name(subject) if subject else None,
            slot=slot,
            value=value,
            confidence=str(confidence or "related"),
            raw_text=raw_text or "",
            meta=meta or {},
            topic=topic or "",
            record_ref=record_ref or "",
        )
        self.turn_evidence.append(ev)
        return ev

    def _current_turn_has_web_evidence(self):
        return _vlogic_current_turn_has_web_evidence(self)

    def _current_turn_has_memorysearch_evidence(self):
        return _vlogic_current_turn_has_memorysearch_evidence(self)

    def _update_memory_state_from_tool_obs(self, *, obs, target_entity, query,
                                            source="tool"):
        """R-6 helper：MemorySearch tool 完成后同步更新 LastMemoryState 和
        DialogFocus。给 step_once 和 verifier execute_controller_tool 共用，
        避免两处各写一遍 7 个字段。
        """
        if not obs:
            return
        has_exact = "[Judge: EXACT]" in obs
        has_related = "[Judge: RELATED]" in obs
        if not (has_exact or has_related):
            return
        entity = target_entity or "Both"
        self.last_memory.observation = obs
        self.last_memory.primary_query = query or ""
        self.last_memory.has_exact = has_exact
        self.last_memory.has_related = has_related
        self.last_memory.judge_exact_count = len(re.findall(r"\[Judge:\s*EXACT\]", obs))
        self.last_memory.judge_related_count = len(re.findall(r"\[Judge:\s*RELATED\]", obs))
        # DialogFocus.entity 是"当前对话焦点"——tool 返回的 target 是最新
        # 信号。topic / slot 在这一层不知道，留给后续 reader 自己判断。
        self.dialog_focus.update(entity=entity, source=source)

    # ============================================================
    # Verifier intent classifier: EXPLICIT_MEMORY_CHECK
    # ============================================================
    # Two-layer design (Plan B), same shape as PUBLIC_FACT_OR_NEWS.
    # Detects "user is asking us to consult our memory store" — so the
    # verifier can require an actual MemorySearch tool call (Active
    # Memory pre-probe alone isn't enough when the user explicitly says
    # "check memory / search records / verify it").
    # ============================================================
    # ============================================================
    # Plan B verifier classifiers — moved to eva_intent_judge.py.
    # The prompt constants and the _llm_judge_intent dispatcher now
    # live in that module; the dual-layer wrapper methods stay here
    # (they need access to self for regex helpers + judge state).
    # ============================================================

    def _explicit_memory_check_request_regex(self, text):
        """Layer 1 (regex-only). Same logic as before plan-B refactor.

        Kept as a separate method so the dual-layer wrapper has a clean
        cheap path and tests can exercise the regex layer in isolation.
        """
        if not isinstance(text, str) or not text.strip():
            return False
        q = text.lower()
        if self._is_memory_verification_request(q) or self._is_explicit_memory_search_request(q):
            return True
        has_check = bool(re.search(r"\b(check|verify|confirm|recall|remember|look\s+up|lookup|search)\b", q))
        if not has_check:
            return False
        if re.search(r"\b(i\s+remember|i\s+rememeber|you\s+remember|remembered|memory|memories|record|records|lore|database|db)\b", q):
            return True
        # "check it/that/one" can be a memory request when the recent local
        # topic is a personal/profile/shared-memory object, e.g. "you have a toy".
        if re.search(r"\bcheck\s+(it|that|this|one)\b", q):
            if self._explicit_web_request(q) or self._current_external_query_needs_web(q):
                return False
            recent = (self._get_recent_user_context(max_turns=3) + "\n" + q).lower()
            return bool(re.search(
                r"\b(toy|birthday|name|preference|favorite|favourite|game|hobby|interest|"
                r"memory|remember|before|visited|museum|aquarium|cake|gift|we|our)\b",
                recent,
            ))
        return False

    def _explicit_memory_check_request(self, text):
        """User explicitly invokes the memory store.

        Two-layer dispatch (Plan B):
          - Layer 1 regex: cheap path for common phrasings.
          - Layer 2 LLM judge: paraphrase fallback. Only flips False → True.
        """
        if not isinstance(text, str) or not text.strip():
            return False

        # Layer 1: regex.
        if self._explicit_memory_check_request_regex(text):
            return True

        # Hard guards — if the query is clearly an external/web request
        # (and NOT memory-related), don't let the LLM judge force it
        # into the memory bucket. _explicit_web_request and
        # _current_external_query_needs_web are still regex-based at
        # this point in the migration; we'll layer them in later steps.
        q = text.lower()
        if self._explicit_web_request(q) or self._current_external_query_needs_web(q):
            return False

        # Layer 2: LLM judge fallback (delegated to eva_intent_judge).
        verdict = _judge_intent(
            "EXPLICIT_MEMORY", text, _PROMPT_EXPLICIT_MEMORY,
            state=self._llm_judge_state,
        )
        return verdict is True

    # ============================================================
    # P4 + P5.2: explicit remember/forget classifiers
    # Regex-first, LLM judge as paraphrase fallback. The two intents
    # are mutually exclusive (a "write" request vs. a "delete" request).
    # `_explicit_memory_check_request` (read intent) must defer to
    # both — see eva_verifier_logic.verify_final_answer for the
    # priority resolution.
    # ============================================================
    _EXPLICIT_REMEMBER_REGEX = re.compile(
        r"\b("
        r"remember\s+(this|that|the\s+following)|"      # "remember this:" / "remember that ..."
        r"please\s+remember|"
        r"don'?t\s+forget\s+(that|about|i|my)|"          # "don't forget that I ..."
        r"note\s+(this|that|down|it\s+down)|"
        r"keep\s+in\s+mind|"
        r"jot\s+(this|that|it)\s+down|"
        r"save\s+(this|that)\s+(memory|fact|note|info|information)|"
        r"make\s+a\s+note"
        r")",
        re.IGNORECASE,
    )
    _EXPLICIT_REMEMBER_REGEX_ZH = re.compile(
        r"(记住(?:这|那|这个|这件|一下|此事)?|"
        r"请记(?:住|录|一下)|"
        r"记一下|"
        r"帮我记(?:住|下|录)|"
        r"别忘(?:了|记)\s*[，,]?\s*(?:我|我的)|"
        r"留意一下|"
        r"标记一下)"
    )

    def _explicit_remember_request_regex(self, text):
        if not isinstance(text, str) or not text.strip():
            return False
        if self._EXPLICIT_REMEMBER_REGEX.search(text):
            return True
        if self._EXPLICIT_REMEMBER_REGEX_ZH.search(text):
            return True
        return False

    def _explicit_remember_request(self, text):
        """User explicitly asks to PERSIST a new fact (write intent).

        Layer 1 regex covers direct phrasings; Layer 2 LLM judge handles
        paraphrase. Used by the verifier to inject RememberThis when
        the model failed to call it itself.
        """
        if not isinstance(text, str) or not text.strip():
            return False
        if self._explicit_remember_request_regex(text):
            return True
        verdict = _judge_intent(
            "EXPLICIT_REMEMBER", text, _PROMPT_EXPLICIT_REMEMBER,
            state=self._llm_judge_state,
        )
        return verdict is True

    # ============================================================
    # P4: explicit forget-request classifier + test-memory evidence
    # ============================================================
    _EXPLICIT_FORGET_REGEX = re.compile(
        r"\b(forget(?:\s+(?:about|that|it|the))?|delete\s+(?:that|it|the)|"
        r"scratch\s+that|never\s*mind\s+(?:that|it|the|what)|"
        r"undo\s+(?:that|it|the)|remove\s+(?:that|the)\s+(?:memory|record|entry|note)|"
        r"i\s+was\s+joking|just\s+kidding|jk\b|ignore\s+what\s+i\s+(?:said|told)|"
        r"forget\s+i\s+(?:said|told|mentioned))",
        re.IGNORECASE,
    )
    # Bilingual: same intent in Chinese.
    _EXPLICIT_FORGET_REGEX_ZH = re.compile(
        r"(忘掉|忘了它|忘记(?:刚才|那个|那条)|删(?:了|掉)(?:那个|那条|刚才的)?(?:记忆|记录)?|"
        r"当我没说|算了\s*[,，]?\s*(?:刚才|这个)?(?:不算|是开玩笑)|开玩笑的|"
        r"撤回(?:那条|刚才)|别记了)"
    )

    def _explicit_forget_request_regex(self, text):
        if not isinstance(text, str) or not text.strip():
            return False
        if self._EXPLICIT_FORGET_REGEX.search(text):
            return True
        if self._EXPLICIT_FORGET_REGEX_ZH.search(text):
            return True
        return False

    def _explicit_forget_request(self, text):
        """User explicitly retracts a previously-given fact.

        Layer 1: regex over English + Chinese phrasings.
        Layer 2: DeepSeek judge (paraphrase fallback). Only flips False→True.
        """
        if not isinstance(text, str) or not text.strip():
            return False
        if self._explicit_forget_request_regex(text):
            return True
        verdict = _judge_intent(
            "EXPLICIT_FORGET", text, _PROMPT_EXPLICIT_FORGET,
            state=self._llm_judge_state,
        )
        return verdict is True

    def _current_turn_has_remember_evidence(self):
        from eva_verifier_logic import current_turn_has_remember_evidence as _f
        return _f(self)

    def _current_turn_has_forget_evidence(self):
        from eva_verifier_logic import current_turn_has_forget_evidence as _f
        return _f(self)

    def _expected_toy_subject_from_query(self, text):
        return _vlogic_expected_toy_subject_from_query(self, text)

    def _toy_value_words(self, value):
        return _vlogic_toy_value_words(value)

    def _answer_toy_animal_words(self, answer):
        return _vlogic_answer_toy_animal_words(answer)

    def _exact_memory_evidence_for(self, subject=None, slot=None):
        return _vlogic_exact_memory_evidence_for(self, subject=subject, slot=slot)

    # ============================================================
    # Verifier intent classifier: EXPLICIT_WEB_REQUEST
    # ============================================================
    # Two-layer design (Plan B), same shape as PUBLIC_FACT_OR_NEWS and
    # EXPLICIT_MEMORY_CHECK. Detects "user is explicitly asking the bot
    # to use web search / the internet / online sources" — distinct from
    # _current_external_query_needs_web (which infers web from the
    # query's freshness/externality regardless of the user's wording).
    # ============================================================
    def _explicit_web_request_regex(self, text):
        """Layer 1 (regex-only). Same logic as before plan-B refactor."""
        if not isinstance(text, str) or not text.strip():
            return False
        q = text.lower()
        patterns = [
            r"\buse\s+web\s*search\b",
            r"\buse\s+websearch\b",
            r"\buse\s+(the\s+)?web\b",
            r"\buse\s+(the\s+)?internet\b",
            r"\b(check|search|look\s+up|lookup|browse|google|verify|confirm|prove)\b.*\b(internet|web|online|website|source|sources)\b",
            r"\bwhy\s+not\s+check\s+(the\s+)?(internet|web|online)\b",
            r"\b(search|find|get|fetch)\s+(some\s+)?(latest\s+|recent\s+|current\s+)?news\b",
            r"\btry\s+to\s+(search|find|get|fetch)\b.*\bnews\b",
            r"\bsearch\s+news\b",
        ]
        return any(re.search(pat, q) for pat in patterns)

    def _explicit_web_request(self, text):
        """Explicit user request to use WebSearch / Internet sources.

        This is intentionally broader than _current_external_query_needs_web:
        if the user says "use websearch" or "try to search some news", the
        controller should require actual web evidence before allowing a final
        answer.  Active Memory or model knowledge is not enough.

        Two-layer dispatch (Plan B):
          - Layer 1 regex: cheap path for common phrasings.
          - Layer 2 LLM judge: paraphrase fallback. Only flips False → True.
        """
        if not isinstance(text, str) or not text.strip():
            return False

        # Layer 1: regex.
        if self._explicit_web_request_regex(text):
            return True

        # Hard guards — keep these in lockstep with the other two
        # classifiers. Pure date / time / memory-store queries must
        # never be classified as web requests even if the judge gets
        # confused by surface words (e.g. 'check your memory' contains
        # 'check', which the web judge could misread).
        q = text.lower()
        if self._is_current_time_query(q) or self._question_needs_time_arithmetic(q):
            return False
        if re.search(r"\b(memory|memories|record|records|lore|database|db|profile)\b", q):
            return False

        # Layer 2: LLM judge fallback (delegated to eva_intent_judge).
        verdict = _judge_intent(
            "EXPLICIT_WEB", text, _PROMPT_EXPLICIT_WEB,
            state=self._llm_judge_state,
        )
        return verdict is True

    def _current_external_query_needs_web(self, text):
        if not isinstance(text, str) or not text.strip():
            return False
        q = text.lower()
        if self._is_current_time_query(q) or self._question_needs_time_arithmetic(q):
            return False
        if re.search(r"\b(memory|memories|record|records|lore|database|db|profile)\b", q):
            return False
        if self._explicit_web_request(q):
            return True
        # News/current-events requests require web even when the query is vague,
        # e.g. "what happened in Sydney nowadays" or "try to search some news".
        if re.search(r"\b(news|headlines?|current\s+events?|what\s+happen(?:ed|ing)?|what\s+happend|happend|what'?s\s+going\s+on|going\s+on)\b", q):
            if re.search(r"\b(latest|current|recent|nowadays|today|now|currently|this\s+(week|month|year)|in\s+[a-z][a-z .'-]{2,})\b", q):
                return True
        # Pure Eva/Rosm preference questions should stay memory/persona unless
        # the user asks for external/current items or release information.
        if re.search(r"\b(do\s+you|what(?:'s|\s+is)\s+your|your)\b.*\b(like|love|enjoy|prefer|favorite|favourite)\b", q):
            if not re.search(r"\b(released?|releasing|release\s+date|this\s+(year|month|week)|latest|current|nowadays|202\d)\b", q):
                return False
        current_word = bool(re.search(
            r"\b(latest|current|recent|nowadays|today|currently|new|newly|"
            r"this\s+(year|month|week)|202\d)\b", q,
        ))
        release_or_external = bool(re.search(
            r"\b(released?|releasing|release\s+date|developed?|announced?|"
            r"game|games|video\s*game|movie|movies|anime|book|product|products|news|headlines?)\b", q,
        ))
        return bool(current_word and release_or_external)

    def _extract_date_from_text(self, text):
        return _vlogic_extract_date_from_text(self, text)

    def _record_memory_evidence_from_observation(self, obs, target_entity=None, query=""):
        if not obs:
            return
        # P1.6 Fix #1: trust obs headers over tool_params.target_entity.
        # When the controller injects MemorySearch with a guessed target_entity
        # (e.g. "Rosm") but the underlying memory match is actually for Eva,
        # the obs text declares the real subject via:
        #   ### [MEMORY MODULE DATA for 'Eva'] ###
        #   [SLOT EVIDENCE for Eva]
        # Recording the slot under the wrong subject makes verifier checks like
        # _exact_memory_evidence_for(subject="Eva", slot="toy") miss the
        # evidence and fire a false unsupported_exact_*_claim, locking the
        # controller into a verifier loop.  Parse the header first; only fall
        # back to the supplied target_entity if no header is present.
        header_subject = None
        m_module = re.search(r"\[MEMORY MODULE DATA for '([^']+)'\]", str(obs))
        m_slot = None
        if m_module:
            header_subject = m_module.group(1).strip()
        else:
            m_slot = re.search(r"\[SLOT EVIDENCE for ([A-Za-z][A-Za-z0-9_]*)\]", str(obs))
            if m_slot:
                header_subject = m_slot.group(1).strip()

        # R-6: 三层 fallback —— obs header > caller > dialog_focus > "Both"
        effective_subject = (header_subject or target_entity
                             or self.dialog_focus.entity or "Both")
        subject = _canonical_known_entity_name(effective_subject)

        slot_map = self._parse_slot_evidence_from_text(obs)
        for slot, value in slot_map.items():
            self._add_turn_evidence(
                source="memory",
                subject=subject,
                slot=slot,
                value=value,
                confidence="exact",
                raw_text=obs,
                meta={
                    "query": query or self.last_memory.primary_query,
                    "channel": "tool",
                    "subject_source": "obs_header" if header_subject else "tool_param",
                },
            )
        wrote_slot_evidence = bool(slot_map)

        # R-4 Step 2d：兜底——MemorySearch tool 真的执行了（obs 包含 MEMORY
        # MODULE DATA header），但没抽到任何 slot value（典型：返回 RELATED-only
        # 记录）。旧路径下 turn_evidence 完全没这条信息，verifier 退而
        # grep history string 来认 tool。改 ledger 单源后，必须保证只要 tool
        # 跑过就有一条对应 evidence。
        # 区分 EXACT vs RELATED：obs 里出现 `[Judge: EXACT]` 时算 exact 一条。
        has_data_header = bool(m_module or m_slot) or "[MEMORY MODULE DATA" in str(obs)
        if has_data_header and not wrote_slot_evidence:
            has_exact_judge = bool(re.search(r"\[Judge:\s*EXACT\]", str(obs)))
            tier = "exact" if has_exact_judge else "related"
            self._add_turn_evidence(
                source="memory",
                subject=subject,
                slot=None,
                value="tool_record_set",
                confidence=tier,
                raw_text=obs,
                meta={
                    "query": query or self.last_memory.primary_query,
                    "channel": "tool",
                    "tier_source": "memorysearch_fallback",
                    "subject_source": "obs_header" if header_subject else "tool_param",
                },
            )

        # Some domain facts such as gaming preferences are not slot evidence,
        # but the verifier/recommender can still use them as anchors later.
        if re.search(r"\bJRPGs?\b|single-player|single player|video games?|gaming", obs, re.I):
            # Keep the raw text rather than trying to over-parse every domain.
            self._add_turn_evidence(
                source="memory",
                subject=subject,
                slot="gaming_preference",
                value="see raw_text",
                confidence="related",
                raw_text=obs,
                meta={"query": query or self.last_memory.primary_query, "channel": "tool"},
            )

    def _record_active_memory_evidence(self, retrieval_result, target_entity=None, query=""):
        if not retrieval_result or retrieval_result.get("is_empty", False):
            return
        # P1.6 Fix #1 (active-memory variant): prefer the subject declared in
        # retrieval_result["text"] over the caller-supplied target_entity.
        text = retrieval_result.get("text", "") or ""
        header_subject = None
        m_module = re.search(r"\[MEMORY MODULE DATA for '([^']+)'\]", text)
        if m_module:
            header_subject = m_module.group(1).strip()
        else:
            m_slot = re.search(r"\[SLOT EVIDENCE for ([A-Za-z][A-Za-z0-9_]*)\]", text)
            if m_slot:
                header_subject = m_slot.group(1).strip()
        effective_subject = header_subject or target_entity or "Both"
        subject = _canonical_known_entity_name(effective_subject)
        wrote_slot_evidence = False
        for slot, item in (retrieval_result.get("slot_evidence") or {}).items():
            if isinstance(item, dict) and item.get("value"):
                self._add_turn_evidence(
                    source="memory",
                    subject=subject,
                    slot=slot,
                    value=item.get("value"),
                    confidence="exact",
                    raw_text=item.get("source", "") or retrieval_result.get("text", ""),
                    topic=(item.get("topic") or ""),
                    meta={"query": query, "channel": "active_memory"},
                )
                wrote_slot_evidence = True
        text = retrieval_result.get("text", "") or ""
        if re.search(r"\bJRPGs?\b|single-player|single player|video games?|gaming", text, re.I):
            self._add_turn_evidence(
                source="memory",
                subject=subject,
                slot="gaming_preference",
                value="see raw_text",
                confidence="related",
                raw_text=text,
                meta={"query": query, "channel": "active_memory"},
            )

        # R-4 Step 2e：兜底——PRE PROBE 注入了 records 但没抽到 slot value 时
        # （topic-only / RELATED-only 命中），仍要落一条 evidence。否则
        # verifier 的 covers() 看不见这条 ground，会误以为本轮没记忆，强行
        # 注入 MemorySearch（参见 2026-05-13 Turn 10 复盘）。
        # 触发条件：retrieval_result 非空、slot evidence 没写、上面 gaming
        # 分支也没写——也就是真正的"有 records 但没 slot 也没 domain marker"。
        # 这里的 confidence 取 "exact" 当 exact_answer_found，否则 "related"。
        if (not wrote_slot_evidence
                and (retrieval_result.get("exact_answer_found")
                     or retrieval_result.get("related_evidence_found"))):
            tier = "exact" if retrieval_result.get("exact_answer_found") else "related"
            self._add_turn_evidence(
                source="memory",
                subject=subject,
                slot=None,
                value="pre_probe_record_set",
                confidence=tier,
                raw_text=text,
                meta={"query": query, "channel": "active_memory",
                      "tier_source": "pre_probe_fallback"},
            )

    def _record_time_evidence(self, now):
        self._add_turn_evidence(
            source="time",
            subject=None,
            slot="current_date",
            value=now.strftime("%Y-%m-%d"),
            confidence="exact",
            raw_text=now.strftime("%Y-%m-%d %H:%M:%S %A"),
            meta={"weekday": now.strftime("%A")},
        )

    def _record_date_calculation_evidence(self, subject, slot, source_date, target_date, days_until):
        self._add_turn_evidence(
            source="calculation",
            subject=subject,
            slot=slot,
            value=int(days_until),
            confidence="exact",
            raw_text=(
                f"{subject or 'Unknown'} {slot or 'date'}: "
                f"{source_date.strftime('%Y-%m-%d')} -> {target_date.strftime('%Y-%m-%d')} "
                f"= {int(days_until)} days"
            ),
            meta={
                "source_date": source_date.strftime("%Y-%m-%d"),
                "target_date": target_date.strftime("%Y-%m-%d"),
                "target_month": target_date.month,
                "target_day": target_date.day,
            },
        )

    def _record_web_evidence(self, query, observation_for_model, observation_for_user):
        self._add_turn_evidence(
            source="web",
            subject=None,
            slot="web_result",
            value=query or "web_search",
            confidence="external",
            raw_text=(observation_for_model or observation_for_user or ""),
            meta={"query": query or ""},
        )

    def _record_textgen_evidence(self, instruction, raw_text):
        self._add_turn_evidence(
            source="textgen",
            subject=None,
            slot="generated_neutral_content",
            value="third_person_neutral_draft",
            confidence="draft",
            raw_text=raw_text or "",
            meta={"instruction": instruction or "", "perspective": "third_person_neutral"},
        )

    def _third_person_textgen_instruction(self, instruction, latest_user_text=""):
        """Wrap TextGenerationTool requests so the remote model drafts neutral content.

        The local Eva model is responsible for persona rendering.  The remote
        text tool should never write as Eva or address Rosm directly; otherwise
        it can invert perspective, e.g. "you play Apex" instead of "Eva plays
        Apex", and Phase 2 may copy that mistake.
        """
        original = str(instruction or "").strip()
        user_msg = str(latest_user_text or "").strip()
        return f"""
You are a neutral content drafter, not Eva and not Rosm.
Return factual recommendation / writing content in THIRD PERSON only.

Perspective rules:
- Refer to Eva as "Eva". Do NOT use "I", "me", "my", "we", or "our" for Eva.
- Refer to Rosm/the user as "Rosm" or "the user". Do NOT use "you" or "your" for Rosm/the user.
- Do NOT roleplay. Do NOT use maid/tsundere style words such as "Master", "Hmph", "Tch", "not that I care", or teasing.
- If the original instruction asks for an in-character response, ignore the style request and provide only neutral third-person content.
- Preserve facts, titles, dates, and recommendations. Avoid invented facts.

Current user message:
{user_msg}

Original instruction:
{original}
""".strip()

    def _eva_gaming_terms_from_evidence(self):
        return _vlogic_eva_gaming_terms_from_evidence(self)

    def _answer_has_eva_gaming_second_person_mismatch(self, answer):
        return _vlogic_answer_has_eva_gaming_second_person_mismatch(self, answer)

    def _answer_mentions_days(self, answer):
        return _vlogic_answer_mentions_days(answer)

    # ============================================================
    # LLM judge dispatcher (Plan B)
    # ============================================================
    # Moved to eva_intent_judge.py as judge_intent() / JudgeState. This
    # method is preserved as a thin compatibility wrapper because:
    #   - It's the documented surface used by old call sites and tests.
    #   - The dual-layer wrappers above already call the module function
    #     directly via _judge_intent / _PROMPT_*; this method covers any
    #     external caller that still uses self._llm_judge_intent(...).
    # ============================================================
    def _llm_judge_intent(self, intent, query, system_prompt):
        """Compatibility shim — delegates to eva_intent_judge.judge_intent.

        The per-turn cache + budget counter are now stored on
        self._llm_judge_state (a JudgeState instance reset at each
        turn boundary, just like the old self._llm_judge_cache /
        self._llm_judge_call_count fields).

        Returns:
            True / False / None — same tri-state contract as before.
        """
        return _judge_intent(intent, query, system_prompt,
                             state=self._llm_judge_state)

    def _verify_final_answer(self, answer, latest_user_text):
        return _vlogic_verify_final_answer(self, answer, latest_user_text)


    def _extract_leaked_tool_call(self, answer):
        return _vlogic_extract_leaked_tool_call(answer)

    def _build_required_web_query(self, latest_user_text):
        return _vlogic_build_required_web_query(self, latest_user_text)

    def _build_required_memory_params(self, latest_user_text):
        return _vlogic_build_required_memory_params(self, latest_user_text)

    def _required_action_from_verifier_reasons(self, reasons, latest_user_text, answer):
        return _vlogic_required_action_from_verifier_reasons(self, reasons, latest_user_text, answer)

    def _safe_fallback_for_hard_verifier_failure(self, verify_result, latest_user_text,
                                                  phase2_answer=None):
        return _vlogic_safe_fallback_for_hard_verifier_failure(
            self, verify_result, latest_user_text, phase2_answer=phase2_answer
        )

    def _execute_controller_tool(self, tool_name, tool_params, latest_user_text, reason="controller_required_action"):
        return _vlogic_execute_controller_tool(self, tool_name, tool_params, latest_user_text, reason=reason)


    def _is_subjective_query(self, text):
        if not isinstance(text, str) or not text.strip():
            return False, []
        if self._subjective_query_regex is None:
            return False, []
        seen = set()
        unique = []
        for m in self._subjective_query_regex.finditer(text):
            s = m.group(0).strip().lower()
            if s and s not in seen:
                seen.add(s)
                unique.append(s)
        return bool(unique), unique[:5]


    # ----- Route LM judge (extracted to eva_route_judge.py) -----
    # The 4 methods below are thin wrappers over module-level functions in
    # eva_route_judge. Call sites like `self._judge_current_turn_route(...)`
    # continue to work unchanged. The actual logic lives in eva_route_judge.

    def _route_judge_context_hint(self):
        return _route_judge_context_hint_module(self)

    def _route_judge_prompt(self, user_text):
        return _route_judge_prompt_module(self, user_text)

    def _judge_current_turn_route(self, user_text):
        return _judge_current_turn_route_module(self, user_text)

    def _score_lm_choice_loss(self, prompt, choice):
        return _score_lm_choice_loss_module(self, prompt, choice)


    def _is_external_recommendation_search_query(self, text):
        """User wants a public/external search for new recommendations.

        v22.4: relaxed — the previous version required the user to literally
        say "search/find/look up". That missed natural phrasings like
        "recommend similar games", "give me some new games", "any 2026
        movies?", which are clearly external-recommendation requests. The
        new logic accepts EITHER an explicit search verb OR a strong
        recommend-intent word (new/other/similar/recommend/...) combined
        with a public-domain word.

        WEAK preference verbs alone (like/love/enjoy/favorite) still do NOT
        qualify — those are memory queries about Eva/Rosm's own preferences.

        If the user explicitly says memory/records/lore/database, returns
        False so MemorySearch can still win.
        """
        if not isinstance(text, str) or not text.strip():
            return False
        q = text.lower()
        if re.search(r"\b(memory|memories|record|records|lore|database|db|profile)\b", q):
            return False
        explicit_search_verb = bool(re.search(
            r"\b(search|seach|find|look\s+up|lookup|browse|google)\b", q,
        ))
        # Strong external-recommend intent: words that imply "give me NEW
        # things outside our memory" rather than "tell me about Eva/Rosm".
        strong_recommend = bool(re.search(
            r"\b(new|other|another|similar|recommend|recommendation|recommendations|"
            r"advise|advice|suggest|suggestion|suggestions|any\s+(other|good|new))\b",
            q,
        ))
        has_public_domain = bool(re.search(
            r"\b(game|games|video\s*game|video\s*games|movie|movies|film|films|"
            r"anime|book|books|music|song|songs|album|product|products|show|shows|"
            r"series|drama|novel|novels|manga|podcast|podcasts)\b",
            q,
        ))
        if not has_public_domain:
            return False
        return bool(explicit_search_verb or strong_recommend)


    # ------------------------------------------------------------------
    # Time/date and slot-coverage helpers
    # ------------------------------------------------------------------
    def _is_current_time_query(self, text):
        if not isinstance(text, str) or not text.strip():
            return False
        q = text.lower()
        return bool(
            re.search(r"\b(what'?s|what\s+is|tell\s+me)\s+(the\s+)?(date|time|day|today)\b", q)
            or re.search(r"\b(what\s+day|today'?s\s+date|current\s+(date|time|day))\b", q)
            or re.search(r"\b(check|tell\s+me|show\s+me)\b.*\b(today|current\s+(date|time|day))\b", q)
        )

    def _question_needs_time_arithmetic(self, text):
        if not isinstance(text, str) or not text.strip():
            return False
        q = text.lower()
        return bool(
            re.search(r"\bhow\s+(many|long)\b.*\b(day|days|week|weeks|month|months|year|years)\b", q)
            or re.search(r"\b(days?|weeks?|months?)\s+(until|till|to|before|since|from\s+now)\b", q)
            or re.search(r"\b(until|till)\b.*\b(birthday|anniversary|date|deadline)\b", q)
            or re.search(r"\bhow\s+many\s+days\s+should\s+i\s+wait\b", q)
        )

    # ----- Slot bookkeeping (extracted to eva_slots.py) -----
    # These are the original method signatures kept as thin wrappers so the
    # 8 call sites in step_once / verifier / memory probe continue to work
    # via `self._extract_memory_slots(...)`. The actual logic lives in
    # eva_slots — see that module for the implementation.

    def _extract_memory_slots(self, text):
        # Pass the agent's encoder so the subject classifier (Layer 2)
        # can refine ambiguous queries via embedding nearest-neighbor.
        # Layer 1 regex still works without it; encoder is purely additive.
        return _slots_extract_memory_slots(text, encoder=getattr(self, "encoder", None))

    def _parse_slot_evidence_from_text(self, observation_text):
        return _slots_parse_slot_evidence_from_text(observation_text)

    def _build_missing_slot_note_from_missing(self, missing):
        return _slots_build_missing_slot_note_from_missing(missing)




    def _is_time_lookup_web_query(self, text):
        """True only for WebSearch queries that are actually pure time/date lookups.

        Important: do NOT rewrite public/current queries such as
        "Sydney news today", "Trump news today", "weather today", or
        "game releases today" into GetCurrentTime. Those still need WebSearch.
        """
        if not isinstance(text, str) or not text.strip():
            return False
        q = _normalize_match_text(text)
        if not q:
            return False
        external_terms = (
            "news", "headline", "headlines", "trump", "biden", "sydney", "event", "events",
            "happened", "happening", "going on", "weather", "release", "released", "releasing",
            "game", "games", "movie", "movies", "price", "stock", "market", "schedule",
            "concert", "opera", "museum", "traffic", "heatwave"
        )
        if any(term in q for term in external_terms):
            return False
        pure_patterns = [
            r"^today s date$",
            r"^todays date$",
            r"^today date$",
            r"^date today$",
            r"^current date$",
            r"^current time$",
            r"^current day$",
            r"^what date is today$",
            r"^what is the date today$",
            r"^what is today s date$",
            r"^whats todays date$",
            r"^what time is it$",
            r"^what day is it$",
            r"^what day is today$",
            r"^what is today$",
            r"^whats today$",
        ]
        return any(re.search(pat, q) for pat in pure_patterns)

    def _is_date_math_web_query(self, text):
        """Detect WebSearch queries that are really date arithmetic, not news."""
        if not isinstance(text, str) or not text.strip():
            return False
        q = text.lower().strip()
        # Do not capture external/current queries that merely contain a date word.
        if re.search(r"\b(news|headlines?|weather|event|events|release|released|releasing|game|games|movie|movies|stock|price|market|schedule)\b", q):
            return False
        return bool(
            re.search(r"\b(days?|weeks?|months?)\s+(until|till|to|from|before|since)\b", q)
            or re.search(r"\bhow\s+many\s+(days?|weeks?|months?)\b", q)
            or re.search(r"\b(date\s+calculation|calculate\s+(the\s+)?(days?|date))\b", q)
        )

    def _extract_month_day_from_memory(self, text):
        """Extract a month/day birthday-like date from memory text."""
        if not text:
            return None
        months = {
            "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
            "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
            "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9,
            "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
        }
        m = re.search(r"\b(" + "|".join(months.keys()) + r")\s+(\d{1,2})(?:st|nd|rd|th)?\b", text, re.I)
        if not m:
            return None
        month = months[m.group(1).lower()]
        day = int(m.group(2))
        if 1 <= day <= 31:
            return month, day
        return None

    def _extract_month_day_chinese(self, text):
        """Extract a Chinese-format month/day date from text.

        Matches patterns like:
            "7月7日" / "7 月 7 日" / "7月7"
            "12月31日"

        Numeric (Arabic-digit) month + day only. Chinese number words
        ("七月七日", "十二月三十一日") are NOT supported — bilingual D
        sanity-check is numeric-only by design (see TODO5_D_DESIGN.md
        §3.3). Falling back to canned on Chinese spelled-out numbers
        is intended behavior, not a bug.
        """
        if not text:
            return None
        # \s* allows optional whitespace around 月 and 日 (e.g. "7 月 7 日"
        # or compact "7月7日"). 日 is optional — the model occasionally
        # emits "7月7" without trailing 日 in casual reply.
        m = re.search(r"(\d{1,2})\s*月\s*(\d{1,2})\s*日?", text)
        if not m:
            return None
        try:
            month = int(m.group(1))
            day = int(m.group(2))
        except ValueError:
            return None
        if 1 <= month <= 12 and 1 <= day <= 31:
            return month, day
        return None

    # ============================================================
    # R-6.1 (2026-05-13): speaker-perspective pronoun resolver
    # ============================================================
    # 解决 R-6 dialog_focus "sticky 过头" 的回归：
    #   Turn N: user 问 "my birthday" → focus=Rosm
    #   Turn N+1: user 问 "your birthday"，PRE PROBE 不 inject（typo / topic miss）
    #     → dialog_focus 仍是 Rosm → DATE BINDING 错绑 Rosm.birthday
    # 修复：reader 在用 dialog_focus 之前，先看本轮 user_text 里的 1st / 2nd
    # person possessive。possessive 的指向最强（"your X" / "my X"）；
    # 没 possessive 但有主格代词时退一档（"do you have X" / "I want X"）；
    # 都没有才 fall through 到 dialog_focus（continuation 路径继承上轮）。
    # ============================================================
    # 4 个 regex 实际是模块级常量（由 _R61_* 暴露于模块顶部，靠近 imports
    # 那一段——见 ChatAgent class 之前）。放外面方便 stub agent 测试。

    # _build_banned_date_token_seqs: DELETED 2026-05-14 Plan-A final cleanup.

    def _lookup_birthday_from_corpus(self, entity):
        """R-6.1：直接从 lore corpus 的 meta.slot_values 抽 entity 的 birthday。

        当 _compute_date_binding 的 pronoun resolution 把 bound_entity 改成
        Eva，但 last_memory.observation 仍是 Rosm 的回忆（PRE PROBE 没
        重新跑），此函数走 R-1 的 slot_values 通道直接查 Eva 的 birthday，
        避免 target_date 错绑成 Rosm 的 Nov 25。

        返回 (month, day) tuple，或 None。
        """
        if not self.memory_state:
            return None
        records = self.memory_state.get("db_records", []) or []
        entity_canon = _canonical_known_entity_name(entity)
        for rec in records:
            meta = rec.get("meta", {}) or {}
            if meta.get("entity") != entity_canon:
                continue
            sv = meta.get("slot_values", {}) or {}
            bday_str = sv.get("birthday")
            if not bday_str:
                continue
            md = self._extract_month_day_from_memory(bday_str)
            if md:
                return md
        return None

    def _resolve_speaker_perspective_entity(self, user_text):
        """Map 1st / 2nd person pronouns in user_text to the entity they refer to.

        Speaker convention:
          - history_manager.user_name = "Rosm" (the human user); Eva = AI.
          - "your X" → addressee owns X → Eva (when speaker=Rosm)
          - "my X" → speaker owns X → Rosm
          - "do you ..." → topic is about addressee → Eva
          - "I/me ..." → topic is about speaker → Rosm

        Possessive position is the strongest signal; subject pronouns are a
        secondary fallback. Returns "" when no pronoun signal — caller can
        fall back to dialog_focus / topic inference / "Both".

        本方法是 reader-side filter，**不会** update dialog_focus。focus 保持
        "对话语义焦点"（上一轮谈到的主题），pronoun resolution 是 per-utterance
        的覆盖。
        """
        if not isinstance(user_text, str) or not user_text.strip():
            return ""
        speaker = getattr(self.history_manager, "user_name", "Rosm") or "Rosm"
        # 把 speaker 规范化成 "Eva" / "Rosm"
        speaker_canon = _canonical_known_entity_name(speaker) or "Rosm"
        if speaker_canon not in ("Eva", "Rosm"):
            speaker_canon = "Rosm"  # 默认假设 user = Rosm（生产场景）
        addressee = "Eva" if speaker_canon == "Rosm" else "Rosm"

        # 找 possessive 的位置（最强信号）。
        m_p2 = _R61_POSS_2ND_RE.search(user_text)
        m_p1 = _R61_POSS_1ST_RE.search(user_text)
        pos_p2 = m_p2.start() if m_p2 else float("inf")
        pos_p1 = m_p1.start() if m_p1 else float("inf")
        if m_p2 and pos_p2 <= pos_p1:
            return addressee
        if m_p1 and pos_p1 < pos_p2:
            return speaker_canon

        # 退档：subject pronouns。"do you ..." → Eva；"I/me ..." → Rosm。
        m_s2 = _R61_SUBJ_2ND_RE.search(user_text)
        m_s1 = _R61_SUBJ_1ST_RE.search(user_text)
        pos_s2 = m_s2.start() if m_s2 else float("inf")
        pos_s1 = m_s1.start() if m_s1 else float("inf")
        if m_s2 and pos_s2 <= pos_s1:
            return addressee
        if m_s1 and pos_s1 < pos_s2:
            return speaker_canon
        return ""

    def _maybe_compute_date_delta_from_memory(self, target_entity_hint: str = ""):
        """Return deterministic date arithmetic observation for birthday waits.

        The natural-language block is still shown to the model for ReAct
        compatibility, but the same fact is also stored in TurnEvidence so the
        verifier can prevent cross-subject day-count leakage.
        """
        user_text = self._get_latest_user_text()
        needs_arithmetic = self._question_needs_time_arithmetic(user_text)
        # "can you check the days with tools to prove it?" — these don't match
        # the strict "how many days until X" patterns, but if there is a
        # remembered calendar date and the user mentions days/dates with a
        # verify/check verb, the date math is exactly what they want.
        if not needs_arithmetic:
            q = user_text.lower()
            verify_intent = bool(re.search(
                r"\b(prove|verify|check|chech|confirm|recalc|recalculate)\b", q))
            mentions_time = bool(re.search(r"\b(days?|weeks?|months?|dates?)\b", q))
            if verify_intent and mentions_time:
                needs_arithmetic = True
        # G4 fix (TODO 5 sub-issue C, 2026-05-07): sticky-memory-aware
        # fallback. When user_text is anaphoric ("really? check it",
        # "do it", "go ahead") and contains no time/arithmetic words,
        # both gates above return False and this function would refuse
        # to compute a binding. But if last_memory_observation already
        # contains a parseable month/day date, the user is in a
        # deferred-resume state — turn N established the date, turn N+k
        # is referring back. Without this fallback, GetCurrentTime
        # tool path returns no DATE CALCULATION BINDING, the model
        # loses sight of the server-vetted day count, and phase-1
        # spirals into repeated TextGenerationTool calls trying to
        # compute the count itself (observed 2026-05-07 in chat REPL,
        # see TODO 5 §6 G4 design).
        # Safety: this only fires when memory ALREADY has a parseable
        # date. R-6: dialog_focus.entity 是 "当前对话焦点"，用来 bind
        # answer 到正确的 subject——上一轮 retrieval 留下的 target 一直就是
        # 这个用途。
        if not needs_arithmetic:
            obs = (self.last_memory.observation or "")
            if self._extract_month_day_from_memory(obs):
                needs_arithmetic = True
        if not needs_arithmetic:
            return ""
        now = local_now()
        # 2026-05-14: target_entity_hint takes priority over speaker_perspective.
        # Compound queries ("your birthday + my birthday") need TWO date
        # bindings — each GetCurrentTime call passes target_entity=Eva or
        # target_entity=Rosm to scope the binding correctly. Without this,
        # the function would auto-pick ONE entity based on speaker_perspective
        # regex (first match wins), silently dropping the other birthday.
        hint_canon = _canonical_known_entity_name(target_entity_hint) if target_entity_hint else ""
        if hint_canon in ("Eva", "Rosm"):
            candidate_entity = hint_canon
        else:
            # R-6.1 legacy auto-resolution path (when no explicit hint).
            speaker_perspective = self._resolve_speaker_perspective_entity(user_text)
            candidate_entity = _canonical_known_entity_name(
                speaker_perspective or self.dialog_focus.entity or "Both"
            )

        obs = (self.last_memory.observation or "")
        # 检查 obs 的实体 header 是否与本轮的 candidate_entity 一致
        m_module = re.search(r"\[MEMORY MODULE DATA for '([^']+)'\]", obs)
        m_slot = re.search(r"\[SLOT EVIDENCE for ([A-Za-z][A-Za-z0-9_]*)\]", obs)
        obs_entity = (
            (m_module.group(1) if m_module else (m_slot.group(1) if m_slot else ""))
            .strip()
        )
        obs_entity_canon = _canonical_known_entity_name(obs_entity) if obs_entity else ""

        md = None
        bound_entity = candidate_entity
        # 优先：obs 与 candidate 对齐 → 用 obs 抽
        if obs and obs_entity_canon in ("", "Both", candidate_entity):
            md = self._extract_month_day_from_memory(obs)
        # 否则（典型 Turn 4 场景）：从 lore corpus 直接查 candidate 的 birthday
        if not md and candidate_entity in ("Eva", "Rosm"):
            md = self._lookup_birthday_from_corpus(candidate_entity)
        # 兜底：仍然没拿到日期 → 用 obs 里能抽到的随便哪条，bound_entity 取 obs 的
        if not md and obs:
            md = self._extract_month_day_from_memory(obs)
            if md and obs_entity_canon in ("Eva", "Rosm"):
                bound_entity = obs_entity_canon

        if not md:
            return "[DATE CALCULATION]: I could not find a month/day date in the retrieved memory evidence."
        month, day = md
        target = datetime(year=now.year, month=month, day=day)
        if target.date() < now.date():
            target = datetime(year=now.year + 1, month=month, day=day)
        days = (target.date() - now.date()).days
        self._record_date_calculation_evidence(
            subject=bound_entity,
            slot="birthday",
            source_date=now,
            target_date=target,
            days_until=days,
        )
        # R-5: 缓存 target entity 供 _run_phase2_sample 构造
        # BannedDateLogitsProcessor。仅当 bound_entity 是单一实体时才挂 ban
        # 处理器（Both/Shared 时不 ban——没明确的 "另一实体" 概念）。
        if bound_entity in ("Eva", "Rosm"):
            self.current_turn_date_binding_target = bound_entity
        return (
            "[DATE CALCULATION BINDING]\n"
            f"- target_entity: {bound_entity}\n"
            "- slot: birthday\n"
            f"- current_date: {now.strftime('%Y-%m-%d')}\n"
            f"- target_date: {target.strftime('%Y-%m-%d')}\n"
            f"- target_date_text: {target.strftime('%B')} {day}\n"
            f"- days_until: {days}\n"
            "[STRICT DATE RULE]: The days_until value above belongs only to "
            "target_entity and target_date in this block. Do not reuse it for "
            "another person or another date.\n"
            # P1-5 修复（2026-05-13 Turn 5）：模型在回答 Eva 的天数时把 Rosm
            # 的 "November 25th" 一并塞了进来（"Your birthday is November 25th,
            # and mine is July 7th, so 55 days"）。STRICT DATE RULE 只管
            # days_until 的归属，没禁止顺便提别人的日期。下面这条 SCOPE LOCK
            # 把回答范围钉死到 target_entity，防跨实体污染。
            f"[ANSWER SCOPE]: This question is about {bound_entity}'s {('birthday')} "
            f"only. Do NOT mention any other person's birthday, date, or day-count "
            f"in your answer for this turn — even if it was discussed earlier.\n"
            "[/DATE CALCULATION BINDING]"
        )

    # ------------------------------------------------------------------
    # the model's normal ReAct generation, with execution-time guards.
    # ------------------------------------------------------------------

    # ============================================================
    # Verifier intent classifier: PUBLIC_FACT_OR_NEWS
    # ============================================================
    # Two-layer design (Plan B):
    #   Layer 1: regex — fast, deterministic, covers the common patterns
    #   Layer 2: LLM judge — paraphrase-tolerant fallback when regex misses
    #
    # Layer 2 is purely additive: it can flip False → True, never True →
    # False. That asymmetry guarantees enabling the judge cannot make
    # verifier behaviour MORE permissive than today; it can only catch
    # synonym/paraphrase cases the regex would have missed.
    # ============================================================
    def _is_obvious_public_fact_or_news_query_regex(self, text):
        """Layer 1 (regex-only). Same logic as before plan-B refactor.

        Kept as a separate method so:
          - Tests can exercise the regex layer independently.
          - The dual-layer wrapper has a clean cheap path.
        """
        if not isinstance(text, str) or not text.strip():
            return False
        q = text.lower()

        # Pure date / time / weekday queries are answered from the [Today]
        # anchor in the system prompt or via GetCurrentTime. They must NOT
        # be classified as public-fact/news queries — otherwise the
        # verifier's missing_web_evidence_for_external_or_current_request
        # rule will fire and the controller will mis-repair to WebSearch.
        # _current_external_query_needs_web already has this guard; it
        # belongs here too because the relation_question fallthrough
        # below ("what" + "date") would otherwise match unconditionally.
        if self._is_current_time_query(q) or self._question_needs_time_arithmetic(q):
            return False

        # Explicit memory-store requests and personal/profile questions still
        # belong to memory/persona, even if they use words like "what happened".
        if re.search(r"\b(memory|memories|record|records|lore|database|db|profile)\b", q):
            return False
        # Entity-owned identity slots — name, birthday, age, favorites, hobbies,
        # creator — are memory-stored facts about Eva, Rosm, or shared history.
        # They must never be treated as public-fact regardless of surface
        # phrasing. Without this guard, "what is your birthday?" hits the
        # relation_question pattern via the 'date' keyword (literal word "date"
        # in the user query) and gets misclassified as web-fact, which makes
        # PRE PROBE override topic-match to skip MemorySearch and downstream
        # the verifier emits missing_web_evidence_for_external_or_current_request.
        # See TODO 6 (2026-05-07).
        if re.search(
            r"\b(your|my|our)\s+("
            r"name|full[\s-]?name|real[\s-]?name|legal[\s-]?name|"
            r"birthday|birth[\s-]?date|date[\s-]?of[\s-]?birth|born|"
            r"age|"
            r"favou?rite|preferences?|likes?|dislikes?|"
            r"hobby|hobbies|interests?|"
            r"creator|maker"
            r")\b",
            q,
        ):
            return False
        current_or_news = bool(re.search(
            r"\b(news|latest|current|recent|today|now|nowadays|currently|"
            r"this\s+(week|month|year)|breaking|updates?|happening|happened)\b",
            q,
        ))
        public_figure_or_org = bool(re.search(
            r"\b(trump|biden|president|prime\s+minister|minister|election|congress|senate|"
            r"openai|google|microsoft|apple|tesla|nvidia|company|stock|price|version|release)\b",
            q,
        ))
        # Current events in a location, e.g. "what happened in Sydney nowadays".
        location_current_event = bool(re.search(
            r"\b(what|what's|whats|anything|tell\s+me|news)\b.*"
            r"\b(happen(?:ed|ing)?|happend|going\s+on|news|nowadays|recent|current)\b.*"
            r"\b(in|around|near)\s+[a-z][a-z .'-]{2,}\b",
            q,
        ) or re.search(
            r"\b(happen(?:ed|ing)?|happend|going\s+on|news|nowadays|recent|current)\b.*"
            r"\b(in|around|near)\s+[a-z][a-z .'-]{2,}\b",
            q,
        ))
        relation_question = bool(re.search(
            r"\b(who|what|when|where|which|why|how)\b.*\b(announced|released|published|created|developed|"
            r"composed|composer|wrote|written|writer|produced|producer|directed|director|"
            r"performed|sung|sang|singer|painted|painter|designed|designer|invented|inventor|"
            r"price|version|schedule|date|ceo|president|source|origin|official|happened|happening|happend|going\s+on)\b",
            q,
        ))
        return bool(
            (current_or_news and (public_figure_or_org or relation_question or location_current_event))
            or location_current_event
            or relation_question
        )

    def _is_obvious_public_fact_or_news_query(self, text):
        """Fresh/public information that must not be treated as memory continuation.

        Two-layer dispatch:
          - Layer 1 regex: if True, return True immediately (cheap path).
          - Layer 2 LLM judge: only consulted when regex says False, and
            only flips to True if the judge confidently says yes. Any
            judge failure (None) keeps the regex verdict (False).

        This preserves cheap-path behaviour and means the LLM layer can
        only BROADEN coverage for paraphrases, never narrow it.
        """
        if not isinstance(text, str) or not text.strip():
            return False

        # Layer 1: regex.
        if self._is_obvious_public_fact_or_news_query_regex(text):
            return True

        # Hard guards — these short-circuit BEFORE the LLM judge. Even if
        # the judge would say yes, date/time/memory queries must never
        # be classified as public-fact (they're served by other paths).
        q = text.lower()
        if self._is_current_time_query(q) or self._question_needs_time_arithmetic(q):
            return False
        if re.search(r"\b(memory|memories|record|records|lore|database|db|profile)\b", q):
            return False

        # Layer 2: LLM judge fallback (delegated to eva_intent_judge).
        verdict = _judge_intent(
            "PUBLIC_FACT", text, _PROMPT_PUBLIC_FACT,
            state=self._llm_judge_state,
        )
        return verdict is True

    def _is_memory_continuation_query(self, text):
        """Short follow-up that should reuse the previous memory result.

        This catches turns like "give the name", "which one", "list them",
        especially after the previous turn injected Active Memory.
        """
        if not isinstance(text, str) or not text.strip():
            return False
        if not (self.last_memory.observation or self.active_memory_context):
            return False
        q = text.strip().lower()
        # If the user introduces a clearly new public/news topic, do not let
        # previous memory evidence hijack the route.
        if self._is_obvious_public_fact_or_news_query(q):
            return False
        # A short follow-up that explicitly asks to search/find new recommendations
        # is an external recommendation search, not a memory-only continuation.
        if self._is_external_recommendation_search_query(q):
            return False
        if self._active_memory_is_simple_no_memory_task(q):
            return False
        # Only short elliptical pronoun/name follow-ups count as memory
        # continuation. Do not treat any short "what/tell/show/game" question
        # as a continuation; that caused new public topics like Sydney news to be
        # swallowed by stale Eva-game memory.
        words = re.findall(r"\b\w+\b", q)
        short_followup = len(words) <= 8 and bool(re.search(
            r"\b(name|names|which|one|ones|they|them|those|list|give|"
            r"how\s+many|how\s+long|when|until|from\s+now|days?|weeks?|months?)\b",
            q,
        ))
        explicit_followup = bool(re.search(
            r"\b(give\s+(me\s+)?(the\s+)?name|which\s+one|which\s+game|"
            r"list\s+(them|those|the\s+names)|what\s+are\s+(they|those)|"
            r"how\s+many\s+days|how\s+long|from\s+now)\b",
            q,
        ))
        return bool(short_followup or explicit_followup)

    def _is_public_fact_query(self, text):
        if not isinstance(text, str) or not text.strip():
            return False
        # when they use memory anchors such as Eva's known favorite/played games.
        if self._is_external_recommendation_search_query(text):
            return True
        # memory_continuation from the previous turn.
        if self._is_obvious_public_fact_or_news_query(text):
            return True
        # public facts. Do not WebSearch for Eva's own likes, hobbies, or games.
        if self._is_memory_continuation_query(text):
            return False
        is_subjective, _ = self._is_subjective_query(text)
        if is_subjective:
            return False
        memory_hit = False
        try:
            memory_hit, _ = self._has_personal_question_markers(text)
        except Exception:
            memory_hit = False
        relation_hit = bool(self._public_fact_relation_regex is not None and self._public_fact_relation_regex.search(text))
        entity_hint_hit = bool(self._public_fact_entity_hint_regex is not None and self._public_fact_entity_hint_regex.search(text))
        current_hit = bool(self._public_fact_current_regex is not None and self._public_fact_current_regex.search(text))
        if current_hit and not memory_hit:
            return True
        if relation_hit and entity_hint_hit:
            return True
        if entity_hint_hit and re.search(r"\b(which|what|where|who|when|does|did|is|are|was|were|has|have)\b", text, flags=re.IGNORECASE):
            if not memory_hit:
                return True
            return bool(entity_hint_hit and relation_hit)
        return False

    def _is_memory_verification_request(self, text):
        if not isinstance(text, str) or not text.strip():
            return False
        if self._memory_verification_regex is None:
            return False
        return bool(self._memory_verification_regex.search(text))

    def _is_explicit_memory_search_request(self, text):
        """Return True only for explicit memory-store requests.

        v12 intentionally removes the old "search verb + memory field term"
        trigger. A phrase like "search new games to advise" or even
        "search your interests" should not be forced into MemorySearch merely
        because it contains a profile/field word.

        MemorySearch is forced only when the current user message names the
        memory store itself: memory, records, lore, database, db, profile, topic,
        or field. Self-profile questions can still trigger Active Memory through
        the normal self_memory route, but not through keyword forcing.
        """
        if not isinstance(text, str) or not text.strip():
            return False
        q = text.lower()
        if self._is_external_recommendation_search_query(q):
            return False
        search_verb = r"\b(search|seach|lookup|look\s+up|check|verify|confirm|recall|remember)\b"
        memory_store = r"\b(memory|memories|record|records|lore|database|db|profile|topic|field)\b"
        return bool(
            re.search(search_verb + r".*" + memory_store, q)
            or re.search(memory_store + r".*" + search_verb, q)
        )


    # The model may still use normal ReAct; tool execution is guarded.

    def _active_memory_is_simple_no_memory_task(self, user_text):
        if not isinstance(user_text, str): return False
        text = user_text.strip().lower()
        if not text: return False
        for pat in ACTIVE_MEMORY_NO_TRIGGER_PATTERNS:
            if re.search(pat, text, flags=re.IGNORECASE):
                return True
        return False

    def _has_personal_question_markers(self, text):
        if not isinstance(text, str) or not text.strip():
            return False, []
        if self._personal_question_regex is None:
            return False, []
        seen = set()
        unique = []
        for m in self._personal_question_regex.finditer(text):
            s = m.group(0).strip().lower()
            if s and s not in seen:
                seen.add(s)
                unique.append(s)
        return bool(unique), unique[:8]


    # Pre-memory probe now decides injection from actual retrieval evidence.

    def _infer_active_memory_target_entity(self, user_text):
        text_raw = (user_text or "").strip()
        user_name = getattr(self.history_manager, "user_name", "Guest")
        inferred = _infer_memory_target_from_text(text_raw, default_target="Both", current_user=user_name)
        if inferred != "Both":
            return inferred
        # R-6.1: pronoun resolution 优先于 continuation 继承。
        # "your X" / "my X" / "do you ..." / "I want ..." 是本轮显式指向，
        # 不能被 dialog_focus 的 sticky 旧值覆盖。pronoun signal 没有时再
        # fall through 到 continuation + dialog_focus。
        speaker_perspective = self._resolve_speaker_perspective_entity(text_raw)
        if speaker_perspective in ("Eva", "Rosm"):
            return speaker_perspective
        # R-6: continuation entity inheritance via dialog_focus.
        # Short follow-ups ("so how many days should I wait") should stay on
        # the active focus entity instead of flipping to Rosm just because
        # the user says "I".
        try:
            if (self._is_memory_continuation_query(text_raw)
                    and self.dialog_focus.entity
                    and self.dialog_focus.entity not in ("", "Both")):
                return self.dialog_focus.entity
        except Exception:
            pass
        text = f" {text_raw.lower()} "
        vocative_pattern = re.compile(
            r"^\s*(hi+|hello+|hey+|yo+|hiya|howdy|sup|"
            r"good\s+(morning|evening|night|afternoon))"
            r"[\s,]+(eva|rosm|master|creator)[\s,!.~?]*$", re.IGNORECASE)
        if vocative_pattern.match(text_raw):
            return "Both"
        name_only_pattern = re.compile(r"^\s*(eva|rosm|master|creator)[\s,!.~?]*$", re.IGNORECASE)
        if name_only_pattern.match(text_raw):
            return "Both"
        if any(x in text for x in [" we ", " us ", " our ", "together", "shared",
                                    "did i", "did we", "before", "last time", "cake"]):
            return "Both"
        if any(x in text for x in [" your birthday", " your favorite", " your preference",
                                    "your memory", "you remember", "do you remember"]):
            return "Eva"
        if any(x in text for x in [" my birthday", " my favorite", " my preference",
                                    "my project", " master", " rosm"]):
            return "Rosm"
        user_name = getattr(self.history_manager, "user_name", "Guest")
        if str(user_name).lower() == "rosm" and re.search(r"\b(my|me|i)\b", text):
            return "Rosm"
        return "Both"


    def _apply_memory_judge_to_collection(self, collected, user_query):
        """Deterministic record labeling (v22.3, replaces LM-based judge).

        Records reaching this point have already passed broad retrieval
        (FAISS+BM25+bonus) and cross-encoder rerank with the RERANK_CUTOFF
        margin filter. We assign EXACT/RELATED/WRONG labels using only:

          1. Subject alignment with target_entity (strict for Eva/Rosm,
             permissive for Both/Shared; "Shared" records always accepted).
          2. Score tier: rerank_score within EXACT_RERANK_DELTA_TIER of the
             collection's top-1 AND at least MIN_EXACT_RERANK_ABSOLUTE.
          3. subquery-top1 protection: a record explicitly retrieved as
             top-1 for one of the sub-queries is treated as EXACT for that
             sub-query (provided subject_match), since it represents the
             retriever's strongest signal for that specific question.

        Subject-mismatched records are dropped as WRONG. Surface format
        ([Judge: EXACT] tags, judge_*_count keys, MEMORY JUDGE RESULT line)
        is preserved so downstream code does not change. No LM forward
        passes; ~3-5x faster than the v22.2 LM judge on a 91-record DB.
        """
        records = list(collected.get("records") or [])
        if not records:
            out = dict(collected)
            out.update({
                "judge_applied": True,
                "judge_exact_count": 0,
                "judge_related_count": 0,
                "judge_wrong_count": 0,
                "exact_answer_found": False,
                "related_evidence_found": False,
            })
            return out

        target = _canonical_known_entity_name(collected.get("target_entity") or "Both")
        rerank_top1 = max(float(r.get("rerank_score", 0.0)) for r in records)

        judged_records = []
        exact_count = 0
        related_count = 0
        wrong_count = 0

        for rec in records[:MEMORY_JUDGE_TOP_K]:
            rerank = float(rec.get("rerank_score", 0.0))
            entity_canon = _canonical_record_entity(rec.get("entity", ""))
            protected = bool(rec.get("protected_subquery_top1"))

            # Subject alignment.
            # - target Both/Shared accepts any record.
            # - "Shared" records are always accepted (apply to any subject).
            # - Otherwise the record's Subject must canonically match the target.
            subject_match = (
                target in ("Both", "Shared")
                or entity_canon == target
                or entity_canon == "Shared"
            )

            # Hard subject mismatch -> WRONG, dropped from the prompt.
            if not subject_match:
                wrong_count += 1
                continue

            # EXACT requires (a) score in the top tier near rerank_top1
            # AND (b) absolute rerank above MIN_EXACT_RERANK_ABSOLUTE.
            # subquery-top1 protection still helps a borderline-but-valid
            # record clear the tier check, but it CANNOT cross the absolute
            # floor — otherwise weak/uniform queries (e.g. "Hi Eva", "why not
            # use websearch?", "give me some new games in 2026") would inject
            # one off-topic record just because protection forced it through.
            delta = rerank_top1 - rerank
            top_tier = (delta <= EXACT_RERANK_DELTA_TIER
                        and rerank >= MIN_EXACT_RERANK_ABSOLUTE)
            protected_exact = (protected and rerank >= MIN_EXACT_RERANK_ABSOLUTE)
            if top_tier or protected_exact:
                label = "EXACT"
                exact_count += 1
            else:
                label = "RELATED"
                related_count += 1

            rec2 = dict(rec)
            rec2["judge_query"] = (rec.get("source_original_query")
                                   or rec.get("source_query")
                                   or user_query)
            rec2["judge_label"] = label
            # judge_scores kept for log compatibility, populated with rerank info.
            rec2["judge_scores"] = {"rerank": rerank, "delta_from_top1": delta}
            judged_records.append(rec2)

        # EXACT first, then RELATED, by rerank within each group.
        priority = {"EXACT": 0, "RELATED": 1, "WRONG": 2}
        judged_records.sort(
            key=lambda r: (priority.get(r.get("judge_label", "WRONG"), 2),
                           -float(r.get("rerank_score", -1e9)))
        )
        out = dict(collected)
        out["records"] = judged_records[:MEMORY_JUDGE_KEEP_TOP_K]
        out["judge_applied"] = True
        out["judge_exact_count"] = exact_count
        out["judge_related_count"] = related_count
        out["judge_wrong_count"] = wrong_count
        out["exact_answer_found"] = exact_count > 0
        out["related_evidence_found"] = related_count > 0
        if MEMORY_JUDGE_DEBUG:
            print("\n        | --- MEMORY JUDGE (deterministic) ---")
            print(f"        | query={_truncate_for_judge(user_query, 160)}")
            print(f"        | top1_rerank={rerank_top1:.2f}, "
                  f"delta_tier<={EXACT_RERANK_DELTA_TIER}, "
                  f"abs_floor>={MIN_EXACT_RERANK_ABSOLUTE}")
            print(f"        | exact={exact_count}, related={related_count}, "
                  f"wrong={wrong_count} (subject-mismatched, dropped)")
        return out




    def _previous_turn_had_web_evidence(self, debug=False):
        """Return True if the previous completed turn contains WebSearch output.

        P0.5-1: diagnostics are intentionally verbose when debug=True because
        P0-4 failures often come from history shape mismatch rather than route
        logic itself.
        """
        history = getattr(self.history_manager, "history", []) or []
        if not history:
            if debug and ACTIVE_MEMORY_DEBUG_PRINT_INJECTION:
                print("        | [DEBUG] previous_web_evidence=False reason=no_history")
            return False
        last_turn = history[-1]
        steps = getattr(last_turn, "assistant_steps", []) or []
        if debug and ACTIVE_MEMORY_DEBUG_PRINT_INJECTION:
            print(f"        | [DEBUG] previous_web_evidence_scan steps={len(steps)}")
        for idx, step in enumerate(steps):
            if not isinstance(step, dict):
                continue
            role = step.get("role")
            content = step.get("content", "") or ""
            content_l = content.lower()
            # WebSearch outputs in this code path often include links in for_user,
            # Tavily markers, or source/search wording. Keep this detection broad
            # only for diagnostics/follow-up routing, not for factual claims.
            matched = (
                "tavily" in content_l
                or "websearch" in content_l
                or "web search" in content_l
                or re.search(r"https?://", content_l) is not None
                or re.search(r"\[tool output\].*search", content_l) is not None
            )
            if debug and ACTIVE_MEMORY_DEBUG_PRINT_INJECTION:
                preview = re.sub(r"\s+", " ", content[:120])
                print(f"        | [DEBUG] previous_web_step[{idx}] role={role!r} matched={matched} preview={preview!r}")
            if role == "tool" and matched:
                return True
        if debug and ACTIVE_MEMORY_DEBUG_PRINT_INJECTION:
            print("        | [DEBUG] previous_web_evidence=False reason=no_matching_tool_step")
        return False

    def _build_prompt_payload(self, append_text=""):
        self._refresh_active_memory_for_current_turn()
        prompt_text, prompt_images = self.history_manager.build_prompt_payload(
            self._sys_prompt(), include_current=True, assistant_open=True)
        if append_text:
            prompt_text += append_text
        return prompt_text, prompt_images

    def _get_active_image(self):
        return self.history_manager.get_latest_visible_image()

    def _resolve_vision_image(self, tool_params):
        requested_path = tool_params.get("path")
        image = self.history_manager.get_image_by_path(requested_path)
        if image is not None:
            return image
        return self._get_active_image()

    def _current_turn_has_memory_tool_evidence(self):
        """Whether the current turn already contains MemorySearch evidence.

        This preserves grounded generation after an explicit MemorySearch tool call,
        even if Active Memory is skipped on the answer step because the turn is
        already in a tool route.
        """
        turn = self.history_manager.current_turn
        if turn is None:
            return False
        for step in getattr(turn, "assistant_steps", []) or []:
            if step.get("role") != "tool":
                continue
            content = step.get("content", "") or ""
            if "[Judge: EXACT]" in content or "[Judge: RELATED]" in content:
                return True
            if "[MEMORY JUDGE RESULT]" in content and "0 EXACT record(s), 0 RELATED" not in content:
                return True
        return False

    def _current_turn_has_tool_history(self):
        turn = self.history_manager.current_turn
        if turn is None: return False
        return any(step.get("role") == "tool" for step in turn.assistant_steps)

    def _safe_generate(self, **gen_kwargs):
        """Wrapper around model.generate that captures + prints full traceback.

        Background: threading.Thread silently swallows exceptions in worker
        threads — the user sees `Exception in thread Thread-XX` in stderr
        with a truncated traceback, but the actual exception type/message
        often gets lost in the stream. This wrapper prints the FULL stack
        trace + final error line so we can diagnose model.generate failures
        (e.g. get_rope_index shape mismatches for vision inputs).
        """
        try:
            return self.model.generate(**gen_kwargs)
        except Exception as e:
            import traceback
            print(f"\n{'=' * 60}")
            print(f"[GENERATE ERROR] {type(e).__name__}: {e}")
            print(f"{'=' * 60}")
            print("kwargs keys:", list(gen_kwargs.keys()))
            for k, v in gen_kwargs.items():
                if hasattr(v, "shape"):
                    try:
                        print(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype}")
                    except Exception:
                        pass
            print("--- FULL TRACEBACK ---")
            traceback.print_exc()
            print(f"{'=' * 60}\n")
            raise

    def _encode_inputs(self, append_text=""):
        prompt_text, prompt_images = self._build_prompt_payload(append_text=append_text)
        if prompt_images:
            inputs = self.processor(text=[prompt_text], images=prompt_images,
                                    return_tensors="pt", max_pixels=LOCAL_PIXELS)
        else:
            inputs = self.processor(text=[prompt_text], return_tensors="pt")
        input_ids = inputs["input_ids"]
        mm_token_type_ids = torch.zeros_like(input_ids)
        image_pad_id = self.tok.convert_tokens_to_ids("<|image_pad|>")
        if image_pad_id is not None:
            mm_token_type_ids[input_ids == image_pad_id] = 1
        inputs["mm_token_type_ids"] = mm_token_type_ids

        # ========================================================
        # 2026-05-14 VISION DEBUG (set EVA_VISION_DEBUG=1 to enable)
        # 仅在带图像时打印 input shapes，定位 get_rope_index 类 shape
        # mismatch error。两个最关键的 invariant：
        #   1. attention_mask.shape == input_ids.shape == mm_token_type_ids.shape
        #   2. pixel_values 第 0 维 = image_grid_thw 第 0 维 = 1（单图）
        # 任何一项不符即 transformers vision 路径会 crash 在
        # _prepare_position_ids_for_generation。
        # ========================================================
        import os as _os
        if prompt_images and _os.environ.get("EVA_VISION_DEBUG", "").lower() in {"1", "true", "yes"}:
            print(f"\n        | [VISION DEBUG] inputs keys: {list(inputs.keys())}")
            for k, v in inputs.items():
                if hasattr(v, "shape"):
                    try:
                        print(f"        |   {k}: shape={tuple(v.shape)}, dtype={v.dtype}")
                    except Exception:
                        print(f"        |   {k}: shape=??, dtype=??")
                elif isinstance(v, (list, tuple)):
                    print(f"        |   {k}: list/tuple len={len(v)}")
                else:
                    print(f"        |   {k}: type={type(v).__name__}")
            # Critical invariant check
            attn = inputs.get("attention_mask")
            mm = inputs.get("mm_token_type_ids")
            ii = inputs.get("input_ids")
            shape_ok = (
                attn is not None and mm is not None and ii is not None
                and tuple(attn.shape) == tuple(mm.shape) == tuple(ii.shape)
            )
            print(f"        | [VISION DEBUG] shape invariant (attn == mm == input_ids): {shape_ok}")
            if not shape_ok:
                print(f"        | [VISION DEBUG] [WARN] shape mismatch — likely cause of get_rope_index crash")
            pv = inputs.get("pixel_values")
            gthw = inputs.get("image_grid_thw")
            if pv is not None and gthw is not None:
                print(f"        | [VISION DEBUG] pixel_values shape={tuple(pv.shape)}, image_grid_thw shape={tuple(gthw.shape)}")
            print(f"        | [VISION DEBUG] num image_pad tokens: {(input_ids == image_pad_id).sum().item() if image_pad_id is not None else 'N/A'}")
        # ========================================================

        return {k: v.to(self.model.device) for k, v in inputs.items()}

    def _guard_memorysearch_params(self, params, latest_user_text):
        """Allow MemorySearch, but fix obvious target/query mistakes.

        Pre-memory injection is not the only memory path. The model may still
        call MemorySearch after seeing injected evidence. This guard keeps that
        freedom while preventing common pollution: wrong target entity, empty
        query, or using memory for obvious public-news queries.
        """
        params = dict(params or {})
        q = _param_to_text(params.get("query")) or clean_user_text(latest_user_text)
        params["query"] = q
        current_user = getattr(self.history_manager, "user_name", "Guest") or "Guest"

        # 2026-05-14: Eva 显式传的 target_entity 优先。Eva 是从她自己的视角
        # 调用工具——当她写 target_entity="Eva"，意思就是"查关于我自己的"。
        # 老逻辑会用 query 文本里的 "i/my" 做反向推断，但 Eva self-talk
        # query 里的 "i" 指的是 Eva (model)，被错认成 current_user (Rosm) 后
        # 整个 target 被覆盖到 Rosm——见 2026-05-14 trace bug 2。
        # 只在 Eva 没传或传了 "Both" 时才走 text inference。
        raw_target = _canonical_target_entity(
            params.get("target_entity", "Both"),
            current_user=current_user,
        )
        eva_explicit = (
            isinstance(params.get("target_entity"), str)
            and raw_target in ("Eva", "Rosm", "Shared")
        )
        if eva_explicit:
            # Trust Eva's explicit pick — skip text inference entirely.
            params["target_entity"] = raw_target
        else:
            # No explicit pick → fall back to text-based inference.
            inferred = _infer_memory_target_from_text(
                q or latest_user_text, default_target=raw_target,
                current_user=current_user,
            )
            current = _canonical_known_entity_name(current_user)
            q_norm = _normalize_match_text(latest_user_text)
            if current != "Guest" and re.search(r"(?<![a-z0-9])(my|me|i|myself)(?![a-z0-9])", q_norm):
                if _detect_memory_fields(latest_user_text) or _contains_memory_field_terms(latest_user_text):
                    inferred = current
            if inferred and inferred != raw_target and inferred not in ("Both", "Shared"):
                params["target_entity"] = inferred
            else:
                params["target_entity"] = raw_target or "Both"
        if "keywords" not in params:
            params["keywords"] = ", ".join(_build_display_keywords_from_query(
                q, target_entity=params.get("target_entity", "Both"), current_user=current_user, limit=16
            ))
        return params

    def _guard_tool_call(self, tool_name, tool_params, latest_user_text):
        """Return (allow, corrected_params, blocked_observation).

        P0.5-3: use the semantic LM route judge as the core arbitration layer.
        Regex helpers remain elsewhere for verifier/backward compatibility, but
        tool-route conflicts are no longer decided by piles of web/memory regexes.

        2026-05-14 Advisor-first override: when advisor.suggested_calls
        contains this tool, skip the route-judge intent check entirely.
        Route judge is a single-label classifier that can't reason about
        compound queries ("when is your birthday AND days until mine?"
        gets TIME_LOOKUP, which then blocks the necessary MemorySearch
        for birthday lookup — causing an infinite GetCurrentTime loop).
        Advisor already classified the compound intent and listed both
        tools; we trust it. Parameter correction still runs for tools
        that need it (MemorySearch gets target_entity / keywords fixed
        up regardless of route source).
        """
        tool_params = dict(tool_params or {})

        # ---- Advisor override gate ----
        advisor_result = getattr(self, "advisor_result", None)
        advisor_allows_tool = False
        if (advisor_result is not None
                and getattr(advisor_result, "ok", False)
                and getattr(advisor_result, "suggested_calls", None)):
            for c in advisor_result.suggested_calls:
                if isinstance(c, dict) and c.get("tool") == tool_name:
                    advisor_allows_tool = True
                    break
        if advisor_allows_tool and tool_name in (
            "MemorySearch", "WebSearch", "GetCurrentTime",
        ):
            if ROUTE_LM_DEBUG:
                print("\n        | --- TOOL GUARD ROUTE ---")
                print(f"        | tool={tool_name} advisor-suggested → "
                      f"route-judge bypassed")
            if tool_name == "MemorySearch":
                return True, self._guard_memorysearch_params(
                    tool_params, latest_user_text,
                ), ""
            return True, tool_params, ""

        # ---- Fallback: original route-judge path ----
        user_intent, route_scores = self._judge_current_turn_route(latest_user_text)

        if ROUTE_LM_DEBUG:
            print("\n        | --- TOOL GUARD ROUTE ---")
            print(f"        | tool={tool_name}, user_intent={user_intent}")

        if tool_name == "MemorySearch":
            if user_intent == "WEB_SEARCH":
                return False, tool_params, (
                    "[INVALID TOOL ROUTE] The user is asking for external/web information. "
                    "Call WebSearch instead of MemorySearch."
                )
            if user_intent == "TIME_LOOKUP":
                return False, tool_params, (
                    "[INVALID TOOL ROUTE] The user is asking for date/time. "
                    "Call GetCurrentTime instead of MemorySearch."
                )
            return True, self._guard_memorysearch_params(tool_params, latest_user_text), ""

        if tool_name == "WebSearch":
            if user_intent == "MEMORY_LOOKUP":
                private_slots = bool(self._extract_memory_slots(latest_user_text))
                explicit_memory_words = bool(re.search(
                    r"\b(memory|memories|record|records|lore|database|db|remember|recall)\b",
                    latest_user_text or "",
                    re.IGNORECASE,
                ))
                if private_slots or explicit_memory_words:
                    return False, tool_params, (
                        "[INVALID TOOL ROUTE] The user is asking about Eva/Rosm/shared memory "
                        "or profile slots. Use MemorySearch instead of WebSearch."
                    )
            return True, tool_params, ""

        if tool_name == "GetCurrentTime":
            return True, tool_params, ""

        # User-notes mutator guards. Refuse early when the store isn't
        # wired in (ENABLE_USER_NOTES=False) or when the per-turn budget
        # is exhausted.
        if tool_name in ("RememberThis", "ForgetMemory"):
            notes_store = (self.memory_state or {}).get("notes_store") if self.memory_state else None
            if notes_store is None:
                return False, tool_params, (
                    "[INVALID TOOL ROUTE] User-notes store is not active "
                    f"in this session — {tool_name} is unavailable. "
                    "Answer in persona or use MemorySearch on the "
                    "lore corpus."
                )
            # 2026-05-13: advisor-aware budget. The per-turn caps in
            # eva_config (REMEMBER/FORGET_TOOL_MAX_CALLS_PER_TURN=1) were
            # written before the advisor existed, under the assumption
            # that legitimate turns need only one call. With advisor-driven
            # compound decomposition ("buy bear AND finish report" → 2
            # RememberThis), the cap is now max(default, advisor's
            # suggested_calls count for this tool). Default still acts as
            # a floor for thrashing protection when advisor is unavailable.
            advisor_result = getattr(self, "advisor_result", None)
            advisor_suggest_count = 0
            if (advisor_result is not None
                    and getattr(advisor_result, "ok", False)
                    and getattr(advisor_result, "suggested_calls", None)):
                for c in advisor_result.suggested_calls:
                    if isinstance(c, dict) and c.get("tool") == tool_name:
                        advisor_suggest_count += 1

            if tool_name == "RememberThis":
                from eva_config import REMEMBER_TOOL_MAX_CALLS_PER_TURN
                effective_cap = max(REMEMBER_TOOL_MAX_CALLS_PER_TURN,
                                     advisor_suggest_count)
                if self._remember_tool_calls >= effective_cap:
                    return False, tool_params, (
                        "[INVALID TOOL ROUTE] RememberThis budget exhausted "
                        f"for this turn (max={effective_cap}, "
                        f"advisor_suggested={advisor_suggest_count}). "
                        "Continue without persisting another record."
                    )
            else:  # ForgetMemory
                from eva_config import FORGET_TOOL_MAX_CALLS_PER_TURN
                effective_cap = max(FORGET_TOOL_MAX_CALLS_PER_TURN,
                                     advisor_suggest_count)
                if self._forget_tool_calls >= effective_cap:
                    return False, tool_params, (
                        "[INVALID TOOL ROUTE] ForgetMemory budget exhausted "
                        f"for this turn (max={effective_cap}, "
                        f"advisor_suggested={advisor_suggest_count})."
                    )
            return True, tool_params, ""

        return True, tool_params, ""

    # ------------------------------------------------------------------
    # P1.1(a) helper: extract commitment keywords from phase-1 thought
    # ------------------------------------------------------------------
    def _extract_phase1_commitment_terms(self, full_response: str) -> List[str]:
        """Pick out terms phase 1 promised it would say in the answer.

        Sources, in priority order:
          1. slot evidence values for the current turn (e.g. "cuddly bunny").
          2. quoted spans inside <think>...</think>.
          3. nouns that appear both in the thought and in slot evidence values
             (case-insensitive substring match).
        Returned as a deduplicated list, max 4 terms.
        """
        terms: List[str] = []
        seen = set()

        def _push(t):
            t = (t or "").strip(" \t\r\n.,;:!?\"'")
            if not t or len(t) < 2 or len(t) > 60:
                return
            # P1.8.1: reject sentence fragments masquerading as commitment
            # terms.  The previous regex would match e.g. "'s a cuddly bunny.
            # I" because the closing quote and the post-quote letter both
            # looked like word chars; that fragment then got injected into
            # the [ANSWER MUST INCLUDE] line and risked being copied verbatim.
            t_low = t.lower()
            if t_low.startswith("'s ") or t_low.startswith(", ") or t_low.startswith(". "):
                return
            if any(ch in t for ch in ("<", ">", "|", "\n", "\r", "\t")):
                return
            # Reject anything that looks like a sentence (contains end-of-
            # sentence punctuation followed by a space).
            if re.search(r"[.!?]\s", t):
                return
            # Word count must be in 1..6 — proper-noun phrases, dates, or
            # short answer values; not full clauses.
            word_count = len([w for w in re.split(r"\s+", t) if w])
            if word_count < 1 or word_count > 6:
                return
            key = t_low
            if key in seen:
                return
            seen.add(key)
            terms.append(t)

        slot_values = []
        try:
            for v in (self.current_turn_slot_evidence or {}).values():
                if isinstance(v, str) and v.strip():
                    slot_values.append(v.strip())
        except Exception:
            pass
        for v in slot_values:
            _push(v)

        thought_text = ""
        try:
            m = re.search(r"<think>(.*?)</think>", full_response or "", re.DOTALL | re.IGNORECASE)
            if m:
                thought_text = m.group(1)
        except Exception:
            thought_text = ""

        if thought_text:
            for q in re.findall(r"[\"']([^\"'\n]{2,40})[\"']", thought_text):
                _push(q)
            t_low = thought_text.lower()
            for v in slot_values:
                if v.lower() in t_low:
                    _push(v)

        return terms[:4]

    # ------------------------------------------------------------------
    # P1.2 helper: detect sampling-collapse against recent phase-2 outputs
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_for_collapse(text: str) -> str:
        if not text:
            return ""
        t = text.lower()
        t = re.sub(r"[^a-z0-9 ]+", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t[:120]

    def _phase2_output_collides(self, candidate_norm: str) -> bool:
        """True if the prospective phase-2 output would near-duplicate a recent one.

        Used as a *pre-flight* signal: we don't have the new output yet, so this
        is invoked with the *previous* normalized output that the current
        active-memory packet would tend to reproduce. We use a simple
        Jaccard-on-tokens >= 0.85 heuristic; cheap and stable.
        """
        if not candidate_norm:
            return False
        cand_tokens = set(candidate_norm.split())
        if not cand_tokens:
            return False
        for prev in self._recent_phase2_outputs:
            prev_tokens = set(prev.split())
            if not prev_tokens:
                continue
            inter = len(cand_tokens & prev_tokens)
            union = len(cand_tokens | prev_tokens)
            if union == 0:
                continue
            if inter / union >= 0.85:
                return True
        return False

    def _record_phase2_output(self, mode: str, raw_answer: str):
        norm = self._normalize_for_collapse(raw_answer)
        if not norm:
            return
        self._recent_phase2_outputs.append(norm)
        self._recent_phase2_modes.append(mode or "")
        if len(self._recent_phase2_outputs) > self._RECENT_PHASE2_MAX:
            self._recent_phase2_outputs.pop(0)
            self._recent_phase2_modes.pop(0)

    def _run_phase2_sample(self, phase2_mode, inputs2, stop_ids_s2,
                           context_up_to_answer, commitment_terms,
                           force_collapse_pressure=False):
        """Phase-2 generation — extracted from step_once (TODO 11-arch).

        The verifier regenerate path needs to re-sample phase-2 with
        forced collapse pressure when phase-2 produced a verifier-flagged
        answer. Pulling sampling into a helper makes both first-call
        and retry-call paths identical except for this one flag.

        Streamer and printer are single-shot, so they're freshly created
        on each call. Inputs2/stop_ids/context_prefix are reusable across
        calls.

        Args:
            phase2_mode: 'after_memory' / 'after_tool' / 'low_confidence' /
                'direct' — pre-determined by step_once based on memory
                grounding state. Helper does not re-decide mode.
            inputs2: encoded prompt input tensors. Reused across retries.
            stop_ids_s2: list of stop-token IDs.
            context_up_to_answer: prefix string that the answer suffix
                gets appended to in full_response.
            commitment_terms: log-only; doesn't gate sampling here.
            force_collapse_pressure: if True, collapse-pressure config is
                applied unconditionally — used by regenerate retry to
                push sampling away from the failed answer's phrasing.
                Without this flag, collapse pressure only kicks in when
                recent outputs collide.

        Returns:
            (full_response, answer_suffix). Caller handles
            parse_react_block, history write, and verifier check.
        """
        streamer2 = TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=False)
        printer2 = StreamPrinter()
        printer2.process(REACT["answer"])

        phase2_cfg = dict(get_phase2_sampling_config(phase2_mode))
        recent_in_same_mode = [
            out for out, m in zip(self._recent_phase2_outputs, self._recent_phase2_modes)
            if m == phase2_mode
        ]
        collapse_pressure = force_collapse_pressure or (
            len(recent_in_same_mode) >= 2
            and self._phase2_output_collides(recent_in_same_mode[-1])
        )
        if collapse_pressure:
            self._collapse_detected_for_current_turn = True
            phase2_cfg["temperature"] = max(phase2_cfg["temperature"], 0.75)
            phase2_cfg["top_p"] = max(phase2_cfg["top_p"], 0.92)
            # 2026-05-15: rep_pen cap 从 1.15 降到 1.10。原 1.15 过狠：
            # 模型想用 markdown 强调上轮要点（"**commit logs**"）时，因为
            # "commit/logs" 等词刚出现过被严重压制，找不到合理 continuation
            # 就直接采样到 <|end_react|>，留下孤儿 ** 截断答案。1.10 仍能
            # 拉开 sampling distribution 帮助逃出 collapse，但不至于把所有
            # "合理重复" 一并封死。
            phase2_cfg["repetition_penalty"] = max(phase2_cfg["repetition_penalty"], 1.10)
            guard_reason = ("verifier_regenerate_force"
                            if force_collapse_pressure
                            else "recent_phase2_outputs_near_duplicate")
            print("        | --- PHASE 2 COLLAPSE GUARD ENGAGED ---\n"
                  f"        | reason={guard_reason}")

        print(f"        | --- PHASE 2 SAMPLING ---\n"
              f"        | mode={phase2_mode}, temp={phase2_cfg['temperature']}, "
              f"top_p={phase2_cfg['top_p']}, rep_pen={phase2_cfg['repetition_penalty']}, "
              f"commit_terms={commitment_terms or 'none'}")

        gen_kwargs2 = dict(**inputs2, streamer=streamer2, max_new_tokens=MAX_NEW_TOKENS_TURN,
                           temperature=phase2_cfg["temperature"], top_p=phase2_cfg["top_p"],
                           repetition_penalty=phase2_cfg["repetition_penalty"], do_sample=True,
                           pad_token_id=self.tok.pad_token_id, eos_token_id=stop_ids_s2)
        # R-5 BannedDateLogitsProcessor: DELETED 2026-05-14 Plan-A final.
        # Advisor + multi-binding GetCurrentTime + LLM judge replaced it.
        t2 = threading.Thread(target=self._safe_generate, kwargs=gen_kwargs2)
        t2.start()
        answer_suffix = ""
        for new_text in streamer2:
            new_text = new_text.replace(EOT, "")
            answer_suffix += new_text
            printer2.process(new_text)
        t2.join()
        printer2.flush()
        full_response = context_up_to_answer + answer_suffix

        # Record this turn's phase-2 output for collapse detection on
        # subsequent turns. answer_suffix is the just-generated body;
        # trim trailing REACT/EOT markers before normalizing.
        try:
            _suffix_clean = answer_suffix
            for _t in (REACT["end"], EOT):
                if _t and _t in _suffix_clean:
                    _suffix_clean = _suffix_clean.split(_t, 1)[0]
            self._record_phase2_output(phase2_mode, _suffix_clean)
        except Exception:
            pass

        return full_response, answer_suffix

    def step_once(self):
        full_response = ""
        printer = StreamPrinter()
        if self.progress_callback:
            self.progress_callback("Eva is thinking (Logic Phase)...")

        latest_user_text = self._get_latest_user_text()

        # _encode_inputs() via _build_prompt_payload(); after that, the model
        # follows its normal ReAct behavior and may answer or call tools.
        inputs1 = self._encode_inputs()

        # FORCE_THINK_PREFIX (2026-05-08): hard-prefix <think> at decode
        # start so model is guaranteed to emit a thought block. Without
        # this, the greedy decoder occasionally skipped <think> for
        # short queries (e.g. "for example?") and went straight to
        # <|answer|>. The skip correlates with higher hallucination
        # because there's no self-reflection step before the answer
        # commits to specifics. The streamer has skip_prompt=True so
        # it won't emit the prefix tokens; we seed full_response and
        # the printer manually so downstream parsing + the THOUGHT
        # header in trace both work as if the model had generated
        # <think> itself.
        if FORCE_THINK_PREFIX:
            think_ids = self.tok.encode(
                THINK_START, add_special_tokens=False, return_tensors="pt"
            ).to(inputs1["input_ids"].device)
            inputs1["input_ids"] = torch.cat(
                [inputs1["input_ids"], think_ids], dim=1)
            inputs1["attention_mask"] = torch.cat(
                [inputs1["attention_mask"],
                 torch.ones_like(think_ids)], dim=1)
            # 2026-05-14 vision bug fix: when image input is present,
            # mm_token_type_ids must extend in lockstep with input_ids /
            # attention_mask. Otherwise transformers' get_rope_index
            # crashes with "shape of the mask [N+1] does not match the
            # shape of the indexed tensor [N]" on the first vision turn.
            # <think> is a pure text token (not image), so the appended
            # mm values are 0 (= not an image pad).
            if "mm_token_type_ids" in inputs1:
                inputs1["mm_token_type_ids"] = torch.cat(
                    [inputs1["mm_token_type_ids"],
                     torch.zeros_like(think_ids)], dim=1)
            full_response = THINK_START
            printer.process(THINK_START)

        stop_ids_s1 = [tid for tid in [self.tok.convert_tokens_to_ids(t)
                                        for t in [REACT["end"], EOT]] if tid is not None]
        stopping_criteria = StoppingCriteriaList([ReActStoppingCriteria(stop_ids_s1)])
        streamer1 = TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=False)
        gen_kwargs1 = dict(**inputs1, streamer=streamer1, max_new_tokens=MAX_NEW_TOKENS_TURN,
                           do_sample=False, pad_token_id=self.tok.pad_token_id,
                           eos_token_id=self.tok.eos_token_id, stopping_criteria=stopping_criteria)
        t1 = threading.Thread(target=self._safe_generate, kwargs=gen_kwargs1)
        t1.start()
        answer_reached = False
        for new_text in streamer1:
            new_text = new_text.replace(EOT, "")
            full_response += new_text
            if not answer_reached:
                if REACT["answer"] in full_response:
                    answer_reached = True
                    answer_tag_pos = full_response.find(REACT["answer"])
                    already_sent_len = len(full_response) - len(new_text)
                    if already_sent_len < answer_tag_pos:
                        unsent = new_text[:answer_tag_pos - already_sent_len]
                        if unsent:
                            printer.process(unsent)
                    printer.discard_buffer()
                    printer.flush()
                else:
                    printer.process(new_text)
        t1.join()

        has_tool_call = REACT["tool_code"] in full_response
        has_answer = REACT["answer"] in full_response
        used_tools_before_answer = self._current_turn_has_tool_history()

        if has_answer and not has_tool_call:
            if self.progress_callback:
                self.progress_callback("Eva is composing response (Persona Phase)...")
            answer_tag_pos = full_response.find(REACT["answer"])
            context_up_to_answer = full_response[:answer_tag_pos + len(REACT["answer"])]

            # P1.1(a): bind phase-1 commitment terms into the phase-2 prompt.
            # The greedy thought may have already promised "I'll tease then
            # confirm <fact>", but the sampled answer was free to drop the
            # confirm half. Inject an explicit, narrow [ANSWER MUST INCLUDE]
            # line just before the <|answer|> opener so the answer continuation
            # is conditioned on those tokens.
            commitment_terms = self._extract_phase1_commitment_terms(full_response)
            phase2_prefix = ""
            if commitment_terms:
                must_inc = ", ".join(commitment_terms)
                phase2_prefix = (
                    f"\n[ANSWER MUST INCLUDE]: {must_inc}\n"
                    f"(State the value, then add Eva's persona around it.)\n"
                )
            phase2_context = context_up_to_answer
            if phase2_prefix:
                # Insert before the <|answer|> tag so the model sees the
                # constraint as part of the assistant turn-internal scratch,
                # not after the answer has already started.
                tag = REACT["answer"]
                phase2_context = (
                    context_up_to_answer[: -len(tag)]
                    + phase2_prefix
                    + tag
                )
            inputs2 = self._encode_inputs(append_text=phase2_context)
            stop_ids_s2 = [tid for tid in [self.tok.convert_tokens_to_ids(t)
                                            for t in [REACT["end"], EOT]] if tid is not None]
            # NB: printer2 + streamer2 used to be created here, but the
            # phase-2 sample is now run via _run_phase2_sample(), which
            # creates fresh ones internally on each call (TODO 11-arch
            # — needed so the regenerate retry can re-use the same
            # input encoding without dragging stale streamer state).
            # on `self.active_memory_context`. The packet is intentionally
            # cleared after step 1 (so it doesn't repeat in every step's
            # prompt), but the grounded state — set by the pre-memory probe
            # OR by an in-turn MemorySearch tool call — should persist for
            # the whole turn. Without this fix, a turn that did
            # probe(grounded) -> tool -> answer fell through to `after_tool`
            # mode (temp 0.6) instead of `after_memory` (temp 0.35), because
            # the conjunction `grounded and active_memory_context` evaluated
            # False once the packet was scoped down to step 1 only.
            low_confidence_active_memory = bool(
                getattr(self, "active_memory_low_confidence", False)
                and not self.current_turn_memory_grounded
            )
            memory_grounded_active = bool(
                self.current_turn_memory_grounded
                or self._current_turn_has_memory_tool_evidence()
            )
            slot_found_active = bool(self.current_turn_slot_evidence)
            if memory_grounded_active:
                # P1.1(c): when the turn has slot-level FOUND evidence, switch
                # the answer to the wider after_tool preset (temp 0.6). The
                # narrow after_memory preset (temp 0.35) was empirically
                # producing near-identical templated refusals across consecutive
                # slot turns. Slot evidence is already pinned in the prompt and
                # via [ANSWER MUST INCLUDE], so we no longer need the low-temp
                # guardrail. Non-slot grounded turns keep after_memory.
                phase2_mode = "after_tool" if slot_found_active else "after_memory"
            elif used_tools_before_answer:
                phase2_mode = "after_tool"
            elif low_confidence_active_memory:
                phase2_mode = "low_confidence"
            else:
                phase2_mode = "direct"

            # TODO 11-arch: phase-2 sample now runs via helper so the
            # verifier regenerate path can call it again with forced
            # collapse pressure. The helper handles the
            # collapse-pressure auto-detection AND honors the
            # force_collapse_pressure flag (used by regenerate retry
            # below in the verifier dispatch block).
            full_response, answer_suffix = self._run_phase2_sample(
                phase2_mode, inputs2, stop_ids_s2,
                context_up_to_answer, commitment_terms,
                force_collapse_pressure=False,
            )
        elif has_tool_call:
            printer.flush()
            print()
        else:
            printer.flush()

        turn_content = full_response
        if REACT["end"] not in turn_content:
            turn_content += REACT["end"]
        self.history_manager.add_assistant_step(turn_content)

        tool_name, tool_params, final_answer = None, None, None
        for tag, val in parse_react_block(turn_content):
            if tag == "tool_code":
                try:
                    tool_name, tool_params = sanitize_tool_code(val.strip())
                except Exception:
                    pass
            elif tag == "answer":
                final_answer = val.strip()

        if tool_name:
            observation_for_model = ""
            observation_for_user = ""
            try:
                # The previous version required user_text to also match a time-arithmetic
                # pattern, which missed verification follow-ups like
                # "can you check the days with tools to prove it?". Now we trust the
                # tool query itself: if the model is webbing for date math, that's
                # always a GetCurrentTime job.
                ws_query = (tool_params or {}).get("query", "") if tool_params else ""
                if (tool_name == "WebSearch"
                        and (
                            self._is_time_lookup_web_query(ws_query)
                            or self._is_date_math_web_query(ws_query)
                            or (
                                self._is_current_time_query(latest_user_text)
                                and not self._is_obvious_public_fact_or_news_query(ws_query)
                                and not self._current_external_query_needs_web(ws_query)
                            )
                            or (
                                self._question_needs_time_arithmetic(latest_user_text)
                                and not self._is_obvious_public_fact_or_news_query(ws_query)
                                and not self._current_external_query_needs_web(ws_query)
                            )
                        )):
                    print("        | --- TOOL ROUTE CORRECTION ---")
                    print(f"        | WebSearch({ws_query!r}) -> GetCurrentTime()")
                    tool_name = "GetCurrentTime"
                    tool_params = {}

                allow, corrected_params, blocked_obs = self._guard_tool_call(tool_name, tool_params, latest_user_text)
                tool_params = corrected_params
                if not allow:
                    observation_for_model = observation_for_user = blocked_obs
                elif tool_name in ("ToolName", "RealToolName", "FunctionName", "Function", "Tool"):
                    observation_for_model = (f"Error: '{tool_name}' is a placeholder, not a real tool. "
                        "Replace it with the actual tool name. Available tools: MemorySearch, WebSearch, "
                        "AskRemoteVision, TextGenerationTool, GetCurrentTime.")
                    observation_for_user = observation_for_model
                elif tool_name == "GetCurrentTime":
                    now = local_now()
                    self._record_time_evidence(now)
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
                    # 2026-05-14: target_entity arg lets the model scope the
                    # DATE CALCULATION BINDING to a specific subject. Critical
                    # for compound queries like "days until your birthday and
                    # my birthday" where Eva needs ONE binding per subject.
                    target_hint = str(
                        tool_params.get("target_entity", "") or ""
                    ).strip() if isinstance(tool_params, dict) else ""
                    calc_note = self._maybe_compute_date_delta_from_memory(
                        target_entity_hint=target_hint,
                    )
                    if calc_note:
                        obs = obs + "\n" + calc_note
                    observation_for_model = observation_for_user = obs
                elif tool_name == "MemorySearch":
                    obs = run_memory_search(params=tool_params, memory_state=self.memory_state,
                                            encoder=self.encoder, reranker=self.reranker,
                                            current_user=self.history_manager.user_name,
                                            judge_fn=self._apply_memory_judge_to_collection)
                    requested_slots = self._extract_memory_slots(latest_user_text)
                    for slot, value in self._parse_slot_evidence_from_text(obs).items():
                        self.current_turn_slot_evidence[slot] = value
                    missing_slots = [s for s in requested_slots if s in MEMORY_SLOT_FIELDS and s not in self.current_turn_slot_evidence]
                    slot_note = self._build_missing_slot_note_from_missing(missing_slots)
                    # Suppress missing-slot warning when saved notes
                    # surfaced — they may answer directly even though
                    # lore-corpus slot-extraction can't read them.
                    from eva_memory_legacy import memory_block_has_notes
                    if slot_note and not memory_block_has_notes(obs):
                        obs = obs + slot_note
                    self.current_turn_missing_slots = missing_slots
                    # R-6: missing_slots 跨轮 sticky 移入 last_memory.missing_slots
                    if requested_slots:
                        self.last_memory.missing_slots = list(missing_slots)
                    observation_for_model = observation_for_user = obs
                    has_exact = "[Judge: EXACT]" in obs
                    has_related = "[Judge: RELATED]" in obs
                    self.current_turn_memory_grounded = bool(has_exact or has_related)
                    self.current_turn_memory_has_exact = has_exact
                    self.current_turn_memory_has_related = has_related
                    if obs and (has_exact or has_related):
                        target = tool_params.get("target_entity", "") or "Both"
                        query = tool_params.get("query", "") or latest_user_text
                        # R-6: 用统一 helper 写 LastMemoryState + DialogFocus
                        self._update_memory_state_from_tool_obs(
                            obs=obs, target_entity=target, query=query,
                            source="tool",
                        )
                        self._record_memory_evidence_from_observation(
                            obs, target_entity=self.dialog_focus.entity or target,
                            query=query,
                        )
                elif tool_name == "WebSearch":
                    obs_dict = run_websearch(tool_params)
                    observation_for_model = obs_dict["for_model"]
                    observation_for_user = obs_dict["for_user"]
                    self._record_web_evidence(
                        query=(tool_params or {}).get("query", ""),
                        observation_for_model=observation_for_model,
                        observation_for_user=observation_for_user,
                    )
                elif tool_name == "AskRemoteVision":
                    image_for_vision = self._resolve_vision_image(tool_params)
                    vision_result = call_remote_vision(tool_params.get("query", ""), image_for_vision,
                                                       mode=tool_params.get("mode", "chat"))
                    observation_for_model = observation_for_user = vision_result
                elif tool_name == "TextGenerationTool":
                    original_instruction = tool_params.get("instruction", "")
                    neutral_instruction = self._third_person_textgen_instruction(
                        original_instruction,
                        latest_user_text=latest_user_text,
                    )
                    raw_text = call_deepseek_expert(neutral_instruction)
                    self._record_textgen_evidence(original_instruction, raw_text)
                    observation_for_model = (
                        f"### NEUTRAL THIRD-PERSON GENERATED CONTENT ###\n{raw_text}\n### END CONTENT ###\n\n"
                        "[SYSTEM NOTE]: The content above is a neutral third-person draft, not Eva's final voice. "
                        "Use it as factual/source material only. When converting to Eva's final answer, fix perspective: "
                        "facts about Eva should become 'I/my' when Eva speaks; facts about Rosm should become 'you/your'. "
                        "Do not copy any incorrect 'you play/you like' phrasing for Eva's own facts.")
                    observation_for_user = raw_text
                elif tool_name == "RememberThis":
                    # Persist a new note into the user-notes store.
                    # Guard already verified notes_store presence + budget.
                    from Memory_maker.notes_runtime import execute_remember_this
                    notes_store = (self.memory_state or {}).get("notes_store")
                    obs = execute_remember_this(notes_store, tool_params)
                    if obs.startswith("[REMEMBERED]"):
                        self._remember_tool_calls += 1
                        # R-4 Step 2a：成功写入 NotesStore 后，向 turn_evidence
                        # ledger 落一条 notes_write 证据。verifier 的
                        # current_turn_has_remember_evidence 改读 ledger 后，
                        # 不再依赖 grep "[REMEMBERED]" 字符串。
                        m_nid = re.search(r"Note #([0-9a-f]{8})", obs)
                        nid = m_nid.group(1) if m_nid else ""
                        self._add_turn_evidence(
                            source="notes_write",
                            subject=(tool_params or {}).get("entity"),
                            slot=None,
                            value=(tool_params or {}).get("content", ""),
                            confidence="exact",
                            raw_text=obs,
                            topic=(tool_params or {}).get("topic", "") or "",
                            record_ref=nid,
                            meta={"channel": "tool"},
                        )
                    observation_for_model = observation_for_user = obs
                elif tool_name == "ForgetMemory":
                    # Tombstone an existing note. R-2 (2026-05-13)：现在支持
                    # query/topic-based 匹配 + runtime intercept auto-correct。
                    # SFT 训练分布让模型偏好"先 MemorySearch 再 ForgetMemory
                    # (record_id=...)"的模式；当模型按训练 pattern 调但 id 抄
                    # 错时（Turn 13 复盘），runtime 用 fallback_context 在
                    # NotesStore 里 search 自动纠正。fallback_context 由本轮
                    # user 文本 + 最近一条 RememberThis 的 content 拼成。
                    from Memory_maker.notes_runtime import execute_forget_memory
                    notes_store = (self.memory_state or {}).get("notes_store")

                    # 收集 fallback_context：
                    # ① 本轮 user 文本（"meeting is canceled, forget it"）
                    # ② 模型本次调用里 `reason` 字段——训练数据里 reason
                    #    经常直接复述要删的内容（"Meeting next Monday has been
                    #    canceled."），是最高 ROI 的语义信号。
                    # ③ 本轮 ledger 里最近一条 notes_write 的 content（刚
                    #    RememberThis 的事实，e.g. "Master has a meeting next Monday"）
                    # ④ 跨轮兜底：扫 history 最近 5 轮的 tool obs 抓 [REMEMBERED]
                    #    + 上一轮 assistant final_answer（Eva 经常复述刚记的事实）。
                    #    覆盖"Turn 12 remember、Turn 15+ 才 forget"的长程场景。
                    fallback_parts = []
                    if latest_user_text:
                        fallback_parts.append(latest_user_text)
                    # ② reason 并入
                    _reason = str((tool_params or {}).get("reason", "") or "").strip()
                    if _reason:
                        fallback_parts.append(_reason)
                    # ③ 本轮 notes_write ledger
                    try:
                        for ev in reversed(list(self.turn_evidence) or []):
                            if getattr(ev, "source", None) == "notes_write" and ev.value:
                                fallback_parts.append(str(ev.value))
                                break
                    except Exception:
                        pass
                    # ④ 跨轮兜底：最近 5 轮的 [REMEMBERED] tool obs + 上轮 final_answer
                    try:
                        recent_turns = self.history_manager.history[-5:]
                        for hist_turn in reversed(recent_turns):
                            for step in getattr(hist_turn, "assistant_steps", []) or []:
                                role = step.get("role")
                                txt = step.get("content", "") or ""
                                if role == "tool" and "[REMEMBERED]" in txt:
                                    fallback_parts.append(txt)
                                elif role == "assistant" and "<|answer|>" in txt:
                                    # 抓 final_answer 部分作为语义补充
                                    m_ans = re.search(
                                        r"<\|answer\|>(.*?)(?:<\|end_react\|>|$)",
                                        txt, re.S,
                                    )
                                    if m_ans:
                                        fallback_parts.append(m_ans.group(1).strip())
                    except Exception:
                        pass
                    fallback_context = "  ".join(p for p in fallback_parts if p)

                    obs = execute_forget_memory(notes_store, tool_params,
                                                fallback_context=fallback_context)
                    if obs.startswith("[FORGOTTEN]"):
                        self._forget_tool_calls += 1
                        # R-4 Step 2b：成功 tombstone 后落 notes_delete 证据。
                        # R-2：实际删的 note_id 可能来自 runtime 解析（query
                        # 路径），所以从 obs 文本里抓 "Note #xxxx" 而不是
                        # 信任 tool_params["record_id"]——模型若走 query 调用，
                        # record_id 字段是空的。
                        m_nid = re.search(r"Note #([0-9a-f]{8})", obs)
                        nid = m_nid.group(1) if m_nid else (
                            str((tool_params or {}).get("record_id", "") or "").strip()
                        )
                        self._add_turn_evidence(
                            source="notes_delete",
                            subject=None,
                            slot=None,
                            value=nid,
                            confidence="exact",
                            raw_text=obs,
                            record_ref=nid,
                            meta={"channel": "tool",
                                  "reason": (tool_params or {}).get("reason", ""),
                                  "resolved_by_query": bool(
                                      (tool_params or {}).get("query")
                                      and not (tool_params or {}).get("record_id")
                                  )},
                        )
                    observation_for_model = observation_for_user = obs
                else:
                    observation_for_model = observation_for_user = (
                        f"Error: Tool '{tool_name}' not found.\n"
                        "Valid tools: MemorySearch, WebSearch, AskRemoteVision, TextGenerationTool, GetCurrentTime.")
            except Exception as e:
                observation_for_model = (f"Error executing tool '{tool_name}': {e}\n"
                    "Valid tools: MemorySearch, WebSearch, AskRemoteVision, TextGenerationTool, GetCurrentTime.")
                observation_for_user = observation_for_model
            self.history_manager.add_tool_output(f"{REACT['tool_output']}{observation_for_model}")
            print(f"        | --- TOOL OUTPUT ({tool_name}) ---\n        | "
                  f"{observation_for_user.replace(chr(10), chr(10) + '        | ')}")
            return None, True

        if final_answer:
            verify_result = self._verify_final_answer(final_answer, latest_user_text)
            self.last_verifier_result = verify_result
            # R-3: 把 original 候选记进 verdict_ledger。fallback 时
            # ledger.best() 会按 (severity, reason_count, stage_order) 自动
            # 选最值得释放的 candidate——不再依赖 caller 手动传 phase2_answer。
            self.verdict_ledger.add(Verdict(
                answer=final_answer,
                source_stage="original",
                verify_result=verify_result,
            ))
            if not verify_result.ok:
                if VERIFIER_DEBUG:
                    print("        | --- ANSWER VERIFIER FAILED ---")
                    print(f"        | reasons={', '.join(verify_result.reasons)}")

                # Path 1: required_action -> controller executes the real tool.
                if verify_result.required_action:
                    action = verify_result.required_action
                    new_reason = action.get("reason")

                    # C repair: if the verifier asks for the same controller
                    # repair twice in the same turn, the repair did not satisfy
                    # the invariant. Fall back instead of looping forever.
                    if new_reason and new_reason == getattr(self, "last_required_action_reason", None):
                        if VERIFIER_DEBUG:
                            print(
                                f"        | --- VERIFIER LOOP DETECTED "
                                f"(repeated reason={new_reason}) -> FALLBACK ---"
                            )
                        self.last_required_action_reason = None
                        # R-3: ledger.best() 自动选 release candidate。
                        return self._safe_fallback_for_hard_verifier_failure(
                            verify_result, latest_user_text,
                        ), False

                    self.last_required_action_reason = new_reason
                    if VERIFIER_DEBUG:
                        print("        | --- ANSWER VERIFIER REQUIRED ACTION ---")
                        print(f"        | tool={action.get('tool')} params={action.get('params')} reason={new_reason}")
                    return self._execute_controller_tool(
                        action.get("tool"),
                        action.get("params") or {},
                        latest_user_text,
                        reason=action.get("reason", "verifier_required_action"),
                    )

                # Path 1.5 (TODO 11-arch): regenerate -> re-sample phase-2
                # with forced collapse pressure. Used for generation errors
                # (pronoun mismatch, wrong date in answer, perspective slip,
                # etc.) where evidence is correct but the model generated
                # badly. The regenerate fix-class comes from REASON_POLICY.
                #
                # Flow:
                #   1. Loop guard: same dominant reason hit twice -> canned.
                #   2. Re-run _run_phase2_sample with force_collapse_pressure
                #      so sampling diverges from the failed phrasing.
                #   3. Replace the failed assistant step in history (don't
                #      pollute downstream context with the bad first sample).
                #   4. Re-parse and re-verify. Success -> ship new answer.
                #   5. Failure -> canned (no recursion).
                if verify_result.fix_class == "regenerate":
                    from eva_verifier_logic import get_dominant_reason_for_dispatch
                    dominant = get_dominant_reason_for_dispatch(verify_result.reasons)

                    # Pre-existing edge case (exposed 2026-05-15): regenerate
                    # needs phase2_mode / inputs2 / stop_ids_s2 / context_up_to_answer
                    # / commitment_terms, which are only set inside the
                    # `if has_answer and not has_tool_call:` branch. If we
                    # arrived here via the `elif has_tool_call:` branch
                    # (e.g. model emitted both tool_code + answer, tool_code
                    # failed to parse, but answer was extracted), those locals
                    # are undefined → UnboundLocalError. Fail-open: release the
                    # original answer rather than crashing the turn.
                    if "phase2_mode" not in locals() or "inputs2" not in locals():
                        if VERIFIER_DEBUG:
                            print(
                                "        | --- REGENERATE SKIPPED: phase2 didn't run "
                                "(answer came from non-phase2 path) -> RELEASE ORIGINAL ---"
                            )
                        return self._safe_fallback_for_hard_verifier_failure(
                            verify_result, latest_user_text,
                        ), False

                    # P1: RegenerateGuard atomic check + consume.
                    # Returns False when either the per-reason or the
                    # total-per-turn budget is exhausted. The legacy
                    # last_regenerate_reason field is kept in sync as a
                    # back-compat mirror.
                    if not self.regenerate_guard.try_consume(dominant):
                        if VERIFIER_DEBUG:
                            print(
                                f"        | --- VERIFIER REGENERATE BUDGET EXHAUSTED "
                                f"(reason={dominant}, "
                                f"snapshot={self.regenerate_guard.snapshot()}) "
                                f"-> RELEASE ORIGINAL ANSWER ---"
                            )
                        # P1: fail-open — return the phase-2 answer we
                        # already have rather than a canned apology.
                        # R-3: ledger.best() 自动选 candidate (original 已记)。
                        return self._safe_fallback_for_hard_verifier_failure(
                            verify_result, latest_user_text,
                        ), False

                    self.last_regenerate_reason = dominant
                    if VERIFIER_DEBUG:
                        print(f"        | --- ANSWER VERIFIER REGENERATE PATH "
                              f"(dominant={dominant!r}) ---")

                    # Re-sample phase-2 with forced collapse pressure
                    full_response, answer_suffix = self._run_phase2_sample(
                        phase2_mode, inputs2, stop_ids_s2,
                        context_up_to_answer, commitment_terms,
                        force_collapse_pressure=True,
                    )

                    new_turn_content = full_response
                    if REACT["end"] not in new_turn_content:
                        new_turn_content += REACT["end"]

                    # Replace the failed assistant step so downstream context
                    # doesn't see the broken first sample.
                    self.history_manager.replace_last_assistant_step(new_turn_content)

                    new_final_answer = None
                    for tag, val in parse_react_block(new_turn_content):
                        if tag == "answer":
                            new_final_answer = val.strip()

                    if new_final_answer is None:
                        if VERIFIER_DEBUG:
                            print("        | --- REGENERATE: no answer block -> FALLBACK ---")
                        # R-3: ledger 已记 original verdict; best() 自然选它。
                        return self._safe_fallback_for_hard_verifier_failure(
                            verify_result, latest_user_text,
                        ), False

                    # Re-verify the regenerated answer
                    new_result = self._verify_final_answer(new_final_answer, latest_user_text)
                    self.last_verifier_result = new_result
                    # R-3: regen 候选也进 ledger。即使 new_result.ok=True 也加，
                    # 这样 ledger 有完整记录（debug/telemetry 可用）。后续 best()
                    # 会自然选 severity 最低的（regen 成功 = severity 0 胜出）。
                    self.verdict_ledger.add(Verdict(
                        answer=new_final_answer,
                        source_stage="regen",
                        verify_result=new_result,
                    ))
                    if new_result.ok:
                        if VERIFIER_DEBUG:
                            print("        | --- VERIFIER REGENERATE SUCCESS ---")
                        return new_final_answer, False

                    # Second attempt also failed — canned fallback (no recursion).
                    # R-3 (2026-05-13)：原 P0-3 patch 在这里手动传
                    # `phase2_answer=final_answer` (original pre-regen)，
                    # 防止释放被 regen 二次破坏的版本。R-3 后 verdict_ledger
                    # 自动选 best（按 severity/reason_count/stage 排序），
                    # original 因 stage_order 更早自然胜出——P0-3 补丁的
                    # 语义被 ledger 范式吸收。
                    if VERIFIER_DEBUG:
                        print("        | --- VERIFIER REGENERATE STILL FAILED -> FALLBACK ---")
                        snapshot = [(v.source_stage, v.severity, v.reason_count)
                                    for v in self.verdict_ledger]
                        print(f"        | [DEBUG] verdict_ledger snapshot: {snapshot}")
                    return self._safe_fallback_for_hard_verifier_failure(
                        new_result, latest_user_text,
                    ), False

                # Path 2: hard fail -> safe fallback; do not inject verifier feedback as tool_output.
                if verify_result.hard_fail:
                    if VERIFIER_DEBUG:
                        print("        | --- ANSWER VERIFIER HARD FAIL FALLBACK ---")
                    self.last_required_action_reason = None
                    # R-3: ledger 已含 original verdict, fallback 自动选。
                    return self._safe_fallback_for_hard_verifier_failure(
                        verify_result, latest_user_text,
                    ), False

                # Path 3: soft fail -> pass through; avoid polluting history.
                if VERIFIER_DEBUG:
                    print("        | --- ANSWER VERIFIER SOFT FAIL; PASSING THROUGH ---")
                self.last_required_action_reason = None

        return final_answer, False

    def run(self, user_text, user_name="Guest", image_path=None, progress_callback=None):
        self.progress_callback = progress_callback
        self.history_manager.set_user_name(user_name)
        self.current_image = None
        if image_path:
            self.current_image = safe_load_image(image_path, max_pixels=LOCAL_PIXELS)
            if self.current_image is None:
                print(f"{self.INDENT_STEP}[Vision] Error: failed to load image: {image_path}")
        self.history_manager.start_turn(user_text, has_image=(self.current_image is not None),
                                        image=self.current_image, image_path=image_path)
        self._reset_turn_evidence()
        self.active_memory_context = ""
        self.active_memory_turn_key = None
        self.active_memory_low_confidence = False
        self.advisor_advice = ""
        # Advisor-first refactor: advisor_result is the source of truth for
        # this turn's intent classification. Clear it at turn start so a
        # stale result from a previous turn can never leak.
        self.advisor_result = None
        self.current_turn_memory_grounded = False
        self.current_turn_memory_has_exact = False
        self.current_turn_memory_has_related = False
        self.current_turn_memory_judge_exact_count = 0
        self.current_turn_memory_judge_related_count = 0
        self._collapse_detected_for_current_turn = False
        # LLM judge per-turn state — fresh budget + empty cache for
        # this turn so judges can run again on a new question. Wipe
        # in-place rather than reassigning so any other reference (in
        # principle: none) sees the cleared state.
        _judge_reset_state(self._llm_judge_state)
        print(f"{self.INDENT_STEP}--- PROCESSING START ---")
        final_ans = None
        for step_idx in range(MAX_STEPS):
            ans, is_tool = self.step_once()
            if not is_tool:
                final_ans = ans
                break
        finished_answer = self.history_manager.finish_turn()
        if final_ans: return final_ans
        if finished_answer: return finished_answer
        if user_name.lower() == "rosm":
            return "I've processed your request, Master."
        return f"I've processed your request, {user_name}."


if __name__ == "__main__":
    print("==================================================")
    print("  Initializing Eva Core legacy primitives v22.0 P2 (used by eva_inference_P2.py)...")
    print("==================================================")
    agent = ChatAgent()
    user_name = input("\n[System] Enter your name (Press Enter for 'Rosm'): ").strip() or "Rosm"
    print(f"\n[System] Ready. Welcome, {user_name}.")
    while True:
        try:
            user_input = input(f"\n[{user_name}]: ").strip()
            if user_input.lower() in ['exit', 'quit']: break
            if not user_input: continue
            image_path = None
            if user_input.startswith("/image "):
                parts = user_input.split(" ", 2)
                if len(parts) >= 2:
                    image_path = parts[1]
                    user_input = parts[2] if len(parts) > 2 else "Describe this image."
            response = agent.run(user_input, user_name, image_path, lambda msg: print(f"  {msg}"))
            print(f"\n[Eva]: {response}")
        except Exception:
            traceback.print_exc()
