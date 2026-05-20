"""
eva_inference_P2.py — Eva inference main entry (P2 refactor).

What this file does:
1. Imports the battle-tested ChatAgent from eva_core.
2. Overrides ONE hot path — `_refresh_active_memory_for_current_turn` —
   to use the new topic-based MemoryModule (eva_memory_v2.py).
3. Boot block creates the agent and runs an interactive REPL.

What is preserved from legacy (do not touch):
- ChatAgent.__init__: model load, processor, FAISS resources
- HistoryManager + image persistence + history compaction
- ReAct two-phase loop (phase-1 plan, phase-2 answer)
- Tool execution / verifier / route judge / collapse guard

What is replaced:
- The PRE MEMORY PROBE entry. Old path:
    _should_skip_memory_probe_obvious  →  _build_active_memory_query_bundle
    →  _run_active_memory_retrieval  →  _active_memory_should_inject
    →  _format_active_memory_evidence_packet
  New path (single MemoryModule call):
    decide(user_text)  →  probe(...)  →  should_inject(...)  →  format_packet(...)

Usage in Colab (multi-cell or single):
    # Cell 1: imports
    import eva_inference_P2 as eva
    agent = eva.build_agent()

    # Cell 2: chat
    eva.run_repl(agent)

Usage in PyCharm:
    python eva_inference_P2.py
"""

import os

# Local imports — ordered so eva_config / eva_prompts initialise before legacy.
from eva_config import TOPIC_KEYWORDS_PATH
from eva_memory_v2 import (
    MemoryModule,
    TopicDictionary,
    KeywordIntentClassifier,
    LLMIntentClassifier,
    LayeredIntentClassifier,
)
from eva_intent_judge import judge_topic_subset as _judge_topic_subset
from eva_intent_judge import synthesize_tool_thought as _synthesize_tool_thought

# Legacy ChatAgent + memory format helpers (we do not subclass — we monkey-patch
# one method to keep all the unchanged behaviour from legacy intact).
from eva_core import (
    ChatAgent as _LegacyChatAgent,
    MEMORY_SLOT_FIELDS,
)


# ============================================================
# 1. Monkey-patch the PRE MEMORY PROBE entry on ChatAgent.
# ============================================================
def _new_refresh_active_memory(self):
    """P2 PRE MEMORY PROBE — driven by topic_keywords.json + dual-path retrieval.

    Replaces _should_skip_memory_probe_obvious + _build_active_memory_query_bundle
    + _run_active_memory_retrieval + _active_memory_should_inject + the legacy
    packet formatter, all in one cohesive flow.
    """
    latest_user_text = self._get_latest_user_text()
    turn_key = (latest_user_text or "").strip()

    # Once the model has emitted any assistant step (typically a tool call),
    # subsequent step prompts no longer carry the active memory packet.
    current_turn = self.history_manager.current_turn
    if current_turn is not None and len(getattr(current_turn, "assistant_steps", []) or []) > 0:
        if os.environ.get("P2_DEBUG_PROBE", "").strip().lower() in {"1", "true", "yes"}:
            print(f"        | [P2-DEBUG] skip: assistant_steps already present (count={len(current_turn.assistant_steps)})")
        self.active_memory_context = ""
        return

    # Same-turn cache: do not re-probe within a single turn.
    if self.active_memory_turn_key == turn_key:
        if os.environ.get("P2_DEBUG_PROBE", "").strip().lower() in {"1", "true", "yes"}:
            print(f"        | [P2-DEBUG] skip: same turn_key already probed ({turn_key[:60]!r})")
        return
    self.active_memory_turn_key = turn_key

    # Reset turn-scoped memory state.
    self.active_memory_context = ""
    self.active_memory_low_confidence = False
    self.current_turn_memory_grounded = False
    self.current_turn_memory_has_exact = False
    self.current_turn_memory_has_related = False
    self.current_turn_memory_judge_exact_count = 0
    self.current_turn_memory_judge_related_count = 0
    self.current_turn_slot_evidence = {}

    if not turn_key:
        return

    # Sync current_user from history_manager BEFORE decide(): the
    # speaker info flows from agent.run(user_name=...) into
    # history_manager.user_name, but MemoryModule.current_user is
    # construction-time bound. Without this sync, the LLM judge
    # always sees speaker="Guest" regardless of actual caller.
    hm_user = getattr(self.history_manager, "user_name", None)
    if hm_user:
        self.memory_module.current_user = hm_user

    # ============================================================
    # Step 0: Advisor FIRST (2026-05-13 Advisor-first refactor)
    # ============================================================
    # Replaces the previous 4-5 DeepSeek judge calls used to classify the
    # turn (EXPLICIT_MEMORY / EXPLICIT_WEB / PUBLIC_FACT / PRE_PROBE_RELEVANCE).
    # Advisor outputs intent + needs_memory_retrieval + needs_web_search in
    # one shot; the runtime trusts it and routes accordingly. Cuts per-turn
    # latency by ~5 DeepSeek round-trips.
    #
    # Failure mode: if advisor returns ok=False (timeout / API down / disabled),
    # fall back to the OLD decide() flow with local LLM intent classifier.
    advisor_result = _run_advisor(self, turn_key)
    self.advisor_result = advisor_result  # accessible to verifier / step_once

    # The advice block goes into the prompt suffix on every path (even fallback
    # when advice happens to be empty). _sys_prompt skips empty blocks.
    self.advisor_advice = _format_advice_block(advisor_result)

    print("\n        | --- P2 PRE MEMORY PROBE ---")
    if advisor_result.ok:
        print(f"        | advisor: intent={advisor_result.intent!r} "
              f"needs_memory={advisor_result.needs_memory_retrieval} "
              f"needs_web={advisor_result.needs_web_search}")

    # ---- Decide whether to run the local memory probe ----
    decision, memory_query, explicit_request = _gate_memory_probe(
        self, turn_key, advisor_result,
    )

    if decision is None:
        # Advisor (or fallback decide) said "no memory probe". advisor_advice
        # is already set; nothing more to do here.
        return

    if decision["matched_topics"]:
        print(f"        | matched_topics={decision['matched_topics']}")

    # R-4 Step 2c：TopicDictionary 命中也是一种"本轮已 ground" 的证据，
    # 哪怕没进 retrieval。verifier 的 covers(subject, topic) 查询要能看到这条。
    # 每个 matched_topic 写一行 source="topic_dict"。subject 取 subject_hint，
    # 多个 hint 时跟 decide() 一样按"单一 hint 才填"原则处理。
    matched = decision.get("matched_topics") or []
    if matched:
        hints = {self.memory_module.topic_dict.subject_hint(t) for t in matched}
        hints.discard(None)
        single_subject = next(iter(hints)) if len(hints) == 1 else None
        for t in matched:
            self._add_turn_evidence(
                source="topic_dict",
                subject=single_subject,
                slot=None,
                value=t,
                confidence="topic",
                raw_text=f"topic_dict matched: {t}",
                topic=t,
                meta={"channel": "pre_probe",
                      "decide_reason": decision.get("reason", "")},
            )

    # ---- Step 2: probe (dual-path) ----
    if self.progress_callback:
        self.progress_callback("Eva is probing relevant memory.")

    try:
        probe_result = self.memory_module.probe(
            user_text=memory_query,
            target_entity=decision["target_entity"],
            matched_topics=decision["matched_topics"],
        )
    except Exception as e:
        import traceback
        print(f"        | probe failed: {type(e).__name__}: {e}")
        print("        | --- traceback ---")
        for line in traceback.format_exc().splitlines():
            print(f"        | {line}")
        self.active_memory_low_confidence = True
        # Advisor already ran at the top; no need to invoke again here.
        return

    print(f"        | exact={probe_result['exact_count']} "
          f"related={probe_result['related_count']} "
          f"topic_hit={probe_result['topic_hit_count']} "
          f"meta_kw_hit={probe_result.get('meta_keyword_hit_count', 0)} "
          f"top1={probe_result['top1_score']:.2f} "
          f"target={probe_result['target_entity']}")

    # ---- Step 3: should we inject? ----
    inject, inject_reason = self.memory_module.should_inject(probe_result)
    print(f"        | inject={inject} reason={inject_reason}")
    if not inject:
        # Advisor already injected advice at the top of this function;
        # nothing more to do here.
        return

    # ---- Step 4: format packet & set state ----
    fmt = self.memory_module.format_packet(
        probe_result=probe_result,
        user_text=turn_key,
        inject_reason=inject_reason,
    )
    packet = fmt["packet"]
    observation = fmt["observation"]

    # Update turn-state flags (verifier + downstream code reads these).
    self.current_turn_memory_has_exact = bool(probe_result["exact_count"] > 0
                                              or probe_result["topic_hit_count"] > 0)
    self.current_turn_memory_has_related = bool(probe_result["related_count"] > 0)
    self.current_turn_memory_judge_exact_count = int(probe_result["exact_count"]
                                                     + probe_result["topic_hit_count"])
    self.current_turn_memory_judge_related_count = int(probe_result["related_count"])
    self.current_turn_memory_grounded = bool(self.current_turn_memory_has_exact
                                             or self.current_turn_memory_has_related)
    self.active_memory_low_confidence = (
        not self.current_turn_memory_has_exact
        and probe_result["top1_score"] < 5.0
    )

    # R-6: 缓存 retrieval 快照 + 更新 dialog focus。PRE PROBE 的 subject_hint
    # 是最有信号的 entity 来源——topic 命中通常带 single-subject hint。
    if observation:
        self.last_memory.observation = observation
        self.last_memory.primary_query = turn_key
        self.last_memory.has_exact = self.current_turn_memory_has_exact
        self.last_memory.has_related = self.current_turn_memory_has_related
        self.last_memory.judge_exact_count = self.current_turn_memory_judge_exact_count
        self.last_memory.judge_related_count = self.current_turn_memory_judge_related_count
        # DialogFocus.entity 更新。topic 留给后续——matched_topics 在 decide
        # 阶段已经写过 evidence ledger 了，这里 dialog_focus 只承载"最近活跃
        # 的实体"。
        self.dialog_focus.update(
            entity=probe_result["target_entity"],
            turn=len(self.history_manager.history),
            source="pre_probe",
        )

    # Slot evidence (legacy verifier still reads these fields).
    requested_slots = list(fmt["requested_slots"])
    self.current_turn_missing_slots = [
        s for s in requested_slots if s in MEMORY_SLOT_FIELDS
    ]
    if requested_slots:
        # R-6: missing_slots 进 last_memory；target_entity 已写 dialog_focus
        self.last_memory.missing_slots = list(self.current_turn_missing_slots)

    # Record evidence for the verifier.
    fake_retrieval_result = {
        "text": observation,
        "exact_answer_found": self.current_turn_memory_has_exact,
        "related_evidence_found": self.current_turn_memory_has_related,
        "judge_exact_count": self.current_turn_memory_judge_exact_count,
        "judge_related_count": self.current_turn_memory_judge_related_count,
        "is_low_confidence": self.active_memory_low_confidence,
        "is_empty": not bool(probe_result["records"]),
        # R-1.1 (2026-05-13)：把 format_packet 透传上来的 slot_evidence 接进
        # 这里，让 _record_active_memory_evidence 能把 slot-level evidence
        # 入账 TurnEvidenceLedger。否则 verifier 的 _exact_memory_evidence_for
        # (subject="Eva", slot="toy") 看不到 PRE PROBE 已经验证过的 slot
        # value，误报 unsupported_exact_toy_claim 强制再跑一次 MemorySearch。
        "slot_evidence": fmt.get("slot_evidence", {}) or {},
        "requested_slots": requested_slots,
    }
    try:
        self._record_active_memory_evidence(
            retrieval_result=fake_retrieval_result,
            target_entity=probe_result["target_entity"],
            query=turn_key,
        )
    except Exception as e:
        print(f"        | [WARN] _record_active_memory_evidence: {type(e).__name__}: {e}")

    # 2026-05-14: prepend a strong "memory pre-fetched" header so Eva
    # doesn't redundantly call MemorySearch when PRE PROBE already pulled
    # the relevant evidence. Without this header, advisor's "search your
    # memory" advice can prompt Eva to call MemorySearch a second time,
    # and the duplicate call's target_entity often gets mis-routed by
    # `_guard_memorysearch_params` text-inference (Eva self-talk "do I
    # have a toy" → "i" parsed as user-Rosm → wrong target → MISSING).
    # The header below is a single-line directive that supersedes advisor
    # tool-call hints when the data is already present.
    if self.current_turn_memory_has_exact or probe_result["top1_score"] >= 5.0:
        packet = (
            "[MEMORY ALREADY RETRIEVED THIS TURN — answer directly from the "
            "data below; do NOT call MemorySearch again unless you need a "
            "DIFFERENT topic. The records here are pre-fetched and authoritative.]\n\n"
            + packet
        )
    self.active_memory_context = packet
    print("        | --- P2 PRE MEMORY INJECTED ---")
    if fmt.get("strict_match_applied"):
        print("        | (strict-match caveat applied)")
    # Diagnostic: show what we actually injected (truncated)
    if os.environ.get("P2_DEBUG_PACKET", "").strip().lower() in {"1", "true", "yes"}:
        preview = (packet or "")[:1200]
        print("        | --- P2 PACKET PREVIEW (first 1200 chars) ---")
        for line in preview.splitlines():
            print(f"        | {line}")
        if len(packet or "") > 1200:
            print(f"        | ... ({len(packet) - 1200} more chars truncated)")

    # End of probe block — advisor already ran at the top.


# ============================================================
# Advisor-first helpers (2026-05-13)
# ============================================================
def _run_advisor(self, user_text: str):
    """Call the remote advisor and return its AdvisorResult.

    Side-effect-free w.r.t. agent state — caller decides what to do with
    the result. On any failure returns an AdvisorResult with ok=False so
    callers can branch on a single attribute.
    """
    from Advisor.advisor_client import AdvisorResult

    try:
        from eva_config import (
            ENABLE_ADVISOR, ADVISOR_MODEL, ADVISOR_TIMEOUT_SECONDS,
            ADVISOR_MAX_CALLS_PER_TURN, ADVISOR_HISTORY_TURNS, ADVISOR_DEBUG,
        )
    except ImportError:
        return AdvisorResult(ok=False, source="fallback_disabled",
                              error="eva_config import failed")
    if not ENABLE_ADVISOR:
        return AdvisorResult(ok=False, source="fallback_disabled",
                              error="advisor disabled via ENABLE_ADVISOR")

    try:
        from Advisor import get_advice
    except Exception as e:
        print(f"        | [Advisor] import failed: {type(e).__name__}: {e}")
        return AdvisorResult(ok=False, source="fallback_error",
                              error=f"import: {type(e).__name__}: {e}")

    # ---- gather context for the advisor payload ----
    history_lines: list[str] = []
    try:
        hist = list(self.history_manager.history)[-ADVISOR_HISTORY_TURNS:]
        for t in hist:
            user_line = (getattr(t, "user_content", "") or "").strip()
            asst_line = ""
            steps = getattr(t, "assistant_steps", None) or []
            if steps:
                last_step = steps[-1]
                if isinstance(last_step, dict):
                    asst_line = (last_step.get("content") or "").strip()
                else:
                    asst_line = (getattr(last_step, "content", "") or "").strip()
            if user_line:
                history_lines.append(f"User: {user_line[:300]}")
            if asst_line:
                history_lines.append(f"Eva: {asst_line[:300]}")
    except Exception:
        history_lines = []

    # ---- Live notes preview for advisor ----
    # Two sources, in priority order:
    #   1. NotesStore.recent_adds — session LRU (newest first). Covers the
    #      common "user just stored a note → user mentions it" pattern.
    #   2. All live notes by created_at desc — covers the "user stored a
    #      note in a previous session, now references it" pattern. Without
    #      this fallback the advisor would have no idea such notes exist
    #      after a session restart (recent_adds is session-scoped).
    # Cap at 5 entries to keep the advisor payload short.
    recent_notes: list[dict] = []
    try:
        ns = self.memory_state.get("notes_store") if self.memory_state else None
        if ns is not None:
            seen_ids: set[str] = set()
            id_to_idx = {m.get("note_id"): i for i, m in enumerate(ns.metas)}

            # Source 1: session recent_adds (newest first)
            for nid in (getattr(ns, "recent_adds", None) or [])[:5]:
                if nid in seen_ids:
                    continue
                idx = id_to_idx.get(nid)
                if idx is None:
                    continue
                m = ns.metas[idx]
                if m.get("deleted"):
                    continue
                recent_notes.append({
                    "note_id": nid,
                    "topic": m.get("topic") or "-",
                    "preview": (ns.contents[idx] or "")[:160],
                })
                seen_ids.add(nid)

            # Source 2: all live notes by created_at desc, fill remaining slots
            if len(recent_notes) < 5:
                all_live = []
                for i, m in enumerate(ns.metas):
                    if m.get("deleted"):
                        continue
                    nid = m.get("note_id")
                    if not nid or nid in seen_ids:
                        continue
                    all_live.append((m.get("created_at") or "", i, m, nid))
                # ISO8601 timestamps are lexicographically sortable
                all_live.sort(key=lambda x: x[0], reverse=True)
                for _, idx, m, nid in all_live[:5 - len(recent_notes)]:
                    recent_notes.append({
                        "note_id": nid,
                        "topic": m.get("topic") or "-",
                        "preview": (ns.contents[idx] or "")[:160],
                    })
                    seen_ids.add(nid)
    except Exception:
        recent_notes = []

    # Advisor-first: at this point the memory probe has NOT run yet, so
    # active_memory_context is empty. relevant_memory stays "" and the
    # advisor decides on its own whether memory retrieval is needed.
    relevant_memory = ""

    # Budget tracking per turn
    if not hasattr(self, "_advisor_budget_state") or self._advisor_budget_state is None:
        self._advisor_budget_state = {"advisor_calls": ADVISOR_MAX_CALLS_PER_TURN}
    if getattr(self, "_advisor_last_turn_key", None) != user_text:
        self._advisor_budget_state["advisor_calls"] = ADVISOR_MAX_CALLS_PER_TURN
        self._advisor_last_turn_key = user_text

    result = get_advice(
        user_text=user_text,
        history_lines=history_lines,
        eva_state=None,
        recent_notes=recent_notes,
        relevant_memory=relevant_memory,
        enabled=True,
        timeout_seconds=ADVISOR_TIMEOUT_SECONDS,
        model=ADVISOR_MODEL,
        budget_state=self._advisor_budget_state,
        debug=ADVISOR_DEBUG,
    )

    print(f"        | --- ADVISOR --- source={result.source} "
          f"ok={result.ok} latency={result.latency_ms}ms")
    if result.ok:
        print(f"        | advice: {result.advice[:200]}")
        print(f"        | intent={result.intent!r} "
              f"needs_memory={result.needs_memory_retrieval} "
              f"mem_hint={result.memory_query_hint!r} "
              f"needs_web={result.needs_web_search} "
              f"web_hint={result.web_query_hint!r}")
        if result.suggested_calls:
            print(f"        | suggested_calls: {result.suggested_calls}")
    elif result.error:
        print(f"        | advisor fallback: {result.error}")

    return result


def _format_advice_block(result) -> str:
    """Wrap _format_advice_block import side-effects in a tiny helper so
    the top-level function reads cleanly."""
    if not result or not getattr(result, "ok", False) or not result.advice:
        return ""
    from Advisor.advisor_client import format_advice_block
    return format_advice_block(result)


def _gate_memory_probe(self, turn_key: str, advisor_result):
    """Decide whether to run the local FAISS probe this turn.

    Returns (decision_dict | None, memory_query: str, explicit_request: bool).
      - decision_dict None means "skip the probe entirely"
      - non-None means "probe with this decision config"

    Priority order:
      1. Advisor ok and confident (intent != "unknown"):
         - intent in {chat, query_external}  AND needs_memory_retrieval=False
           → skip probe (return None)
         - any intent with needs_memory_retrieval=True
           → probe; use memory_query_hint if given, else turn_key
         - intent={remember, forget, query_memory} but advisor said no memory
           → still probe (these intents almost always benefit from context),
             use raw turn_key
      2. Advisor failed (ok=False) → fall back to OLD decide() flow with
         local LLM intent classifier. Preserves pre-refactor behaviour as
         a safety net for advisor outages.
    """
    if advisor_result and advisor_result.ok and advisor_result.intent != "unknown":
        intent = advisor_result.intent
        needs_mem = advisor_result.needs_memory_retrieval
        hint = advisor_result.memory_query_hint

        # Skip probe: chat / query_external without memory hint
        if intent in ("chat", "query_external") and not needs_mem:
            print(f"        | action=skip reason=advisor_intent={intent}_no_memory_needed")
            return None, "", False

        # 2026-05-16: 只对 LIST-SAVED-NOTES 模式跳过 probe。其他时候 advisor
        # 仍仅作建议，PRE PROBE 保留 belt-and-suspenders 自由度。详细原因
        # 见 Advisor.advisor_client.is_list_saved_notes_pattern 注释。
        from Advisor.advisor_client import is_list_saved_notes_pattern
        if is_list_saved_notes_pattern(advisor_result):
            print(
                "        | action=skip reason=advisor_list_saved_notes_pattern "
                "(intent=query_memory + needs_mem=False + no MemorySearch in suggested)"
            )
            return None, "", False

        # Probe path: use hint if provided, else raw turn_key
        memory_query = hint if hint else turn_key

        # Memory-relevant intents always benefit from a probe, even if the
        # advisor forgot to set needs_memory_retrieval=True. This is a small
        # belt-and-suspenders to catch advisor under-specification.
        if intent in ("remember", "forget", "query_memory") or needs_mem:
            # 2026-05-14: enrich the synthetic decision so FAISS path-A
            # (topic-direct curated record pull) and target_entity filter
            # actually work. Without this, the probe sees target="Both" +
            # matched_topics=[] and degrades to pure FAISS-on-short-string,
            # which loses entries like the Eva-toy bunny record on a
            # one-word query "toy". Three signal sources, priority order:
            #   1. advisor.suggested_calls[i].args.target_entity (explicit)
            #   2. TopicDict.subject_hint(matched_topics) (single hint wins)
            #   3. "Both" (fallback)
            matched_topics = []
            try:
                matched_topics = self.memory_module.match_topics(memory_query)
            except Exception:
                matched_topics = []

            target_entity = "Both"

            # Source 1: advisor explicitly named target_entity for this tool
            for c in (advisor_result.suggested_calls or []):
                if not isinstance(c, dict):
                    continue
                if c.get("tool") != "MemorySearch":
                    continue
                args = c.get("args") or {}
                t = args.get("target_entity") if isinstance(args, dict) else None
                if isinstance(t, str) and t.strip():
                    from eva_memory_legacy import _canonical_target_entity
                    canon = _canonical_target_entity(
                        t.strip(),
                        current_user=(self.history_manager.user_name or "Guest"),
                    )
                    if canon in ("Eva", "Rosm", "Shared", "Both"):
                        target_entity = canon
                        break

            # Source 2: TopicDict subject_hint when topic match is unambiguous
            if target_entity == "Both" and matched_topics:
                try:
                    hints = {
                        self.memory_module.topic_dict.subject_hint(t)
                        for t in matched_topics
                    }
                    hints.discard(None)
                    if len(hints) == 1:
                        target_entity = next(iter(hints))
                except Exception:
                    pass

            reason_suffix = (
                f"+topics={','.join(matched_topics[:3])}"
                if matched_topics else ""
            )
            decision = {
                "action": "probe",
                "reason": f"advisor:intent={intent}{reason_suffix}",
                "target_entity": target_entity,
                "matched_topics": matched_topics,
            }
            print(f"        | action=probe reason=advisor:intent={intent} "
                  f"query_hint={memory_query!r} target={target_entity} "
                  f"topics={matched_topics}")
            return decision, memory_query, (intent in ("query_memory", "remember", "forget"))

        # Default: skip
        print(f"        | action=skip reason=advisor_intent={intent}_default")
        return None, "", False

    # ---------- Fallback: advisor unavailable / unknown intent ----------
    try:
        from eva_config import EVA_ADVISOR_FALLBACK_MODE
    except ImportError:
        EVA_ADVISOR_FALLBACK_MODE = "judges"

    if EVA_ADVISOR_FALLBACK_MODE == "chat":
        # Strict mode: when advisor fails, treat as chat. Cheapest, may miss
        # memory-bound queries. Use only when latency budget is tight.
        print("        | action=skip reason=advisor_failed_fallback_mode=chat")
        return None, "", False

    if EVA_ADVISOR_FALLBACK_MODE == "strict":
        # Strict mode: hard-fail without advisor. Returns None so probe is
        # skipped and Eva runs with whatever in-prompt context she has.
        print("        | action=skip reason=advisor_failed_fallback_mode=strict")
        return None, "", False

    # Default: "judges" — run the original local LLM intent classifier flow
    # so behaviour degrades to pre-refactor levels when advisor is down.
    explicit_request = bool(self._explicit_memory_check_request(turn_key))
    decision = self.memory_module.decide(
        user_text=turn_key,
        explicit_memory_request=explicit_request,
    )
    print(f"        | action={decision['action']} reason={decision['reason']} (advisor_fallback=judges)")

    # Preserve public-fact override in the fallback path.
    if (decision["action"] == "probe"
            and decision["reason"].startswith("topic_match")
            and not explicit_request
            and self._is_obvious_public_fact_or_news_query(turn_key)):
        print("        | action=skip reason=public_fact_query_overrides_topic_match")
        return None, "", explicit_request

    if decision["action"] == "skip":
        return None, "", explicit_request

    return decision, turn_key, explicit_request


_LegacyChatAgent._refresh_active_memory_for_current_turn = _new_refresh_active_memory


# ============================================================
# 2. Build agent factory + REPL
# ============================================================
class ChatAgent(_LegacyChatAgent):
    """Eva inference agent (P2).

    Inherits the full ChatAgent from legacy. The only behavioural
    difference is the PRE MEMORY PROBE entry, monkey-patched above.

    On __init__ we additionally:
    - load topic_keywords.json
    - construct MemoryModule and bind it as self.memory_module
    """

    def __init__(self):
        super().__init__()
        topic_dict = TopicDictionary(TOPIC_KEYWORDS_PATH)

        # Build layered intent classifier (TODO 4 Step 3):
        #   keyword baseline -> LLM judge subset filter
        #
        # The judge_fn closes over self._llm_judge_state so PRE PROBE
        # judge calls share the per-turn cache + budget pool with the
        # Plan B verifier judge calls. This was the locked design
        # decision from conversation 2026-05-06: state is owned by
        # ChatAgent (option A), not module-level.
        #
        # Speaker (current_user) is forwarded to judge_topic_subset so
        # the dual-subject prompt (Eva + Rosm profiles) can disambiguate
        # first-person queries: 'what do I like' from Rosm probes
        # Rosm's profile, the same query from a Guest rejects candidates.
        # Step 4 fix (post-Step-4 regression: rosm_hobby fail).
        #
        # Failure mode: if judge_topic_subset returns None (disabled,
        # over budget, or DeepSeek error), LLMIntentClassifier degrades
        # to "all input candidates relevant" — equivalent to keyword-
        # only baseline. PRE PROBE behaviour cannot become MORE
        # restrictive than today as a result of judge failure.
        def _pre_probe_judge_fn(query, candidates, speaker="Rosm"):
            return _judge_topic_subset(
                query, candidates,
                state=self._llm_judge_state,
                speaker=speaker,
            )

        intent_classifier = LayeredIntentClassifier(
            KeywordIntentClassifier(),
            LLMIntentClassifier(_pre_probe_judge_fn),
        )

        self.memory_module = MemoryModule(
            memory_state=getattr(self, "memory_state", None),
            encoder=getattr(self, "encoder", None),
            reranker=getattr(self, "reranker", None),
            topic_dict=topic_dict,
            current_user="Guest",
            intent_classifier=intent_classifier,
        )
        print("[P2] MemoryModule wired. Topic-based PRE PROBE active "
              "(Layered: Keyword + LLM).")

    def _synthesize_repair_thought(self, query, tool_name, tool_args_summary):
        """Generate an in-character thought for verifier-injected tool repair.

        Used by `eva_verifier_logic.execute_controller_tool` (Step 5
        trace rewriting) to fill the `<think>...</think>` block when
        rewriting the last assistant turn from answer-shape to
        tool-call-shape.

        Returns:
            str | None — the synthesised thought (max 200 chars,
                post-validated). None on failure / disabled / over
                budget. The verifier site MUST handle None by
                substituting a hardcoded per-tool template; never
                emit an empty <think></think>.

        Closes over self._llm_judge_state so this DeepSeek call shares
        the per-turn cache + budget pool with judge_intent and
        judge_topic_subset (LLM_JUDGE_MAX_CALLS_PER_TURN=6 across all
        three).
        """
        # P5 (2026-05-08): pass recent turns so the judge can resolve
        # pronoun follow-ups ('check it', 'do that again') against
        # the most recent (user, assistant) exchanges. Falls through
        # to legacy behaviour (no history) if history_manager is
        # missing or recent_turns raises.
        recent = None
        try:
            if hasattr(self, "history_manager") and self.history_manager is not None:
                recent = self.history_manager.recent_turns(n=2)
        except Exception:
            recent = None

        return _synthesize_tool_thought(
            query, tool_name, tool_args_summary,
            state=self._llm_judge_state,
            recent_turns=recent,
        )


def build_agent() -> ChatAgent:
    """Construct and return a ready-to-use Eva agent.

    Use this in Colab cell 1 to load the model + memory + topic dict.
    Subsequent cells can call `agent.run(user_text=..., user_name=..., image_path=...)`.

    User-notes (RememberThis / ForgetMemory) are enabled by default via
    `eva_config.ENABLE_USER_NOTES = True`. Notes persist across sessions
    in `eva_config.NOTES_DIR` (default `Notes/`). To disable entirely
    (e.g. for ablation), set `ENABLE_USER_NOTES = False` in eva_config
    and rebuild the agent. To wipe accumulated notes at runtime call
    `agent.memory_state["notes_store"].discard()`.
    """
    print("=" * 60)
    print("  Eva inference P2 — booting")
    print("=" * 60)
    agent = ChatAgent()

    from eva_config import ENABLE_USER_NOTES, NOTES_DIR
    if ENABLE_USER_NOTES:
        from pathlib import Path
        from Memory_maker.notes_runtime import NotesStore

        if agent.encoder is None:
            print("[notes] WARN: agent.encoder is None — NotesStore will "
                  "lazy-load its own mpnet on first add/search. This wastes "
                  "memory; install sentence-transformers to share.")
            shared_encoder = None
        else:
            # Share the production mpnet encoder; ensures notes vectors
            # live in the exact same embedding space as the lore corpus.
            def shared_encoder(texts):
                emb = agent.encoder.encode(texts, normalize_embeddings=True)
                # SentenceTransformer returns numpy already; coerce shape.
                import numpy as _np
                return _np.asarray(emb, dtype="float32")

        # Embedding dim of mpnet-base-v2 is 768; align notes store to match.
        notes_store = NotesStore(
            root=Path(NOTES_DIR),
            encoder=shared_encoder,
            dim=768,
        )
        if agent.memory_state is None:
            agent.memory_state = {}
        agent.memory_state["notes_store"] = notes_store
        st = notes_store.status()
        print(f"[notes] Store ready at {NOTES_DIR}/ — "
              f"live={st['live']} deleted={st['deleted']} "
              f"session_id={st['session_id']}")

    return agent


def run_repl(agent: ChatAgent) -> None:
    """Simple stdin REPL. Use Ctrl+C / 'exit' to quit."""
    print("\nEva is ready. Type 'exit' to quit, 'image:<path>' to attach an image.")
    pending_image: str = ""
    while True:
        try:
            user_text = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[bye]")
            return
        if not user_text:
            continue
        if user_text.lower() in ("exit", "quit"):
            print("[bye]")
            return
        if user_text.lower().startswith("image:"):
            pending_image = user_text[6:].strip().strip('"\'')
            print(f"[image queued for next turn] {pending_image}")
            continue
        try:
            agent.run(user_text=user_text,
                      user_name="Guest",
                      image_path=pending_image or None)
            pending_image = ""
        except Exception as e:
            import traceback
            print(f"[ERR] {type(e).__name__}: {e}")
            traceback.print_exc()


# ============================================================
# 3. Direct invocation (PyCharm `python eva_inference_P2.py`)
# ============================================================
if __name__ == "__main__":
    agent = build_agent()
    run_repl(agent)
