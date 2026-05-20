"""Advisor client — DeepSeek call with cache, timeout, and silent fallback.

Single public entry point: `get_advice(context_dict) -> AdvisorResult`.

设计原则：
  1. **永不抛错**。Advisor 是辅助层；任何失败都返回 ok=False + 空 advice，
     上游照常推理。
  2. **延迟可控**。硬 timeout（默认 4s），超过即放弃；本轮无 advice。
  3. **同输入缓存**。同一个 (user_text, history_digest) 在 60s 窗口内重用，
     防止 regenerate 路径反复打 API。
  4. **预算上限**。每个 ChatAgent 实例可选维护一个 budget counter（passed
     in via state dict），用完该轮也不再调用。

输出格式约定（DeepSeek 端 system prompt 强制）：
    {
      "advice": "1-3 行自然语言建议",
      "suggested_calls": [{"tool": "...", "args": {...}}, ...]  # optional
    }

local 模型只读 "advice" 文本块，不解析 suggested_calls——后者只用于 logging /
未来 A/B 比较。
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AdvisorResult:
    """Outcome of a single advisor call.

    Attributes:
        ok: True iff the call returned valid advice (non-empty `advice` text).
        advice: The natural-language hint to inject into Eva's prompt. Empty
                string if ok=False.
        intent: Coarse intent label that the runtime uses to short-circuit
                local intent classifiers (EXPLICIT_MEMORY / EXPLICIT_WEB /
                PUBLIC_FACT etc.). One of:
                  "chat"            — pure conversation, no tool needed
                  "remember"        — user wants something stored (RememberThis)
                  "forget"          — user wants something deleted (ForgetMemory)
                  "query_memory"    — user wants Eva to recall stored facts
                  "query_external"  — user wants external/current info (WebSearch)
                  "mixed"           — compound input spanning multiple intents
                  "unknown"         — advisor not confident; runtime falls back
        needs_memory_retrieval: Should the local FAISS probe run this turn?
                Independent from intent — "chat" can still want background
                retrieval ("tell me about yourself"). True when Eva would
                benefit from seeing relevant memory in her prompt.
        memory_query_hint: Optional query string for the FAISS probe. None
                means "use user_text directly". Set when advisor wants the
                probe to use a more focused query than the raw input.
        needs_web_search: Hint to Eva that WebSearch is appropriate this turn.
        web_query_hint: Optional refined WebSearch query (None = use raw input).
        suggested_calls: Optional structured tool suggestions (logged + may
                drive runtime hints; Eva still emits her own tool_code).
        source: "live" | "cache" | "fallback_disabled" | "fallback_timeout"
                | "fallback_error" — for telemetry.
        latency_ms: How long the call took (excluding cache hit).
        error: Brief reason on failure (None on success).
    """
    ok: bool = False
    advice: str = ""
    intent: str = "unknown"
    needs_memory_retrieval: bool = False
    memory_query_hint: Optional[str] = None
    needs_web_search: bool = False
    web_query_hint: Optional[str] = None
    suggested_calls: list = field(default_factory=list)
    source: str = "fallback_disabled"
    latency_ms: int = 0
    error: Optional[str] = None


# ============================================================
# In-process cache (per-session, 60s TTL)
# ============================================================
# Cache key = sha1(user_text + recent_history_digest). Keeps regenerate /
# verifier retry loops from re-hitting DeepSeek with the same payload.
_ADVISOR_CACHE: dict[str, tuple[float, AdvisorResult]] = {}
_CACHE_TTL_SECONDS = 60.0


def _cache_key(user_text: str, history_digest: str) -> str:
    h = hashlib.sha1()
    h.update((user_text or "").encode("utf-8"))
    h.update(b"||")
    h.update((history_digest or "").encode("utf-8"))
    return h.hexdigest()


def _cache_get(key: str) -> Optional[AdvisorResult]:
    item = _ADVISOR_CACHE.get(key)
    if not item:
        return None
    ts, result = item
    if time.time() - ts > _CACHE_TTL_SECONDS:
        _ADVISOR_CACHE.pop(key, None)
        return None
    cached = AdvisorResult(
        ok=result.ok,
        advice=result.advice,
        intent=result.intent,
        needs_memory_retrieval=result.needs_memory_retrieval,
        memory_query_hint=result.memory_query_hint,
        needs_web_search=result.needs_web_search,
        web_query_hint=result.web_query_hint,
        suggested_calls=list(result.suggested_calls),
        source="cache",
        latency_ms=0,
        error=result.error,
    )
    return cached


def _cache_put(key: str, result: AdvisorResult) -> None:
    if not result.ok:
        return  # never cache failures
    _ADVISOR_CACHE[key] = (time.time(), result)


def reset_cache() -> None:
    """Clear the advisor cache. Called between Colab cells / tests."""
    _ADVISOR_CACHE.clear()


# ============================================================
# Advisor system prompt (DeepSeek-side)
# ============================================================
_ADVISOR_SYSTEM_PROMPT = """You are the SINGLE decision authority for a small local LLM named Eva.

Eva is a tsundere-maid AI assistant. You are her remote planning brain.
The runtime trusts your output to skip all other intent classifiers, so you
must classify the user's turn AND advise Eva, in one shot.

# Eva's tools

- MemorySearch(query, target_entity?)        — search Eva's curated lore + user notes
- WebSearch(query)                            — external / current info
- RememberThis(content, entity, topic, keywords, category?, slot_values?,
               event_date?, event_time?, event_type?, expires_at?, participants?)
- ForgetMemory(query?, record_id?, topic?, reason?)
- GetCurrentTime(target_entity?)              — current date/time; pass
                                                  target_entity="Eva" or "Rosm"
                                                  to scope a days-until-birthday
                                                  calc to ONE subject. For
                                                  compound queries ("days
                                                  until your AND my birthday"),
                                                  call it twice — once per entity.
- AskRemoteVision(query, image)               — image understanding
- TextGenerationTool(prompt)                  — long-form writing

# Eva's strengths and weaknesses

Strong:  natural language, persona, conversation flow
Weak:    decomposing compound input, choosing the right tool, tracking note ids

# Your job (per turn)

Output STRICT JSON with the following schema. ALL fields required.

{
  "advice": "<1-3 line natural-language hint for Eva — what to do this turn>",
  "intent": "<one of: chat | remember | forget | query_memory | query_external | mixed | unknown>",
  "needs_memory_retrieval": <true | false>,
  "memory_query_hint": "<string | null>",
  "needs_web_search": <true | false>,
  "web_query_hint": "<string | null>",
  "suggested_calls": [
    {"tool": "<ToolName>", "args": {<key params>}}
  ]
}

# Field semantics

- intent:
    "chat"            — pure conversation / greeting / opinion / persona play
    "remember"        — user explicitly wants something stored (Remember this:, 记一下, don't forget I…)
    "forget"          — user wants something deleted (forget X, scratch that, 忘了那件事)
    "query_memory"    — user wants Eva to recall a stored fact (my birthday, your toy)
    "query_external"  — user wants outside info (news, today's prices, who composed X)
    "mixed"           — compound input touching MULTIPLE DIFFERENT intent classes
                        in one message. Examples:
                          - "remember X and forget Y"      → remember + forget
                          - "what's my name and store this" → query_memory + remember
                        DO NOT use "mixed" for: two questions of the same kind
                        (e.g. "when is your birthday and my full name" is
                        query_memory + query_memory — emit "query_memory",
                        not "mixed", and put 2 MemorySearch in suggested_calls).
    "unknown"         — you are not confident; runtime will fall back

- needs_memory_retrieval:
    true if Eva should run a local FAISS lookup before answering (any "about
    Eva / Master / stored facts" question). Set even for chat-intent turns
    when relevant background would help Eva stay grounded.

- memory_query_hint:
    a SHORT query string (1-4 keywords) for the local FAISS probe. null =
    use raw user input. The probe runs against a sentence-embedding index
    (mpnet); verbose phrases like "Eva's toy or plush ownership history"
    DILUTE the signal and return zero results. Prefer "toy", "birthday",
    "favourite food", "next monday meeting" — concrete nouns + minimal
    qualifiers. When in doubt, leave null.

- needs_web_search / web_query_hint:
    same idea for WebSearch. Only true for "query_external" intent.

- suggested_calls:
    **SOURCE OF TRUTH** for verifier alignment. List every tool that should
    fire this turn, and ONLY those tools. The runtime verifier checks Eva's
    actual tool calls against this list — if you suggest WebSearch and Eva
    doesn't call it, the runtime flags a misalignment; if Eva calls a tool
    you didn't suggest, that's tolerated (Eva can exceed your guidance) but
    omissions are flagged.
    Rules:
      - For pure chat: []
      - For compound input ("buy bear AND finish report"): list TWO entries
        (e.g. RememberThis × 2 with different args)
      - When user asks N independent questions: list N MemorySearch entries
      - Use canonical tool names exactly: MemorySearch, WebSearch,
        RememberThis, ForgetMemory, GetCurrentTime, AskRemoteVision,
        TextGenerationTool

# Critical rules

1. COMPOUND DECOMPOSITION: if user input contains N independent facts/tasks
   ("buy a bear AND finish my report"), output exactly N items in
   suggested_calls (typically N x RememberThis), intent="mixed", and the
   advice text MUST say "call RememberThis N times".
   Special case — COMPOUND DATE QUERIES ("days until your birthday and my
   birthday", "when is X's birthday and Y's birthday"): each subject needs
   its own GetCurrentTime call with target_entity= explicitly set, plus
   one MemorySearch per subject for the actual birthday date. Without
   target_entity, GetCurrentTime defaults to one subject and the other
   gets silently dropped (causes "54 days" being misattributed across
   two different birthdays).

2. USE [recent_notes] AGGRESSIVELY: the [recent_notes] block in the user
   payload lists Eva's currently-stored user notes (note_id + topic +
   content preview). Whenever the user's question could be answered by
   any note in that list, you MUST:
     a) Reference it in your advice ("you have a note about X").
     b) For "forget X" intent: put the exact record_id in
        suggested_calls.args.record_id.
     c) For "check/recall/what about X" intent (SPECIFIC topic mentioned):
        suggest a MemorySearch whose query uses keywords pulled DIRECTLY
        from the matching note's content, not the user's wording. Example:
        note says "Master wants to buy Eva a birthday gift" and user asks
        "check note about buying" → query="birthday gift" (will hit),
        not query="buying" (often misses on mpnet).
     d) For "is there any record/note about X": if a note matches, say
        "yes, you have Note #abc12345 about X"; if none matches, say
        "no relevant note found".

3. **LIST-ALL PATTERN — CRITICAL**: detect semantically, not by phrase
   matching. Trigger when the user wants to **view / read / audit / be shown**
   their stored content without naming a specific topic. Phrases vary
   wildly across languages ("show me your notes", "let me see what you
   wrote down", "把你的笔记给我看", "what have you marked so far",
   "read me what you've got"); the defining signal is **viewing intent +
   no topic narrowing**.

   Semantic indicators (any combination is enough):
     a) Verbs of revealing/displaying ("show", "list", "display",
        "enumerate", "read out", "let me see", "let me check",
        "tell me what you have", "把…给我看", "给我看下你的…")
     b) Object = "your notes / records / reminders / memos / tips /
        笔记 / 记录 / 提醒"
     c) The user is auditing what Eva has stored — often comes right
        after a RememberThis turn ("did it stick?", "is it really
        saved?")
     d) Asking for totality: "all", "everything", "everything you have",
        "全部", "所有"

   When this signal fires (one or more of a-d, AND no specific topic
   keyword narrows it down to a single note):
     - intent = "query_memory"
     - needs_memory_retrieval = **false** (no FAISS probe needed —
       the [Eva's Saved Notes Index — ...] block in Eva's system prompt
       already lists every live note authoritatively)
     - suggested_calls = **[]** (no tool call needed)
     - advice = "Display ALL live notes from your Saved Notes Index in
       bullet format: '📌 <imperative verb + object> — <time/context>'
       (one bullet per note). Wrap the bullets with tsundere persona at
       top and bottom (NOT interleaved per line). Do NOT print raw 'Note
       #hex [Topic]:' fields (internal). Do NOT narrow to a single note
       (even one note still uses a bullet). Do NOT call MemorySearch."

   Ambiguity rule: when the user says something vague like "show me
   yours" / "tell me what you have" right after the user just asked
   you to remember something — **default to list-all** (safer than
   inventing a narrow MemorySearch query).

   Anti-hallucination flag: if `recent_notes` in your input shows live
   notes exist, you MUST acknowledge them. NEVER advise Eva to claim
   notes are "gone" / "deleted" / "forgotten" / "not saved" when the
   notes are clearly listed. If Eva keeps repeating that lie in recent
   history (visible in your input), explicitly call it out in advice:
   "Eva has been falsely claiming the note is gone — the note IS in
   her Saved Notes Index. Make her list it this turn."

4. **COMPLEX MATH → TextGenerationTool, NEVER local-only reasoning**:
   Eva's local 9B model is good at PERSONA and EXPLAINING known concepts
   but **bad at carrying multi-step computation reliably**. When the
   user's request requires precise numerical/symbolic work, delegate to
   TextGenerationTool (which routes to a stronger remote model). Triggers:

     - Matrix / linear algebra operations (multiplication, inverse,
       eigenvalues, SVD)
     - Derivatives, gradients, partial derivatives, chain rule
       application on specific functions
     - Integrals (definite or indefinite)
     - Solving equations or systems of equations
     - Numerical computation with specific values that need exact results
       (e.g. "what is 17! / 12!", "P(A|B) when P(A)=0.3, P(B|A)=0.7, ...")
     - Multi-step probability / statistics with numbers
     - Algebraic simplification of non-trivial expressions
     - Series / sequences requiring computation of specific terms
     - Optimization (find min/max of a given function)

   What's NOT this category (Eva alone is fine):
     - Trivial mental arithmetic ("what's 2+2?")
     - Stating a KNOWN formula symbolically ("write the softmax formula")
     - CONCEPTUAL explanations ("what does centering mean in DINO?")
     - Showing a derivation template at the concept level (NOT computing
       with specific numbers)

   When this signal fires:
     - intent = "chat" (TextGen is a chat-side capability, not memory/web)
     - needs_memory_retrieval = false (unless ALSO needs prior context)
     - suggested_calls = [{"tool": "TextGenerationTool",
                           "args": {"prompt": "<the math problem, restated
                                                cleanly + any constants
                                                the user gave>",
                                    "requirements": "<style: e.g. show
                                                     intermediate steps,
                                                     use Unicode math>"}}]
     - advice = "Call TextGenerationTool with the math problem — the
       local model will fumble multi-step computation. Restate the
       problem cleanly in the prompt arg. Wrap the tool's output in
       Eva's tsundere voice but include the full computed result
       verbatim (per TextGenerationTool's discipline)."

   Examples:
     ✓ User: "compute the gradient of L=(Wx-y)^2 wrt W" → TextGen
     ✓ User: "what's 47! mod 13?" → TextGen
     ✓ User: "if P(rain)=0.3 and P(traffic|rain)=0.8 ..." → TextGen
     ✗ User: "explain backprop intuitively" → Eva alone, intent=chat
     ✗ User: "write the cross-entropy formula" → Eva alone (known formula)

5. NEVER include raw tool-call syntax inside `advice`. Just describe what
   to do in plain English. Eva will format the call.

6. ALWAYS stay terse. advice ≤ 3 short lines.

7. UNKNOWN: if you genuinely cannot classify (very ambiguous), set
   intent="unknown" and provide a generic chat advice. The runtime will
   fall back to its own light classifier.
"""


# ============================================================
# Public entry
# ============================================================
def get_advice(
    *,
    user_text: str,
    history_lines: list[str],
    eva_state: Optional[dict] = None,
    recent_notes: Optional[list[dict]] = None,
    relevant_memory: Optional[str] = None,
    enabled: bool = True,
    timeout_seconds: float = 4.0,
    model: str = "deepseek-v4-flash",
    budget_state: Optional[dict] = None,
    debug: bool = False,
) -> AdvisorResult:
    """Call the remote advisor.

    Args:
        user_text:        latest user input (required).
        history_lines:    list of "speaker: text" strings, oldest-first.
                          Trim caller-side to last 3-5 turns.
        eva_state:        optional "正在做 X" state dict
                          ({current_activity, started_at, context}).
        recent_notes:     optional list of {note_id, topic, preview} for
                          forget/edit hints. Newest-first.
        relevant_memory:  optional pre-retrieved memory text block (one
                          paragraph).
        enabled:          if False, return immediate disabled fallback.
        timeout_seconds:  hard per-call timeout.
        model:            DeepSeek model name.
        budget_state:     optional dict with key "advisor_calls"; if set,
                          decremented per call; when ≤ 0 we skip.
        debug:            print payload / response if True.

    Returns:
        AdvisorResult — never raises.
    """
    if not enabled:
        return AdvisorResult(
            ok=False, source="fallback_disabled",
            error="advisor disabled via flag",
        )
    if not user_text or not user_text.strip():
        return AdvisorResult(
            ok=False, source="fallback_disabled",
            error="empty user_text",
        )

    # Budget check
    if budget_state is not None:
        remaining = budget_state.get("advisor_calls", 0)
        if remaining <= 0:
            return AdvisorResult(
                ok=False, source="fallback_disabled",
                error="advisor budget exhausted this turn",
            )

    # Build the user-payload (separate from system prompt).
    from .build_advisor_prompt import build_advisor_prompt
    payload = build_advisor_prompt(
        user_text=user_text,
        history_lines=history_lines or [],
        eva_state=eva_state,
        recent_notes=recent_notes or [],
        relevant_memory=relevant_memory or "",
    )

    # Cache check
    history_digest = hashlib.sha1(
        "\n".join(history_lines or []).encode("utf-8")
    ).hexdigest()[:16]
    ck = _cache_key(user_text, history_digest)
    cached = _cache_get(ck)
    if cached is not None:
        if debug:
            print(f"        | [Advisor] cache hit: {ck[:8]}")
        return cached

    # Budget consume (only after cache miss — cache hit is free)
    if budget_state is not None:
        budget_state["advisor_calls"] = budget_state.get("advisor_calls", 0) - 1

    # Live call
    t0 = time.time()
    try:
        # Lazy import to avoid hard dep on openai at module load.
        from openai import OpenAI
        from eva_config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL

        if not DEEPSEEK_API_KEY:
            return AdvisorResult(
                ok=False, source="fallback_error",
                error="DEEPSEEK_API_KEY not set",
                latency_ms=int((time.time() - t0) * 1000),
            )

        # max_retries=0：禁用 OpenAI SDK 默认的 2 次自动重试。否则单次
        # timeout=4s 实际可能跑到 12-14s（4s × 3 次尝试）。Advisor 失败时
        # 我们已有 fallback 路径（走 judges），重试反而拖慢主路径。
        # timeout 适度放宽给单次请求多一点头羽。
        client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
            timeout=timeout_seconds,
            max_retries=0,
        )
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _ADVISOR_SYSTEM_PROMPT},
                {"role": "user", "content": payload},
            ],
            temperature=0.2,  # mostly deterministic; small variance OK
            response_format={"type": "json_object"},
            stream=False,
        )
        raw_text = (resp.choices[0].message.content or "").strip()
        if debug:
            print(f"        | [Advisor] raw response: {raw_text[:300]}")
        try:
            parsed = json.loads(raw_text)
        except Exception as e:
            return AdvisorResult(
                ok=False, source="fallback_error",
                error=f"json_parse_failed: {e}",
                latency_ms=int((time.time() - t0) * 1000),
            )

        advice = str(parsed.get("advice", "") or "").strip()
        suggested = parsed.get("suggested_calls") or []
        if not isinstance(suggested, list):
            suggested = []
        if not advice:
            return AdvisorResult(
                ok=False, source="fallback_error",
                error="empty advice field",
                latency_ms=int((time.time() - t0) * 1000),
            )

        # New fields (Advisor-first refactor). All defaults are deliberately
        # safe: when advisor omits the field, runtime treats it as "no
        # information" and falls back to local heuristics.
        intent = str(parsed.get("intent", "unknown") or "unknown").strip().lower()
        _VALID_INTENTS = {
            "chat", "remember", "forget", "query_memory",
            "query_external", "mixed", "unknown",
        }
        if intent not in _VALID_INTENTS:
            intent = "unknown"

        def _to_bool(v, default=False):
            if isinstance(v, bool):
                return v
            if isinstance(v, str):
                return v.strip().lower() in {"true", "yes", "1"}
            return default

        def _to_opt_str(v):
            if v is None:
                return None
            s = str(v).strip()
            return s if s and s.lower() != "null" else None

        needs_memory = _to_bool(parsed.get("needs_memory_retrieval"), default=False)
        memory_hint = _to_opt_str(parsed.get("memory_query_hint"))
        needs_web = _to_bool(parsed.get("needs_web_search"), default=False)
        web_hint = _to_opt_str(parsed.get("web_query_hint"))

        result = AdvisorResult(
            ok=True,
            advice=advice,
            intent=intent,
            needs_memory_retrieval=needs_memory,
            memory_query_hint=memory_hint,
            needs_web_search=needs_web,
            web_query_hint=web_hint,
            suggested_calls=suggested,
            source="live",
            latency_ms=int((time.time() - t0) * 1000),
        )
        _cache_put(ck, result)
        return result

    except Exception as e:
        return AdvisorResult(
            ok=False, source="fallback_error",
            error=f"{type(e).__name__}: {e}",
            latency_ms=int((time.time() - t0) * 1000),
        )


def is_list_saved_notes_pattern(result: Optional[AdvisorResult]) -> bool:
    """LIST-SAVED-NOTES pattern detection — single source of truth.

    Returns True iff the advisor's structured output matches the
    list-saved-notes signal:

        intent              == "query_memory"
        needs_memory        == False
        suggested_calls does NOT include MemorySearch

    This trio is structurally unique to the "list/show my saved notes"
    request — every other query_memory case has either needs_memory=True
    or MemorySearch in suggested_calls (e.g. 'what's my birthday',
    'tell me about my Toy').

    Used by 3 call sites (all should branch on this same predicate):
      1. eva_inference_P2.py — skip PRE PROBE
      2. format_advice_block (below) — inject BINDING DIRECTIVE
      3. eva_core._sys_prompt — append the detailed Notes Index block
    """
    if not result or not getattr(result, "ok", False):
        return False
    if getattr(result, "intent", None) != "query_memory":
        return False
    if getattr(result, "needs_memory_retrieval", False):
        return False
    wants_memsearch = any(
        isinstance(c, dict) and c.get("tool") == "MemorySearch"
        for c in (getattr(result, "suggested_calls", None) or [])
    )
    return not wants_memsearch


def format_advice_block(result: AdvisorResult) -> str:
    """Format the advice for injection into Eva's system prompt suffix.

    Returns "" when ok=False so the suffix builder can naturally skip
    via `if block:` truth check.

    2026-05-16: BINDING DIRECTIVE is **narrowly scoped** to the
    LIST-SAVED-NOTES pattern (advisor 把 saved notes 列表当答案的场景).
    Other empty-suggested-calls cases (pure chat, ambiguous unknowns,
    etc.) only get the soft hint wrapper — they don't need hard binding
    because they wouldn't trigger tool calls anyway, or the model legitimately
    needs flexibility.

    The architectural principle: advisor is ADVISORY by default. The single
    exception is "list saved notes" where Eva's SFT distribution strongly
    biases her toward calling MemorySearch (which then pulls lore records
    and confuses them with saved notes — see the 2026-05-16 'show me your
    notes' incident).
    """
    if not result.ok or not result.advice:
        return ""

    advice_text = result.advice.strip()
    base = (
        "[Advisor Hint — this turn only, not in history]\n"
        f"{advice_text}\n"
    )

    # LIST-SAVED-NOTES pattern: structurally identifiable trio. See the
    # is_list_saved_notes_pattern() helper above for the canonical predicate.
    if is_list_saved_notes_pattern(result):
        return base + (
            "[BINDING DIRECTIVE — NO TOOL CALL THIS TURN]\n"
            "Advisor identified a list-saved-notes request. NO "
            "<|tool_code|> this turn. Your <think> may briefly acknowledge "
            "the situation, then you MUST emit <|answer|> directly. The "
            "answer comes ONLY from the [Eva's Saved Notes Index — ...] "
            "block in your system prompt. Override the default instinct "
            "to call MemorySearch — fetching here surfaces unrelated lore "
            "records and you'd mistake them for saved notes."
        )

    # All other turns: advisor is advisory only.
    return base + (
        "(Eva, this hint is from your remote advisor. Follow it unless it "
        "contradicts user intent. Stay in your tsundere-maid persona.)"
    )
