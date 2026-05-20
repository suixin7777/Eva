"""
eva_intent_judge.py — LLM-as-judge for query intent classification.

Owns the DeepSeek-driven binary classifiers used by:
  - Verifier (Plan B, completed 2026-05-06): three intent classifiers
    that supplement regex for paraphrase-tolerant detection of
    PUBLIC_FACT / EXPLICIT_MEMORY / EXPLICIT_WEB intents.
  - PRE PROBE (TODO 4 Step 3, in progress): topic-relevance subset
    judgement for MemoryModule.decide.

This module was extracted from eva_core.py during TODO 4 Step 2
(2026-05-06). Before extraction, the dispatcher was a method on
ChatAgent and the prompts were class-level string constants. The
extraction was triggered by TODO 3's stated condition: "if/when TODO 4
lands an LLM-judge layer for PRE PROBE, the Plan B classifier family
becomes a natural co-tenant; splitting both into a shared module then
is the right time".

Design contract (locked in conversation 2026-05-06):

  - Cache and budget counter live on the CALLER (ChatAgent), not in
    this module. judge_intent() takes a `state` argument carrying both.
    Rationale: per-turn lifecycle is owned by the agent (ChatAgent
    resets cache+counter on each new turn); making the state caller-
    owned avoids stale-cache bugs and keeps the module stateless and
    safe under hypothetical multi-agent concurrency.

  - judge_intent() returns tri-state (True / False / None). True/False
    are confident verdicts; None means "judge unavailable, errored, or
    over budget — caller MUST defer to its regex verdict". This is the
    same contract the caller's regex-or-judge fallback was designed
    against; do not change the semantics without auditing every caller.

  - Same (intent, query) within a turn returns the cached verdict.
    Cache stores None too: a failed judge call within a turn will
    likely fail again, so retrying within the same turn is wasted.

  - Per-turn budget caps total judge calls. Once exhausted, every
    subsequent call returns None. ChatAgent resets the counter at
    each new turn boundary.

Public surface:
    JudgeState                 — dataclass holding cache + counter
    new_state()                — fresh JudgeState (call once per turn)
    reset_state(state)         — wipe an existing state in-place
    judge_intent(...)          — binary classifier dispatcher (Plan B)
    judge_topic_subset(...)    — multi-label subset classifier (TODO 4 Step 3)
    synthesize_tool_thought(...) — generate Eva-voice thought for verifier
                                   trace rewrite (Step 5)
    PROMPT_PUBLIC_FACT         — verifier prompt: external/news/fact
    PROMPT_EXPLICIT_MEMORY     — verifier prompt: memory-store invocation
    PROMPT_EXPLICIT_WEB        — verifier prompt: web/internet invocation
    PROMPT_PRE_PROBE_TOPIC_RELEVANCE — PRE PROBE prompt: topic subset
    PROMPT_REWRITE_THOUGHT     — Step 5 prompt: in-character thought synthesis
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Lazy-imported inside judge_intent() — see the function for why.

__all__ = [
    "JudgeState",
    "new_state",
    "reset_state",
    "announce_pending_llm",
    "judge_intent",
    "judge_topic_subset",
    "synthesize_tool_thought",
    "PROMPT_PUBLIC_FACT",
    "PROMPT_EXPLICIT_MEMORY",
    "PROMPT_EXPLICIT_REMEMBER",
    "PROMPT_EXPLICIT_FORGET",
    "PROMPT_EXPLICIT_WEB",
    "PROMPT_PRE_PROBE_TOPIC_RELEVANCE",
    "PROMPT_REWRITE_THOUGHT",
]


# ============================================================
# Caller-owned state
# ============================================================
@dataclass
class JudgeState:
    """Per-turn cache + budget counter for LLM judge calls.

    cache:
        Map (intent, query_lc) -> Optional[bool]. None entries mean
        "this call failed within the current turn; do not retry".

    call_count:
        Number of judge calls issued in the current turn. Bounded by
        LLM_JUDGE_MAX_CALLS_PER_TURN; once exhausted, judge_intent
        returns None for the remainder of the turn.

    pronoun_call_count:
        P6: independent budget counter for the pronoun resolver. Lives
        on the same JudgeState (so callers don't need to thread a
        second state object through the agent), but bounded by its own
        flag PRONOUN_RESOLVER_MAX_CALLS_PER_TURN — not by
        LLM_JUDGE_MAX_CALLS_PER_TURN. Rationale: PRE PROBE / Plan B
        exhausting the global pool must not starve pronoun resolution,
        which has its own well-bounded per-turn cost (≤2 calls).

    Reset at turn boundaries via reset_state() or by replacing the
    instance with new_state().
    """
    cache: Dict[Tuple[str, str], Optional[bool]] = field(default_factory=dict)
    call_count: int = 0
    pronoun_call_count: int = 0


def new_state() -> JudgeState:
    """Construct a fresh per-turn judge state."""
    return JudgeState()


def reset_state(state: JudgeState) -> None:
    """Reset an existing state in-place. Useful when the caller wants
    to keep the same state object across turns (rare; new_state() is
    the typical pattern).
    """
    state.cache.clear()
    state.call_count = 0
    state.pronoun_call_count = 0


# ============================================================
# UX placeholder for blocking DeepSeek calls (2026-05-08)
# ============================================================
def announce_pending_llm(label: str) -> None:
    """Emit a one-line placeholder before a 2-5s DeepSeek call.

    Why: Eva's verifier-repair path stacks two synchronous DeepSeek
    calls — pronoun resolver (~3s) + synthesize_tool_thought (~3s) —
    between `--- ANSWER VERIFIER FAILED ---` and the rewrite block.
    Without a marker the operator stares at dead air for 5-6 seconds
    and assumes the agent hung. The placeholder turns dead air into
    a visible "in flight" signal; the actual result line that follows
    (e.g. `[PRONOUN] q=... source=llm`) supersedes it visually.

    Stdout-only on purpose: SDK consumers wanting structured progress
    events should hook agent.progress_callback (set at run() entry,
    fired at higher-level milestones in step_once). This helper
    serves console/Colab-stream contexts where the trace IS the UI.

    Unconditional — not gated by a debug flag. The whole point is
    that operators in non-debug runs see the latency cause; gating
    would defeat the purpose.

    Caller contract: invoke RIGHT BEFORE the synchronous call.
    Don't invoke if the call will be skipped (cache hit, budget
    exhausted) — those paths have no latency to absorb.
    """
    short = label[:120] + ("…" if len(label) > 120 else "")
    print(f"        | [PENDING DeepSeek] {short}")


# ============================================================
# Verifier prompts (Plan B)
# Originally lived as class constants on ChatAgent. Same wording —
# moving them here is a code-organisation change, not a content change.
# ============================================================
PROMPT_PUBLIC_FACT = (
    "You are a strict binary classifier for a chatbot's verifier. "
    "Your job: decide whether the user query is asking about a "
    "PUBLIC, EXTERNAL, FACTUAL piece of information that lives "
    "outside the chatbot's persona and outside its long-term memory "
    "of personal interactions. The query must specifically ask for "
    "information ABOUT something in the world (a person, product, "
    "release, role, current event). Math, creative requests, "
    "image/vision tasks, jokes, and casual chat are NOT public-fact "
    "queries even when they start with 'what is'.\n\n"
    "Examples that ARE PUBLIC_FACT (return true):\n"
    "  - 'who composed the soundtrack for NieR Automata' (authorship)\n"
    "  - 'what is the latest stable Python version' (current public fact)\n"
    "  - 'who is the current president of France' (current public role)\n"
    "  - 'when was GPT-5 announced' (release/announcement date)\n"
    "  - 'what's the news today about AI regulation' (current news)\n"
    "  - 'who directed Inception' (authorship of a public work)\n"
    "  - 'what is the price of Tesla stock' (current market data)\n\n"
    "Examples that are NOT public-fact (return false):\n"
    "  - 'do you remember what we talked about' (personal memory)\n"
    "  - 'what's your favorite game' (chatbot's own persona)\n"
    "  - 'do you like me' (subjective/relational)\n"
    "  - 'what's today's date' (answered from a built-in time anchor)\n"
    "  - 'how many days until your birthday' (date arithmetic on persona facts)\n"
    "  - 'who am I to you' (relationship)\n"
    "  - 'tell me about your personality' (persona introspection)\n"
    "  - 'what's 2 + 2', 'what is 100 / 5' (arithmetic)\n"
    "  - 'tell me a joke', 'write me a haiku' (creative request)\n"
    "  - 'describe this image', 'read the text in this picture' (vision task)\n"
    "  - 'translate this paragraph' (text transformation)\n"
    "  - any short greeting, emoji, or chitchat\n\n"
    "Output STRICT JSON with exactly this shape:\n"
    '  {"ok": true}   if the query is PUBLIC_FACT\n'
    '  {"ok": false}  otherwise\n'
    "No prose, no explanation, no other keys. JSON only."
)


PROMPT_EXPLICIT_MEMORY = (
    "You are a strict binary classifier for a chatbot's verifier. "
    "Your job: decide whether the user's query is an EXPLICIT request "
    "for the chatbot to consult its long-term memory / records / "
    "stored profile, as opposed to an ordinary question that the bot "
    "happens to know. "
    "Return true ONLY when the user asks the bot to actively look "
    "something up in its own memory store, verify it from records, "
    "recall a stored fact, or check the database. The wording must "
    "INVOKE the memory store directly — words like 'memory', "
    "'records', 'database', 'remember', 'recall', 'look up', "
    "'search your X', 'consult'. Plain past-tense recall questions "
    "('did you do X', 'what did I tell you', 'who am I to you') "
    "are NOT explicit memory checks — those are answered from the "
    "bot's normal active context.\n\n"
    "Examples that ARE explicit memory checks (return true):\n"
    "  - 'do you remember what we talked about last time'\n"
    "  - 'check your memory for what I said before'\n"
    "  - 'search your records for our anniversary'\n"
    "  - 'verify that fact from your database'\n"
    "  - 'recall what game we played'\n"
    "  - 'look up what you have on Rosm'\n"
    "  - 'what do you remember about my birthday'\n"
    "  - 'consult your records'\n"
    "  - 'pull up our shared history'\n\n"
    "Examples that are NOT explicit memory checks (return false):\n"
    "  - 'what's your favorite game' (persona, not memory invocation)\n"
    "  - 'tell me about your personality' (introspection)\n"
    "  - 'who composed Inception' (public fact)\n"
    "  - 'do you like me' (subjective)\n"
    "  - 'search the web for new games' (web search, not memory)\n"
    "  - 'look up the latest Python version' (web)\n"
    "  - 'how are you today' (chitchat)\n"
    "  - 'did you dance with me before' (implicit past-event recall — answered from active memory, not store invocation)\n"
    "  - 'what do I like to do' (preference question — persona/memory background, not store invocation)\n"
    "  - 'who am I to you' (relationship question, not memory invocation)\n"
    "  - 'what color dress did you wear' (recall question — answer from active memory)\n"
    "  - 'tell me about our last visit' (story/recap, not store check)\n"
    "  - any greeting, emoji, or short reaction\n\n"
    "Output STRICT JSON with exactly this shape:\n"
    '  {"ok": true}   if it IS an explicit memory check\n'
    '  {"ok": false}  otherwise\n'
    "No prose, no explanation, no other keys. JSON only."
)


PROMPT_EXPLICIT_REMEMBER = (
    "You are a strict binary classifier for a chatbot's verifier. "
    "Your job: decide whether the user is EXPLICITLY asking the chatbot "
    "to PERSIST a NEW piece of information into long-term memory — i.e. "
    "a WRITE request, not a recall/lookup. The user must clearly direct "
    "the bot to remember/note/save a specific fact going forward.\n\n"
    "Examples that ARE explicit remember/write requests (return true):\n"
    "  - 'remember this: I just adopted a cat named Peach'\n"
    "  - 'remember that the meeting is at 3pm tomorrow'\n"
    "  - \"don't forget that I'm vegetarian\"\n"
    "  - 'note this down: my new password is X'\n"
    "  - 'keep in mind that I work nights now'\n"
    "  - 'jot this down — Tom said yes'\n"
    "  - 'save this fact'\n"
    "  - '记住, 我刚买了辆车'\n"
    "  - '请记一下我的生日是七月七日'\n"
    "  - '别忘了我喜欢咖啡'\n\n"
    "Examples that are NOT explicit remember requests (return false):\n"
    "  - 'do you remember what we talked about' (recall query)\n"
    "  - 'I remember last summer' (user statement, not a directive)\n"
    "  - 'forget about the cat' (DELETE request)\n"
    "  - 'remind me to call mom' (reminder/scheduling, not memory write)\n"
    "  - 'what should I remember' (question, not directive)\n"
    "  - 'I forgot my keys' (user state, not a directive)\n"
    "  - any greeting, chitchat, or question\n\n"
    "Output STRICT JSON with exactly this shape:\n"
    '  {"ok": true}   if it IS an explicit remember/write request\n'
    '  {"ok": false}  otherwise\n'
    "No prose, no explanation, no other keys. JSON only."
)


PROMPT_EXPLICIT_FORGET = (
    "You are a strict binary classifier for a chatbot's verifier. "
    "Your job: decide whether the user is EXPLICITLY asking the chatbot "
    "to delete, undo, or retract a piece of information it previously "
    "remembered. The user must clearly retract — not merely say "
    "'goodbye', 'never mind' as filler, or change topic.\n\n"
    "Examples that ARE explicit forget/delete requests (return true):\n"
    "  - 'forget what I just told you about the cat'\n"
    "  - 'actually, forget about that'\n"
    "  - 'delete that memory'\n"
    "  - 'I was joking, ignore what I said'\n"
    "  - 'scratch that'\n"
    "  - 'remove that record'\n"
    "  - 'never mind, that was wrong'\n"
    "  - 'undo that'\n"
    "  - '忘掉刚才那个吧'\n"
    "  - '把那条记忆删了'\n"
    "  - '当我没说'\n"
    "  - '算了，刚才是开玩笑的'\n\n"
    "Examples that are NOT explicit forget requests (return false):\n"
    "  - 'remember this: ...' (this is an ADD request)\n"
    "  - 'do you remember what I said' (recall query)\n"
    "  - 'never mind' alone with no referent (filler)\n"
    "  - 'forget it, let's talk about something else' (topic change, "
    "    not memory deletion — must reference a prior fact)\n"
    "  - 'I forgot what I was going to say' (user forgot, not asking us to)\n"
    "  - any question, greeting, or chitchat\n\n"
    "Output STRICT JSON with exactly this shape:\n"
    '  {"ok": true}   if it IS an explicit forget/delete request\n'
    '  {"ok": false}  otherwise\n'
    "No prose, no explanation, no other keys. JSON only."
)


PROMPT_EXPLICIT_WEB = (
    "You are a strict binary classifier for a chatbot's verifier. "
    "Your job: decide whether the user is EXPLICITLY asking the bot "
    "to use web search, the internet, an online source, or to verify "
    "something against external sources. The user's wording must "
    "actually invoke web/online/internet, or use a search-style verb "
    "directed at external information. Mere questions about external "
    "facts (without wording that invokes the web) do NOT count — those "
    "are handled by a separate classifier.\n\n"
    "Examples that ARE explicit web requests (return true):\n"
    "  - 'use websearch to verify this'\n"
    "  - 'check the internet for me'\n"
    "  - 'google the latest python release'\n"
    "  - 'look it up online'\n"
    "  - 'browse a few sources and confirm'\n"
    "  - 'fetch some recent news'\n"
    "  - 'can you search the web for new games'\n"
    "  - 'verify that on a website'\n"
    "  - 'go online and find out'\n\n"
    "Examples that are NOT explicit web requests (return false):\n"
    "  - 'who composed Inception' (public-fact, no web wording)\n"
    "  - 'what is the latest python version' (current fact, no web verb)\n"
    "  - 'do you remember our last visit' (memory)\n"
    "  - 'what's your favorite song' (persona)\n"
    "  - 'what is today's date' (time anchor)\n"
    "  - 'check your memory for the toy' (memory check, not web)\n"
    "  - 'search your records' (memory store, not web)\n"
    "  - any greeting, emoji, chitchat\n\n"
    "Output STRICT JSON with exactly this shape:\n"
    '  {"ok": true}   if the query EXPLICITLY invokes web/internet/online\n'
    '  {"ok": false}  otherwise\n'
    "No prose, no explanation, no other keys. JSON only."
)


# ============================================================
# Dispatcher
# ============================================================
def judge_intent(
    intent: str,
    query: str,
    system_prompt: str,
    *,
    state: JudgeState,
) -> Optional[bool]:
    """Ask DeepSeek 'is this query of type <intent>?' Return tri-state.

    Args:
        intent: short tag for cache key + debug output (e.g. "PUBLIC_FACT").
        query: user query text. Trimmed, lowercased only for cache key
            (the full text is sent to DeepSeek).
        system_prompt: the binary-classifier system prompt for this
            intent. Pass one of PROMPT_PUBLIC_FACT / PROMPT_EXPLICIT_MEMORY /
            PROMPT_EXPLICIT_WEB, or any future prompt that follows the
            same `{"ok": bool}` schema.
        state: caller-owned JudgeState (cache + budget counter). The
            caller is responsible for lifecycle — typically
            ChatAgent.run() calls reset_state() at turn start.

    Returns:
        True   — judge confidently said yes (route as if regex matched).
        False  — judge confidently said no.
        None   — judge unavailable / errored / over budget. Caller MUST
                 defer to its regex verdict; treating None as False is
                 the right default for an additive fallback.

    Module-level lazy imports:
        eva_config (LLM_JUDGE_* flags) and eva_tools_runtime
        (call_deepseek_judge) are imported inside the function so
        eva_intent_judge stays import-safe even when the consumer
        doesn't intend to use the judge (e.g. when running offline
        unit tests that stub eva_config).
    """
    # Short-circuit when disabled.
    from eva_config import (
        ENABLE_LLM_VERIFIER_JUDGE,
        LLM_JUDGE_DEBUG,
        LLM_JUDGE_MAX_CALLS_PER_TURN,
    )
    if not ENABLE_LLM_VERIFIER_JUDGE:
        return None
    if not isinstance(query, str) or not query.strip():
        return None

    # Cache lookup — cached value (including None) wins.
    key = (intent, query.strip().lower())
    if key in state.cache:
        return state.cache[key]

    # Per-turn budget — once exhausted, every subsequent call returns
    # None until the caller resets the state.
    if state.call_count >= LLM_JUDGE_MAX_CALLS_PER_TURN:
        if LLM_JUDGE_DEBUG:
            print(f"        | [JUDGE] budget exhausted "
                  f"({state.call_count}/{LLM_JUDGE_MAX_CALLS_PER_TURN}); "
                  f"intent={intent} -> None")
        return None

    # Issue the call. Increment counter BEFORE the network call so a
    # raised exception can't cause an "infinite retry" within a turn —
    # we'd be over budget by the time control returned anyway.
    state.call_count += 1
    from eva_tools_runtime import call_deepseek_judge
    result = call_deepseek_judge(system_prompt, query, debug=LLM_JUDGE_DEBUG)

    # Parse verdict. Contract: judge MUST return {"ok": bool, ...}
    # on success; anything else is treated as unknown.
    verdict: Optional[bool] = None
    if isinstance(result, dict):
        ok_field = result.get("ok")
        if isinstance(ok_field, bool):
            verdict = ok_field

    if LLM_JUDGE_DEBUG:
        short_q = query[:60] + ("…" if len(query) > 60 else "")
        if verdict is None:
            err = result.get("error") if isinstance(result, dict) else "?"
            print(f"        | [JUDGE] intent={intent} q={short_q!r} "
                  f"-> None  (err={err})")
        else:
            print(f"        | [JUDGE] intent={intent} q={short_q!r} "
                  f"-> {verdict}")

    # Cache the verdict (including None — see module docstring).
    state.cache[key] = verdict
    return verdict


# ============================================================
# PRE PROBE topic-relevance prompt (TODO 4 Step 3)
# ============================================================
# Schema differs from the Plan B binary classifiers: this judge
# returns a JSON OBJECT with a "relevant" array (subset of the
# input candidates that are actually relevant to the user's intent).
# Topics not present in input MUST NOT be invented — that's the
# explicit non-choice in the TODO 4 design (situation (a) only).
PROMPT_PRE_PROBE_TOPIC_RELEVANCE = (
    "You are a strict topic-relevance filter for a chatbot's "
    "long-term memory system. The chatbot is Eva, a tsundere maid AI. "
    "Eva's memory is DUAL-SUBJECT — it stores curated facts about TWO "
    "specific people:\n"
    "  - Eva herself (her preferences, hobbies, daily life, personality)\n"
    "  - Rosm, Eva's Creator and Master, a specific named person whose "
    "profile Eva has memorised (his hobbies, communication style, "
    "preferences, daily life, etc.)\n"
    "Memory does NOT store profiles of other people (guests, strangers).\n\n"
    "The keyword scanner has matched CANDIDATE topics for the user's "
    "current message. The 'speaker' field tells you WHO is talking to "
    "Eva. Your job: decide which candidates are ACTUALLY relevant to "
    "this query, given who the speaker is.\n\n"
    "When the speaker is Rosm:\n"
    "  - First-person queries ('what do I like', 'when is my birthday') "
    "ask about Rosm himself. Eva's memory has Rosm's profile — KEEP "
    "the candidates.\n"
    "  - Second-person queries ('what do you like', 'tell me about "
    "yourself') ask about Eva. Eva's memory has her own profile — "
    "KEEP the candidates.\n"
    "  - 'we'/'us'/'our' queries ('did we visit X', 'what did we do') "
    "ask about shared history — KEEP the candidates.\n\n"
    "When the speaker is anyone else (Guest, stranger):\n"
    "  - First-person queries ('what do I like') ask about the speaker, "
    "whom Eva does NOT have a profile for — REJECT the candidates.\n"
    "  - Second-person queries ('what do you like') still ask about Eva — "
    "KEEP the candidates.\n"
    "  - Rosm-related queries by guests are unusual; KEEP candidates if "
    "the query names Rosm.\n\n"
    "Reject candidates regardless of speaker when the keyword match is "
    "incidental and the actual intent is:\n"
    "  - Public-fact / external information (real-world person, work, "
    "product, news) — Eva's persona/Rosm memory is irrelevant.\n"
    "  - Creative request (joke, poem, story) about a topic — user "
    "wants generated content, not Eva's memory.\n"
    "  - General knowledge / how-to (mentions topic as subject matter "
    "without asking for Eva's or Rosm's preference/history).\n"
    "  - Translation / formatting / arithmetic / utility tasks.\n\n"
    "Examples:\n"
    "  speaker: 'Rosm', query: 'what do I like to do in my free time?'\n"
    "  candidates: ['Leisure', 'Likes']\n"
    "  output: {\"relevant\": [\"Leisure\", \"Likes\"]}   "
    "(Rosm asking about his own profile; Eva's memory has it)\n\n"
    "  speaker: 'Rosm', query: 'when is my birthday?'\n"
    "  candidates: ['Birthday']\n"
    "  output: {\"relevant\": [\"Birthday\"]}   "
    "(Rosm asking about his own birthday; Eva's memory has it)\n\n"
    "  speaker: 'Rosm', query: \"what's your favorite game?\"\n"
    "  candidates: ['Gaming']\n"
    "  output: {\"relevant\": [\"Gaming\"]}   "
    "(asking Eva's persona preference)\n\n"
    "  speaker: 'Guest', query: 'what do I like to do?'\n"
    "  candidates: ['Leisure']\n"
    "  output: {\"relevant\": []}   "
    "(guest asking about themselves; Eva has no profile for guests)\n\n"
    "  speaker: 'Rosm', query: 'who composed the soundtrack for the game NieR Automata?'\n"
    "  candidates: ['Gaming']\n"
    "  output: {\"relevant\": []}   "
    "(public-fact about a real game's composer; not Eva's persona)\n\n"
    "  speaker: 'Rosm', query: 'tell me a joke about games'\n"
    "  candidates: ['Gaming']\n"
    "  output: {\"relevant\": []}   "
    "(creative request; Eva's gaming memory irrelevant)\n\n"
    "  speaker: 'Rosm', query: \"what's a good game development engine?\"\n"
    "  candidates: ['Gaming']\n"
    "  output: {\"relevant\": []}   "
    "(technical product advice; not about Eva or Rosm)\n\n"
    "  speaker: 'Rosm', query: 'do you like ballet and music?'\n"
    "  candidates: ['Dancing', 'Music', 'Likes']\n"
    "  output: {\"relevant\": [\"Dancing\", \"Music\", \"Likes\"]}   "
    "(asking Eva's preferences across all candidates)\n\n"
    "  speaker: 'Rosm', query: 'what music plays at the gym?'\n"
    "  candidates: ['Music']\n"
    "  output: {\"relevant\": []}   "
    "(general factual question, not about Eva or Rosm)\n\n"
    "Rules:\n"
    "  1. The 'relevant' array MUST be a subset of the input candidates. "
    "Never add a topic not in the input.\n"
    "  2. If unsure whether the speaker's intent matches Eva's or "
    "Rosm's stored profile, KEEP the candidate (lean toward inclusion "
    "to avoid under-injection).\n"
    "  3. Empty array is valid output and means 'skip memory injection'.\n\n"
    "The user query, speaker, and candidate topics will be provided in JSON. "
    "Output STRICT JSON with exactly this shape:\n"
    '  {"relevant": ["Topic1", "Topic2", ...]}\n'
    "No prose, no explanation, no other keys. JSON only."
)


# ============================================================
# judge_topic_subset — PRE PROBE topic-relevance dispatcher
# ============================================================
def judge_topic_subset(
    query: str,
    candidates: List[str],
    *,
    state: JudgeState,
    speaker: str = "Rosm",
) -> Optional[List[str]]:
    """Ask DeepSeek to filter candidate topics to actually-relevant ones.

    Args:
        query: the raw user query.
        candidates: keyword-matched topic candidates from
            TopicDictionary. MUST be non-empty (caller's responsibility).
        state: caller-owned JudgeState (cache + budget counter).
            Shared with judge_intent (verifier classifiers) so PRE PROBE
            and verifier judge calls draw from the same per-turn pool.
        speaker: WHO is talking to Eva. Default "Rosm" since most
            production traffic is from the Master. The dual-subject
            prompt uses this to disambiguate first-person queries:
            'what do I like' from Rosm probes Rosm's stored profile;
            same query from a Guest should reject the candidates
            (Eva has no Guest profile).

    Returns:
        List[str] — subset of `candidates` the judge confirmed relevant.
                    May be empty (= skip memory injection).
        None      — judge unavailable / errored / over budget / disabled.
                    Caller MUST treat as "judge silent" and degrade to
                    the keyword-only verdict (= return candidates as-is).

    The (intent, query, speaker) cache key uses intent="PRE_PROBE_RELEVANCE"
    so PRE PROBE entries don't collide with the binary verifier intents,
    and so the same query asked by Rosm vs Guest receives independent
    verdicts (otherwise the first speaker's verdict would be served to
    the second from cache, which is wrong by design).

    Lazy imports of eva_config + eva_tools_runtime so this module stays
    import-safe in offline tests that stub those modules.
    """
    from eva_config import (
        ENABLE_LLM_PRE_PROBE_JUDGE,
        ENABLE_LLM_VERIFIER_JUDGE,
        LLM_JUDGE_DEBUG,
        LLM_JUDGE_MAX_CALLS_PER_TURN,
    )

    # Independent enable flag — operators can disable PRE PROBE judge
    # without touching the verifier. ENABLE_LLM_VERIFIER_JUDGE is also
    # checked because it gates the underlying network call_deepseek_judge
    # path; if the operator killed the verifier judge globally, we
    # respect that.
    if not (ENABLE_LLM_PRE_PROBE_JUDGE and ENABLE_LLM_VERIFIER_JUDGE):
        return None
    if not isinstance(query, str) or not query.strip():
        return None
    if not candidates:
        return None

    # Cache key: intent tag + speaker + query + sorted candidate tuple.
    # Sorted because candidate order from regex is not guaranteed
    # stable across runs and cache should be order-independent.
    # Speaker is part of the key because the dual-subject prompt
    # produces speaker-dependent verdicts (see docstring above).
    intent = "PRE_PROBE_RELEVANCE"
    speaker_norm = (speaker or "Rosm").strip()
    cand_key = ",".join(sorted(candidates))
    key = (intent, f"{speaker_norm}|{query.strip().lower()}|{cand_key}")
    if key in state.cache:
        cached = state.cache[key]
        # cache stores the parsed list (or None); judge_intent's cache
        # stores bool/None. Different value types in the same dict is
        # safe because the keys are disjoint by intent tag.
        return cached

    if state.call_count >= LLM_JUDGE_MAX_CALLS_PER_TURN:
        if LLM_JUDGE_DEBUG:
            print(f"        | [JUDGE] budget exhausted "
                  f"({state.call_count}/{LLM_JUDGE_MAX_CALLS_PER_TURN}); "
                  f"intent={intent} -> None")
        return None

    state.call_count += 1
    from eva_tools_runtime import call_deepseek_judge
    import json as _json
    payload = _json.dumps(
        {"speaker": speaker_norm,
         "query": query,
         "candidates": list(candidates)},
        ensure_ascii=False,
    )
    result = call_deepseek_judge(PROMPT_PRE_PROBE_TOPIC_RELEVANCE, payload,
                                 debug=LLM_JUDGE_DEBUG)

    # Parse result. Contract: judge returns {"relevant": [str, ...]}.
    # Anything else (including the {"ok": None, "error": ...} sentinel
    # from call_deepseek_judge on failure) is treated as None.
    parsed: Optional[List[str]] = None
    if isinstance(result, dict):
        rel = result.get("relevant")
        if isinstance(rel, list) and all(isinstance(t, str) for t in rel):
            # Whitelist filter: only keep topics that were in the input.
            # Defensive against the model hallucinating a topic name.
            cand_set = set(candidates)
            parsed = [t for t in rel if t in cand_set]

    if LLM_JUDGE_DEBUG:
        short_q = query[:60] + ("…" if len(query) > 60 else "")
        if parsed is None:
            err = result.get("error") if isinstance(result, dict) else "?"
            print(f"        | [JUDGE] intent={intent} speaker={speaker_norm} "
                  f"q={short_q!r} cand={candidates} -> None  (err={err})")
        else:
            print(f"        | [JUDGE] intent={intent} speaker={speaker_norm} "
                  f"q={short_q!r} cand={candidates} -> relevant={parsed}")

    state.cache[key] = parsed
    return parsed


# ============================================================
# Step 5 — Trace rewriting on verifier repair
# ============================================================
# When the answer-verifier hard-fails and injects a repair tool
# (controller_inject path), the resulting trajectory is:
#
#   assistant: <think>I know X is Y</think><|answer|>X is Y<|end_react|>
#   tool:      <|tool_output|>...real-truth...</|tool_output|>
#
# This shape does not occur in SFT data. SFT trajectories with tool
# output ALWAYS have the form:
#
#   assistant: <think>...</think><|tool_code|>RealTool(...)<|end_react|>
#   tool:      <|tool_output|>...</|tool_output|>
#
# Step 5 fixes this by rewriting the assistant turn from answer-shape
# to tool-call-shape just before injecting tool output. The new
# `<think>...</think>` is generated here by DeepSeek so it matches
# Eva's voice; if DeepSeek is silent (disabled / over budget /
# errored), the verifier site falls back to a hardcoded template.
#
# Why DeepSeek-generated thoughts beat hardcoded templates:
#   - Tone matches SFT data (first-person Eva narration), reducing
#     phase-2 distribution shift further.
#   - Thought can specifically describe WHY the tool was needed
#     ('this is a public-fact question about a real game' vs the
#     generic 'I should consult WebSearch').
#   - Light variance prevents the fallback from looking obviously
#     templated to the model in long-running sessions.
# ============================================================
PROMPT_REWRITE_THOUGHT = (
    "You are rewriting an internal reasoning thought for an AI maid "
    "named Eva. Eva is mid-conversation and has decided she needs to "
    "call a tool to answer the user's question. You are writing what "
    "Eva is thinking RIGHT BEFORE she invokes the tool — a brief, "
    "first-person internal monologue explaining WHY the tool is "
    "appropriate.\n\n"
    "Style requirements:\n"
    "  - First-person from Eva's perspective ('I should...', 'Master "
    "is asking...', 'This needs...').\n"
    "  - 1-2 sentences, max 200 characters.\n"
    "  - Explain WHY the tool was chosen (e.g. 'this is a public-fact "
    "question', 'this is about Master\\'s personal preferences').\n"
    "  - Do NOT commit to any factual answer — the tool will provide "
    "the answer. Do NOT name specific entities (people, products, "
    "dates) that the tool will return.\n"
    "  - Match Eva's actual speech style: thoughtful, sometimes a "
    "touch tsundere, but functional. Avoid heavy roleplay markup.\n"
    "  - Do NOT use 'Hmph', 'Tch', or other in-character interjections "
    "— this is internal reasoning, not spoken dialogue.\n\n"
    "Style examples (real Eva thoughts from training data):\n"
    "  - 'Master is asking about something personal to them. I should "
    "check my memory for Emma's favorite color first.'\n"
    "  - 'This is a public-fact question about a real game's composer "
    "— I should use WebSearch for the authoritative answer.'\n"
    "  - 'Master sent an image and asked what it is. I should use "
    "AskRemoteVision in OCR mode to extract the details.'\n"
    "  - 'Master needs the current date to compute this. I should "
    "call GetCurrentTime first.'\n\n"
    "PRONOUN RESOLUTION (P5):\n"
    "  - The input may include a `recent_turns` field listing the "
    "last few (user, assistant) exchanges. If the user query "
    "contains pronouns ('it', 'that', 'them', 'this', 'those', "
    "'check it', 'do it again', etc.), you MUST resolve the "
    "antecedent from `recent_turns` BEFORE writing the thought.\n"
    "  - Reflect the resolved antecedent in your reasoning. Example: "
    "if the assistant just said 'I could show you my special "
    "collection' and the user replied 'can you check it?', the "
    "thought should be 'Master wants me to check what I just "
    "mentioned \u2014 my special collection. I should search my memory.'\n"
    "  - If `recent_turns` is empty or no clear antecedent exists, "
    "write a generic thought without inventing one.\n\n"
    "Input is JSON with the user query, the tool name that was "
    "decided, a short summary of tool args, and optionally "
    "recent_turns (newest last). Output STRICT JSON with exactly "
    "this shape:\n"
    '  {"thought": "<1-2 sentence first-person reasoning>"}\n'
    "No prose, no preamble, no other keys. JSON only."
)


def synthesize_tool_thought(
    query: str,
    tool_name: str,
    tool_args_summary: str,
    *,
    state: JudgeState,
    recent_turns: Optional[list] = None,
) -> Optional[str]:
    """Synthesise an in-character thought for verifier-injected tool.

    Args:
        query: the user query that triggered the verifier repair.
        tool_name: the tool the verifier decided to call (e.g.
            'WebSearch', 'MemorySearch'). Required so the prompt
            can ground its reasoning around the right tool.
        tool_args_summary: short human-readable summary of args
            (e.g. 'query="NieR Automata composer"'). Used purely
            for the prompt's context; not parsed.
        state: caller-owned JudgeState (cache + budget counter),
            shared with judge_intent / judge_topic_subset for
            single per-turn budget pool.

    Returns:
        str — synthesised thought, max 200 chars, post-validated.
        None — judge unavailable / errored / over budget / disabled
                / failed validation. Caller MUST fall back to a
                hardcoded per-tool template.

    Cache key: (intent="REWRITE_THOUGHT", lower(query)|tool|args_norm).
    Cache returns the same parsed string (or None) on repeat hits.

    Validation rules applied to the LLM output before returning:
      1. Must be a non-empty string.
      2. Length <= 200 characters (truncate at last sentence
         boundary if over, fall through if can't truncate cleanly).
      3. Must NOT contain the user's query verbatim (sign that the
         model echoed input instead of synthesising).
      4. Must NOT contain raw `Hmph` / `Tch` / asterisk-action
         tokens — these are spoken-dialogue markers, not
         internal-reasoning markers.
    """
    from eva_config import (
        ENABLE_LLM_PRE_PROBE_JUDGE,
        ENABLE_LLM_VERIFIER_JUDGE,
        LLM_JUDGE_DEBUG,
        LLM_JUDGE_MAX_CALLS_PER_TURN,
    )

    # Reuse the same enable flags as PRE PROBE judge — Step 5 is part
    # of the same DeepSeek-judge family. If operators disable that
    # subsystem, Step 5 falls through to hardcoded templates and the
    # rest of the trace rewrite still works.
    if not (ENABLE_LLM_PRE_PROBE_JUDGE and ENABLE_LLM_VERIFIER_JUDGE):
        return None
    if not isinstance(query, str) or not query.strip():
        return None
    if not isinstance(tool_name, str) or not tool_name.strip():
        return None

    intent = "REWRITE_THOUGHT"
    args_norm = (tool_args_summary or "").strip().lower()

    # P5: include a fingerprint of recent_turns in the cache key so
    # different conversation contexts don't collide on the same
    # query+tool. Only the assistant side is hashed because the user
    # side is already reflected in `query`.
    turns_fp = ""
    if recent_turns:
        try:
            joined = "|".join(
                (t.get("assistant") or "")[:64].strip().lower()
                for t in recent_turns[-3:]
            )
            turns_fp = joined[:200]
        except Exception:
            turns_fp = ""

    key = (intent,
           f"{query.strip().lower()}|{tool_name.strip()}|{args_norm}|{turns_fp}")
    if key in state.cache:
        return state.cache[key]

    if state.call_count >= LLM_JUDGE_MAX_CALLS_PER_TURN:
        if LLM_JUDGE_DEBUG:
            print(f"        | [JUDGE] budget exhausted "
                  f"({state.call_count}/{LLM_JUDGE_MAX_CALLS_PER_TURN}); "
                  f"intent={intent} -> None (use template fallback)")
        return None

    state.call_count += 1
    from eva_tools_runtime import call_deepseek_judge
    import json as _json

    # UX placeholder for the operator-visible ~3s wait. Caller will
    # print [JUDGE] result line once the call returns.
    short_q = query[:60] + ("…" if len(query) > 60 else "")
    announce_pending_llm(
        f"synthesize_tool_thought: tool={tool_name} q={short_q!r}"
    )

    payload_dict = {
        "query": query,
        "tool_name": tool_name,
        "tool_args": tool_args_summary or "",
    }
    # P5: only attach recent_turns when non-empty to avoid bloating
    # the prompt for the common case (no follow-up).
    if recent_turns:
        compact_turns = []
        for t in recent_turns[-3:]:
            u = (t.get("user") or "").strip()
            a = (t.get("assistant") or "").strip()
            if not (u or a):
                continue
            compact_turns.append({
                "user": u[:300],
                "assistant": a[:400],
            })
        if compact_turns:
            payload_dict["recent_turns"] = compact_turns

    payload = _json.dumps(payload_dict, ensure_ascii=False)
    result = call_deepseek_judge(PROMPT_REWRITE_THOUGHT, payload,
                                 debug=LLM_JUDGE_DEBUG)

    # Parse and validate.
    parsed: Optional[str] = None
    if isinstance(result, dict):
        thought_raw = result.get("thought")
        if isinstance(thought_raw, str):
            t = thought_raw.strip()
            # Validation 1: non-empty
            if t:
                # Validation 2: length cap (truncate at sentence
                # boundary if possible)
                if len(t) > 200:
                    cut = t[:200]
                    last_period = max(cut.rfind("."), cut.rfind("!"),
                                      cut.rfind("?"))
                    if last_period > 50:
                        t = cut[:last_period + 1]
                    else:
                        # Couldn't find clean sentence boundary;
                        # reject rather than truncate mid-word
                        t = ""
                if t:
                    # Validation 3: must not echo full query
                    q_norm = query.strip().lower()
                    if q_norm and q_norm in t.lower():
                        t = ""
                # Validation 4: no spoken-dialogue markers
                if t:
                    forbidden = ["hmph", "tch", "*", "~"]
                    if any(f in t.lower() for f in forbidden):
                        t = ""
                if t:
                    parsed = t

    if LLM_JUDGE_DEBUG:
        short_q = query[:50] + ("…" if len(query) > 50 else "")
        if parsed is None:
            err = result.get("error") if isinstance(result, dict) else "?"
            print(f"        | [JUDGE] intent={intent} q={short_q!r} "
                  f"tool={tool_name} -> None  (err={err})")
        else:
            short_t = parsed[:80] + ("…" if len(parsed) > 80 else "")
            print(f"        | [JUDGE] intent={intent} q={short_q!r} "
                  f"tool={tool_name} -> thought={short_t!r}")

    state.cache[key] = parsed
    return parsed
