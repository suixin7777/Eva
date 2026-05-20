"""
eva_route_judge.py — Forced-choice route judge using the local LM.

Extracted from eva_core.py during the post-Plan-B cleanup. This module
houses the routing-only intent classifier that uses the local Qwen-VL
model (the same one that does inference) to score one of four labels:

    MEMORY_LOOKUP  — user wants Eva/Rosm/shared remembered facts
    WEB_SEARCH     — user wants public/external/current info
    TIME_LOOKUP    — current date / time / date arithmetic
    DIRECT         — casual chat / persona / creative / no tool needed

It complements the regex+Plan-B judge layer used by the verifier:

  - The verifier judges INTENT for "is this a public-fact request" /
    "is this an explicit memory request" / "is this an explicit web
    request" — three orthogonal binary classifiers via DeepSeek.
  - The ROUTE judge here is one 4-way forced choice via the LOCAL model,
    used at controller time to nudge the model away from picking a tool
    it would otherwise have skipped.

The two layers are independent: route judge runs in step_once before
generation; the verifier judges run after generation as repair gates.

# Why module-level functions instead of an extracted class
# -------------------------------------------------------
# These four operations need access to a lot of ChatAgent state
# (model/processor/tok for the LM forward pass; last_memory_observation
# and _get_recent_user_context for the prompt context hint;
# _route_judge_cache for memoisation across turns). Rather than build
# a mixin or pass 7 individual fields, every function takes the agent
# as an `agent` parameter. ChatAgent retains thin wrapper methods so
# call sites like `self._judge_current_turn_route(...)` still work.
#
# This is "extract method to module" not "extract class" — same code
# motion, but the agent stays a single object.

Public surface (used by ChatAgent wrapper methods):
    route_judge_context_hint(agent)              -> str
    route_judge_prompt(agent, user_text)         -> str (ChatML wire format)
    judge_current_turn_route(agent, user_text)   -> (label, scores_dict)
    score_lm_choice_loss(agent, prompt, choice)  -> float (LM loss; lower is better)
"""

import numpy as np
import torch

from eva_config import (
    ENABLE_ROUTE_LM_JUDGE,
    ROUTE_LM_CHOICES,
    ROUTE_LM_DEBUG,
    MEMORY_JUDGE_DEBUG,
)
from eva_render import wrap_chatml, clean_user_text
from eva_memory_legacy import _normalize_match_text, _truncate_for_judge


__all__ = [
    "route_judge_context_hint",
    "route_judge_prompt",
    "judge_current_turn_route",
    "score_lm_choice_loss",
]


# ============================================================
# Context hint
# ============================================================
def route_judge_context_hint(agent):
    """Compact routing-only context for ambiguous follow-ups.

    The LM judge classifies the CURRENT user message, but messages like
    "check it" or "try again" need to know whether the previous completed
    topic was web/external or memory/profile. Keep this compact so the
    judge does not become a second conversation prompt.
    """
    hints = []
    try:
        if agent._previous_turn_had_web_evidence():
            hints.append("Previous completed route: WEB_SEARCH evidence was used.")
    except Exception:
        pass
    try:
        # R-6: 读 LastMemoryState dataclass。
        last_obs = getattr(getattr(agent, "last_memory", None), "observation", "") or ""
        if last_obs.strip():
            hints.append("Previous memory state: MemorySearch evidence exists from an earlier turn.")
    except Exception:
        pass
    recent = ""
    try:
        recent = agent._get_recent_user_context(max_turns=2).strip()
    except Exception:
        recent = ""
    if recent:
        hints.append("Recent user context: " + _truncate_for_judge(recent, 240))
    return "\n".join(hints).strip() or "No prior routing context."


# ============================================================
# Prompt construction
# ============================================================
def route_judge_prompt(agent, user_text):
    """Build the ChatML-wrapped 4-way classification prompt."""
    system = (
        "You are a strict tool-routing judge. You are NOT Eva. "
        "Do not answer the user. Choose exactly one label."
    )
    context_hint = route_judge_context_hint(agent)
    user = f"""
Task: classify the current user message for tool routing.

Labels:
MEMORY_LOOKUP = User asks about Eva/Rosm/shared remembered facts, prior conversations,
  or explicitly says to check memory/records/lore/database.
  Examples: "what game do you like", "search your memory about your interests",
  "check it with memory", "do you remember when".

WEB_SEARCH = User asks for public/external/current/recent information, news,
  live facts, prices, schedules, OR explicitly tells the assistant to use
  web/internet/online sources, OR asks the assistant to verify/recheck/research/look up
  external information.
  Examples: "news about Trump", "what happened in Sydney recently",
  "search new games similar to Apex", "use websearch", "try websearch to recheck",
  "look it up online", "check the internet", "go verify this", "get fresh data",
  "you should search the web", "actually search this".

TIME_LOOKUP = User asks for current date, current time, weekday, today, or date
  arithmetic such as "how many days until...".

DIRECT = Casual chat, persona response, creative content, or a request that does
  not need memory, web, or time tools.

Important:
- Judge the CURRENT user message intent, not merely the previous topic.
- If the user explicitly tells you to use a specific tool/source type (websearch,
  internet, online, memory, records, lore, database), honor that intent.
- "Recheck/verify/look up again" usually means WEB_SEARCH if the previous turn
  was about external info; MEMORY_LOOKUP if it was about Eva/Rosm/shared memory.
- Do not choose MEMORY_LOOKUP just because the previous turn involved memory;
  if the current message names web/internet/news/online/fresh data, choose WEB_SEARCH.
- Choose only one label: MEMORY_LOOKUP, WEB_SEARCH, TIME_LOOKUP, or DIRECT.

Routing context:
{context_hint}

Current user message:
{_truncate_for_judge(user_text, 500)}

Label:
""".strip()
    return (
        wrap_chatml("system", system, complete=True)
        + wrap_chatml("user", user, complete=True)
        + wrap_chatml("assistant", "Label:", complete=False)
    )


# ============================================================
# LM-loss scoring (forced-choice forward pass)
# ============================================================
def score_lm_choice_loss(agent, prompt, choice):
    """Score a fixed continuation by LM loss; lower is better.

    This avoids unreliable JSON/free-form generation. The local model only
    compares continuations such as " EXACT", " RELATED", " WRONG", or
    " MEMORY_LOOKUP", " WEB_SEARCH" etc.
    """
    try:
        full_text = prompt + choice
        prompt_inputs = agent.processor(text=[prompt], return_tensors="pt")
        full_inputs = agent.processor(text=[full_text], return_tensors="pt")
        prompt_len = int(prompt_inputs["input_ids"].shape[1])
        labels = full_inputs["input_ids"].clone()
        labels[:, :prompt_len] = -100
        # Keep the same multimodal token-type convention as normal inference.
        input_ids = full_inputs["input_ids"]
        mm_token_type_ids = torch.zeros_like(input_ids)
        image_pad_id = agent.tok.convert_tokens_to_ids("<|image_pad|>")
        if image_pad_id is not None:
            mm_token_type_ids[input_ids == image_pad_id] = 1
        full_inputs["mm_token_type_ids"] = mm_token_type_ids
        full_inputs = {k: v.to(agent.model.device) for k, v in full_inputs.items()}
        labels = labels.to(agent.model.device)
        with torch.no_grad():
            out = agent.model(**full_inputs, labels=labels)
        loss = float(out.loss.detach().float().cpu().item())
        if not np.isfinite(loss):
            return float("inf")
        return loss
    except Exception as e:
        if MEMORY_JUDGE_DEBUG:
            print(f"[WARN] Memory judge LM scoring failed for choice={choice!r}: {e}")
        return float("inf")


# ============================================================
# Top-level dispatch
# ============================================================
def judge_current_turn_route(agent, user_text):
    """Forced-choice current-turn route judge.

    This replaces profile/domain keyword-routing for ambiguous turns. It is
    intentionally not used for explicit hard guards such as images,
    explicit memory-store commands, obvious news/current public facts, or
    external recommendation search commands.
    """
    if not ENABLE_ROUTE_LM_JUDGE:
        return "DIRECT", {}
    q = clean_user_text(user_text)
    if not q:
        return "DIRECT", {}
    cache = getattr(agent, "_route_judge_cache", None)
    if cache is None:
        agent._route_judge_cache = {}
        cache = agent._route_judge_cache
    key = _normalize_match_text(q)
    if key in cache:
        if ROUTE_LM_DEBUG:
            print("\n        | --- ROUTE JUDGE CACHE HIT ---")
            print(f"        | label={cache[key][0]}, key={key[:80]}")
        return cache[key]
    prompt = route_judge_prompt(agent, q)
    scores = {}
    for label in ROUTE_LM_CHOICES:
        scores[label] = score_lm_choice_loss(agent, prompt, " " + label)
    if not scores or all(not np.isfinite(v) for v in scores.values()):
        # Conservative fallback: do not force MemorySearch without explicit
        # memory-store wording. Public/current hard cases are handled before
        # this judge, so DIRECT is the safest fallback.
        result = ("DIRECT", {"heuristic": 1.0})
    else:
        result = (min(scores, key=scores.get), scores)
    cache[key] = result
    if ROUTE_LM_DEBUG:
        label, dbg = result
        try:
            dbg_s = "/".join(f"{k}:{float(v):.3f}" for k, v in dbg.items())
        except Exception:
            dbg_s = str(dbg)
        print("\n        | --- ROUTE JUDGE ---")
        print(f"        | label={label}, scores={dbg_s}")
    return result
