"""Build the user-payload that goes to the remote advisor.

The advisor system prompt (in advisor_client.py) describes Eva's tool set and
output JSON format. This module is responsible for assembling the per-turn
*context* the advisor consumes: latest user text, recent history,
EvaState (if any), recent_notes list (for forget hints), and a relevant_memory
block (if any was just retrieved by PRE PROBE).

Keep payload terse: DeepSeek pricing is per-token and the advisor needs to
respond in <4s. Aim for ≤ 1500 input tokens total in normal cases.
"""
from __future__ import annotations

from typing import Optional


_MAX_HISTORY_LINES = 6        # last 3 (user,assistant) pairs
_MAX_RECENT_NOTES = 5
_MAX_MEMORY_BLOCK_CHARS = 800
_MAX_USER_TEXT_CHARS = 1200


def _trim(s: str, cap: int) -> str:
    if not s:
        return ""
    s = s.strip()
    if len(s) <= cap:
        return s
    return s[: cap - 20].rstrip() + " …[truncated]"


def _format_history(history_lines: list[str]) -> str:
    if not history_lines:
        return "(no prior turns in this session)"
    tail = history_lines[-_MAX_HISTORY_LINES:]
    out = []
    for line in tail:
        line = (line or "").strip()
        if line:
            out.append(_trim(line, 400))
    return "\n".join(out) if out else "(no prior turns)"


def _format_recent_notes(recent_notes: list[dict]) -> str:
    """Format `recent_notes` for the advisor.

    Each entry: {"note_id": "abc12345", "topic": "Pet", "preview": "..."}.
    The list is expected newest-first. The advisor uses this when the user
    says "forget that thing I just told you" — it can suggest the right
    record_id rather than letting Eva guess.
    """
    if not recent_notes:
        return "(no recent notes in this session)"
    out = []
    for n in recent_notes[:_MAX_RECENT_NOTES]:
        nid = (n.get("note_id") or "").strip()
        topic = (n.get("topic") or "-").strip()
        preview = _trim(n.get("preview") or n.get("content") or "", 80)
        if nid:
            out.append(f"- Note #{nid} [topic={topic}]: {preview}")
    return "\n".join(out) if out else "(no recent notes)"


def _format_eva_state(eva_state: Optional[dict]) -> str:
    if not eva_state or not isinstance(eva_state, dict):
        return "(idle — no current activity)"
    activity = (eva_state.get("current_activity") or "").strip()
    if not activity:
        return "(idle)"
    ctx = eva_state.get("context") or {}
    extras = ""
    if isinstance(ctx, dict) and ctx:
        bits = [f"{k}={v}" for k, v in list(ctx.items())[:4]]
        extras = f" ({', '.join(bits)})"
    return f"Eva is currently in: {activity}{extras}"


def build_advisor_prompt(
    *,
    user_text: str,
    history_lines: list[str],
    eva_state: Optional[dict] = None,
    recent_notes: Optional[list[dict]] = None,
    relevant_memory: str = "",
) -> str:
    """Assemble the advisor user-payload (single string).

    The system prompt (advisor_client._ADVISOR_SYSTEM_PROMPT) tells the
    advisor how to interpret these blocks and what JSON to output.
    """
    user_text = _trim(user_text or "", _MAX_USER_TEXT_CHARS)
    history_block = _format_history(history_lines or [])
    state_block = _format_eva_state(eva_state)
    notes_block = _format_recent_notes(recent_notes or [])
    mem_block = _trim(relevant_memory or "", _MAX_MEMORY_BLOCK_CHARS)
    if not mem_block:
        mem_block = "(no relevant memory pre-retrieved this turn)"

    payload = f"""[user_input]
{user_text}

[recent_history]
{history_block}

[eva_state]
{state_block}

[recent_notes]
{notes_block}

[relevant_memory]
{mem_block}

[task]
Output the JSON described in the system prompt. Be concise.
"""
    return payload
