"""
eva_chat_colab.py — Multi-turn conversation helpers for Colab.

Drop this file next to your Eva files (or paste it into a notebook
cell). Usage:

    from eva_chat_colab import ChatSession
    sess = ChatSession(agent, user_name="Rosm")
    sess.send("Hi Eva, when's your birthday?")
    sess.send("Do you remember the games we played?")
    sess.attach_image("/content/photo.jpg")
    sess.send("What's in this image?")

    # Inspect what Eva remembers about this conversation:
    sess.show_history()

    # Reset between test scenarios:
    sess.reset()

The session wraps `agent.run()` and persists the agent's
history_manager state across turns. That's it — the heavy lifting
(turn boundaries, history compression, memory probe) is already
inside agent.run.
"""

from typing import Optional


class ChatSession:
    """Stateful wrapper around `agent.run()` for multi-turn dialogue.

    A single ChatAgent instance can host many ChatSessions sequentially
    by calling `session.reset()` between scenarios — useful for ablation
    testing where you want to compare Eva's behavior across personas
    (Rosm vs Guest vs different guest names) without rebooting the model.

    Note on multi-session interleaving: `agent.history_manager` is a
    SHARED state owned by the agent. If you create two ChatSessions
    bound to the same agent and interleave their .send() calls, they
    will trample each other's history. For most testing this is fine
    (you only test one scenario at a time); the .reset() pattern below
    is the supported way to switch scenarios.
    """

    def __init__(self, agent, user_name: str = "Rosm"):
        """
        Args:
            agent: a built ChatAgent (output of build_agent()).
            user_name: speaker name. "Rosm" makes Eva treat the user as
                Master and unlock dual-subject memory probing. Any
                other name puts Eva in Guest mode (warmer, more
                distant; no Rosm-personal memory).
        """
        self.agent = agent
        self.user_name = user_name
        self._pending_image: Optional[str] = None
        self._turn_count = 0

    def attach_image(self, image_path: str) -> None:
        """Queue an image for the NEXT send() call.

        Eva's vision tool only sees the image attached to the current
        user turn. After that turn finishes, the image is gone (though
        Eva can still reference what she described it as in earlier
        turns via history).
        """
        self._pending_image = image_path
        print(f"[image queued for next turn] {image_path}")

    def send(self, user_text: str, *, verbose: bool = True) -> str:
        """Send one user turn, get Eva's final answer back.

        Args:
            user_text: the message to send to Eva.
            verbose: if False, suppresses the trace logs Eva normally
                prints (THOUGHT / TOOL CODE / ANSWER blocks). The final
                answer is still returned. Useful for batch testing.

        Returns:
            Eva's final answer as a string.
        """
        self._turn_count += 1
        image_path = self._pending_image
        self._pending_image = None  # consume

        if verbose:
            print(f"\n{'='*60}")
            print(f"  Turn {self._turn_count} — You ({self.user_name}):")
            print(f"{'='*60}")
            print(f"  {user_text}")
            if image_path:
                print(f"  [📎 image: {image_path}]")
            print()

        # Suppress agent's own prints if verbose=False
        if not verbose:
            import io
            import contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                answer = self.agent.run(
                    user_text=user_text,
                    user_name=self.user_name,
                    image_path=image_path,
                )
        else:
            answer = self.agent.run(
                user_text=user_text,
                user_name=self.user_name,
                image_path=image_path,
            )

        if verbose:
            print(f"\n{'='*60}")
            print(f"  Eva:")
            print(f"{'='*60}")
            print(f"  {answer}")
            print()

        return answer

    def reset(self, user_name: Optional[str] = None) -> None:
        """Wipe conversation state. Optionally switch persona.

        After reset, Eva has no memory of the previous turns in this
        session. Her long-term memory DB (the 91-record knowledge base)
        is untouched — only the per-conversation context is cleared.

        Use this between test scenarios so they don't bleed into each
        other. E.g.:

            sess.send("Hi Eva!")
            sess.reset()                       # turn 1 forgotten
            sess.reset(user_name="Alice")      # also switch to Guest mode
            sess.send("Hi Eva!")               # Eva sees a fresh stranger
        """
        hm = getattr(self.agent, "history_manager", None)
        if hm is not None:
            hm.history = []
            hm.current_turn = None
            if hasattr(hm, "compressed_kv"):
                hm.compressed_kv = []
            if hasattr(hm, "image_registry"):
                hm.image_registry = {}
            if hasattr(hm, "image_order"):
                hm.image_order = []

        # Also clear P2 turn-cache if present (idempotent — these may
        # not exist on all agent variants).
        for attr in ("active_memory_turn_key", "active_memory_context"):
            if hasattr(self.agent, attr):
                setattr(self.agent, attr, "")
        # R-6: last_memory 现在是 dataclass，调用 reset() 而不是赋空字符串。
        last_mem = getattr(self.agent, "last_memory", None)
        if last_mem is not None and hasattr(last_mem, "reset"):
            last_mem.reset()
        focus = getattr(self.agent, "dialog_focus", None)
        if focus is not None and hasattr(focus, "reset"):
            focus.reset()

        self._turn_count = 0
        self._pending_image = None
        if user_name is not None:
            self.user_name = user_name
        print(f"[session reset] user_name={self.user_name}")

    def show_history(self, *, full: bool = False) -> None:
        """Print the conversation so far.

        Args:
            full: if True, includes Eva's internal ReAct trace
                (thoughts, tool calls). If False, just the user-visible
                conversation.
        """
        hm = getattr(self.agent, "history_manager", None)
        if hm is None:
            print("[no history_manager on agent]")
            return

        history = list(hm.history) + ([hm.current_turn] if hm.current_turn else [])
        if not history:
            print("[no turns yet]")
            return

        for i, turn in enumerate(history, 1):
            print(f"\n--- Turn {i} ---")
            print(f"  {self.user_name}: {turn.user_content}")
            if full:
                # Show all assistant steps + tool outputs
                for step in turn.assistant_steps:
                    role = step.get("role", "?")
                    content = step.get("content", "")
                    short = content[:200] + ("..." if len(content) > 200 else "")
                    print(f"  [{role}] {short}")
            else:
                # Just the final visible answer
                final = turn.get_final_answer()
                if final:
                    print(f"  Eva: {final}")

        # Also note compressed history if any
        if getattr(hm, "compressed_kv", None):
            n_compressed = len(hm.compressed_kv)
            print(f"\n  [+ {n_compressed} earlier turn(s) compressed into "
                  f"summary context]")


# ============================================================
# Convenience: a one-shot interactive REPL for Colab.
# ============================================================
def _handle_notes_command(agent, line: str) -> None:
    """Operator-side dump of NotesStore content. Zero LLM calls.

    Subcommands:
      /notes              — live notes (id + entity + topic + preview)
      /notes full         — include tombstoned (deleted=True) notes
      /notes raw <id>     — full JSON for one note (meta + content)
    """
    ns = (agent.memory_state or {}).get("notes_store") if agent.memory_state else None
    if ns is None:
        print("[notes] No NotesStore wired into this agent.")
        return

    parts = line.split(None, 2)
    subcmd = parts[1].strip().lower() if len(parts) > 1 else ""

    if subcmd == "raw" and len(parts) >= 3:
        target_id = parts[2].strip().lstrip("#")
        for content, meta in zip(ns.contents, ns.metas):
            if meta.get("note_id") == target_id:
                import json
                print(f"[notes] --- raw note {target_id} ---")
                print(f"content: {content}")
                print("meta:")
                print(json.dumps(meta, ensure_ascii=False, indent=2))
                return
        print(f"[notes] No note with id {target_id!r} found.")
        return

    include_deleted = (subcmd == "full")
    rows = ns.list_notes(include_deleted=include_deleted)
    status = ns.status()
    print(f"\n[notes] Store at {status['root']}/  "
          f"session_id={status['session_id']}  "
          f"live={status['live']}  deleted={status['deleted']}  "
          f"total={status['total']}")
    if not rows:
        print("[notes] (no notes to show)")
        return
    print(f"[notes] {len(rows)} note(s):")
    for r in rows:
        tag = "x" if r["deleted"] else "."
        print(f"  [{tag}] #{r['note_id']}  "
              f"[{r['entity']!s:6}]  [{(r['topic'] or '-')!s:12}]  "
              f"{r['content_preview']}")
    if not include_deleted and status["deleted"] > 0:
        print(f"[notes] ({status['deleted']} tombstoned not shown — use /notes full)")


def chat(agent, user_name: str = "Rosm") -> None:
    """Interactive REPL inside a Colab cell.

    Usage:
        from eva_chat_colab import chat
        chat(agent, user_name="Rosm")

    Special inputs:
        exit / quit / Ctrl-C        — end session
        /image <path>               — attach image for next turn
        /reset [name]               — wipe history (optional new name)
        /history                    — show what Eva remembers so far
        /history full               — show full ReAct trace too
        /quiet on | /quiet off      — toggle verbose Eva trace logs
        /notes                      — list live notes (id + topic + preview)
        /notes full                 — also include deleted/tombstoned notes
        /notes raw <id>             — print full JSON for one note by id

    All other input is sent to Eva normally.
    """
    sess = ChatSession(agent, user_name=user_name)
    verbose = True
    print(f"\nEva is ready. Speaking as: {user_name}")
    print("Special: /image <path> /reset [name] /history [full] "
          "/quiet on|off /notes [full|raw <id>], or 'exit'.\n")
    while True:
        try:
            line = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[bye]")
            return
        if not line:
            continue
        if line.lower() in ("exit", "quit"):
            print("[bye]")
            return
        if line.startswith("/image "):
            sess.attach_image(line[len("/image "):].strip().strip('"\''))
            continue
        if line.startswith("/reset"):
            parts = line.split(None, 1)
            new_name = parts[1].strip() if len(parts) > 1 else None
            sess.reset(user_name=new_name)
            continue
        if line == "/history":
            sess.show_history(full=False)
            continue
        if line == "/history full":
            sess.show_history(full=True)
            continue
        if line == "/quiet on":
            verbose = False
            print("[verbose=False — Eva's trace logs hidden]")
            continue
        if line == "/quiet off":
            verbose = True
            print("[verbose=True — Eva's trace logs visible]")
            continue
        if line == "/notes" or line.startswith("/notes "):
            _handle_notes_command(agent, line)
            continue
        try:
            sess.send(line, verbose=verbose)
        except Exception as e:
            import traceback
            print(f"[ERR] {type(e).__name__}: {e}")
            traceback.print_exc()
