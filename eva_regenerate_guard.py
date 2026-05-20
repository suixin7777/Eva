"""
eva_regenerate_guard.py — Per-turn regenerate budget tracker.

Generalises the legacy `last_regenerate_reason` single-slot guard
that lived as an attribute on ChatAgent. The original guard could
only catch the case "same dominant reason fired twice in a row";
it could NOT catch:

  reasonA -> regenerate -> answer'
  reasonB -> regenerate -> answer''      (different reason; passes)
  reasonA -> regenerate again            (different reason in between, so
                                          last_regenerate_reason was reset)

That hop-around loop never converges. RegenerateGuard fixes it by
maintaining a per-turn ledger:

  - attempts[reason]   - count of regenerate attempts spent on each reason
  - total              - count of all regenerate attempts in the turn

Dispatch checks BOTH limits before allowing a regenerate, and the
caller is expected to release the original phase-2 answer (NOT a
canned apology) when either limit is reached. This matches the P0
semantic-reason fail-open posture: when the verifier can't make
forward progress, prefer the model's answer over a fabricated one.

Why a class instead of a few attributes:
  - State is reset on turn boundary; a class makes the lifecycle
    explicit (one instance per agent, .reset_for_new_turn() at the
    top of step_once).
  - Tests can construct one in isolation without pulling ChatAgent.
  - The legacy `last_regenerate_reason` attribute on ChatAgent stays
    around as a back-compat read-only mirror of `most_recent_reason`
    for a release while the dispatcher migrates.
"""

from typing import Optional, Dict, List


class RegenerateGuard:
    """Tracks regenerate attempts within a single user turn.

    Usage from the verifier dispatcher (eva_core.step_once):

        if not agent.regenerate_guard.try_consume(dominant_reason):
            # budget exhausted — release the original phase-2 answer
            return phase2_answer, False

        # ... do the regenerate ...

    The class does NOT decide what to do on exhaustion — that policy
    lives in the dispatcher. It only answers the yes/no question.
    """

    def __init__(self,
                 max_per_reason: int = 1,
                 max_total: int = 2):
        self._max_per_reason = max(0, int(max_per_reason))
        self._max_total = max(0, int(max_total))
        self._attempts: Dict[str, int] = {}
        self._total: int = 0
        self._history: List[str] = []   # debug-only ordered ledger
        self._most_recent_reason: Optional[str] = None

    # ------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------
    def reset_for_new_turn(self) -> None:
        """Clear all counters. Call at the start of each user turn,
        from the same place that resets agent.turn_evidence."""
        self._attempts.clear()
        self._total = 0
        self._history.clear()
        self._most_recent_reason = None

    # ------------------------------------------------------------
    # Quotas
    # ------------------------------------------------------------
    def can_regenerate(self, reason: Optional[str]) -> bool:
        """Pure check — does NOT consume budget. Returns True iff
        another regenerate for this reason would stay within both
        the per-reason cap and the total cap."""
        if reason is None:
            # Unknown / missing reason: allow IF total budget remains.
            return self._total < self._max_total
        used = self._attempts.get(reason, 0)
        return used < self._max_per_reason and self._total < self._max_total

    def try_consume(self, reason: Optional[str]) -> bool:
        """Atomic: check + increment. Returns True on success
        (caller may proceed with the regenerate); False on budget
        exhaustion (caller must NOT regenerate)."""
        if not self.can_regenerate(reason):
            return False
        key = reason or "<unknown>"
        self._attempts[key] = self._attempts.get(key, 0) + 1
        self._total += 1
        self._history.append(key)
        self._most_recent_reason = reason
        return True

    # ------------------------------------------------------------
    # Read accessors (debug + back-compat with last_regenerate_reason)
    # ------------------------------------------------------------
    @property
    def most_recent_reason(self) -> Optional[str]:
        """The reason for the most recent successful try_consume call,
        or None if no regenerate has been attempted this turn. Mirrors
        the semantics of ChatAgent.last_regenerate_reason for callers
        that haven't migrated."""
        return self._most_recent_reason

    @property
    def total_attempts(self) -> int:
        return self._total

    def attempts_for(self, reason: str) -> int:
        return self._attempts.get(reason, 0)

    def snapshot(self) -> dict:
        """Pure-data view for debug logging. Safe to JSON-serialize."""
        return {
            "total": self._total,
            "max_total": self._max_total,
            "max_per_reason": self._max_per_reason,
            "attempts": dict(self._attempts),
            "history": list(self._history),
        }

    def __repr__(self) -> str:
        return (f"RegenerateGuard(total={self._total}/{self._max_total}, "
                f"attempts={self._attempts})")
