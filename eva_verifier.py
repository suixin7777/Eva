"""
eva_verifier.py — Post-hoc answer verifier for Eva.

The verifier inspects the final assistant answer against turn evidence
(slot facts, tool history, web results, time arithmetic) and either:
- approves, or
- raises one of HARD_VERIFIER_REASONS so the controller injects the
  required tool call and regenerates.

Implementation stays in eva_core.py — those methods are bound to
ChatAgent (they read self.turn_evidence / self.history_manager). This
module documents the surface and exposes the data classes so external
code can introspect verifier results.

Verifier methods (on ChatAgent, defined in legacy):
- _verify_final_answer(answer, latest_user_text)
- _required_action_from_verifier_reasons(reasons, latest_user_text, answer)
- _safe_fallback_for_hard_verifier_failure(verify_result, latest_user_text)
- _execute_controller_tool(tool_name, tool_params, latest_user_text, reason)
"""

from eva_core import (
    TurnEvidence,
    VerifyResult,
)
from eva_config import (
    HARD_VERIFIER_REASONS,
    ENABLE_ANSWER_VERIFIER,
    VERIFIER_DEBUG,
)

__all__ = [
    "TurnEvidence",
    "VerifyResult",
    "HARD_VERIFIER_REASONS",
    "ENABLE_ANSWER_VERIFIER",
    "VERIFIER_DEBUG",
]
