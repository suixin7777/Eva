"""Advisor — remote-LLM advisor that injects per-turn planning hints
into Eva's local model prompt.

Architecture (设计哲学):
  本地 9B 模型不擅长"分解 compound input / 选择正确工具序列 / 跟踪 hex ID"，
  这些事情在 SFT 分布之外仍然脆弱。Advisor 把这部分**决策**外包给远端
  DeepSeek-V4-Pro，并把建议**注入每轮 prompt 顶部**——不进历史文本，
  不持久化，每轮重新生成。

  本地模型保留它擅长的：自然语言理解、Eva 傲娇人设、回复生成。
  远端模型只输出几行自然语言建议（"本轮你应该调 RememberThis 两次，分别记
  ① 和 ② ..."），本地模型大概率会跟，但允许跳脱（fail-soft）。

  Advisor 失败 / 超时 / DeepSeek API down 都不破坏推理——静默 fallback
  到无 advice 模式，保留的少量 verifier 兜底仍能挡掉灾难。

Public surface:
    build_advisor_prompt(...)  see build_advisor_prompt.py
    get_advice(...)            see advisor_client.py
"""

from .advisor_client import get_advice, AdvisorResult
from .build_advisor_prompt import build_advisor_prompt

__all__ = ["get_advice", "AdvisorResult", "build_advisor_prompt"]
