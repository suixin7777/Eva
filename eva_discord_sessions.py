"""
eva_discord_sessions.py — 多用户 Discord 会话隔离工具。

从 eva_discord.py 抽出来的纯逻辑层，目的：
1. 让 tests/test_session_isolation.py 能 offline 测，不用加载 9B 模型；
2. 把 swap-in / swap-out 的契约固化在一个小模块里，便于回归测。

依赖：仅 stdlib + eva_history（轻量，只用三个 dataclass）。
"""

from __future__ import annotations

import copy
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional


# ============================================================
#  跨轮 sticky 状态白名单
#
#  ChatAgent 上还有几十个字段，但绝大多数都是 per-turn 的，会在
#  agent.run() / _reset_turn_evidence() 开头被覆写。下面这 5 个槽
#  是真正"上一轮留下、影响下一轮"的：
# ============================================================
STICKY_SLOTS = (
    "history_manager",          # HistoryManager 对象，history + compressed_kv + image_registry
    "last_memory",              # LastMemoryState dataclass
    "dialog_focus",              # DialogFocus dataclass
    "_recent_phase2_outputs",   # 采样塌缩检测环形 list
    "_recent_phase2_modes",     # 同上
)

MAX_TRACKED_SESSIONS = 100  # LRU 上限


@dataclass
class UserSession:
    """单个 Discord 用户的 Eva 对话状态。

    state 是 snapshot_agent_state() 的返回 dict（含 5 个 sticky 槽），
    或者 None（首轮，restore_agent_state 时建空白状态）。
    """
    user_name: str
    state: Optional[dict] = None
    turn_count: int = 0


def snapshot_agent_state(agent_obj) -> dict:
    """把 agent 当前的 5 个跨轮 sticky 状态深拷贝出来。

    deepcopy 是必须的：history_manager.image_registry 含 PIL Image 引用，
    list/dict 也是可变的；如果浅拷贝，后一个用户的 mutate 会污染前一个
    用户的 snapshot。
    """
    return {
        "history_manager": copy.deepcopy(agent_obj.history_manager),
        "last_memory": copy.deepcopy(agent_obj.last_memory),
        "dialog_focus": copy.deepcopy(agent_obj.dialog_focus),
        "recent_phase2_outputs": list(agent_obj._recent_phase2_outputs),
        "recent_phase2_modes": list(agent_obj._recent_phase2_modes),
    }


def restore_agent_state(agent_obj, snap: Optional[dict]) -> None:
    """把快照写回 agent；snap=None 时把状态清成一个全新会话。

    None 路径用于：
      - 首轮（用户从未来过）
      - /reset 命令之后的下一轮
    """
    # 局部 import 避免在 offline 测试 import 这个模块时连带触发 transformers。
    from eva_history import HistoryManager, LastMemoryState, DialogFocus

    if snap is None:
        agent_obj.history_manager = HistoryManager()
        agent_obj.last_memory = LastMemoryState()
        agent_obj.dialog_focus = DialogFocus()
        agent_obj._recent_phase2_outputs = []
        agent_obj._recent_phase2_modes = []
    else:
        agent_obj.history_manager = snap["history_manager"]
        agent_obj.last_memory = snap["last_memory"]
        agent_obj.dialog_focus = snap["dialog_focus"]
        agent_obj._recent_phase2_outputs = list(snap["recent_phase2_outputs"])
        agent_obj._recent_phase2_modes = list(snap["recent_phase2_modes"])


class SessionStore:
    """LRU 字典 + helper。封装是为了测试里能创建独立实例。

    主程序 (eva_discord.py) 用一个模块级全局实例就够。
    """

    def __init__(self, max_sessions: int = MAX_TRACKED_SESSIONS):
        self.max_sessions = max_sessions
        self._d: "OrderedDict[int, UserSession]" = OrderedDict()
        self.evictions = 0  # 测试可观察

    def __len__(self) -> int:
        return len(self._d)

    def __contains__(self, user_id: int) -> bool:
        return user_id in self._d

    def get_or_create(self, user_id: int, user_name: str) -> UserSession:
        """取 user_id 的 session；不存在就建一个空的，并维持 LRU 大小。"""
        sess = self._d.get(user_id)
        if sess is None:
            sess = UserSession(user_name=user_name)
            self._d[user_id] = sess
            while len(self._d) > self.max_sessions:
                self._d.popitem(last=False)
                self.evictions += 1
        else:
            if sess.user_name != user_name:
                sess.user_name = user_name
        self._d.move_to_end(user_id)
        return sess

    def touch(self, user_id: int) -> None:
        """显式标记访问；用于 swap-out 后更新 LRU 顺序。"""
        if user_id in self._d:
            self._d.move_to_end(user_id)

    def reset(self, user_id: int) -> bool:
        """删掉一个用户的 session。返回是否真的删掉了。"""
        return self._d.pop(user_id, None) is not None

    def peek(self, user_id: int) -> Optional[UserSession]:
        """只读访问，不更新 LRU 顺序。仅用于测试断言。"""
        return self._d.get(user_id)


def inject_resumed_turns(agent_obj, turns: list) -> int:
    """把外部恢复的 (user_text, assistant_text) 对注入 agent.history_manager。

    用法场景：bot 重启后内存里没有这个用户的 session。我们从 Discord channel
    history 拉到他最近的对话，伪造成 ConversationTurn 塞进 history，然后让
    HistoryManager 自带的"超过 KEEP_FULL_TURNS 就压进 compressed_kv"逻辑
    自动接管 —— 这样最近 N 轮保留完整，更早的自动变 summary。

    Args:
        agent_obj: 已经 restore 到空白 session 的 ChatAgent。
        turns: 按时间顺序（oldest-first）的 (user_text, assistant_text) 列表。

    Returns:
        实际注入的 turn 数（与 len(turns) 相同；compress 不计入）。
    """
    from eva_history import ConversationTurn
    from eva_config import REACT

    hm = agent_obj.history_manager
    injected = 0
    for user_text, assistant_text in turns:
        # 跳过任一端空白的对，避免污染历史
        if not (user_text and str(user_text).strip() and
                assistant_text and str(assistant_text).strip()):
            continue
        turn = ConversationTurn(user_content=str(user_text).strip())
        # 包成 <|answer|>...<|end_react|>，跟 _prompt 构建器对历史 turn 的预期
        # 格式匹配；to_compact_messages() 会把它当 final answer 渲染。
        wrapped = f"{REACT['answer']}{str(assistant_text).strip()}{REACT['end']}"
        turn.add_assistant_step(wrapped)
        hm.history.append(turn)
        injected += 1

    # 复用 HistoryManager._finalize_current_turn 的压缩逻辑：
    # 超过 KEEP_FULL_TURNS 的最老 turn 转成 summary 进 compressed_kv。
    while len(hm.history) > hm.KEEP_FULL_TURNS:
        oldest = hm.history.pop(0)
        qa_pair = oldest.to_summary_text(hm.user_name)
        hm.compressed_kv.append(qa_pair)
        if len(hm.compressed_kv) > hm.MAX_COMPRESSED_QA:
            hm.compressed_kv.pop(0)

    return injected
