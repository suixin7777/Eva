"""
eva_history.py — Conversation history, turn state, ReAct stopping, evidence.

Extracted from eva_core.py during the P2 step-6 refactor. This module owns:

- StreamPrinter         live ReAct stream pretty-printer (THOUGHT/TOOL/ANSWER
                        framing with proper indentation)
- ConversationTurn      one user turn = user content + assistant ReAct steps
                        + tool outputs (+ optional image)
- HistoryManager        full chat state: image registry across turns + history
                        compaction so long sessions don't blow up the prompt
                        with stale ReAct traces
- ReActStoppingCriteria HF stopping-criteria that watches for ReAct end tokens
- TurnEvidence          dataclass — structured per-source evidence the
                        controller-side verifier consumes (raw text still
                        flows to the model unchanged)
- VerifyResult          dataclass — verifier verdict + suggested repair action

Key behaviours preserved (per user requirement):
- Image persistence across turns: HistoryManager.image_registry +
  get_image_by_path() + get_latest_visible_image() let the model refer to
  images attached in earlier turns by path or "latest".
- History compaction: past turns auto-compress to user+final-answer pairs
  (KEEP_FULL_TURNS=3, MAX_COMPRESSED_QA=10) so the prompt stays bounded.

Also re-exports clean_visual_tags / render_messages_to_prompt so callers
that import them from eva_history (legacy pattern) keep working.
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from transformers import StoppingCriteria

from eva_config import REACT, EOT, LOCAL_PIXELS
from eva_render import clean_visual_tags, render_messages_to_prompt


__all__ = [
    "StreamPrinter",
    "ConversationTurn",
    "HistoryManager",
    "ReActStoppingCriteria",
    "TurnEvidence",
    "VerifyResult",
    # Re-exported render helpers (legacy import surface)
    "clean_visual_tags",
    "render_messages_to_prompt",
]


# ============================================================
# StreamPrinter — live pretty-printer for the ReAct token stream
# ============================================================
class StreamPrinter:
    def __init__(self):
        self.buffer = ""
        self.current_indent = ""
        self.is_block_start = True
        self.has_printed_content = False
        self.TAG_MAP = {
            "<think>": ("    | --- THOUGHT ---", "    | "),
            "<|tool_code|>": ("        | --- TOOL CODE ---", "        | "),
            "<|tool_output|>": ("        | --- TOOL OUTPUT ---", "        | "),
            "<|answer|>": ("          | --- ANSWER ---", "          | "),
        }
        self._max_tag_len = max(len(t) for t in self.TAG_MAP)

    def process(self, new_text):
        new_text = new_text.replace("</think>", "")
        self.buffer += new_text
        while True:
            first_tag, first_pos = None, -1
            for tag in self.TAG_MAP:
                p = self.buffer.find(tag)
                if p != -1 and (first_pos == -1 or p < first_pos):
                    first_tag, first_pos = tag, p
            if first_tag is None:
                break
            pre, post = self.buffer.split(first_tag, 1)
            if pre:
                self._print(pre)
            header, indent = self.TAG_MAP[first_tag]
            prefix = "\n" if self.has_printed_content else ""
            print(f"{prefix}{indent.replace('| ', '')}{header.strip()}")
            self.current_indent = indent
            self.buffer = post
            self.is_block_start = True
            self.has_printed_content = True
        if not self.buffer:
            return
        tail_start = max(0, len(self.buffer) - self._max_tag_len)
        lt_pos = self.buffer.find("<", tail_start)
        if lt_pos == -1:
            self._print(self.buffer)
            self.buffer = ""
        elif lt_pos > 0:
            self._print(self.buffer[:lt_pos])
            self.buffer = self.buffer[lt_pos:]

    def _print(self, text):
        if not text: return
        if self.is_block_start:
            sys.stdout.write(self.current_indent)
            self.is_block_start = False
        sys.stdout.write(text.replace("\n", f"\n{self.current_indent}"))
        sys.stdout.flush()

    def flush(self):
        if self.buffer:
            self._print(self.buffer)
            self.buffer = ""
        print("")

    def discard_buffer(self):
        self.buffer = ""


# ============================================================
# ConversationTurn — one user turn record
# ============================================================
class ConversationTurn:
    def __init__(self, user_content, has_image=False, image=None, image_path=None, image_id=None):
        self.user_content = user_content
        self.has_image = has_image
        self.assistant_steps = []
        self.image = image
        self.image_path = image_path
        self.image_id = image_id

    def add_assistant_step(self, content):
        self.assistant_steps.append({"role": "assistant", "content": content})

    def replace_last_assistant_step(self, content):
        """Replace the last assistant step's content. Used by the verifier
        regenerate path (TODO 11-arch) when phase-2 is re-sampled — the
        failed first sample must not pollute history. Returns True if a
        prior assistant step existed and was replaced; False otherwise."""
        for i in range(len(self.assistant_steps) - 1, -1, -1):
            if self.assistant_steps[i]["role"] == "assistant":
                self.assistant_steps[i]["content"] = content
                return True
        return False

    def add_tool_output(self, content):
        self.assistant_steps.append({"role": "tool", "content": content})

    def get_final_answer(self):
        for step in reversed(self.assistant_steps):
            if step["role"] == "assistant" and REACT["answer"] in step["content"]:
                return step["content"].split(REACT["answer"])[-1].replace(REACT["end"], "").replace(EOT, "").strip()
        return ""

    def to_messages(self):
        return [{"role": "user", "content": self.user_content, "has_image": self.has_image,
                 "image": self.image, "image_path": self.image_path, "image_id": self.image_id}] + \
               [dict(step) for step in self.assistant_steps]

    def to_compact_messages(self):
        """Compact view for FINISHED past turns.

        Past turns must not expose planner/tool ReAct traces to the model,
        otherwise it can copy old planner thoughts, tool calls, observations,
        and recommendation answers into a new turn. Keep only the user message
        and the final assistant answer. The current in-flight turn still uses
        to_messages() so multi-tool planning can see current observations.
        """
        msgs = [{"role": "user", "content": self.user_content, "has_image": self.has_image,
                 "image": self.image, "image_path": self.image_path, "image_id": self.image_id}]
        final = self.get_final_answer()
        if final:
            msgs.append({"role": "assistant",
                         "content": f"{REACT['answer']}{final}{REACT['end']}",
                         "complete": True})
        return msgs

    def to_summary_text(self, user_name):
        user_text = self.user_content
        if self.has_image:
            image_label = self.image_path or self.image_id or "image"
            user_text = f"[Image attached: {image_label}] {user_text}".strip()
        parts = [f'{user_name}: "{user_text}"']
        for step in self.assistant_steps:
            if step["role"] == "assistant" and REACT["answer"] in step["content"]:
                ans = step["content"].split(REACT["answer"])[-1].replace(REACT["end"], "").strip()
                parts.append(f'Eva: "{ans}"')
        return " -> ".join(parts)


# ============================================================
# HistoryManager — full chat state
# ============================================================
class HistoryManager:
    KEEP_FULL_TURNS = 3
    MAX_COMPRESSED_QA = 10

    def __init__(self, user_name="Guest"):
        self.user_name = user_name
        self.compressed_kv = []
        self.history = []
        self.current_turn = None
        self.image_registry = {}
        self.image_order = []
        self._image_counter = 0

    def set_user_name(self, user_name):
        self.user_name = user_name

    def _register_image(self, image, image_path=None):
        if image is None: return None
        self._image_counter += 1
        image_id = f"image_{self._image_counter}"
        keys = {image_id}
        if image_path:
            path_str = str(image_path)
            keys.add(path_str)
            keys.add(os.path.basename(path_str))
            try: keys.add(os.path.abspath(path_str))
            except Exception: pass
        for key in keys:
            if key:
                self.image_registry[key] = image
        self.image_order.append(image_id)
        return image_id

    def get_image_by_path(self, requested_path):
        if not requested_path: return None
        key = str(requested_path).strip().strip("\"'")
        if not key: return None
        if key.lower() in {"latest", "current", "attached", "image"}:
            return self.get_latest_visible_image()
        candidates = [key, os.path.basename(key)]
        try: candidates.append(os.path.abspath(key))
        except Exception: pass
        for cand in candidates:
            if cand in self.image_registry:
                return self.image_registry[cand]
        return None

    def start_turn(self, user_content, has_image=False, image=None, image_path=None):
        if self.current_turn is not None:
            self._finalize_current_turn()
        image_id = None
        if has_image and image is not None:
            image_id = self._register_image(image=image, image_path=image_path)
        self.current_turn = ConversationTurn(user_content=user_content, has_image=has_image,
                                             image=image, image_path=image_path, image_id=image_id)

    def add_assistant_step(self, content):
        if self.current_turn:
            self.current_turn.add_assistant_step(content)

    def replace_last_assistant_step(self, content):
        """Delegate to current Turn (TODO 11-arch). Used by step_once's
        regenerate path to overwrite the failed phase-2 sample."""
        if self.current_turn:
            return self.current_turn.replace_last_assistant_step(content)
        return False

    def add_tool_output(self, content):
        if self.current_turn:
            self.current_turn.add_tool_output(content)

    def _finalize_current_turn(self):
        if self.current_turn is None: return
        self.history.append(self.current_turn)
        self.current_turn = None
        while len(self.history) > self.KEEP_FULL_TURNS:
            oldest_full_turn = self.history.pop(0)
            qa_pair = oldest_full_turn.to_summary_text(self.user_name)
            self.compressed_kv.append(qa_pair)
            if len(self.compressed_kv) > self.MAX_COMPRESSED_QA:
                self.compressed_kv.pop(0)

    def finish_turn(self):
        if self.current_turn is None: return ""
        final_answer = self.current_turn.get_final_answer()
        self._finalize_current_turn()
        return final_answer

    # ------------------------------------------------------------
    # P1 (verifier-v2): recent_turns
    #
    # SemanticVerifier (eva_verifier_semantic.py) needs the last N
    # (user_text, assistant_final_answer) pairs to do pronoun /
    # perspective / referent disambiguation. Returning a flat list
    # of dicts keeps the verifier independent of ConversationTurn's
    # internal layout (assistant_steps, react tags, image fields).
    #
    # Includes the in-flight current_turn ONLY if include_current=True
    # AND the turn already has at least one assistant step — otherwise
    # the verifier is asked to judge the very answer it would be paired
    # with, which leaks the answer back into its own context.
    # ------------------------------------------------------------
    def recent_turns(self, n=4, include_current=False):
        """Return up to N most recent finished turns as a flat list:
            [{'user': str, 'assistant': str}, ...]
        Oldest first. Empty assistant strings indicate a turn that
        produced no answer block (rare; tool-only or interrupted)."""
        out = []
        finished = list(self.history)
        if include_current and self.current_turn is not None and self.current_turn.assistant_steps:
            finished = finished + [self.current_turn]
        for turn in finished[-max(0, int(n)):]:
            try:
                final = turn.get_final_answer() or ""
            except Exception:
                final = ""
            out.append({
                "user": (turn.user_content or "").strip(),
                "assistant": final.strip(),
            })
        return out

    def _clean_visual_tags(self, text, replacement=""):
        return clean_visual_tags(text, replacement=replacement)

    def get_latest_visible_image(self):
        if (self.current_turn is not None and self.current_turn.has_image and self.current_turn.image is not None):
            return self.current_turn.image
        for turn in reversed(self.history):
            if turn.has_image and turn.image is not None:
                return turn.image
        if self.image_order:
            latest_key = self.image_order[-1]
            return self.image_registry.get(latest_key)
        return None

    def build_prompt_messages(self, sys_prompt, include_current=True):
        context_str = "\n".join(self.compressed_kv)
        sys_content = (f"{sys_prompt}\n\n[Context]\n{context_str}" if context_str else sys_prompt)
        messages = [{"role": "system", "content": sys_content, "complete": True}]
        for turn in self.history:
            # Past turns are always compact: user + final answer only.
            # Never include old tool traces in the prompt.
            for msg in turn.to_compact_messages():
                role = msg.get("role", "")
                if role == "user":
                    messages.append({"role": "user",
                                     "content": self._clean_visual_tags(msg.get("content", "")),
                                     "complete": True,
                                     "has_image": bool(msg.get("has_image")) and msg.get("image") is not None,
                                     "image": msg.get("image"),
                                     "image_path": msg.get("image_path"),
                                     "image_id": msg.get("image_id")})
                else:
                    messages.append({"role": role,
                                     "content": self._clean_visual_tags(msg.get("content", ""), replacement="[Image attached]"),
                                     "complete": True})
        if include_current and self.current_turn:
            for msg in self.current_turn.to_messages():
                role = msg.get("role", "")
                if role == "user":
                    messages.append({"role": "user",
                                     "content": self._clean_visual_tags(msg.get("content", "")),
                                     "complete": True,
                                     "has_image": bool(msg.get("has_image")) and msg.get("image") is not None,
                                     "image": msg.get("image"),
                                     "image_path": msg.get("image_path"),
                                     "image_id": msg.get("image_id")})
                else:
                    messages.append({"role": role,
                                     "content": self._clean_visual_tags(msg.get("content", ""), replacement="[Image attached]"),
                                     "complete": True})
        return messages

    def build_prompt_payload(self, sys_prompt, include_current=True, assistant_open=True):
        messages = self.build_prompt_messages(sys_prompt, include_current=include_current)
        effective_system_prompt = messages[0]["content"] if messages else sys_prompt
        prompt_text, prompt_images = render_messages_to_prompt(
            effective_system_prompt, messages[1:],
            include_assistant_prefix=assistant_open, max_pixels=LOCAL_PIXELS, skip_system_messages=True)
        return prompt_text, prompt_images


# ============================================================
# Generation control
# ============================================================
class ReActStoppingCriteria(StoppingCriteria):
    def __init__(self, stop_token_ids):
        self.stop_token_ids = stop_token_ids

    def __call__(self, input_ids, scores, **kwargs):
        if input_ids.shape[-1] == 0: return False
        return input_ids[0, -1].item() in self.stop_token_ids


# ============================================================
# Verifier dataclasses
# ============================================================
@dataclass
class TurnEvidence:
    """Structured evidence collected during one user turn.

    The model still receives the old natural-language tool outputs for ReAct
    compatibility. This structure is for controller-side verification only.

    R-4 (2026-05-13): 加 `topic` 和 `record_ref` 两个字段。
      - topic: 来自 topic_keywords.json 命中或 record meta.topic。用于"本轮已
        覆盖某 topic"的检查（替代用 raw_text 字符串搜的旧路径）。
      - record_ref: 跨证据对账锚——note_id（来自 RememberThis/ForgetMemory）
        或 record content hash。让 verifier 能判断 "PRE PROBE 命中的 record
        和 MemorySearch 返回的 record 是不是同一条"。
    """
    source: str          # "memory" | "topic_dict" | "notes_write" | "notes_delete"
                         # | "web" | "time" | "calculation" | "textgen"
    subject: Optional[str] = None
    slot: Optional[str] = None
    value: Any = None
    confidence: str = "related"   # "exact" | "related" | "topic" | "missing"
                                  # | "external" | "draft"
    raw_text: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)
    topic: str = ""               # R-4 新增
    record_ref: str = ""          # R-4 新增


class TurnEvidenceLedger:
    """Per-turn evidence ledger — single source of truth for what's grounded.

    R-4 设计目标：verifier / phase-2 / PRE PROBE 三处过去各自维护"本轮已知
    什么"的理解，导致同一证据有时被 verifier 看见、有时看不见（典型：PRE
    PROBE 注入 EXACT 但 verifier 仍要求 MemorySearch）。这个类把所有证据写入
    点统一到一个对象，所有读取走 query API。

    Backward-compatible：实现了 list-like 接口（len/iter/getitem/append），
    现有 `for ev in agent.turn_evidence` / `len(...)` 代码无需改动。

    Query API：
      - covers(subject, slot, topic, min_tier) — 本轮是否已覆盖某条问询
      - best_for(subject, slot) — 取该 (subject, slot) 上 tier 最强的一条
      - by_source(*sources) — 按 source 过滤
    """

    # tier 排序：用于 covers / best_for 的比较。
    _TIER_RANK = {
        "exact": 4,
        "external": 4,   # WebSearch / API 结果按 exact 同级处理
        "related": 3,
        "topic": 2,
        "draft": 1,
        "missing": 0,
    }

    def __init__(self):
        self._items: List["TurnEvidence"] = []

    # ---------------- list-compat ----------------
    def __len__(self):
        return len(self._items)

    def __iter__(self):
        return iter(self._items)

    def __getitem__(self, idx):
        return self._items[idx]

    def __bool__(self):
        return bool(self._items)

    def append(self, ev: "TurnEvidence") -> "TurnEvidence":
        self._items.append(ev)
        return ev

    def clear(self) -> None:
        self._items.clear()

    # ---------------- query API ----------------
    def by_source(self, *sources) -> List["TurnEvidence"]:
        if not sources:
            return list(self._items)
        s = set(sources)
        return [e for e in self._items if e.source in s]

    def has_source(self, *sources) -> bool:
        s = set(sources)
        return any(e.source in s for e in self._items)

    def covers(self, *, subject=None, slot=None, topic=None,
               min_tier: str = "topic") -> bool:
        """True iff at least one evidence covers the given filter at >= min_tier.

        过滤维度（任何一项为 None 即 wildcard）：
          - subject: 实体 ("Eva" / "Rosm" / "Shared")
          - slot: slot 名 ("toy" / "birthday" / ...)
          - topic: topic 标签 ("Toy" / "Birthday" / ...)

        min_tier 默认 "topic"：只要有 topic-only 级别的命中就算覆盖。要求更强
        证据时传 "related" 或 "exact"。
        """
        min_rank = self._TIER_RANK.get(min_tier, 2)
        for e in self._items:
            if subject and e.subject != subject:
                continue
            if slot and e.slot != slot:
                continue
            if topic and e.topic != topic:
                continue
            if self._TIER_RANK.get(e.confidence, 0) < min_rank:
                continue
            return True
        return False

    def best_for(self, subject: str, slot: str) -> Optional["TurnEvidence"]:
        """Strongest evidence row for (subject, slot). None if no match."""
        best = None
        best_rank = -1
        for e in self._items:
            if e.subject != subject or e.slot != slot:
                continue
            rank = self._TIER_RANK.get(e.confidence, 0)
            if rank > best_rank:
                best, best_rank = e, rank
        return best


@dataclass
class LastMemoryState:
    """R-6 (2026-05-13)：跨轮 sticky 的 retrieval 快照。

    把过去散落在 ChatAgent 上的 7 个 last_memory_* 字段（observation /
    primary_query / has_exact / has_related / judge_exact_count /
    judge_related_count / missing_slots）收敛到一个 dataclass。subject /
    entity 维度归 DialogFocus 管，这里只放"上一次 retrieval 留下了什么原始
    数据"。

    写入 3 处：PRE PROBE / step_once MemorySearch / verifier execute tool。
    读取 ~15 处（date binding / continuation / verifier missing-evidence 等）。
    """
    observation: str = ""
    primary_query: str = ""
    has_exact: bool = False
    has_related: bool = False
    judge_exact_count: int = 0
    judge_related_count: int = 0
    missing_slots: List[str] = field(default_factory=list)

    def reset(self) -> None:
        self.observation = ""
        self.primary_query = ""
        self.has_exact = False
        self.has_related = False
        self.judge_exact_count = 0
        self.judge_related_count = 0
        self.missing_slots = []


@dataclass
class DialogFocus:
    """R-6 (2026-05-13)：跨轮 sticky 的 dialog focus（关注实体/slot/topic）。

    与 LastMemoryState 的区别：
      - LastMemoryState 是 retrieval 的"原始快照"（这次搜了什么 query、
        拿到什么 obs、判定 EXACT/RELATED 多少条）；
      - DialogFocus 是 conversation-level"语义焦点"——可以由 user 显式提名
        覆盖、由 pronoun followup 继承、由 PRE PROBE 设置。

    取代旧字段：`last_memory_target_entity` / `last_missing_slot_target_entity`
    （两者实际上一直是同一个值，R-6 一并合并）。

    更新源（source 字段记录）：
      - "user_named":      用户句中显式提名 ("Rosm's birthday")
      - "pre_probe":       PRE PROBE subject_hint 命中
      - "tool":            MemorySearch tool 返回 target_entity
      - "pronoun_inherit": pronoun resolver 继承上一轮 focus（R-6 让 pronoun
                           resolver 不再自己推断 target，统一读 focus）
    """
    entity: str = ""        # "Eva" | "Rosm" | "Shared" | "Both" | ""
    slot: str = ""          # "toy" | "birthday" | "" (无具体 slot)
    topic: str = ""         # "Toy" | "Birthday" | "" (无具体 topic)
    set_at_turn: int = -1
    source: str = ""

    def is_set(self) -> bool:
        return bool(self.entity)

    def reset(self) -> None:
        self.entity = ""
        self.slot = ""
        self.topic = ""
        self.set_at_turn = -1
        self.source = ""

    def update(self, *, entity: str = "", slot: str = "",
               topic: str = "", turn: int = -1, source: str = "") -> None:
        """部分字段更新——传空字符串/-1 不覆盖原值，便于增量写入。

        典型用法：
          PRE PROBE 命中：focus.update(entity=hint, topic=t, source="pre_probe")
          tool 完成：    focus.update(entity=target, source="tool")
          pronoun 继承：仅写 source，不动 entity（已是上一轮的值）。
        """
        if entity:
            self.entity = entity
        if slot:
            self.slot = slot
        if topic:
            self.topic = topic
        if turn >= 0:
            self.set_at_turn = turn
        if source:
            self.source = source


@dataclass
class VerifyResult:
    ok: bool
    reasons: List[str] = field(default_factory=list)
    required_action: Optional[Dict[str, Any]] = None
    hard_fail: bool = False
    # TODO 11-arch (2026-05-07): fix_class is the policy-derived class
    # that controls how step_once dispatches the failure. Possible
    # values from REASON_POLICY in eva_verifier_logic:
    #   "inject_tool"     — controller runs a repair tool
    #   "regenerate"      — phase-2 re-sampled with collapse pressure
    #   "canned_fallback" — terminal failsafe canned message
    #   None              — verifier passed (ok=True)
    # Unlisted reasons fall back to DEFAULT_POLICY ("canned_fallback").
    fix_class: Optional[str] = None


# ============================================================
# R-3 (2026-05-13): Verdict + VerdictLedger
# ============================================================
# 把"safe_fallback 该释放哪个候选答案"从 callers 手动 phase2_answer= 传参
# 改成结构化 ledger 选择。每次 verify_final_answer 完成都 append 一条
# Verdict；fallback 时按 (severity, reason_count, stage_order) 排序选 best。
#
# 解决的根本问题（参见 2026-05-13 Turn 5 复盘）：
#   - original "55 days until my birthday" 被 regex 误报 date_math_target_date_mismatch
#   - regen "my birthday is way later" 被 LLM-judge 真错 pronoun_referent_mismatch
#   - 旧 fallback 释放 regen（最新一次），用户看到比 original 还差的答案
#   - R-3 ledger 按 (severity, reason_count, stage) 排序，自然选 original
#     （同 severity 同 reason_count，但 original 是更早 stage）
# ============================================================

# 用于 best() 排序的 stage 优先级。越靠前优先级越高（数值越小）。
# original = 模型最自然的输出；regen = 强制 collapse pressure 后的二次生成。
# regen 是修复尝试，质量预期比 original 差——同 severity 时偏好 original。
_VERDICT_STAGE_ORDER = {
    "original": 0,
    "after_tool": 1,        # phase-2 跑完 tool 后的 final answer
    "regen": 2,             # 强制 regenerate 路径产出
    "budget_exhausted": 3,  # regen budget 用尽时的兜底
}


def _classify_verdict_reasons(reasons) -> str:
    """把 reason 列表归类为 'regex' / 'llm_semantic' / 'mixed' / 'none'。

    用于 VerdictLedger.has_disagreement() 检测跨 verifier 不一致。
      - "llm_semantic": LLM-judge SemanticVerifier 推上来的（前缀
        'semantic_verifier_fail:'）
      - "regex": 其它所有 regex / 启发式判定
      - "mixed": 同一个 verdict 里两类都有
      - "none": 没有任何 reason（应当 ok=True，但防御）
    """
    if not reasons:
        return "none"
    has_llm = any(
        isinstance(r, str) and r.startswith("semantic_verifier_fail:")
        for r in reasons
    )
    has_regex = any(
        isinstance(r, str) and not r.startswith("semantic_verifier_fail:")
        for r in reasons
    )
    if has_llm and has_regex:
        return "mixed"
    if has_llm:
        return "llm_semantic"
    if has_regex:
        return "regex"
    return "none"


@dataclass
class Verdict:
    """One verifier verdict on one candidate phase-2 answer.

    多个 verdict（original / regen / after_tool）累积进 VerdictLedger。
    safe_fallback 通过 ledger.best() 选 release candidate，不再依赖 caller
    手动传 phase2_answer。
    """
    answer: str
    source_stage: str           # 见 _VERDICT_STAGE_ORDER
    verify_result: VerifyResult

    @property
    def severity(self) -> int:
        """0 = clean / 1 = soft fail / 2 = hard fail."""
        vr = self.verify_result
        if vr is None or vr.ok:
            return 0
        if vr.hard_fail:
            return 2
        return 1

    @property
    def reason_count(self) -> int:
        return len((self.verify_result.reasons if self.verify_result else None) or [])

    @property
    def reason_class(self) -> str:
        return _classify_verdict_reasons(
            (self.verify_result.reasons if self.verify_result else None) or []
        )


class VerdictLedger:
    """Per-turn verdict ledger. List-compat for iteration / len, plus query API."""

    def __init__(self):
        self._candidates: List[Verdict] = []

    # list-compat
    def __len__(self):
        return len(self._candidates)

    def __iter__(self):
        return iter(self._candidates)

    def __getitem__(self, idx):
        return self._candidates[idx]

    def __bool__(self):
        return bool(self._candidates)

    @property
    def candidates(self) -> List[Verdict]:
        return list(self._candidates)

    def add(self, verdict: Verdict) -> Verdict:
        self._candidates.append(verdict)
        return verdict

    def clear(self) -> None:
        self._candidates.clear()

    # query
    def best(self) -> Optional[Verdict]:
        """Pick the verdict with lowest severity → fewest reasons → earliest stage.

        Turn 5 复盘对照：
          original verdict: severity=2, reason_count=1 (regex)
          regen verdict:    severity=2, reason_count=1 (llm)
          tie on (severity, reason_count) → stage 排序 → original 胜
        """
        if not self._candidates:
            return None
        return min(self._candidates, key=lambda v: (
            v.severity,
            v.reason_count,
            _VERDICT_STAGE_ORDER.get(v.source_stage, 99),
        ))

    def has_disagreement(self) -> bool:
        """True iff at least one failed verdict is regex-only AND another is
        llm-only. 用作 telemetry——R-3 当前**不**自动 canned，仅记录现象，
        让 best() 的 (severity, reason_count, stage) 排序自然选 less-flagged
        candidate。
        """
        failed = [v for v in self._candidates if v.severity > 0]
        if len(failed) < 2:
            return False
        classes = {v.reason_class for v in failed}
        return "regex" in classes and "llm_semantic" in classes
