# R-11 设计文档：TaskStore + β 数据 + Full FT（2026-05-13）

## 文档定位

R-11 是把 Eva 的 user-notes 子系统从"flat prose store"升级为"facts + tasks 双
范畴 + lifecycle"的根本性架构改造。它**不是**对 R-1~R-10 任何一条的延伸补丁
——而是一个跨三层（数据 / runtime / 模型权重）的协同改造。

本 doc 是冻结版的设计契约，所有后续 implementation / review / 重构都以此为
准。R-1~R-10 + 各 hot-fix 的关系详见 `TODO_2026-05-13_root_fixes.md`。

---

## 1. 背景与必要性

### 1.1 R-1~R-10 + hot-fix 的边界

至 2026-05-13，已落地 7 项"根本方案"（R-1~R-7）+ 3 项 hot-fix（R-1.1/R-2.1/R-3.2/R-6.1）。三轮 Colab 实跑后所有 turn-级 hot-fix
都成功——但**新一轮实跑（13:55-15:25 那一组）暴露了一个 R-1~R-10 全部范围
之外的根本问题**：

```
user (Turn 6): "I will buy a new bear for you and I have to finish my final
                report in this Saturday, just remind me buy your toy and finish
                the document"

model: RememberThis(content="Master must buy Eva a toy and finish his final
                              report by Saturday.",
                    entity="Shared", topic="Task",
                    slot_values={"toy": "cuddly bunny"})   # ← 一条 note
       → Note #ec451435

user (Turn 12): "the final report have been finished now, so forget it"

model: ForgetMemory(record_id="ec451435", ...)
       → Note #ec451435 tombstoned
       → 买熊任务也没了        ← 数据丢失，下游用户感知到 bug
```

### 1.2 这不是补丁能解决的

| 层 | 该问题对应的"补丁式修法" | 为什么不够 |
|---|---|---|
| Prompt | "encourage model to split compound input" | SFT 分布锚定，prompt 是弱信号 |
| Verifier | 加规则 detect compound RememberThis 拒绝 | 规则永远追不上句式变体 |
| Runtime | execute_remember_this 内部拆 content | runtime "暗自" 拆，model thought 与实际不一致；下游 search 难对账 |
| **训练数据** | **重训** | **0 个 multi-call 样本，模型从未见过这种 pattern** |

数据集统计（9261 samples）：
- 含 RememberThis: 188
- 含 ForgetMemory: 78
- **同一 dialogue 内多次 RememberThis: 0**
- 含 R-1 `slot_values` 字段: 0
- 含 R-7 `event_date / event_type`: 0

→ 任何"只动 runtime / prompt"的修法都收敛于 ≤ ceiling 的 SFT 分布。
真正的根本必须**三层同时动**：

  数据层（β） → 加 ~570 个新样本，教 multi-call + lifecycle
  Runtime 层（R-11） → 加 TaskStore + 4 工具 + lifecycle 状态机
  模型层 → Full FT 重训 v3 数据集

### 1.3 立场

- R-1~R-10 + hot-fix 全部**保留**，作 fact 路径 + 老 SFT fallback
- R-11 新增 **task 路径**（独立 store，独立工具，独立 lifecycle）
- β 数据生成与 R-11 实现**严格同步**——任何一边 schema 漂移都会让训练样本和 dispatcher obs 对不上

---

## 2. R-11 Runtime 架构

### 2.1 数据范畴分离

```
Memory_maker/
├── notes_runtime.py     ← R-1/2/7 现有: NotesStore (prose facts)
└── tasks_runtime.py     ← R-11 新增: TaskStore (structured tasks)
```

| 范畴 | 例子 | 走 | 关键差异 |
|---|---|---|---|
| 永久事实 | "Master likes chocolate" | NotesStore + RememberThis | 不可变 + 偏好性 |
| 偏好属性 | "Master is lactose intolerant" | NotesStore + RememberThis | 同上 |
| 一次性任务 | "buy a bear by Saturday" | TaskStore + CreateTask | 有 completion + lifecycle |
| 过期事件 | "meeting next Monday 2pm" | TaskStore + CreateTask | 自动 expire |
| 偏好+任务 mixed | "I love jazz AND want to go to festival" | 两个 tool call (1 RememberThis + 1 CreateTask) | 范畴分流 |

### 2.2 TaskStore 物理布局

```
项目根/
├── Memory/                    lore corpus (read-only)
├── Notes/                     R-1/2/7 NotesStore
│   ├── notes.index, notes.jsonl, ...
└── Tasks/                     R-11 NEW
    ├── tasks.index            FAISS for action search
    ├── tasks.jsonl            源真值
    ├── tasks_content.json     content 拼接
    ├── tasks_meta.json        meta dicts（含 state/due/...）
    └── audit_<session>.log    create/done/cancel/expire 全记录
```

**决策**：
- 物理独立目录（不复用 NotesStore FAISS）—— 避免互相污染 search 排序
- 共享 mpnet encoder 实例（不浪费 GPU/CPU 资源）
- 双 store 各有 `recent_adds: list[str]` LRU（R-2.1 范式复用）

### 2.3 Task 数据结构

```python
@dataclass
class Task:
    # 标识
    task_id: str              # 8-char hex (uuid4 hex[:8])

    # 内容
    entity: str               # "Eva" | "Rosm" | "Shared"
    action: str               # 短动词短语（CreateTask 必填）
    detail: str               # 可选 prose 补充

    # 时间
    due_date: str             # ISO YYYY-MM-DD or ""
    due_time: str             # HH:MM or ""

    # 生命周期（4 states）
    state: str                # "pending" | "done" | "canceled" | "expired"
    created_at: str           # ISO timestamp
    completed_at: str         # state→done 时填
    canceled_at: str          # state→canceled 时填
    canceled_reason: str
    expired_at: str           # state→expired 时填

    # 关系
    parent_task: str          # 父 task_id; "" 为顶级
    keywords: list[str]

    # session 元数据
    origin: str               # f"session_{session_id}"
    deleted: bool             # **hard-delete tombstone**——独立于 state
```

**关键设计**：state 和 deleted **两个维度**：
- `state ∈ {pending, done, canceled, expired}` 是 lifecycle，永远保留在 store 内
- `deleted = True` 是运维 tombstone（expire 后 7+ 天清理），FAISS / list_by_state 跳过

类比：state ≈ 数据库的 "status enum"；deleted ≈ "soft_delete flag"。

### 2.4 TaskStore 公共 API

| 方法 | 入参 | 返回 | 调用方 |
|---|---|---|---|
| `create(action, entity, due_date="", due_time="", detail="", parent_task="", keywords=None)` | task_id | execute_create_task |
| `mark_done(task_id, reason="")` | bool | execute_complete_task |
| `mark_canceled(task_id, reason="")` | bool | execute_cancel_task |
| `list_by_state(state="pending", entity="", limit=10)` | list[Task] | execute_list_tasks + MemorySearch render |
| `search(query, top_k=5, states=("pending",))` | list[Task] | auto-correct fallback |
| `find_by_id(task_id)` | Task or None | direct id |
| `expire_due(now=None)` | list[task_id] | 定期 / lazy |
| `hard_delete(task_id)` | bool | 运维 |

### 2.5 4 个新工具（模型暴露）

```python
def CreateTask(action: str, entity: str,
               due_date: str = "", due_time: str = "",
               detail: str = "", parent_task: str = "",
               keywords: str = "") -> str:
    """Save a TASK Eva should remind the user about.
    For compound input ('do X AND do Y'), make TWO separate calls.
    Returns the task_id."""

def CompleteTask(task_id: str = "", query: str = "", reason: str = "") -> str:
    """Mark task done when user says 'finished X' / 'X is done'.
    Pass task_id (preferred) or query (description)."""

def CancelTask(task_id: str = "", query: str = "", reason: str = "") -> str:
    """Mark task canceled when user says 'X canceled' / 'forget X'.
    Use this instead of ForgetMemory for tasks."""

def ListTasks(entity: str = "", state: str = "pending", limit: int = 5) -> str:
    """List tasks. Default: pending for any entity."""
```

### 2.6 Tool observation schema（与 β 训练样本 byte-level 一致）

```
[TASK CREATED] Task #abc12345 (action="buy a bear toy", entity=Rosm,
state=pending, due_date=2026-05-18, due_time=none).
Future ListTasks / CompleteTask / CancelTask can reference this id.

[TASK DONE] Task #abc12345 marked completed.
Reason: "<reason>". Original action: "<action>".

[TASK CANCELED] Task #abc12345 canceled.
Reason: "<reason>". Original action: "<action>".

[TASKS] 2 pending task(s) for Rosm:
  - Task #abc12345 (action="buy a bear", due=2026-05-18)
  - Task #def67890 (action="finish report", due=2026-05-16)
[if N==0]: No pending tasks.

[TASK NOT FOUND] No live task matches "<query|id>".
Maybe already done/canceled. Use ListTasks to inspect.

[TASK ERROR] <validation reason>.
```

**字段顺序 / 引号 / 括号 / "none" 字面 / 换行**全锁定——β 样本 obs 字符串与
dispatcher 实际返回必须 byte-level 一致。schema_validate 脚本在数据生成
管线 enforce。

### 2.7 MemorySearch 双区渲染

R-11 后 MemorySearch 输出新增 PENDING TASKS section（与 SAVED NOTES 并列）：

```
### [MEMORY MODULE DATA for 'Rosm'] ###

Record 1 [Lore] [Subject: Rosm] [Topic: Pet]: ...

>>> SAVED NOTES <<<
  Note 1 [Note #...]: Master is lactose intolerant.
>>> END SAVED NOTES <<<

>>> PENDING TASKS <<<                 ← R-11 NEW
  Task 1 [Task #abc12345] [Subject: Rosm] [pending]: buy a bear toy
    [due] 2026-05-18
  Task 2 [Task #def67890] [Subject: Rosm] [pending]: finish final report
    [due] 2026-05-16
>>> END PENDING TASKS <<<
```

**决策**：自动包含（不需要单独 ListTasks）。优点：
- 模型每次 MemorySearch 都看见 pending tasks，减少"忘了有事"
- ListTasks 仍存在，给 state≠pending 的查询（如 done / canceled）

包含规则：
- Notes：走 NotesStore.search（既有 R-2.1）
- Tasks：TaskStore.search(query, states=("pending",)) + TaskStore.list_by_state("pending", entity, limit=3) 合并去重

### 2.8 ForgetMemory 在 task_id 上的自动 reroute

R-11 后 ForgetMemory 在 dispatch 入口检查 record_id 是否实际指向 task：

```
1. record_id 给了：
   a. NotesStore.find → 命中 → tombstone notes ✓
   b. TaskStore.find → 命中 → reroute 到 CancelTask(task_id, reason)
      obs: "[FORGET → CANCEL] Note path miss; record_id #xxx matches a
            task. Rerouting to CancelTask. <CancelTask result>"
   c. 都 miss → R-2.1 auto-correct（NotesStore.recent_adds + fallback_context）
```

**为什么 reroute 而不报错？** 兼容老 SFT 模型——它们只知 ForgetMemory，
不知 CancelTask。reroute 让它们能误用过去。β 训练后模型自然走 CancelTask。

### 2.9 与 TurnEvidenceLedger (R-4) 接线

新增 3 个 evidence source：

| source | trigger | meta |
|---|---|---|
| `task_create` | execute_create_task 成功 | `{task_id, action, due_date, ...}` |
| `task_done` | execute_complete_task 成功 | `{task_id, reason}` |
| `task_cancel` | execute_cancel_task 成功 | `{task_id, reason}` |

verifier 新 helper：

```python
def current_turn_has_task_evidence(agent, kind="any"):
    """kind ∈ {create, done, cancel, any}"""
    target = f"task_{kind}" if kind != "any" else None
    return any(
        ev.source.startswith("task_") and
        (target is None or ev.source == target)
        for ev in agent.turn_evidence
    )
```

### 2.10 与 DialogFocus (R-6) 接线

`DialogFocus` 新增 `task_id` 字段：

```python
@dataclass
class DialogFocus:
    entity: str = ""
    slot: str = ""
    topic: str = ""
    task_id: str = ""    # R-11 新增
    set_at_turn: int = -1
    source: str = ""
```

用途：用户说 "mark it done" 但没指 id 时，runtime fallback 用
`dialog_focus.task_id` 兜底。auto-correct 路径同 R-2.1。

### 2.11 Verifier 3 个新 reason

加入 `REASON_POLICY`：

```python
"task_request_not_handled": {
    "severity": "hard", "fix": "inject_tool",
    "canned": "I should have saved that as a task. Let me do it now.",
    "trigger": (
        "EXPLICIT_TASK_REQUEST intent + "
        "current_turn_has_task_evidence('create') == False"
    ),
},
"task_completion_not_handled": {
    "severity": "hard", "fix": "inject_tool",
    "canned": "I should mark that done. Let me update the task.",
    "trigger": (
        "EXPLICIT_TASK_COMPLETION intent + "
        "current_turn_has_task_evidence('done') == False"
    ),
},
"task_cancellation_not_handled": {
    "severity": "hard", "fix": "inject_tool",
    "canned": "I should cancel that task. Let me do so.",
    "trigger": (
        "EXPLICIT_TASK_CANCELLATION intent + "
        "current_turn_has_task_evidence('cancel') == False"
    ),
},
```

新 3 个 intent classifier（DeepSeek judge）：
- EXPLICIT_TASK_REQUEST: "remind me to X" / "I need to do X" / "我要做 X"
- EXPLICIT_TASK_COMPLETION: "I finished X" / "X is done" / "我做完 X 了"
- EXPLICIT_TASK_CANCELLATION: "skip X" / "forget X" / "我不做 X 了"（要和"forget the fact" 区分）

注入修复（auto-correct 模式）：
- task_request：runtime 调 CreateTask with action 从 user_text 抽（heuristic）
- task_completion：fallback_context + TaskStore.search 找候选 → mark_done
- task_cancellation：同上 → mark_canceled

### 2.12 Hedge fallback：runtime 内部拆 compound

主流程不做 compound 拆分（信任 β 训练后模型自拆）。但保留一个降级路径：
verifier 检测 CreateTask 的 `action` 字段含 ` AND ` / ` 和 ` 等连接词 →
报 `task_compound_undecomposed` reason → runtime 内部拆 → 多次调
TaskStore.create → obs 提示模型"compound was split into N tasks"。

这条**仅**给老 SFT 模型 + 训练偶发漏抓时用。β 之后应当几乎不触发。

---

## 3. β 数据生成

### 3.1 样本类型 + 数量预算

| Type | 用途 | 数量 | 优先级 |
|---|---|---|---|
| **T1** Simple CreateTask | 单任务基线 | 80 | ★★★ |
| **T2** CreateTask + 相对日期 | 日期解析 | 50 | ★★★ |
| **T3** Compound 拆分（X AND Y → 2 calls）| **核心** | 80 | ★★★ |
| **T4** CompleteTask | lifecycle "done" | 60 | ★★★ |
| **T5** CancelTask | lifecycle "canceled" | 50 | ★★★ |
| **T6** ListTasks | retrieval pending | 40 | ★★ |
| **T7** task vs fact 区分 contrastive | 范畴分流 | 60 (30 对) | ★★★ |
| **T8** "I forget" 反例 | 防误删 | 30 | ★★★ |
| **T9** 多 turn lifecycle (create → list → complete) | 端到端 | 40 | ★★ |
| **T10** parent_task 子任务 | 任务树 | 20 | ★ |
| **T11** due_time（不只 due_date） | 精确时间 | 30 | ★ |
| **T12** Mixed compound（fact + task） | 跨范畴 | 30 | ★★ |

**总：~570 新样本**（占 v3 总集约 6%）。

### 3.2 JSONL 样本 schema（v3）

```jsonc
{
  "user_name": "Rosm",
  "dialogue": [
    {"role": "user", "content": "..."},
    {"role": "assistant",
     "thought": "<必须显式说出 task/fact 判定依据>",
     "action": "<CreateTask | RememberThis | ...>",
     "action_input": "<key=\"val\", key=\"val\" 形式>"},
    {"role": "tool",
     "tool_name": "<同上>",
     "observation": "<§2.6 标准化 obs>"},
    // compound 时继续 N 个 assistant→tool pair
    {"role": "assistant", "thought": "...", "final_answer": "..."}
  ],
  "_custom_tools": "<v3 stub: 5 旧 + 4 新>",
  "_meta_strategy": "tasks_lifecycle | tasks_compound | fact_vs_task | ...",
  "_meta_source": "generate_task_samples",
  "_sample_type": "T1_SIMPLE | T3_COMPOUND | ...",
  "_task_id": "tasksT3_0042",
  "_attempt": 1,
  "_audited": true | false
}
```

### 3.3 复合形态（Q1=A 锁定）

```
user: "buy a bear AND finish report"
assistant: CreateTask(action="buy a bear", ...)
tool: [TASK CREATED] #abc...
assistant: CreateTask(action="finish report", ...)
tool: [TASK CREATED] #def...
assistant: final_answer
```

**关键学习信号**（thought 必须显式）：
- "Two distinct tasks joined with 'and'"
- "save them as SEPARATE tasks"
- "each can be marked done individually later"

### 3.4 task vs fact 边界规则

模型 thought 必须显式判定。生成 prompt 强制：

| 输入特征 | 判定 | 工具 |
|---|---|---|
| stative verb (love/like/prefer) + 无 completion | fact | RememberThis |
| be-verb + 属性 ("Master is X") | fact | RememberThis |
| 习惯句 ("usually X / every X") | fact (preference) | RememberThis |
| action verb (buy/finish/call/visit/...) + 单次 completion | task | CreateTask |
| future tense + due ("will X by Y") | task with date | CreateTask |
| "want to X sometime" | task (no date) | CreateTask |
| "remind me to X" | task | CreateTask |
| "I'm doing X on Y" | task with date | CreateTask |

### 3.5 旧 188 RememberThis 样本审核（Q3=审核+重打标）

预估分布：
- 纯 fact (preference / past event 类)：~130 条 → 保留
- task-shaped (buy / do / call 类)：~40 条 → 重写为 CreateTask 形态
- ambiguous：~18 条 → 上下文判定

工作量：~2-3 小时人工 review。

### 3.6 生成 pipeline（Q4=hybrid）

```
generate/generate_task_samples.py:

1. for type in [T1...T12]:
     seeds = load(f"seeds/{type}.jsonl")  # 12 files, ~5-10 seeds each = ~100 hand-written
     target_n = TYPE_BUDGET[type]
     
2. for batch of 5 seeds:
     prompt = build_expansion_prompt(type, batch, target_per_seed=8)
     # prompt 强制:
     #   - 输出 JSON list
     #   - 严格按 seed schema
     #   - thought 含判定信号
     #   - action_input 严格 key="val" 格式
     #   - 不重复 seed 的 action/entity/date
     expanded = deepseek_call(prompt)
     for sample in expanded:
         if validate_schema(sample):
             write_to(f"datasets/v3/{type}_generated.jsonl")

3. validate_schema 必填检查:
   - schema 字段完整
   - dialogue 角色顺序合法 (user → assistant → tool → ...)
   - action_input 是 key="val" 形式可 parse
   - tool obs 完全匹配 §2.6 标准 (字符串 startswith / contains 检查)
   - thought 含"task" 或 "fact" 字样

4. human_review_queue:
   - 抽 20% 分层采样
   - 输出 review.html 可视化
   - 人工标 _audited=true/false
```

成本：~$15 DeepSeek API + 2-3 day 人工。

### 3.7 v3 数据集合并

```
final_dataset_ready_v3.jsonl =
    9073 老样本 (9261 - 188 待审)
  + 188 重打标样本 (~140 保留 + ~48 转 CreateTask)
  + 570 新生成样本
  ≈ 9831 条
```

---

## 4. Full FT 训练配置（Q7 锁定）

| 项 | 配置 |
|---|---|
| 基础模型 | 现有 Eva-Qwen3.5-VL-9B-Merged |
| 数据 | v3 全集 (~9831 条) |
| 老数据保留比例 | 100% (防 catastrophic forgetting) |
| Batch size | effective ~64 |
| Epochs | 2-3 |
| Learning rate | 5e-6 ~ 1e-5（增量 FT，比 from-scratch 小一档） |
| Eval checkpoint | 每 500 steps |
| Early stop | 新行为命中 ≥85% AND 回归 ≥95% |

---

## 5. 验证集设计

3 个独立 bucket：

### 5.1 新行为命中（60 条手写）

任务输入 → 期望 model 调 CreateTask（含 compound → 多 call）。

| 子类 | 数量 | 验收 |
|---|---|---|
| Simple task | 20 | ≥18 走 CreateTask |
| Compound | 15 | ≥12 走多个 CreateTask call |
| Mixed compound (fact + task) | 10 | ≥8 走 RememberThis + CreateTask |
| ListTasks 触发 | 5 | ≥4 走 ListTasks |
| Complete / Cancel | 10 | ≥8 走对应工具 |

整体通过率 ≥85%。

### 5.2 回归（50 条从老 188 抽样）

纯 fact 输入 → 期望仍走 RememberThis。通过率 ≥95%。

### 5.3 复盘 case（4 条手写）

- Turn 6: "buy bear AND finish report" → 期望 2 个 CreateTask call
- Turn 7: "remind me to buy your toy and finish the document" → 期望 2 个 CreateTask call
- Turn 20: "well, I want to know what need to do, maybe I forget something" → 期望走 MemorySearch 或 ListTasks，**不**走 CancelTask
- 任意 "I forget X" 句式（无 imperative）→ 不走 CancelTask / ForgetMemory

复盘 100% 通过。

---

## 6. 5 周执行节奏

```
Week 1 ─ R-11 骨架 + β 独立 seeds
  D1-2  R-11: tasks_runtime.py 完整 + 单测
  D1-3  β:    T1 / T7 / T8 seeds (不依赖 dispatcher)

Week 2 ─ Dispatcher + 多 turn seeds
  D3-4  R-11: 4 dispatcher + step_once 路由 + ledger 写入
  D5    R-11: eva_prompts 4 stub + RememberThis docstring 调整
  D4-7  β:    T3 / T4 / T5 / T9 seeds (依赖 dispatcher obs)

Week 3 ─ MemorySearch 渲染 + verifier + 长尾 seeds
  D6-7  R-11: PENDING TASKS 渲染 + 3 verifier reason
  D8-9  β:    T2 / T6 / T10 / T11 / T12 seeds
  D10   β:    188 旧样本审核 + 重打标

Week 4 ─ 集成 + 训练准备
  D11   schema 一致性脚本 (R-11 obs ↔ β 样本字符串)
  D12   β: v3 合并 + 验证集准备
  D13   R-11: 全套 unit + 集成测试 (stub mode)

Week 5+ ─ SFT + 验证
  Full FT v3; checkpoint 评估; 实跑 Colab 复盘 Turn 6/20
```

---

## 7. Interlock checkpoints

| 截止 | R-11 必须 ready | β 必须 ready |
|---|---|---|
| W1 end | tasks_runtime.py 单测全过 | T1 / T7 / T8 seeds (~25 条) |
| W2 end | 4 dispatcher + 路由 OK | T3 / T4 / T5 / T9 seeds (~230 条) |
| W3 end | PENDING TASKS 渲染 + verifier reasons | 12 类 seeds 完工，DeepSeek expansion 启动 |
| W4 end | 完整集成测试通过 | v3 数据生成完，验证集准备完 |
| W5 end | — | Full FT 第一个 checkpoint |

任一边没达成 → 不进下一周。

---

## 8. Schema 一致性 enforce

`scripts/validate_r11_schema.py`（W4 D11 实现）：

1. 对每条 v3 新样本：
   - 跑 dispatcher in stub mode（无模型）
   - 比较生成的 tool obs 字符串与样本里 observation 字段
   - 任何 mismatch 报告差异行号

2. 对 R-11 dispatcher：
   - 检查每个工具返回的 obs 严格匹配 §2.6 模板
   - regex 校验：`r"^\[TASK CREATED\] Task #[0-9a-f]{8} \(action=\"[^\"]+\""`

任何 mismatch 阻断 commit。

---

## 9. 撤补丁清单（R-11 完成后）

| 补丁 | 状态 |
|---|---|
| P0-1 toy regex (eva_slots) | 保留作 fallback (R-1 已降级) |
| P1-5 ANSWER SCOPE prompt | 保留作 cumulative defense (R-5 已加 logits guard) |
| P2-7 _resolve_relative_dates | 保留作 fallback (R-7 已升结构化字段) |
| R-3.2 subjective skip | 保留（与 R-11 正交） |
| R-6.1 pronoun resolver | 保留（与 R-11 正交） |
| **R-1.1 / R-2.1 / R-3.2 / R-6.1 全部 hot-fix** | **保留** |

R-11 不撤任何既有补丁——它**新增** task 路径，不与 fact 路径冲突。

---

## 10. 风险 + 缓解

| 风险 | 缓解 |
|---|---|
| **catastrophic forgetting** lore 行为退化 | 100% 保留老数据 + 低 LR (5e-6 ~ 1e-5) + 2-3 epoch + checkpoint 评估 |
| 模型把 fact 误分类成 task | T7 contrastive pair ≥30 对 + 188 旧 RememberThis 全部保留 |
| Compound 拆分过度（误拆 "I love pizza and pasta"）| T7 / T12 反例样本明确"和"的语义边界 + thought 显式判定 |
| Full FT 训出来对 lore 答题 regression | 验证集 bucket 2 严格 ≥95% |
| 数据生成 DeepSeek 输出 schema 错乱 | validate_schema 严格 + 失败丢弃重生 |
| R-11 dispatcher obs 与 β 样本对不上 | scripts/validate_r11_schema.py 在 commit 时阻断 |
| **运维 expire_due 逻辑误删未完成任务** | expire 仅在 due_date 过期 ≥ 7 天 + state=pending 时触发 |
| **TaskStore + NotesStore 数据迁移到生产时不一致** | tasks/ 和 notes/ 独立目录 + 各自 audit log |

---

## 11. 已锁定决策清单

R-11 Runtime：

| ID | 决策 | 选项 |
|---|---|---|
| R-11.A | MemorySearch 是否自动包含 pending tasks | 是（追加 PENDING TASKS section）|
| R-11.B | ForgetMemory(task_id) 处理 | 自动 reroute 到 CancelTask |
| R-11.C | state 分类 | 4 状态 (pending / done / canceled / expired) |
| R-11.D | Compound 拆分位置 | 由模型 SFT 后自拆，runtime 不内部拆（Hedge fallback 例外） |

β 数据：

| ID | 决策 | 选项 |
|---|---|---|
| β.Q1 | Compound 形态 | A: 模型连发 N 个独立 CreateTask call |
| β.Q3 | 旧 188 RememberThis | 审核 + 重打标 |
| β.Q4 | 生成策略 | C: hybrid (seeds + DeepSeek + 抽审) |
| β.Q7 | 重训方式 | Full FT 增量 |

未锁定的开放问题（执行中再决定）：

- Seed 撰写优先级（T3/T7/T8 先 vs 全 type 并行）
- 验证集 60 条手写 user_name 分布（沿用 9261 还是新平衡）
- Full FT GPU 预算 / 云资源选型
- v3 是否保留 v2 全集 vs 下采样

---

## 12. 文档版本

- 2026-05-13 v0.1 初稿。
- 后续版本变更需在此处追加，**不修改已锁定决策**——新决策走 v0.2 / v0.3。
