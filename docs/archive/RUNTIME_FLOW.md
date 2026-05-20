# Eva Runtime Flow — How a User Turn Becomes an Answer

> 写给：第一次接触本项目的工程师，或者一段时间没看代码、需要快速回忆"用户
> 输入到底经过了多少环节"的人。
>
> 阅读时长：~15 min。看完后应该能回答："为什么用户问 X，trace 里出现了
> 这十几个 JUDGE / TOOL OUTPUT / VERIFIER 块？"

---

## 〇、Eva 是什么

一个**多轮对话推理系统**，约 13K 行 Python。核心能力：

- 多轮对话 + 长期个人记忆（生日、玩具、共同事件等）
- 智能工具路由（MemorySearch / WebSearch / GetCurrentTime / TextGeneration / RemoteVision）
- **两阶段推理**：Phase-1 贪心规划 + Phase-2 采样答案；中间可插入工具调用
- **验证 + 自我修复**：回答会过 Verifier，不达标自动注入工具重生成

主入口：[eva_chat_colab.py:ChatSession](../eva_chat_colab.py) → [eva_core.py:ChatAgent.run()](../eva_core.py)

---

## 一、High-Level Data Flow

```
user input
   │
   ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Step 0: history append + reset judge state                          │
│   eva_history.HistoryManager 记录新 turn                              │
│   eva_intent_judge.JudgeState.reset() 清缓存 + 预算计数              │
└─────────────────────────────────────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Step 1: Pre-Memory Probe (PRE PROBE)                                │
│   决定 "这一轮该不该提前注入记忆"                                     │
│   ├─ judge_intent (DeepSeek): EXPLICIT_WEB / EXPLICIT_MEMORY        │
│   ├─ topic_keywords match (regex word-boundary)                     │
│   ├─ judge_topic_subset (DeepSeek): 过滤误匹配的话题                 │
│   ├─ FAISS 检索 + slot 提取                                          │
│   └─ 决策: action ∈ {skip, probe-no-evidence, probe-inject}         │
└─────────────────────────────────────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Step 2: Phase-1 Generation (greedy)                                 │
│   model.generate(do_sample=False, prefix="<think>")                 │
│   ├─ FORCE_THINK_PREFIX 注入 → 必然产出 <think>...</think>          │
│   ├─ 输出形态二选一:                                                 │
│   │   <think>...</think><|tool_code|>ToolName(...)                  │
│   │   <think>...</think><|answer|>...                               │
│   └─ ReActStoppingCriteria 在 <|end_react|> 处停                    │
└─────────────────────────────────────────────────────────────────────┘
   │
   ├─ 出现 <|tool_code|>           ─┐
   │  ▼                             │ 循环：执行工具 → 输出注回历史 →
   │  Step 3: Tool Execution        │ 回 Step 2 重生 (max ~5 轮)
   │   ├─ MemorySearch              │
   │   ├─ WebSearch (Tavily)        │
   │   ├─ GetCurrentTime            │
   │   ├─ TextGenerationTool        │
   │   └─ AskRemoteVision           │
   │  ▼                             │
   │  tool_output 注回 history ────┘
   │
   ▼ (出现 <|answer|>)
┌─────────────────────────────────────────────────────────────────────┐
│ Step 4: Phase-2 Generation (sampled)                                │
│   model.generate(do_sample=True, temp=...)                          │
│   ├─ commit_terms 绑定 (Phase-1 承诺过的事实必须出现)                │
│   ├─ Phase-2 Collapse Guard 防止重复生成                             │
│   └─ 模式选择: direct / after_tool / after_memory                   │
└─────────────────────────────────────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Step 5: Verifier                                                    │
│   ├─ 语义 verifier (DeepSeek): 代词指代、视角一致性                   │
│   ├─ 逻辑 verifier (regex/谓词): 工具证据完整性、日期算术、记忆引用  │
│   │     注：这层 regex 是闭集合错误检查（保留），区别于 P6.4 要删的   │
│   │     pronoun-followup 识别 regex（开放语言枚举，不收敛）          │
│   └─ 失败 → required_action (注入工具) → 回 Step 3 修复              │
└─────────────────────────────────────────────────────────────────────┘
   │
   ▼ (verifier pass 或 regenerate budget 耗尽)
┌─────────────────────────────────────────────────────────────────────┐
│ Step 6: Output                                                      │
│   answer 落到 history.assistant_steps                               │
│   trace 行 close                                                     │
│   返回字符串给用户                                                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 二、各步骤详解

### Step 0 — 进入 turn

**职责**：把用户输入挂上 history，重置每轮 judge 状态。

**关键文件**：
- [eva_history.py](../eva_history.py) — `HistoryManager.append_turn()`
- [eva_intent_judge.py](../eva_intent_judge.py) — `JudgeState.reset()`

**`JudgeState`** 是一个 per-turn 的缓存 + 预算容器。挂在
`agent._llm_judge_state`。包含：

- `cache`: dict[(intent, query_lc)] → 上次判决（避免同 turn 内重复问 DeepSeek）
- `call_count`: 全局 LLM judge 调用次数（cap：`LLM_JUDGE_MAX_CALLS_PER_TURN=6`）
- `pronoun_call_count`: P6 pronoun resolver **独立预算池**（cap：`PRONOUN_RESOLVER_MAX_CALLS_PER_TURN=2`）

设计理由：跨子系统的 LLM 调用共享一个全局 budget，但 pronoun resolver
独立——避免 PRE PROBE 吃光预算后 resolver 被牵连。

---

### Step 1 — Pre-Memory Probe

**职责**：在模型读 prompt 前，决定要不要把记忆数据"提前注入"对话历史。
注入后模型在 Phase-1 看到的就是带证据的 prompt，能直接生成正确答案而不是
先调 MemorySearch 再答。这是一个**性能优化层**——避免每次都走"工具调用"
路径。

**关键文件**：
- [eva_memory_v2.py](../eva_memory_v2.py) — `MemoryModule.decide()` / `probe()`
- [eva_intent_judge.py](../eva_intent_judge.py) — `judge_intent()` / `judge_topic_subset()`
- [topic_keywords.json](../topic_keywords.json) — 话题→关键词映射

**决策链**（Layered Intent Classifier）：

```
1. 关键词匹配 (regex word-boundary)
   → 找出候选 topics (如 ["Birthday", "Gifts"])
   ↓
2. judge_topic_subset (DeepSeek): 用语义判断
   → 过滤误匹配（如 "do you like games" 不算 Gaming 话题，是 Likes）
   → 输出 relevant subset (可能为空)
   ↓
3. 如果 subset 非空：
   → FAISS embedding 检索（top_k=20，subject_filter）
   → slot extractor 抽取 (birthday/full_name/age/toy 等结构化字段)
   → 决策: inject={True/False, reason=exact_evidence|no_evidence}
```

**Trace 示例**：

```
| --- P2 PRE MEMORY PROBE ---
| action=probe reason=topic_match[layered:keyword>llm]:Birthday
| matched_topics=['Birthday']
| exact=1 related=0 topic_hit=1 meta_kw_hit=0 top1=8.51 target=Eva
| inject=True reason=exact_evidence
| --- P2 PRE MEMORY INJECTED ---
```

**hard guard 路径**：某些 query 如 `"how old are you?"` / `"what's your name?"`
被 `_is_personal_identity_question` regex 命中后**强制 probe**，不需要走
keyword 路径——这是兜底，避免 topic dict 漏掉这种 slot 类查询。

---

### Step 2 — Phase-1 Generation (greedy)

**职责**：让模型决定"要回答还是要调用工具"。贪心解码保证决策可重复。

**关键文件**：
- [eva_core.py:1938 step_once()](../eva_core.py)
- [eva_prompts.py](../eva_prompts.py) — FORMAT_RULES / IDENTITY_MASTER_INFERENCE

**输出格式**（ReAct 协议）：

```
<think>brief reasoning</think><|tool_code|>ToolName(args)<|end_react|>
              ↑                              ↑
              内部规划                     选择调工具
                                              或
<think>brief reasoning</think><|answer|>final response<|end_react|>
                                              ↑
                                            直接回答（之后还会进 Phase-2）
```

**FORCE_THINK_PREFIX**（2026-05-08 加的）：

```python
think_ids = self.tok.encode(THINK_START, add_special_tokens=False, return_tensors="pt")
inputs1["input_ids"] = torch.cat([inputs1["input_ids"], think_ids], dim=1)
```

把 `<think>` token 硬塞进 input 末尾，模型从 think 块**内部**开始解码——
不能跳过推理直接出 answer。背景：实测发现 greedy decoder 偷懒，对简单
query（如 `"for example?"`）跳过 think，导致缺失 self-reflection 引发幻觉。

---

### Step 3 — Tool Execution

**职责**：执行 Phase-1 决定调用的工具，把输出注回 history 里下次 generate 时就能看见。

**关键文件**：
- [eva_tools.py](../eva_tools.py) — 工具门面
- [eva_tools_runtime.py](../eva_tools_runtime.py) — Tavily / DeepSeek expert / vision / parser
- [eva_core.py](../eva_core.py) — `_execute_controller_tool()`

**5 个工具**：

| 工具 | 用途 | 注入 prompt 上下文 |
|---|---|---|
| `MemorySearch` | 查记忆库（FAISS + BM25 + 重排） | `### [MEMORY MODULE DATA for 'X'] ###` 块 |
| `WebSearch` | Tavily 搜索网络（外部事实） | `[Tavily AI Answer]` + URLs |
| `GetCurrentTime` | 当前日期 + 日期算术（如 "几天到生日"） | `[TIME BINDING]` + `[DATE CALCULATION BINDING]` 块 |
| `TextGenerationTool` | DeepSeek expert 处理 Phase-1 不擅长的复杂文本生成 | 直接返回字符串 |
| `AskRemoteVision` | 远程视觉理解（图片识别） | 描述字符串 |

**工具路由保护**（一个常见的反模式）：用户问 "what's the news today" 但
Phase-1 误调 MemorySearch——`run_memory_search` 会返回
`[INVALID TOOL ROUTE] The user is asking for external/web information. Call WebSearch instead.`，
模型读到这个会自我纠错改调 WebSearch。

---

### Step 4 — Phase-2 Generation (sampled)

**职责**：把 Phase-1 的 plan / tool output 转成自然语言 final answer。
采样解码出风味（tsundere 语气、措辞变化），但被 Phase-1 commit_terms 绑定
不能漏掉关键事实。

**关键文件**：
- [eva_core.py](../eva_core.py) — `_phase2_generate()`
- [eva_history.py](../eva_history.py) — `_record_phase2_output()`

**5 种 phase-2 mode**（影响采样参数）：

| mode | temp | top_p | rep_pen | 触发条件 |
|---|---|---|---|---|
| `direct` | 0.82 | 0.9 | 1.08 | 简单回答，无工具/记忆注入 |
| `after_tool` | 0.6 | 0.85 | 1.06 | Phase-1 调过工具，要严格按 tool output 回 |
| `after_memory` | 0.35 | 0.8 | 1.05 | PRE PROBE 注入了记忆，要锚定记忆事实 |
| Collapse-Guard 触发 | 0.75 / 0.92 / 1.15 | — | — | 检测到最近输出重复，加大随机性破环 |

**commit_terms**：[eva_core.py:_extract_phase1_commitment_terms](../eva_core.py)
从 Phase-1 的 thought 里提取必须出现的实体名（"July 7th"、"music box"
等），通过 `[ANSWER MUST INCLUDE]` prompt 注入 Phase-2 输入，让采样的
答案不会漏掉关键事实。

**Phase-2 Collapse Guard**：检测最近 3 个 Phase-2 输出是否近似重复
（编辑距离 + 词频）。若是，提高 temp + rep_pen 强制不同表达。防止
"Hmph! Don't act surprised, Master!" 这种短语在多轮里反复刷屏。

---

### Step 5 — Verifier

**职责**：审查 Phase-2 的 final answer 是否合格。不合格则**注入工具修复**
重新生成。这是 Eva 的"质检员"。

**关键文件**：
- [eva_verifier.py](../eva_verifier.py) — 门面
- [eva_verifier_logic.py](../eva_verifier_logic.py) — 主体逻辑（1869 行）
- [eva_verifier_semantic.py](../eva_verifier_semantic.py) — DeepSeek 语义校验
- [eva_regenerate_guard.py](../eva_regenerate_guard.py) — 重生预算

**两层 verifier**：

#### A. Semantic Verifier (DeepSeek)

判断：
- **代词指代错误**：用户问"你的生日"但答"你的生日是 7/7"（pronoun_referent_mismatch）
- **视角不一致**：第一人称/第二人称混乱
- **persona 漂移**：tsundere 维持失败

3 档 severity（warn / hard）；hard 触发 fail。

#### B. Logic Verifier (regex / 谓词)

检查约 10 类问题，列在 [REASON_POLICY](../eva_verifier_logic.py)：

| reason | 含义 | required_action |
|---|---|---|
| `missing_memorysearch_for_explicit_memory_check` | 用户要求查记忆但没调 MemorySearch | 注入 MemorySearch |
| `missing_web_evidence_for_external_or_current_request` | 用户要查公开事实但没调 WebSearch | 注入 WebSearch |
| `missing_date_calculation_evidence` | 答案有日期算术但没用 GetCurrentTime 数据 | 注入 GetCurrentTime |
| `eva_self_birthday_pronoun_mismatch` | Eva 把自己的生日说成用户的 | 重 Phase-2 + 修 prompt |
| `unsupported_exact_toy_claim` | toy 字段答了但记忆里没证据 | 注入 MemorySearch |
| ... | ... | ... |

**P6 接入点**：当 verifier 决定要注入 MemorySearch 时，会调用
[`build_required_memory_params`](../eva_verifier_logic.py)，里面调
`resolve_pronoun()` 解析 follow-up query 的 antecedent，把 antecedent
作为 keyword 加到 MemorySearch 的 query 里——这是 P6 重构的关键路径。

---

### Step 5b — Repair Loop

verifier fail 时不直接放弃，而是：

1. 把 `required_action` 转成 tool call（`MemorySearch(query="...")`）
2. 用 `synthesize_tool_thought` (DeepSeek) 生成"为什么要调这个工具"的 thought
3. 拼回 Phase-1 输出形态：`<think>thought</think><|tool_code|>ToolName(...)`
4. 走 Step 3 → Step 4 → Step 5 完整循环
5. **预算保护**（[eva_regenerate_guard.py](../eva_regenerate_guard.py)）：
   每个 reason 有重生上限，超限则 fail-open 接受原答案

---

## 三、Sub-systems

### 3.1 Memory System

**架构**：

```
8.memory_optimized.jsonl  ←── Memory_maker/Memory.py 
    │
    ▼ encode (mpnet-base-v2)
Memory/memory.index    ──┐
Memory/memory_content.json├── eva_memory_legacy.run_memory_search()
Memory/memory_meta.json ──┘
                              │
                              ├─ FAISS top_k=20
                              ├─ stemmed BM25 retrieval
                              ├─ CrossEncoder rerank
                              ├─ subject filter (Eva/Rosm/Shared/Both)
                              └─ slot extractor
                                    │
                                    ▼
                         结构化输出:
                           >>> ANSWER VALUE <<<
                           [SLOT EVIDENCE]
                           - birthday: FOUND = July 7th
                           Record 1 [Lore][Subject:X][Topic:Y][Judge:EXACT]: ...
```

**Topic Dict** ([eva_memory_v2.py](../eva_memory_v2.py))：把用户 query 关键词
映射到话题。**word-boundary 正则**保证 `'yo'` 不会误匹 `'do you'`。

```python
re.compile(rf"(?<![a-z0-9]){alias}(?![a-z0-9])", re.I)
```

**Slot extraction**：从 retrieved records 里抽 4 个槽位 (`birthday`, `full_name`,
`age`, `toy`)，结构化输出给模型。如果某 slot 没找到，标 `MISSING` 提示
模型不要编造。

工具：
- [Memory_maker/rewrite_memory.py](../Memory_maker/rewrite_memory.py) — record 源代码
- [Memory_maker/Memory.py](../Memory_maker/Memory.py) — 重建 FAISS 索引
- [Memory_maker/add_memory.py](../Memory_maker/add_memory.py) — LLM 驱动追加新记忆

---

### 3.2 Judge Family

所有"DeepSeek 二元/多元判断"统一在 [eva_intent_judge.py](../eva_intent_judge.py)：

| Judge | 用途 | 调用点 |
|---|---|---|
| `judge_intent("PUBLIC_FACT", q)` | "这是公开事实查询吗？" | Plan B verifier 修复路径 |
| `judge_intent("EXPLICIT_MEMORY", q)` | "用户在显式要求查记忆吗？" | PRE PROBE + verifier |
| `judge_intent("EXPLICIT_WEB", q)` | "用户在显式要求搜网吗？" | 同上 |
| `judge_topic_subset(q, candidates, speaker)` | "这些候选 topic 哪些真相关？" | PRE PROBE 第二层 |
| `synthesize_tool_thought(q, tool, args, recent_turns)` | 生成 Eva 第一人称的"我要调这个工具"思考 | Verifier 修复时改写 Phase-1 trace |

所有 judge 共享 `JudgeState.cache` + `JudgeState.call_count` 全局预算。

---

### 3.3 Pronoun Resolver (P6)

**问题**：用户说 `"check it"` 时，"it" 指什么？regex 之前枚举语言形状
（`really? check it`、`hold on, check it`）总是漏；改 LLM。

**位置**：[eva_pronoun_resolver.py](../eva_pronoun_resolver.py)

**调用点**：`build_required_memory_params(agent, latest_user_text)` 内部
（仅在 verifier 决定要注入 MemorySearch 时）。

**三阶段**：

```
1. Cheap gate: ≤8 词 + 含 it/that/them/this/again/too 等触发词
   不通过 → source="skip"
2. LLM main: DeepSeek 直接回答 needs_resolution + antecedents[1..3] + confidence
   通过 → source="llm"
3. Regex fallback (P6.4 后删除): 只在 LLM 失败时跑，调用 legacy 的
   _is_pronoun_followup + _extract_topical_nouns_from_recent_turns
   → source="regex"
```

参见 [P6_pronoun_resolver_refactor_v4.md](P6_pronoun_resolver_refactor_v4.md)。

---

### 3.4 History Manager

**位置**：[eva_history.py:HistoryManager](../eva_history.py)

**职责**：
- 维护 `ConversationTurn` 列表
- 历史压缩（每轮超 N 之后压缩到 KV summary 释放上下文窗口）
- 图像注册表（`<|image_id|>` 引用机制）
- `recent_turns(n=2)` 给 pronoun resolver / verifier 用

**`ConversationTurn`** 不是简单的 (user, assistant) tuple，而是：

```python
{
  "user_content": "...",
  "assistant_steps": [
    {"role": "assistant", "content": "<think>...</think><|tool_code|>..."},
    {"role": "tool_output", "content": "..."},
    {"role": "assistant", "content": "<think>...</think><|answer|>..."},
  ],
}
```

每个 step 是一次 Phase-1 或 Tool Output 或 Phase-2。多个 step 组成一轮。

---

### 3.5 Regex 全景图 + Prompt 演进

**TL;DR**：regex 在 Eva 里出现在 **5 类触点**，每个 turn 大约触发 ~10 次
regex match。对照下面的"prompt 演进"小节看一遍，能彻底理解每个 regex
**何时被调用、改了什么、模型最终看到什么**。

#### 3.5.A — 五类 regex 触点

| 类别 | 文件 / 函数 | 用途 | 删/留 |
|---|---|---|---|
| **输入清洗** | `eva_render.py: clean_user_text` | 去 HTML / 控制符 / 多余空格 | 永久保留 |
| **Topic 路由** | `eva_memory_v2.py: TopicDictionary._compile_patterns` | word-boundary 正则匹配 58 个话题的 alias | 永久保留 |
| **Slot question hard-guard** | `eva_memory_legacy.py: _is_personal_identity_question` | "what's your X" 类问句强制 probe | 永久保留 |
| **ReAct 协议解析** | `eva_tools_runtime.py: parse_react_block (TAG_RE)` | 拆分 `<think>` `<\|tool_code\|>` `<\|answer\|>` `<\|end_react\|>` | 永久保留 |
| **Verifier 谓词** | `eva_verifier_logic.py: ~30 个 reason 专用正则` | 答案合法性闭集合检查 | 永久保留 |
| **P5 pronoun follow-up** ⚠️ | `eva_verifier_logic.py: _PRONOUN_FOLLOWUP_PATTERNS` | "really? check it" 识别 | **P6.4 删** |
| **slot value extraction** | `eva_slots.py / eva_memory_legacy.py` | 从记忆文本里抽 birthday/full_name/age/toy 值 | 永久保留 |

注意：`_PRONOUN_FOLLOWUP_PATTERNS` 是 P6 重构里**唯一**要删的 regex（开放
语言枚举，不收敛）。其他全部保留——它们做的是闭集合任务（话题路由、协议
解析、谓词验证），是 Eva 性能的"骨架"。

#### 3.5.B — Prompt 演进示例（一条 turn 走完）

用例：用户输入 `"do you have a toy?"`（无前置历史）。

##### 阶段 0 — 用户输入到达

```
原始 input: "do you have a toy?"
```

###### regex 触点 A：clean_user_text

```python
# eva_render.py
"do you have a toy?"
# 这里 regex 移除：HTML 标签 / 控制字符 / 多余空白
# 输入已经很干净 → 不变
→ "do you have a toy?"
```

##### 阶段 1 — PRE PROBE 决策

###### regex 触点 B：TopicDictionary.match（58 个 word-boundary 模式）

```python
# eva_memory_v2.py:113
# 每个 alias 编译成: re.compile(rf"(?<![a-z0-9]){alias}(?![a-z0-9])", re.I)
# 对 "do you have a toy?" 跑 58 个 topic：

"Toy" topic 的 alias "do you have a toy"  → 命中 ✓
"Toy" topic 的 alias "toy"                 → 命中 ✓
"Greetings" topic 的 alias "yo"            → "do **yo**u" 中 "yo" 后面跟 "u"
                                              → word-boundary 救场 → 不匹配 ✓
"Likes" topic 的 alias "like"              → 不匹配 (no "like" in query)
... (其他 55 个 topic 全 miss)

→ matched_topics = ["Toy"]
```

###### regex 触点 C：`_is_personal_identity_question` hard-guard

```python
# eva_memory_legacy.py
# regex: r"\b(your|my)\s+(name|age|birthday|toy|...)\b"
# 命中 "your toy"（在 "do you have a toy" 里 "your" 没出现，但
#  "have a toy" 也是 toy slot 询问）
# 实际实现里有更宽松的 "do you have a X" 模式
→ hard_guard = True  (与 topic match 同时触发，强化 probe 决策)
```

###### regex 触点 D：FAISS 检索后的 slot extraction

```python
# eva_memory_legacy.py: extract_birthday_value / extract_toy_value 等
# 从最高分 record 的 content 里 regex 抽出 slot value:
# 命中 record 87: "Eva's favorite toy has always been a cuddly bunny..."
# regex: r"favorite (toy|childhood toy) (?:is|was|has always been)\s+a?\s*([\w\s-]+?)(?:[,.\n]|$)"
→ toy = "cuddly bunny"
```

##### 阶段 2 — Phase-1 prompt 构造

PRE PROBE 决定 inject=True，把记忆作为 system 块拼到 prompt 里。
**模型实际看到的完整 prompt**（ChatML 格式简化版）：

```
<|im_start|>system
[FORMAT_RULES]
Reply in exactly one form:
<think>brief reasoning</think><|tool_code|>RealToolName(...)<|end_react|>
or
<think>brief reasoning</think><|answer|>your answer<|end_react|>

[IDENTITY_MASTER_INFERENCE]
You are Eva (full name: Eva Louisa) - a tsundere maid speaking to Rosm,
your Creator and Master.
# Top of mind (no recall needed)
- Your name: Eva Louisa
- Your birthday: July 7th
- The ONE fact you refuse: your age
...

### [MEMORY MODULE DATA for 'Eva'] ###          ← PRE PROBE 注入
>>> ANSWER VALUE (use this as the fact) <<<
[SLOT EVIDENCE for Eva]
- toy: FOUND = cuddly bunny
  Source: Eva's favorite toy has always been a cuddly bunny — soft,
  slightly worn at the ears, with one button eye that's been re-sewn.
>>> END ANSWER VALUE <<<

Record 1 [Lore][Subject:Eva][Topic:Toy][Judge:EXACT]: Eva's favorite
toy has always been a cuddly bunny — soft, slightly worn at the ears,
with one button eye that's been re-sewn twice. She's had it since her
earliest days and still keeps it tucked on her shelf...
<|im_end|>
<|im_start|>user
do you have a toy?
<|im_end|>
<|im_start|>assistant
<think>                                          ← FORCE_THINK_PREFIX 注入的 token
```

##### 阶段 3 — Phase-1 generation

模型从 `<think>` 后面开始生成。greedy decode 输出（举例）：

```
<think>Master is asking about my toy. The injected memory says my
favorite is a cuddly bunny. I should answer in tsundere voice.</think><|answer|>Hmph! Of course I have one—a cuddly bunny, since you must
ask. Don't make a fuss about it!<|end_react|>
```

###### regex 触点 E：parse_react_block（拆分 ReAct token）

```python
# eva_tools_runtime.py
TAG_RE = re.compile(
    r"(<think>|</think>|<\|tool_code\|>|<\|tool_output\|>|"
    r"<\|answer\|>|<\|end_react\|>)"
)
# 把上面字符串拆成有序的 (tag, content) 列表：
[
    ("thought", "Master is asking about my toy. The injected memory..."),
    ("answer", "Hmph! Of course I have one—a cuddly bunny..."),
]
# StreamPrinter 用同样的 TAG_RE 决定何时打印 "--- THOUGHT ---" 头
```

##### 阶段 4 — Phase-2 prompt 构造

Phase-2 的 prompt = Phase-1 的 prompt + Phase-1 已生成的
`<think>...</think>` 块 + commit_terms binding + `<|answer|>` 前缀，
让模型从 `<|answer|>` 后面继续生成最终答案：

```
... (前面所有内容，包括 PRE PROBE 注入的记忆) ...
<|im_start|>assistant
<think>Master is asking about my toy. The injected memory says my
favorite is a cuddly bunny. I should answer in tsundere voice.</think>
[ANSWER MUST INCLUDE: cuddly bunny]                ← commit_terms 注入
<|answer|>                                          ← Phase-2 从这里开始
```

模型采样输出：
```
Hmph! Of course I do—it's a cuddly bunny. I've had it forever, and
no, you can't see it. Don't get any ideas~<|end_react|>
```

##### 阶段 5 — Verifier

###### regex 触点 F：`answer_toy_animal_words`（multiple verifier 谓词之一）

```python
# eva_verifier_logic.py
# regex: r"\b(bunny|teddy|plush|cat|dog|...)\b"
# answer 中命中 "bunny"
→ answer 提到了具体玩具 → 触发后续检查
```

###### regex 触点 G：`exact_memory_evidence_for(turn, slot="toy")`

```python
# 扫历史 history 找 [SLOT EVIDENCE for Eva] toy=cuddly bunny
# (PRE PROBE 注入的就是这个) 
# regex: r"toy:\s*FOUND\s*=\s*([\w\s-]+)"
→ True (有 evidence 支持答案中的 "cuddly bunny" 断言)
```

###### regex 触点 H：`current_turn_has_memorysearch_evidence`（验证 reason 适用性）

```python
# 因为本 turn 没显式调过 MemorySearch（PRE PROBE 注入不算 tool 调用）
# 但 answer 没断言"我查了记忆"，所以这条 reason 不适用
→ skip
```

所有谓词通过 → **verifier pass** ✓ → 答案返回给用户。

##### 阶段 6 — 整轮 regex 调用统计

| 阶段 | regex 用途 | 文件 | 大致次数 |
|---|---|---|---|
| 0 | 输入清洗 | eva_render.py | 3-5 |
| 1a | Topic 路由（58 个 alias 模式） | eva_memory_v2.py | 58 次（多数 miss） |
| 1b | Slot question hard-guard | eva_memory_legacy.py | 1 |
| 1c | Slot value extraction | eva_memory_legacy.py / eva_slots.py | 4 (4 个 slot) |
| 3 | ReAct token 拆分 | eva_tools_runtime.py | 1 (扫一次输出) |
| 4 | StreamPrinter 流式拆分 | eva_history.py | N (流式调用) |
| 5 | Verifier 谓词（每条 reason 一次） | eva_verifier_logic.py | 5-15 |

**~80-100 次 regex match** 一轮。每次 < 1 ms，总计 < 100 ms。
对比：每次 LLM 调用 2-5 秒。Regex 在 Eva 性能预算里几乎免费。

#### 3.5.C — Prompt 形状规律

每次 generate 调用，prompt 都是**上一次的 superset**：

```
[初始 system + identity 模板]                  Phase-1 第 1 次
   + [PRE PROBE 注入：记忆数据]
   + [user turn]
   + [<think> 前缀]                             ← FORCE_THINK_PREFIX

   + [Phase-1 生成: <think>...</think>          Phase-1 第 N 次
      <|tool_code|>... 或 <|answer|>...]
   + [tool_output 注回历史]                     ← 如果调了工具
   ↓ 下一个 generate 看到的 prompt 包含上面所有

   + [Phase-2: commit_terms binding]            Phase-2
   + [<|answer|> 前缀]                          ← Phase-2 续写起点
```

修复路径例外：verifier fail 触发 trace rewrite 时，**把上一次失败的
Phase-2 answer 从 history 里改写成 tool call shape**——这是 prompt
唯一会"缩短/改写"的场景，由 `synthesize_tool_thought` (DeepSeek) 完成。

#### 3.5.D — P6 resolver 不在 prompt 路径里

值得专门强调：[eva_pronoun_resolver.py](../eva_pronoun_resolver.py) 不会
直接修改模型 prompt。它只在**verifier 决定要注入 MemorySearch 时被调用**
（`build_required_memory_params` 内部），把 antecedent 作为 keyword 写进
MemorySearch **工具参数**：

```
P6 resolver 输出: antecedents=["tiny plushie", "plushie"]
   ↓
build_required_memory_params 拼出:
   query    = "really? Check it tiny plushie plushie"
   keywords = "tiny plushie, plushie, eva, rosm, shared, toy, ..."
   ↓
MemorySearch 调用 → tool_output 注回 history → 下次 Phase-1 看到正确 evidence
```

所以 P6 影响 prompt 是**间接**的：通过改善工具输入 → 改善工具输出 →
改善后续 prompt。这就是为什么"resolver 错了" 不会立刻让答案错，但会
**让记忆召回崩**，进而让答案逐步漂移。

---

## 四、Worked Examples — 两条真实 turn

两个例子覆盖了 Eva 最常见的两条主路径：
- **4.1**：Phase-1 主动决定调工具（**proactive tool call**）
- **4.2**：Phase-1 直接出 answer，但 verifier fail 触发修复，途中调 P6 resolver（**verifier-driven repair + P6**）

二者机制差别很大；看完两条对照才完整。

---

### 4.1 — Proactive tool call: `"so when is your birthday? How many days should I wait?"`

```
Step 0 — turn 入口
   history.append_turn(user_content=...)
   judge_state.reset()
   ▼
Step 1 — PRE PROBE
   judge_intent("EXPLICIT_WEB", q) → False     [LLM call #1]
   judge_intent("EXPLICIT_MEMORY", q) → False  [LLM call #2]
   hard_guard_slot_identity_question = True   (regex)
   topic_match = ["Birthday"]                  (keyword)
   FAISS retrieve target=Eva → Record [Eva/Birthday]
   slot extract: birthday: FOUND = July 7th
   inject=True, reason=exact_evidence
   ▼
Step 2 — Phase-1 (greedy, with <think> prefix)
   model.generate() 输出:
     <think>Master is asking about my birthday again. I know it's July
     7th. Today is May 7th, so I can calculate days.</think>
     <|tool_code|>MemorySearch(query="Eva birthday", target_entity="Eva", 
                               keywords="birthday, Eva")<|end_react|>
   ▼
Step 3 — Tool: MemorySearch
   FAISS + rerank → top1=8.93 EXACT
   tool_output: "[SLOT EVIDENCE for Eva] birthday: FOUND = July 7th
                 Record 1 [Lore][Subject:Eva][Topic:Birthday][Judge:EXACT]:..."
   ▼ (tool_output 注回 history)
Step 2' — Phase-1 again
   model.generate() 输出:
     <think>My birthday is July 7th. Today is May 7th, so I need to
     calculate days. May has 31, so May 7 to May 31 = 24 days. June=30. 
     July 1-7 = 7. Total: 24+30+7 = 61 days.</think>
     <|tool_code|>TextGenerationTool(instruction="Calculate days from
                  May 7th to July 7th. Show the breakdown")<|end_react|>
   ▼
Step 3' — Tool: TextGenerationTool (DeepSeek expert)
   tool_output: "Total: 24 + 30 + 7 = 61 days. ..."
   ▼
Step 2'' — Phase-1 again
   model.generate() 输出:
     <think>61 days until July 7th. I'll give the exact count with a 
     tsundere twist.</think>
     <|answer|>Hmph! Your birthday is July 7th, Master. And if you're
     counting—there are exactly 61 days until then.<|end_react|>
   ▼
Step 4 — Phase-2 (mode=after_tool, temp=0.6, commit_terms=['July 7th'])
   model.generate() 输出:
     "Hmph! Your birthday is July 7th, Master. And if you're counting—
     there are exactly 61 days until then. Don't waste them on 
     trivialities~"
   ▼
Step 5 — Verifier
   semantic verifier (DeepSeek) → fail!
     issue: pronoun_referent_mismatch
     "The answer says 'Your birthday is July 7th' when the user asked
      about Eva's birthday, incorrectly attributing Eva's own birthday 
      to the user."
   logic verifier:
     reasons = ['missing_date_calculation_evidence',
                'semantic_verifier_fail:pronoun_referent_mismatch']
     hard_fail = True, fix_class = 'inject_tool'
   required_action: GetCurrentTime()
   ▼
Step 5b — Repair (RegenerateGuard 还有预算)
   synthesize_tool_thought(q, "GetCurrentTime", "")
     → "Master needs the current date or time anchor. I should call
        GetCurrentTime first."
   trace rewrite: 把上一次的 Phase-2 答案改写为 tool call shape
   ▼
Step 3 — Tool: GetCurrentTime
   tool_output:
     [TIME BINDING] current_date: 2026-05-07
     [DATE CALCULATION BINDING]
       target_entity: Rosm
       slot: birthday
       target_date: 2026-07-07
       days_until: 61
   ▼
Step 4' — Phase-2 again (mode=after_tool, commit_terms=['July 7th'])
   "My birthday is July 7th. Since today is May 7th, you'll have to wait
    exactly 61 days. Try not to forget it, Master!"
   ▼
Step 5 — Verifier
   semantic: pass ✓
   logic: pass ✓
   ▼
Step 6 — Output
   返回给用户。history 中新 turn 包含 6 个 assistant_steps
   (Phase-1 ×3 + tool_output ×3 + Phase-2 ×1)
```

整个过程一共触发：
- 5+ 次 DeepSeek judge 调用
- 3 次 generation step（Phase-1 ×2 + Phase-2 ×2）
- 3 次工具调用（MemorySearch / TextGen / GetCurrentTime）
- 2 次 verifier 评估

时间预算：~10-30 秒（取决于网络和 GPU）。

---

### 4.2 — Verifier-driven repair + P6 resolver: `"really? Check it"`

**前置**：上一轮 Eva 说过 `"a tiny plushie shaped like me wouldn't be half bad"`。
现在用户跟进：

```
Step 0 — turn 入口
   history.append_turn(user_content="really? Check it")
   judge_state.reset()
   ▼
Step 1 — PRE PROBE
   judge_intent("EXPLICIT_WEB", q) → False   [LLM call #1]
   action=probe reason=explicit_memory_request   ← regex 命中"check"
   FAISS retrieve target=Both → top1=4.63 但无 exact
   inject=False, reason=no_evidence              ← 短句没足够锚点
   ▼
Step 2 — Phase-1 (greedy, with <think> prefix)
   model.generate() 输出（注意：模型不知道 "it" 指什么，编了个回答）：
     <think>Master's being cheeky again—showing off without even giving
     it to me yet. I should scold lightly while showing I'm secretly 
     curious.</think>
     <|answer|>Hmph! What are you showing me, Master? A whole store
     full of toys? Fine, I'll pretend to care.<|end_react|>
   ▼
Step 4 — Phase-2 (mode=direct, temp=0.82)
   "Hmph! What are you showing me, Master? A whole store full of toys?
    …Fine, I'll pretend to care. Just make sure it's not garbage—you'll 
    be letting me decide, right?"
   ▼
Step 5 — Verifier
   semantic verifier (DeepSeek) → pass ✓
   logic verifier:
     检测到 "check it" → 用户在显式要求查记忆 (regex)
     但本轮没调用过 MemorySearch
     → reason: missing_memorysearch_for_explicit_memory_check
     → required_action: MemorySearch(...)
     → hard_fail = True
   ▼
Step 5b — Repair: build_required_memory_params
   现在到 P6 介入的地方:
   resolve_pronoun(q="really? Check it", recent_turns=[...])
     ├─ Stage 1 cheap gate: 3 词 + 含 "it" → 通过
     ├─ Stage 2 LLM main path: DeepSeek 调用
     │    输入 query + 最近 2 turn 历史
     │    返回: {"needs_resolution": true,
     │            "antecedents": ["tiny plushie", "plushie"],
     │            "confidence": 0.92}
     └─ source="llm" ✓                                  [LLM call #2]
   
   trace 行:
     | [DEBUG] P6 pronoun-followup detected: source=llm
     |          antecedents=['tiny plushie', 'plushie'] conf=0.92
   
   build_required_memory_params 用 antecedents 拼出:
     query = "really? Check it tiny plushie plushie"
     keywords = "tiny plushie, plushie, eva, rosm, shared, toy, ..."
     target_entity = "Both"
   ▼
Step 5c — synthesize_tool_thought
   再调一次 DeepSeek 生成 Eva-voice 的解释 thought      [LLM call #3]
   输出: "Master is referring to the tiny plushie I just mentioned —
          I should search my memory for details."
   trace rewrite: 把上面 Step 4 的 answer 改写成 tool call shape
   ▼
Step 3 — Tool: MemorySearch
   FAISS + slot 抽取 → 找到 [Eva/Toy] 的 cuddly bunny 记录
   tool_output: "[SLOT EVIDENCE for Both] toy: MISSING (target=Both 看不到 Eva-only)
                 Record 1 [Lore][Subject:Shared][Topic:Gifts]: ..."
                 注：因 target=Both，slot 找不到 Eva-only 数据；
                 但 Records 里有相关 lore (Eva 珍惜 handmade gifts)
   ▼
Step 4' — Phase-2 again (mode=after_memory, temp=0.35, commit_terms=none)
   "Hmph! Don't go getting silly ideas, Master—I'm not some kid who 
    needs a toy. …Though if you really insist on making one, I 
    suppose I wouldn't complain. Just don't expect me to play with it~"
   ▼
Step 5 — Verifier
   semantic: pass ✓
   logic: pass ✓ (这次有 MemorySearch evidence 了)
   ▼
Step 6 — Output
   答案落 history。
```

整个过程触发：
- **3 次 DeepSeek judge 调用**（intent + P6 resolver + synthesize_thought）
- 2 次 generation（Phase-1 ×1 + Phase-2 ×2）
- 1 次工具调用（修复时注入的 MemorySearch）
- 2 次 verifier（第一次 fail + 第二次 pass）

**关键观察 vs 4.1**：

| 维度 | 4.1 (proactive) | 4.2 (verifier-driven) |
|---|---|---|
| Phase-1 决定 | 立即调工具 | 直接出 answer（错的） |
| 工具触发源 | 模型自己（看到 PRE PROBE 注入的记忆） | Verifier 判 fail 后注入 |
| P6 resolver 触发 | 不触发 | 触发（在 `build_required_memory_params` 里） |
| 第一次 Phase-2 答案 | 大概率正确 | 大概率错（但被 verifier 抓住） |
| 总 turn 时长 | 长（多次 Phase-1） | 中（只一次 Phase-1，但 1 次 LLM judge + 1 次 resolver + 1 次 thought 改写） |

→ 这就是为什么 P6 resolver 的 76% rescue rate 重要——它就是这条路径上
解析 antecedent 的关键步骤；resolver 错了，verifier 修复路径会带着错误的
keywords 去查 memory，召回率崩。

---

## 五、模块映射

| 文件 | 行数 | 职责 |
|---|---|---|
| [eva_core.py](../eva_core.py) | 2430 | 核心 ChatAgent；Phase-1/2 生成；总编排 |
| [eva_inference_P2.py](../eva_inference_P2.py) | 412 | P2 推理入口；MemoryModule 单调用 |
| [eva_chat_colab.py](../eva_chat_colab.py) | 265 | ChatSession Colab 入口 |
| [eva_history.py](../eva_history.py) | 423 | HistoryManager / ConversationTurn / StreamPrinter |
| [eva_render.py](../eva_render.py) | 189 | ChatML 渲染 / clean_user_text |
| [eva_prompts.py](../eva_prompts.py) | 202 | 系统提示库 (FORMAT_RULES / IDENTITY_*) |
| [eva_config.py](../eva_config.py) | 600+ | 全局配置中心 |
| [eva_model_loader.py](../eva_model_loader.py) | 68 | 模型权重加载 |
| **Pre-probe / Memory** | | |
| [eva_memory_v2.py](../eva_memory_v2.py) | 1173 | MemoryModule.decide/probe；TopicDictionary；LayeredIntentClassifier |
| [eva_memory_legacy.py](../eva_memory_legacy.py) | 1495 | FAISS / BM25 / CrossEncoder / slot extractor |
| **Judges** | | |
| [eva_intent_judge.py](../eva_intent_judge.py) | 795 | DeepSeek 判断器统一入口 |
| [eva_route_judge.py](../eva_route_judge.py) | 241 | 本地 LM 路由分类（MEMORY/WEB/TIME/DIRECT） |
| **Pronoun Resolver (P6)** | | |
| [eva_pronoun_resolver.py](../eva_pronoun_resolver.py) | 395 | 三阶段解析：cheap gate / LLM / regex fallback |
| **Verifier** | | |
| [eva_verifier.py](../eva_verifier.py) | 38 | 门面 |
| [eva_verifier_logic.py](../eva_verifier_logic.py) | 1869 | REASON_POLICY / 修复 dispatch / build_required_memory_params |
| [eva_verifier_semantic.py](../eva_verifier_semantic.py) | 417 | DeepSeek 语义校验 |
| [eva_regenerate_guard.py](../eva_regenerate_guard.py) | 131 | 重生预算保护 |
| **Tools** | | |
| [eva_tools.py](../eva_tools.py) | 43 | 工具门面 |
| [eva_tools_runtime.py](../eva_tools_runtime.py) | 288 | Tavily / DeepSeek expert / vision / parser |
| [eva_slots.py](../eva_slots.py) | 264 | 插槽定义 |

---

## 六、配置 Touchpoints

只想调一个 dial 改行为，找这些 flag（[eva_config.py](../eva_config.py)）：

### 性能 / 成本

| Flag | 默认 | 影响 |
|---|---|---|
| `LLM_JUDGE_MAX_CALLS_PER_TURN` | 6 | 每轮 DeepSeek 调用上限 |
| `PRONOUN_RESOLVER_MAX_CALLS_PER_TURN` | 2 | resolver 独立预算 |
| `LLM_JUDGE_TIMEOUT_SECONDS` | 8 | 每次 judge 超时（短超时保延迟） |
| `MAX_NEW_TOKENS_TURN` | 500 | 每次 generate 最大 token |

### 行为开关

| Flag | 默认 | 影响 |
|---|---|---|
| `FORCE_THINK_PREFIX` | `True` | 硬塞 `<think>` 防 greedy 跳过 |
| `PRONOUN_RESOLVER_MODE` | `"llm_first"` | P6.3 cutover 后；可改 `"regex_only"` 回滚 |
| `ENABLE_LLM_VERIFIER_JUDGE` | `True` | Plan B 三元判断器总开关 |
| `ENABLE_LLM_PRE_PROBE_JUDGE` | `True` | judge_topic_subset 总开关 |

### 调试 / 观察

| Flag | 默认 | 影响 |
|---|---|---|
| `VERIFIER_DEBUG` | `True` | verifier trace 是否打印 |
| `LLM_JUDGE_DEBUG` | `True` | DeepSeek 调用 payload + 结果 |
| `PRONOUN_RESOLVER_DEBUG` | `False` | resolver LLM raw response |
| `SEMANTIC_VERIFIER_DEBUG` | `True` | 语义 verifier 完整 issue 列表 |

---

## 七、读懂一条 trace 的 cheat sheet

```
| --- PROCESSING START ---             ← Step 0 完成
| --- CALLING JUDGE ---                ← LLM judge 调用 (DeepSeek)
| [System] DeepSeek judge: payload=... ← 发出去的内容
| [JUDGE] intent=X q=... -> True/False ← 返回结果
| --- P2 PRE MEMORY PROBE ---           ← Step 1 决策块
| action=skip|probe reason=...         ← 跳过 / 探针的具体原因
|     | --- THOUGHT ---                ← Phase-1 输出的 <think> 块
|         | --- TOOL CODE ---           ← Phase-1 决定调用工具
|         | --- TOOL OUTPUT ---         ← Step 3 工具返回
|     | --- THOUGHT ---                ← Phase-1 再次思考（看到 tool output）
|           | --- ANSWER ---            ← Phase-1 决定回答
|         | --- PHASE 2 SAMPLING ---    ← Step 4 采样
|         | mode=... temp=... commit_terms=... ← 采样参数 + 绑定项
|         | --- ANSWER VERIFIER FAILED ---     ← Step 5 verifier fail
|         | reasons=...                         ← 失败原因
|         | --- ANSWER VERIFIER REQUIRED ACTION ---  ← Step 5b 修复指令
|         | --- STEP-5 TRACE REWRITE ---        ← 改写 Phase-1 输出
|         | --- CONTROLLER TOOL EXECUTION (...) --- ← 修复路径调工具
| --- PHASE 2 COLLAPSE GUARD ENGAGED ---       ← 检测到重复，加大随机性
| reason=recent_phase2_outputs_near_duplicate
```

trace 里看到 `[PRONOUN-SHADOW]` 行 = P6.2 shadow 模式开了；`source=llm/regex/skip` = resolver 走的路径。

trace 里看到 `[DEBUG] P6 pronoun-followup detected: q=...` = `build_required_memory_params` 调了 resolver。

---

## 八、相关文档

- [SESSION_ARCHIVE_2026-05-08.md](SESSION_ARCHIVE_2026-05-08.md) — 本会话所有改动归档
- [P6_pronoun_resolver_refactor_v4.md](P6_pronoun_resolver_refactor_v4.md) — P6 重构正式方案
- [P6_4_deletion_patch.md](P6_4_deletion_patch.md) — P6.4 待执行 patch
- [P6_2_shadow_runbook.md](P6_2_shadow_runbook.md) — Shadow 测试 Colab 操作手册

---

## 九、新工程师 Onboarding 路径建议

1. 读本文档（一遍）
2. 跑一次 `python tests/test_p6_pronoun_resolver.py` — 验证环境通
3. 在 Colab 上跑 build_agent + 一两轮对话 — 看真实 trace
4. 拿一条 trace 对照本文档 § 七的 cheat sheet 逐行解读
5. 试 `python Memory_maker/add_memory.py "你的记忆"` — 体会 LLM 工具流
6. 改一个 Flag（比如把 `FORCE_THINK_PREFIX = False` 改回去）跑同一句话对比

第 5 步走完了就基本理解了主要数据流。第 6 步走完了就能开始改代码。
