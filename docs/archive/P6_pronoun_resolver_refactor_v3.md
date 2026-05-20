# P6 — Pronoun Resolver 重构方案 v3

> 基于 v2 评审追问修订。核心方向（LLM 主路径）不变；
> 主要变更是 **P6.4 改为完全删除 regex 路径**——v2 的"瘦身保留作 safety net"
> 是自欺式兜底（不被触发 → 不被测试 → 真用时大概率已坏），不如承认 LLM 是
> 唯一主路径，不可用时退化到 pre-P5 行为。
> 带 **[v3]** 标记的章节为相对 v2 的实质性变更。

---

## 一、为什么重构（不变）

`_PRONOUN_FOLLOWUP_PATTERNS` 在 P5 → P5.1 共改了两次：

| 版本 | 失败案例 |
|---|---|
| P5 初版 | `"really? Check it"` 不匹配（前缀 `really?`） |
| P5.1 补丁 | 加 `(?:really\??\|wait,\|hmm,\|huh,)?` 前缀 |
| 下一次 | 必然又会出现 `"hold on, check it"` / `"sorry, check it"` / `"um... check it"` |

根因：**用 regex 枚举语言形状永远不收敛**。每次新对话风格都要改正则、发版、回归测试。
更致命的是 regex 不解析 antecedent——只回答"是不是跟进句"，真正的指代物
（"music box" / "special collection"）要靠 `_extract_topical_nouns_from_recent_turns`
继续猜，两层 heuristic 串联累积错误。

LLM 路径基础设施已经齐全（`JudgeState` + `synthesize_tool_thought` 已经接 `recent_turns`），
只欠把"是否跟进 + antecedent"合并问 LLM 一次。

---

## 二、最终架构  **[v3 — Stage 3 仅在迁移期存在]**

```
                ┌─────────────────────────────────────────────┐
  user_text ──► │  eva_pronoun_resolver.py                     │
  recent_turns  │                                              │
                │  ┌────────────────────────────────────────┐  │
                │  │ Stage 1: cheap gates                    │  │
                │  │   - empty / >8 词 / 无指代触发词        │  │
                │  │   - 直接 return source="skip"           │  │
                │  └────────────────────────────────────────┘  │
                │                                              │
                │  ┌────────────────────────────────────────┐  │
                │  │ Stage 2: LLM main path                  │  │
                │  │   - PROMPT_PRONOUN_RESOLVER             │  │
                │  │   - DeepSeek judge (独立 budget)        │  │
                │  │   - 返回 needs / antecedents[1..3] / conf │
                │  └────────────────────────────────────────┘  │
                │                                              │
                │  ┌────────────────────────────────────────┐  │
                │  │ Stage 3: regex fallback  [迁移期 only]  │  │
                │  │   - P6.0–P6.3 期间存在                  │  │
                │  │   - P6.4 起完全删除                     │  │
                │  │   - LLM 失败时 → needs=False, source=skip│ │
                │  └────────────────────────────────────────┘  │
                │                                              │
                └─────────────┬────────────────────────────────┘
                              │
                              ▼
                  PronounResolution(
                      needs_resolution=True,
                      antecedents=["music box"],
                      confidence=0.92,
                      source="llm",        # llm | skip | (regex 仅迁移期)
                      reasoning="..."      # 仅 PRONOUN_RESOLVER_DEBUG=True
                  )
```

**P6.4 后的 LLM 失败行为**：直接返回
`PronounResolution(needs_resolution=False, antecedents=[], source="skip")`，
那一轮 verifier 用原始 query 走默认内存搜索——**等价于 pre-P5 行为**。
P5 是优化项，不是必需项，退化是可接受的。

调用方变化：

```python
# eva_verifier_logic.build_required_memory_params
resolution = resolve_pronoun(latest_user_text, recent, state=agent._llm_judge_state)
if resolution.needs_resolution and resolution.antecedents:
    head = " ".join(resolution.antecedents[:2])
    q_for_target = f"{q} {head}".strip()
    keywords_extra = list(resolution.antecedents)
```

---

## 三、文件清单  **[v3 — P6.4 删除项扩大]**

| 路径 | 状态 | 改动 |
|---|---|---|
| `eva_pronoun_resolver.py` | **新建** | 单文件 ~250 行 |
| `eva_config.py` | 编辑 | 新增 5 个 flag（见 § 四） |
| `eva_verifier_logic.py` | 编辑 | `build_required_memory_params` 改为调用 `resolve_pronoun`；P6.4 时**删除** `_PRONOUN_FOLLOWUP_PATTERNS` / `_FOLLOWUP_NOUN_STOPWORDS` / `_is_pronoun_followup` / `_extract_topical_nouns_from_recent_turns` 四个 symbol 及其调用点 |
| `eva_intent_judge.py` | 编辑 | `synthesize_tool_thought` 接收 `resolved_antecedent` 参数 |
| `eva_inference_P2.py` | 编辑 | `_synthesize_repair_thought` 把 resolution 结果向下传 |

---

## 四、配置 flag  **[v3 — 移除 regex_only 模式]**

```python
# eva_config.py 追加
ENABLE_PRONOUN_RESOLVER = True
PRONOUN_RESOLVER_MODE = "llm_first"          # "llm_first" | "off"  [v3]
PRONOUN_RESOLVER_MIN_CONFIDENCE = 0.60
PRONOUN_RESOLVER_DEBUG = False
PRONOUN_RESOLVER_MAX_WORDS = 8
PRONOUN_RESOLVER_MAX_CALLS_PER_TURN = 2     # 独立预算池，不与全局 judge 共享
```

降级矩阵（v3）：

| flag 值 | 行为 |
|---|---|
| `MODE="llm_first"` | LLM → 失败时 source="skip" → needs=False |
| `MODE="off"` | 全部跳过（实验或紧急回滚） |
| `ENABLE=False` | 等价 `MODE="off"` |

**[v3] 移除 `MODE="regex_only"`**：v2 保留它有两个目的——(a) P6.1 起步用以保持现行为；
(b) 成本敏感部署的离线档位。(a) 在 P6.1–P6.2 期间确实需要，但属于**迁移期临时档位**，
P6.4 删除 regex 代码时一并删除该 mode。(b) 当前没有产品需求，不应为推测性需求养代码。

实现要点：`PronounResolver` 在传入的 `JudgeState` 上**用独立 counter 字段**
（如 `state.pronoun_call_count`）记账，不递增 `state.call_count`。
`JudgeState` 上新增对应字段，`reset_state()` 同时清零。

---

## 五、PROMPT 设计（不变）

```text
You are a pronoun-resolution assistant for an AI maid named Eva.
Given the user's most recent message and the conversation history,
decide whether the user is referring back to something Eva or the
user mentioned earlier, and if so, what the referents are.

Output STRICT JSON with this exact shape:
{
  "needs_resolution": true|false,
  "antecedents": ["<noun phrase>", ...] | [],   // 1..3 phrases, ranked
  "confidence": 0.0-1.0
}

Decision rules:
  - needs_resolution=true ONLY when the user message cannot be
    understood without conversation history (e.g. contains "it",
    "that", "them", "this", "those", or is a short follow-up like
    "really?", "do it again", "check it").
  - antecedents are BARE noun phrases WITHOUT articles
    ("music box" not "the music box", "photo" not "the photo",
     "Master's birthday" keeps the possessive).
  - Return 1 phrase when the referent is unambiguous; return up to
    3 ranked candidates (most likely first) when multiple referents
    are plausible. This widens recall for the downstream FAISS/BM25
    lookup.
  - If history is empty or no clear referent exists, return
    needs_resolution=false, antecedents=[], confidence=1.0.
  - confidence reflects YOUR certainty in BOTH the detection AND
    the antecedent extraction. Below 0.6 means "I'm guessing."

Examples:
  Input:
    user: "can you check it?"
    history: [..., assistant: "I could show you my special collection."]
  Output:
    {"needs_resolution": true, "antecedents": ["special collection"],
     "confidence": 0.95}

  Input:
    user: "sorry, check that"
    history: [..., assistant: "Here is the photo Master sent yesterday."]
  Output:
    {"needs_resolution": true, "antecedents": ["photo"],
     "confidence": 0.90}

  Input:
    user: "what's the weather today?"
    history: [...]
  Output:
    {"needs_resolution": false, "antecedents": [], "confidence": 1.0}

  Input:
    user: "really? what date is today?"
    history: [..., assistant: "Your birthday is in 202 days."]
  Output:
    {"needs_resolution": false, "antecedents": [], "confidence": 0.85}
```

---

## 六、迁移步骤  **[v3 — P6.4 改为完全删除]**

| Phase | 改动 | 风险 | 回滚 |
|---|---|---|---|
| **P6.0** | 新建 `eva_pronoun_resolver.py` + offline test | 0 | 删文件 |
| **P6.1** | 接到 `build_required_memory_params`，**MODE="regex_only"** 起步（迁移期临时档位） | 低 | flag 改 `"off"` |
| **P6.2** | flag 切 **`"llm_first"`**，shadow 模式：LLM 跑但不替换 regex 决策，只 log 差异 | 低 | flag 切回 |
| **P6.3** | shadow 指标达标（见 § 七）后切 LLM 主路径，regex 降级为 fallback | 中 | flag 切回 P6.2 |
| **P6.4** **[v3]** | **完全删除** `_PRONOUN_FOLLOWUP_PATTERNS` / `_FOLLOWUP_NOUN_STOPWORDS` / `_is_pronoun_followup` / `_extract_topical_nouns_from_recent_turns` 四个 symbol；移除 `PRONOUN_RESOLVER_MODE` 中的 `"regex_only"` 选项；resolver Stage 3 改为 `return needs_resolution=False, source="skip"` | 中 | git revert |

**v3 关键变更**：v2 计划在 P6.4 把 regex "瘦身保留作 safety net"。复盘后这条不成立——
regex 在 99% 时间不被触发，长期不会有人维护，**真到 LLM 不可用那天它大概率已经因为某次
重构悄悄失效了**。一个不被测试的兜底比没兜底更危险，因为它给人虚假安全感。
直接删除、LLM 失败时退化到 pre-P5 行为，是更诚实的选择。

**前置条件**：P6.4 启动前必须确认 P6.3 在生产稳定运行 ≥ 30 天，且 LLM 调用成功率
持续 ≥ 98%。任一不满足，停在 P6.3。

---

## 七、Shadow 验收指标（不变）

P6.2 shadow 期间统计：

| 指标 | 来源 | 通过阈值 |
|---|---|---|
| **§ 八验收用例覆盖率** | 8 个 fixture 全跑 | 必须 100% 通过（硬门） |
| **vs regex 一致率** | 生产 trace 中 `_is_pronoun_followup=True` 的样本 | LLM 判断 needs=True 的比例 ≥ 95% |
| **antecedent 重叠率** | 上同样本，比对 LLM antecedents[0] 与 regex `_extract_topical_nouns_from_recent_turns()[0:2]` | Jaccard 相似度 ≥ 0.5（lemma 后比较） |
| **LLM 调用成功率** | shadow 期间所有 resolver 调用 | ≥ 98%（剩 2% 走 regex fallback 可接受） |
| **平均延迟** | DeepSeek 调用耗时 | P95 ≤ 800ms（超过则需要异步化） |

**P6.3 升级条件**：上述 5 项全部达标，且 shadow 至少跑满 48 小时 / 200 个样本（取大者）。
任一不达标就停在 P6.2 调 prompt 或代码，不强行升级。

新增 trace 行：
```
| [PRONOUN] q='really? Check it' source=llm
|           needs=True antecedents=['music box'] conf=0.92
```

shadow 模式额外打印对照行：
```
| [PRONOUN-SHADOW] regex_needs=True llm_needs=True
|                  regex_terms=['music', 'box', 'master'] llm_ants=['music box']
|                  agree=True overlap=0.50
```

---

## 八、验收用例（不变）

| 输入 | history 末尾 | 期望 needs | 期望 antecedents[0] |
|---|---|---|---|
| `"can you check it?"` | "...special collection..." | True | "special collection" |
| `"really? Check it"` | "...music box..." | True | "music box" |
| `"hold on, check it"` | "...music box..." | True | "music box" |
| `"sorry, check that"` | "...the photo..." | True | "photo" |
| `"do it again"` | "...joke..." | True | "joke" |
| `"really? what date is today?"` | (任意) | False | — |
| `"tell me about your hobbies"` | (任意) | False | — |
| `"what's the weather"` | "...music box..." | False | — |

---

## 九、与 P5 的兼容性  **[v3 — 终态明确]**

P6.0–P6.3 期间：P5 的 `_extract_topical_nouns_from_recent_turns` 在 LLM 模式下不再被调用，
但保留为 regex_fallback 路径。P5 现有 trace 行 `[DEBUG] P5 pronoun-followup probe` 保留，便于审计。

**P6.4 之后**：P5 的 4 个 symbol 全部删除；LLM 不可用时 resolver 直接跳过。
P5 trace 行同步删除。`build_required_memory_params` 中"pronoun-followup antecedent
resolution"段落被新的 `resolve_pronoun()` 调用取代——文件净行数显著下降
（regex + stopword 表 + helper 共约 160 行被删，新 resolver 调用约 5 行）。

---

## 十、版本演进摘要

| # | v1 | v2 | v3 |
|---|---|---|---|
| 1 预算池 | 共享全局 | 独立 counter | 独立 counter |
| 2 cheap gate 词数 | ≤12 | ≤8 | ≤8 |
| 3 antecedent | 单 string | list 1..3 | list 1..3 |
| 4 noun phrase 格式 | 不约束 | 裸 noun phrase | 裸 noun phrase |
| 5 reasoning 字段 | schema 内 | 移除 | 移除 |
| 6 shadow 验收 | 定性 | 5 项量化 + 时长门槛 | 5 项量化 + 时长门槛 |
| 7 P6.4 终态 | 硬删 regex | 瘦身保留作 safety net | **完全删除 regex 全部 4 个 symbol** |
| 8 `MODE="regex_only"` | 永久 | 永久 | **迁移期专用，P6.4 删除** |

### v2 → v3 核心论证

v2 留 regex 作 safety net 的隐含假设：**"regex 至少能给个答案，比啥都没有强"**。
但这个假设在以下几点上不成立：

1. **不被触发的代码不被测试**——regex 99% 时间不跑，半年后是否还能正确执行没人知道
2. **依赖链早就绑死 LLM**——`judge_intent`、`synthesize_tool_thought`、`judge_topic_subset`
   都依赖 DeepSeek。LLM 挂的时候整个验证器修复回路已经降级，pronoun resolver 不特殊
3. **退化目标本来就可接受**——退化到 pre-P5 不是灾难，P5 是优化不是必需
4. **虚假兜底比没兜底更糟**——给人安全感，但真用时大概率已坏，掩盖问题

诚实的做法：承认 LLM 是唯一主路径，不可用时退化到无 pronoun resolution，
让运维监控盯着 LLM 可用性而不是养一坨从来不跑的 fallback 代码。
