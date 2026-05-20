# P6 — Pronoun Resolver 重构方案 v2

> 基于 v1 评审反馈修订。核心方向（LLM 主路径 + regex 降级）不变；
> 修订集中在**预算隔离、cheap gate 阈值、antecedent 召回面、shadow 验收指标、
> 最终回退策略**五处。带 **[v2]** 标记的章节为相对 v1 的实质性变更。

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

## 二、最终架构（v2 微调）

```
                ┌─────────────────────────────────────────────┐
  user_text ──► │  eva_pronoun_resolver.py                     │
  recent_turns  │                                              │
                │  ┌────────────────────────────────────────┐  │
                │  │ Stage 1: cheap gates  [v2]              │  │
                │  │   - empty / >8 词 / 无指代触发词        │  │
                │  │   - 直接 return source="skip"           │  │
                │  └────────────────────────────────────────┘  │
                │                                              │
                │  ┌────────────────────────────────────────┐  │
                │  │ Stage 2: LLM main path                  │  │
                │  │   - PROMPT_PRONOUN_RESOLVER             │  │
                │  │   - DeepSeek judge (独立 budget)  [v2]  │  │
                │  │   - 返回 needs / antecedents[1..3] / conf │
                │  └────────────────────────────────────────┘  │
                │                                              │
                │  ┌────────────────────────────────────────┐  │
                │  │ Stage 3: regex fallback                 │  │
                │  │   - LLM 不可用 / budget 用尽 / parse 错  │  │
                │  │   - 复用现有 _PRONOUN_FOLLOWUP_PATTERNS  │  │
                │  │   - antecedent 走旧 _extract_topical_*   │  │
                │  └────────────────────────────────────────┘  │
                │                                              │
                └─────────────┬────────────────────────────────┘
                              │
                              ▼
                  PronounResolution(
                      needs_resolution=True,
                      antecedents=["music box"],   # [v2] list, 1..3
                      confidence=0.92,
                      source="llm",
                      reasoning="..."  # 仅 debug 模式填充 [v2]
                  )
```

调用方变化：

```python
# eva_verifier_logic.build_required_memory_params
resolution = resolve_pronoun(latest_user_text, recent, state=agent._llm_judge_state)
if resolution.needs_resolution and resolution.antecedents:
    head = " ".join(resolution.antecedents[:2])
    q_for_target = f"{q} {head}".strip()
    keywords_extra = list(resolution.antecedents)   # [v2] 多锚点保留
```

---

## 三、文件清单（不变）

| 路径 | 状态 | 改动 |
|---|---|---|
| `eva_pronoun_resolver.py` | **新建** | 单文件 ~250 行 |
| `eva_config.py` | 编辑 | 新增 6 个 flag（见 § 四，v2 增加 2 个） |
| `eva_verifier_logic.py` | 编辑 | `build_required_memory_params` 改为调用 `resolve_pronoun`；旧 helper 降级为 fallback |
| `eva_intent_judge.py` | 编辑 | `synthesize_tool_thought` 接收 `resolved_antecedent` 参数 |
| `eva_inference_P2.py` | 编辑 | `_synthesize_repair_thought` 把 resolution 结果向下传 |

---

## 四、配置 flag  **[v2 — 预算隔离]**

```python
# eva_config.py 追加
ENABLE_PRONOUN_RESOLVER = True
PRONOUN_RESOLVER_MODE = "llm_first"          # "llm_first" | "regex_only" | "off"
PRONOUN_RESOLVER_MIN_CONFIDENCE = 0.60
PRONOUN_RESOLVER_DEBUG = False               # [v2] 控制 reasoning 字段是否填充
PRONOUN_RESOLVER_MAX_WORDS = 8               # [v2] cheap gate 词数上限

# [v2] 关键变更：独立预算池，不与 judge_intent / synthesize_tool_thought
# 共享 LLM_JUDGE_MAX_CALLS_PER_TURN。原方案让 resolver 共享全局池，
# 一旦 PRE PROBE 吃光预算 resolver 就降级到 regex，P6.4 删除 regex 的
# 前提就破了。独立池让 resolver 的可用性可独立保证。
PRONOUN_RESOLVER_MAX_CALLS_PER_TURN = 2
```

实现要点：`PronounResolver` 在传入的 `JudgeState` 上**用独立 counter 字段**
（如 `state.pronoun_call_count`）记账，不递增 `state.call_count`。
`JudgeState` 上新增对应字段，`reset_state()` 同时清零。

降级矩阵不变。

---

## 五、PROMPT 设计  **[v2 — antecedents 列表 + 格式约束]**

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

变化要点：
- **antecedents 是数组**（1..3 项），保留 regex 路径"多锚点"的召回优势
- **格式统一为裸 noun phrase**（无冠词），与 `_FOLLOWUP_NOUN_STOPWORDS` 过滤一致，避免 `"the photo"` 被下游剥成空串
- **`reasoning` 字段从 schema 中删除**，节省 token、减少 parse 失败概率；
  debug 信息走 `PRONOUN_RESOLVER_DEBUG=True` 时打印 LLM raw response

---

## 六、迁移步骤  **[v2 — P6.4 重定义]**

| Phase | 改动 | 风险 | 回滚 |
|---|---|---|---|
| **P6.0** | 新建 `eva_pronoun_resolver.py` + offline test | 0 | 删文件 |
| **P6.1** | 接到 `build_required_memory_params`，**MODE="regex_only"** 起步 | 低 | flag 改 `"off"` |
| **P6.2** | flag 切 **`"llm_first"`**，shadow 模式：LLM 跑但不替换 regex 决策，只 log 差异 | 低 | flag 切回 |
| **P6.3** | shadow 指标达标（见 § 七）后切 LLM 主路径，regex 降级 | 中 | flag 切回 P6.2 |
| **P6.4** **[v2]** | 把 `_PRONOUN_FOLLOWUP_PATTERNS` 缩减为**最小稳定子集**（保留 P5.1 三条主 pattern），删除 `_FOLLOWUP_NOUN_STOPWORDS` 中的 Eva 风格填充词等明显冗余项；regex 永久作为 LLM 不可用时的 safety net 保留，**不硬删** | 低 | git revert |

**v2 关键变更**：v1 计划在 P6.4 硬删 regex。LLM 限流 / API 超时 / 服务降级是真实场景，
删掉 fallback 等于丢兜底。改为"瘦身保留"——既偿还了"regex 不收敛"的技术债（不再追加新
pattern，只裁剪现有），又保住了可用性底线。

---

## 七、Shadow 验收指标  **[v2 — 新增量化标准]**

P6.2 shadow 期间统计：

| 指标 | 来源 | 通过阈值 |
|---|---|---|
| **§ 八验收用例覆盖率** | 8 个 fixture 全跑 | 必须 100% 通过（硬门） |
| **vs regex 一致率** | 生产 trace 中 `_is_pronoun_followup=True` 的样本 | LLM 判断 needs=True 的比例 ≥ 95% |
| **antecedent 重叠率** | 上同样本，比对 LLM antecedents[0] 与 regex `_extract_topical_nouns_from_recent_turns()[0:2]` | Jaccard 相似度 ≥ 0.5（lemma 后比较，避免单复数 / 大小写干扰）|
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

注意 `"sorry, check that"` 的期望从 v1 的 `"the photo"` 调整为 `"photo"`，与 § 五的"裸 noun phrase"约束一致。

---

## 九、与 P5 的兼容性（不变）

P5 的 `_extract_topical_nouns_from_recent_turns` 在 LLM 模式下不再被调用，
但永久保留为 regex_fallback 路径（v2 不再设删除 milestone）。
P5 现有 trace 行 `[DEBUG] P5 pronoun-followup probe` 在 shadow 阶段（P6.2）继续保留，
便于审计；P6.3 之后只在 regex fallback 触发时打印。

---

## 十、v1 → v2 变更摘要

| # | v1 | v2 | 理由 |
|---|---|---|---|
| 1 | 共享 `LLM_JUDGE_MAX_CALLS_PER_TURN` | 独立 `PRONOUN_RESOLVER_MAX_CALLS_PER_TURN` + `state.pronoun_call_count` | PRE PROBE 吃满全局预算时 resolver 不能被牵连降级 |
| 2 | cheap gate ≤12 词 | ≤8 词（`PRONOUN_RESOLVER_MAX_WORDS`） | 保持与现有 `_is_pronoun_followup` ≤6 词同量级，避免长句进 LLM 浪费成本 |
| 3 | `antecedent: str \| null` | `antecedents: list[str]`（1..3 项） | 保留 regex 路径多锚点的召回优势，避免 LLM 切换后 FAISS/BM25 召回面收窄 |
| 4 | antecedent 格式不约束 | 强制裸 noun phrase（无冠词） | `"the photo"` 在下游 stopword 过滤后变空，统一格式避免后端对接出错 |
| 5 | PROMPT 含 `reasoning` 字段 | 移除；debug 时打印 raw response | 减少 token & parse 失败面，trace 行已能反推 |
| 6 | shadow "看 1-2 天日志确认 LLM 更准" | 5 项量化指标 + 48h/200 样本 | 没有量化标准的升级是凭感觉 |
| 7 | P6.4 硬删 `_PRONOUN_FOLLOWUP_PATTERNS` 与 `_extract_topical_nouns_*` | 缩减为最小稳定子集，永久保留 fallback | LLM 不可用是真实场景，留底线 |
