# Eva Regex 合理性审计

> 写于 2026-05-08。对全 codebase 中所有 regex 使用做一次系统性审计，
> 判断哪些合理保留、哪些应该考虑 LLM 替代。结论可作为后续讨论"要不要再删
> 一批 regex"时的基线。
>
> 阅读时长：~10 min。

---

## 〇、判定原则

P6 重构（删 `_PRONOUN_FOLLOWUP_PATTERNS`）的核心教训：

| 任务性质 | 工具 | 理由 |
|---|---|---|
| **闭集合**：协议 token、自产格式、固定字段、关键词字典 | regex | 收敛、几乎免费（< 1ms/次） |
| **开放枚举**：把用户语言所有可能形态写成正则 | LLM | 永远不收敛——每加一种说法就要补正则 |

LLM 替代只在**第二类**有正当理由。其他场景换 LLM 是赔本：
- 多 2-5 秒延迟
- 多花 API 钱
- 多了 LLM 不可用风险

P6 之所以能换，是因为 n=50 实测 `llm_rescue_rate=76%`——regex 漏检率高到无法接受。
**没有等价证据，不动。**

---

## 一、按类别审计

### A. 协议 / Token 解析（5 处）— 必须保留

| 位置 | 模式 | 用途 |
|---|---|---|
| [eva_render.py:67](../eva_render.py) | `<\|vision_start\|>.*?<\|vision_end\|>` | 剥 vision protocol token |
| [eva_render.py:74-76](../eva_render.py) | `<\|image\|>` / `<\|image_pad\|>` 字符串 replace | 剥 image protocol |
| [eva_tools_runtime.py:62](../eva_tools_runtime.py) | `^\s*([A-Za-z_]\w*)\s*\((.*)\)\s*$` | 解析 ReAct tool call 语法 `Func(...)` |
| [eva_tools_runtime.py:77](../eva_tools_runtime.py) | `,\s*(?=\w+\s*=)` | 拆分 kwargs 参数 |
| [eva_tools_runtime.py:99](../eva_tools_runtime.py) | `TAG_RE.finditer` | 拆 ReAct tag 块 |

**操作对象**：Eva **自己**产出的 ChatML / ReAct 字符串。这是有限状态文法。
换 LLM 等于让 LLM 解析自己的 token——荒谬。

**判定**：✅ 保留。无替代必要。

---

### B. 闭集合值抽取（~20 处）— 保留

主要在 [eva_slots.py](../eva_slots.py)、[eva_memory_legacy.py](../eva_memory_legacy.py)。

代表性模式：

| 位置 | 模式 | 用途 |
|---|---|---|
| [eva_slots.py:87](../eva_slots.py) | `([A-Z][a-z]+\.?\s+\d{1,2}(?:st\|nd\|rd\|th)?)` | 抽 birthday "July 7th" |
| [eva_slots.py:109-123](../eva_slots.py) | `([A-Z][A-Za-z][A-Za-z0-9_ '\-]{1,80})` | 抽 full_name |
| [eva_slots.py:146](../eva_slots.py) | `\b{pref}\s+age\s+(?:is\|:)\s+(\d{1,3})\b` | 抽 age 数字 |
| [eva_slots.py:168-179](../eva_slots.py) | `toy\s+(?:was\|is\|:)\s+(?:a\|an\|the)?\s*([^,.;!]+)` | 抽 toy 值 |
| [eva_memory_legacy.py:119](../eva_memory_legacy.py) | `\b\w{2,}\b` | BM25 分词 |
| [eva_memory_legacy.py:186-196](../eva_memory_legacy.py) | `(?<=[?!])\s+`、`\s+(?:and\|or\|also\|...)\s+` | 复合句拆分 |
| [eva_memory_legacy.py:210-212](../eva_memory_legacy.py) | `[_\-–—/]+`、`[^a-z0-9一-鿿]+`、`\s+` | 字符 normalize |
| [eva_memory_legacy.py:240](../eva_memory_legacy.py) | `(?<![a-z0-9])X(?![a-z0-9])` | word-boundary 短语匹配 |

**操作对象**：我们自己写的记忆 record。Record 格式由 [Memory_maker/rewrite_memory.py](../Memory_maker/rewrite_memory.py) 控制，slot extractor 跟 record 写法是配套设计的。

**为什么不换 LLM**：
1. PRE PROBE 阶段对延迟敏感——每轮跑 4 次 slot extract，换 LLM = 多 8-20 秒
2. 抽不到时 fallback 是 `MISSING`，verifier 会兜底
3. 真正的语义判断已经在上层有 `judge_topic_subset`（DeepSeek）

**风险点**：记忆库重写后部分模式可能要校准（[SESSION_ARCHIVE_2026-05-08.md](SESSION_ARCHIVE_2026-05-08.md) § 四提过 vector_text 改造已做过一轮）。

**判定**：✅ 保留。

---

### C. Verifier 谓词（~30 处）— 保留

[eva_verifier_logic.py](../eva_verifier_logic.py) ~30 个 reason 专用 regex + [eva_core.py](../eva_core.py) 路由谓词。

代表性模式（按用途分组）：

| 用途类别 | 代表模式 | 文件:行 |
|---|---|---|
| 答案 token 检查 | `\b\d+\s+day(s)?\b` | [eva_verifier_logic.py:380](../eva_verifier_logic.py) |
| 动物词集合 | `\b[a-z]{3,}\b` 配 `{cat, dog, bunny, ...}` 集合 | [eva_verifier_logic.py:396](../eva_verifier_logic.py) |
| Eva 自指代检查 | `(?<![a-z0-9])(eva\|maid\|your)(?![a-z0-9]).*\bgaming\b` | [eva_verifier_logic.py:610-622](../eva_verifier_logic.py) |
| 生日 pronoun mismatch | `\byour\s+birthday\b` | [eva_verifier_logic.py:1086](../eva_verifier_logic.py) |
| 玩具否定语 | `\b(no\s+record\|don'?t\s+know\|unknown\|...)\b` | [eva_verifier_logic.py:1099](../eva_verifier_logic.py) |
| 中英文日期 | `\b(\d+)\s+day(?:s)?\b` / `(\d{1,4})\s*天` | [eva_verifier_logic.py:1143/1339/1347](../eva_verifier_logic.py) |
| Tool call 泄漏 | `^\`+\|`+$` 加 `function\s+call` | [eva_verifier_logic.py:528-531](../eva_verifier_logic.py) |
| 解析 evidence 头 | `\[MEMORY MODULE DATA for '([^']+)'\]` | [eva_core.py:613-665](../eva_core.py) |
| 月日抽取 | `\b(jan\|feb\|...)\s+(\d{1,2})(?:st\|...)?\b` | [eva_core.py:1015](../eva_core.py) |
| 中文日期 | `(\d{1,2})\s*月\s*(\d{1,2})\s*日?` | [eva_core.py:1042](../eva_core.py) |
| 思考块抽取 | `<think>(.*?)</think>` | [eva_core.py:1784](../eva_core.py) |

**操作对象**：Eva 自己生成的 answer + 我们自己注入的 evidence 块。结构都是我们设计的。

**关键设计**：这层 regex 跟 [eva_verifier_semantic.py](../eva_verifier_semantic.py)（DeepSeek 语义校验）**互补**：
- regex 层：判 "具体 token 有没有出现"（闭集合）
- 语义层：判 paraphrase / 视角一致 / persona 漂移（开放）

把 regex 谓词换成 LLM 等于让 semantic verifier 跑两遍。

**判定**：✅ 保留。[RUNTIME_FLOW.md](RUNTIME_FLOW.md) § 三.5.A 已明确"永久保留"。

---

### D. 用户输入上的开放语言匹配（最值得审视）

**这是唯一可能值得替代的类**。共 3 处。

#### D1. `_HARD_GUARD_REGEX` ([eva_memory_v2.py:135-154](../eva_memory_v2.py))

15+ 模式枚举 slot/identity 问句：

```
\bwhat(?:'s|\s+is)\s+(?:your|my)\s+(?:full\s+)?name\b
\bwhen\s+(?:is|was)\s+(?:your|my)\s+birthday\b
\bdo\s+you\s+have\s+(?:a|any)\s+(?:toy|pet|hobby)\b
...
```

**性质**：和被删的 `_PRONOUN_FOLLOWUP_PATTERNS` 形态一致——枚举用户语言形状。

**但是**它是 **fail-safe 的快路径**：

| 命中? | 后果 |
|---|---|
| 命中 | 强制 probe；错的代价是浪费一次 FAISS 调用，**不是错答** |
| 没命中 | 走正常 keyword + `judge_topic_subset` (LLM) 路径，PRE PROBE 仍能触发 |

**判定**：⚠️ **保留，但值得监控**。

监控信号：trace 计数 "新形态的 slot 问句被漏过去并导致错答的次数"。如果到达 ~5%，
再考虑像 P6 那样下决心删。**目前没有等价于 P6 76% 的证据**。

#### D2. `_RELATIONAL_PREDICATES` ([eva_memory_v2.py:173-177](../eva_memory_v2.py))

```
\b(with\s+me|with\s+you|together|with\s+us)\b
\bin\s+front\s+of\b   (record 端)
```

**性质**：4-5 个固定关系谓词，闭集合。

**判定**：✅ 保留。换 LLM 是杀鸡用牛刀。

#### D3. [eva_core.py](../eva_core.py) 路由谓词

`PUBLIC_FACT_RELATION_PATTERNS` / `MEMORY_VERIFICATION_PATTERNS` / `\b(check|verify|confirm|recall|remember|...)\b` 等约 10 处。

**性质**：**已经分层**——
- 第一层：regex 快筛（这些）
- 第二层：[eva_intent_judge.py](../eva_intent_judge.py) `judge_intent("EXPLICIT_MEMORY"/"EXPLICIT_WEB", ...)` LLM 校验

架构本身已经吸收了 regex 不收敛的风险。这是教科书式正确分工。

**判定**：✅ 保留。

---

### E. 输入清洗（2 处）— 必须保留

| 位置 | 模式 | 用途 |
|---|---|---|
| [eva_render.py:67-76](../eva_render.py) | vision/image protocol tag 剥除 | ChatML 净化 |
| [eva_tools_runtime.py:119](../eva_tools_runtime.py) | `User avatar\|Skip to content\|Powered by phpBB\|...` | 剥 web 搜索 boilerplate |
| [eva_tools_runtime.py:121](../eva_tools_runtime.py) | `\s+` | 折叠空白 |

**判定**：✅ 保留。LLM 清 HTML 是浪费。

---

### F. Topic 关键词字典（[eva_memory_v2.py:109](../eva_memory_v2.py)）— 保留

word-boundary 正则 `(?<![a-z0-9])alias(?![a-z0-9])` 跑 58 个 topic alias。

**架构**：已是最优 "regex 快筛 + LLM 兜底"：
- 第一层：本 regex 找候选 topics（58 alias × ~10ms = ~快）
- 第二层：`judge_topic_subset` (DeepSeek) 过滤误匹配

换 LLM 做第一层 = 每轮多调一次 DeepSeek，纯赔本。

**判定**：✅ 保留。

---

## 二、汇总表

| 类别 | 数量 | 替换 LLM？ | 核心理由 |
|---|---|---|---|
| A 协议解析 | 5 | ❌ | 操作我们自己的 token |
| B 闭集合抽取 | ~20 | ❌ | 我们自己的 record 格式；4×/turn 延迟敏感 |
| C Verifier 谓词 | ~30 | ❌ | 已和 semantic verifier 互补分工 |
| D1 hard-guard | 15+ | ⚠️ 监控 | 形态像 P5，但 fail-safe；漏一条不致命 |
| D2 relational | 2 | ❌ | 闭集合 |
| D3 路由谓词 | ~10 | ❌ | 已是 "regex+LLM 双层" |
| E 清洗 | 2 | ❌ | HTML / 噪声 |
| F Topic 字典 | 1 (×58 alias) | ❌ | 已是 "regex+LLM 双层" |

---

## 三、总判断

**P6 删除是有原则的特例，不是普遍模式**。

整个 codebase 的 regex 用法基本都在闭集合域，每条都有合理位置。

**唯一值得监控**：D1 (`_HARD_GUARD_REGEX`)——它是与 P5 同一形态的开放枚举，
但因为 fail-safe 设计目前不构成生产风险。

**触发下一次 P6-style 重构的信号**：trace 里出现 "用户问 slot 问句但 PRE PROBE
skip 导致错答" 的比例到 ~5%。**目前没有这个证据。**

---

## 四、行动项（可选）

### 短期（可不做）

无。当前 regex 全部位置合理。

### 中期（监控）

- [ ] 在 [eva_memory_v2.py](../eva_memory_v2.py) hard_guard 路径加计数：
  - `hard_guard_hit_count` / `hard_guard_miss_with_keyword_hit_count`
  - 如果 miss-but-keyword-hit 比例长期 > 5%，可能存在新形态 slot 问句
- [ ] verifier 修复路径里加 "miss → 错答" 的 trace 标记，便于事后审计

### 长期（仅在数据支持时）

- [ ] 如果监控显示 D1 漏检率 > 5% 持续 30 天，启动 D1 重构（参考 P6 方法论：
  shadow → effective_quality 评估 → cutover → 30 天后删除）
- [ ] 不要为"理论上更优雅"去换。**没有 P6 那种 76% 实证证据，不动**。

---

## 五、相关文档

- [RUNTIME_FLOW.md](RUNTIME_FLOW.md) § 三.5 — Regex 全景图 + Prompt 演进
- [P6_pronoun_resolver_refactor_v3.md](P6_pronoun_resolver_refactor_v3.md) — 删除 regex 的方法论
- [P6_pronoun_resolver_refactor_v4.md](P6_pronoun_resolver_refactor_v4.md) — effective_quality 评估指标
- [SESSION_ARCHIVE_2026-05-08.md](SESSION_ARCHIVE_2026-05-08.md) — 记忆库 vector_text 改造
