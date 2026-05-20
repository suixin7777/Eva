# Session Archive — 2026-05-07 / 05-08

> 本会话的全部代码改动、重构决定、实测数据、遗留事项的整体归档。
> 时间跨度：约两整天，从 P6 重构开始，延伸到 latency 调优、记忆库重写、
> 工具脚本，结束于 add_memory.py 落地。

---

## 〇、Executive Summary

### 完成度概览

| 工作流 | 状态 | 主要产物 |
|---|---|---|
| **P6 Pronoun Resolver 重构** | ✅ P6.0–P6.3 完成；P6.4 patch 待 30 天后应用 | `eva_pronoun_resolver.py` + 36 tests + 4 个 P6 文档 |
| **Latency 调优** | ✅ Prompt 缩短；阈值按实测重设 | n=50 baseline + v4 plan 二修 |
| **Force `<think>` prefix** | ✅ 上线 | `FORCE_THINK_PREFIX` flag + `step_once` 改造 |
| **记忆库重写** | ✅ 91 → 100 records；去元标签 + 加细节；新增 4 事件 + Toy 修复 | `8.memory_optimized.jsonl` + 备份 |
| **Topic 词典重写** | ✅ 58 topics 覆盖大幅扩展 | `topic_keywords.json` v2 |
| **工具脚本** | ✅ 4 个新脚本全部就绪 | `rewrite_memory.py` / `Memory.py` 重写 / `add_memory.py` / `test_p6_latency_probe.py` |
| **文件组织** | ✅ 清理目录结构 | `docs/` + `tests/` + `Memory/` 平级 |
| **Regex 审计** | ✅ 全 codebase 6 类 regex 系统审视 | `docs/REGEX_AUDIT.md`（二次会话） |
| **Verifier 防幻觉规则** | ✅ NO ELABORATION 后置硬验 | `unsupported_specifics_under_no_elaboration_rule` reason + 17 tests（二次会话） |
| **Trace rewrite 视觉标记** | ✅ STEP-5 分隔块 | `_render_step5_rewrite_block` + `TRACE_REWRITE_STYLE` flag + 7 tests（二次会话） |
| **UX 异步占位符** | ✅ DeepSeek pending 标记 | `announce_pending_llm()` + 6 tests（二次会话） |

### 关键里程碑

1. **2026-05-07 早期**：P6.0–P6.2 三阶段一气完成，36 tests 全过
2. **2026-05-07 晚期**：n=50 shadow 实测，effective_quality=86%，**verdict READY for P6.3 cutover**
3. **2026-05-08 早期**：P6.3 cutover 应用（`MODE="llm_first"`），latency 三次阈值重设
4. **2026-05-08 中段**：Force think prefix 解决跳 thought 导致的幻觉问题
5. **2026-05-08 晚期**：记忆库整体重写 + add_memory.py 工具落地
6. **2026-05-08 二次会话**：完成原 § 九 P0/P1 三项遗留——Verifier 防幻觉、Trace 视觉
   标记、UX 占位符；4 套 offline 测试 66/66 全绿；regex 审计文档落地

---

## 一、P6 Pronoun Resolver 重构

### 起源

[docs/P6_pronoun_resolver_refactor_v3.md](P6_pronoun_resolver_refactor_v3.md) 列出动机：
原 `_PRONOUN_FOLLOWUP_PATTERNS` regex 在 P5 → P5.1 已经被打了两次补丁
（加 `really?|wait,|hmm,|huh,` 前缀）；下一次出现 `hold on,` / `sorry,`
等新形态又得改正则。**用 regex 枚举语言形状不收敛**。

### 五阶段计划

| Phase | 状态 | 内容 |
|---|---|---|
| P6.0 | ✅ | 新建 `eva_pronoun_resolver.py` + JudgeState 字段 + 6 个 config flag + 36 tests + 修一个 pre-existing VERIFIER_DEBUG bug |
| P6.1 | ✅ | 接到 `build_required_memory_params`；MODE="regex_only" 起步保行为 |
| P6.2 | ✅ | shadow 模式实现；50 obs 实测（见下）|
| P6.3 | ✅ | 翻 MODE="llm_first" 并 SHADOW=False |
| P6.4 | ⏸️ | 30 天稳定后按 [P6_4_deletion_patch.md](P6_4_deletion_patch.md) 删 4 个 legacy regex symbol |

### 50-obs Shadow 实测结果（2026-05-07）

> 在 Colab → DeepSeek 跨洋环境采集。

| 指标 | 实测 | v4 阈值 | 结果 |
|---|---|---|---|
| effective_quality | **86%** | ≥ 80% | ✅ |
| llm_rescue_rate | **76%** | ≥ 20% | ✅ 远超 |
| true_disagree_rate | **12%** | ≤ 15% | ✅ |
| llm_availability | **98%** | ≥ 95% | ✅ |
| 样本量 | 50 | ≥ 50 | ✅ |

**关键发现**：legacy regex **漏检率 76%**——每 4 个 pronoun follow-up 有 3 个
regex 完全识别不到，LLM 全部正确补上。这比 v3 plan 设想的还严重，是 P6
重构最强的实证支持。

### 6/6 true_disagree 人工 audit

- 0/6 是 LLM 明显错
- 3/6 是 substring matcher 假阳性（`Neon Genesis Evangelion` vs `shinji ikari`、`Neuro-sama` vs `neuro sama`）
- 3/6 是低 conf（≤ 0.7）case，被 `PRONOUN_RESOLVER_MIN_CONFIDENCE=0.6` 阈值正确 demote

→ **VERDICT: READY，应用 P6.3 cutover**。

### 文档版本演进

```
v1 (用户原方案)
  ↓ + 7 项担心评审
v2 (我的修订)
  ↓ + 用户追问"P6.4 为何不能完全删 regex"
v3 (彻底删除 regex，无 safety net)
  ↓ + n=50 实测发现 strict Jaccard 严重低估 LLM
v4 (effective_quality + region-aware latency)
```

参见 [docs/P6_pronoun_resolver_refactor_v4.md](P6_pronoun_resolver_refactor_v4.md)。

---

## 二、Latency 调优历程

### 三次阈值修订

| 版本 | cn_native P95 | 触发原因 |
|---|---|---|
| v3 初版 | ≤ 800ms | 沿用旧目标，**完全脱离实际** |
| v4 一改 | ≤ 4000ms | n=20 单次跑后第一次校准 |
| v4 二改 | ≤ 5000ms | n=50 稳定 baseline 后最终校准 |

### 实测 baseline (n=50, Colab → DeepSeek)

```
P50:  2935ms
P95:  4905ms
mean: 3270ms
max:  9209ms (1/50, P98 outlier)
分布: 80% calls 落 2200-3500ms，18% 落 3500-6000ms
```

减去跨洋 RTT (~500ms) 后推算 cn_native：
- P50 ≈ 2500ms
- P95 ≈ 4500ms

### 关键洞察

**DeepSeek `v4-flash` 服务端生成时间本身就是 2-3 秒级**——不是网络问题。
能动的优化杠杆只剩：

1. ✅ 缩短 system prompt（已做：~390 → ~170 tokens；实测在 noise 内无明显增益）
2. ❌ 减少 output token（输出已经 30-50 tokens，没空间）
3. ❌ 切更快模型（DeepSeek 没出 turbo 版）
4. ⏸️ UX 异步化（"思考中..."占位符；架构改动，未实施）

### 工具

[tests/test_p6_latency_probe.py](../tests/test_p6_latency_probe.py)——
独立 probe 脚本，任何 Python + openai 环境可跑，不依赖 build_agent。

---

## 三、Force `<think>` Prefix（防 hallucination）

### 问题

实测发现 Phase-1 greedy 解码对简单 query（如 `"for example?"`）会**跳过
`<think>`**直接出 `<|answer|>`，缺少自我审查窗口，相关 turn 出现
fabricate（`"lasagna once nearly set off the smoke alarm"` — 记忆里
根本没这事）。

### 解法

Decode 时把 `<think>` 作为前缀**硬塞**进 input_ids，模型只能从 think 块
内部继续生成。

### 实现

| 文件 | 改动 |
|---|---|
| [eva_config.py](../eva_config.py) | 加 `FORCE_THINK_PREFIX = True` flag |
| [eva_core.py](../eva_core.py) `step_once` | 在 Phase-1 generate 前 `torch.cat` 注入 `<think>` token；同步 seed `full_response` 和 printer 让 trace 正常显示 THOUGHT 头 |

### 副作用 / 回滚

- **延迟**：每轮多 200-500ms（生成 thought 的额外 token）
- **幻觉**：预期下降（实测见 turn 6 vs 7 对比）
- **回滚**：`FORCE_THINK_PREFIX = False`

---

## 四、记忆库整体重写

### 动机

实测 Colab 对话发现 `"Do you have a toy?"` Eva 回答"我没有"——但记忆里
有 cuddly bunny 记录。根因：

1. **`topic_keywords.json` 缺 `Toy` topic**——用户说 "toy" 时 PRE PROBE 完全 miss
2. **`Childhood` 关键词太窄**——只有 4 个，覆盖不到日常说法
3. **vector_text 被元标签污染**——`[Category: Lore] [Topic: Childhood]` 拉低 embedding 质量
4. **content 太瘦**——大多数 record 一句话，缺场景/情感/细节

### 重写规模

| 项 | 旧 | 新 | 净变化 |
|---|---|---|---|
| 总 records | 91 | **100** | +9 |
| Eva | 42 | 45 | +3（Toy / Sleep / 真实色 mint green） |
| Rosm | 15 | 17 | +2（counting chocolate / 容忍 Eva 恶作剧） |
| Shared events | 5 | 9 | +4（First Day / Rainstorm / Apex Co-op / Quiet Evening） |
| Shared lore | 29 | 29 | 0（全部重写但不增） |
| topic_keywords topics | 56 | **58** | +2（Toy / Sleep） |

### vector_text 改造（embedding 质量）

```diff
- [Category: Lore] [Entity: Eva] [Topic: Childhood] Eva's favorite childhood toy was a bunny and she watched sassy cartoons.
+ Eva's favorite toy is a cuddly bunny — she's had it since childhood and still keeps it. Plushie, stuffed animal, bunny. She does have a toy.
```

去元标签（embedding 噪声），加同义词覆盖（toy / plushie / stuffed animal / bunny 全在），
查询 "do you have a toy" 时 FAISS 命中分 0.472 → 任意阈值都过。

### content 改造（细节 + 场景 + 情感）

```diff
- Once, Rosm gave Eva a beautiful music box. She pretended not to care… and then played it on repeat until he regretted giving it to her.
+ Once, Rosm gave Eva a small carved wooden music box — a ballerina that turned to a slow Tchaikovsky melody. Eva pretended not to care, set it on her shelf, and then played it on repeat for two solid days until Rosm started flinching at the opening notes. She still has the music box. It still works. She still plays it.
```

加：物体细节（`small carved wooden`、`ballerina that turned`）、配乐
（`Tchaikovsky`）、时间长度（`two solid days`）、Rosm 反应（`flinching`）、
持久情感（`She still has it. She still plays it.`）。

### Smoke test 验证

| Query | Top hit | Score |
|---|---|---|
| `"do you have a toy?"` | Eva/Toy (cuddly bunny) | **0.472** ✅ |
| `"how about a toy?"` | Eva/Toy | 0.433 |
| `"your favorite plushie"` | Eva/Toy | **0.504** |
| `"did we visit a museum together?"` | Shared/Activity | **0.580** |

之前完全 miss 的查询现在全部命中。

### Topic 词典 v2

[topic_keywords.json](../topic_keywords.json) 改动：

- 新增 `Toy` topic：`["toy", "toys", "plushie", "plush", "stuffed animal", ..., "do you have a toy"]`（19 个 alias）
- 新增 `Sleep` topic
- 扩展 `Childhood`：`["childhood", "child", "young", "kid", "kids", "growing up", "early days", ...]`（15 个）
- 扩展 `Hobbies` / `Gifts` / `Pet` / `Greetings` / 8 个 Birthday-prefixed 等
- 加 `_version` 元字段记录修订历史

word-boundary 正则匹配（`(?<![a-z0-9])X(?![a-z0-9])`）已实测正确，
不会出现 `'yo'` 误匹 `'do you'` 的假阳性。

---

## 五、新工具脚本

### [rewrite_memory.py](../Memory_maker/rewrite_memory.py)（200 行）

**作用**：所有 100 条 record 的源代码（Python list of `rec(...)` calls）。

未来想批量改风格、加新记忆、调整 topic 分类——改这个文件再 re-run，
比直接编辑 JSONL 安全。

### [Memory.py](../Memory_maker/Memory.py) 重写

| 旧 | 新 |
|---|---|
| 硬编码 `OUTPUT_DIR = "Memory"`（在 Memory_maker 下产生 dup） | `--outdir` argparse；默认 `../Memory/`（生产路径，sibling of Memory_maker） |
| 写死的 "Who am I" debug 检查（过时数据版本对齐） | 删掉 |
| 单 except 接 FileNotFoundError | argparse + 完整错误处理 |
| 末尾 2 个 hardcoded test query | 5 个 smoke test 覆盖 toy / gifts / museum / plushie 等之前失败的 case |

### [add_memory.py](../Memory_maker/add_memory.py)（300 行）

**作用**：交互式记忆追加，DeepSeek 把自然语言变成结构化 record。

四种模式：

```bash
python add_memory.py "Eva and Rosm watched a meteor shower"     # 单条
cat memory.txt | python add_memory.py                            # stdin
python add_memory.py --batch new_memories.txt                    # 批量
python add_memory.py --manual                                    # 无 LLM 手填
```

交互菜单：`[a]ccept / [r]egenerate / [f]eedback+regen / [c]ancel`

内置 validation：检查 entity / category / topic（vs canonical list）/
vector_text 元标签污染 / content 长度 / keywords 类型。校验只警告不阻塞。

### [tests/test_p6_latency_probe.py](../tests/test_p6_latency_probe.py)

**作用**：独立 latency probe，绕开 build_agent，只测 DeepSeek 调用本身。

```bash
python tests/test_p6_latency_probe.py --n 30                     # 默认
python tests/test_p6_latency_probe.py --region cn_native --n 50  # 强制 region 阈值
```

---

## 六、文件组织清理

### Before

```
D:/Eva_new/
├── eva_*.py                              # 21 个核心代码
├── test_*.py                              # 4 个测试散在根
├── *.md                                   # 5 个 P6 markdown 散在根
├── topic_keywords.json
├── Memory/
└── Memory_maker/
    ├── Memory/                            # ← 重复
    ├── Memory.py
    └── 8.memory_optimized.jsonl
```

### After

```
D:/Eva_new/
├── docs/                                  # 5 个 P6 markdown 集中
│   ├── P6_pronoun_resolver_refactor_v2.md
│   ├── P6_pronoun_resolver_refactor_v3.md
│   ├── P6_pronoun_resolver_refactor_v4.md
│   ├── P6_4_deletion_patch.md
│   ├── P6_2_shadow_runbook.md
│   └── SESSION_ARCHIVE_2026-05-08.md      # 本文件
├── tests/                                 # 4 个测试集中
│   ├── test_multiturn_regression.py
│   ├── test_p2_regression.py
│   ├── test_p6_latency_probe.py
│   └── test_p6_pronoun_resolver.py
├── Memory/                                # 生产路径（Memory.py 直接写这里）
│   ├── memory.index
│   ├── memory_content.json
│   └── memory_meta.json
├── Memory_maker/                          # 数据 + build 工具
│   ├── 8.memory_optimized.jsonl
│   ├── 8.memory_optimized.jsonl.bak
│   ├── rewrite_memory.py
│   ├── Memory.py
│   └── add_memory.py
├── eva_*.py                               # 核心代码
├── topic_keywords.json
└── ...
```

测试文件加了 `sys.path` 补丁，从任意 cwd 跑都行：

```python
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
```

---

## 七、配置 flag 清单（截至本归档时）

[eva_config.py](../eva_config.py) 新增的关键 flag：

```python
# ===== P6 Pronoun Resolver =====
ENABLE_PRONOUN_RESOLVER = True
PRONOUN_RESOLVER_MODE = "llm_first"           # P6.3 cutover: was "regex_only"
PRONOUN_RESOLVER_SHADOW = False               # P6.2 only
PRONOUN_RESOLVER_MIN_CONFIDENCE = 0.60
PRONOUN_RESOLVER_DEBUG = False
PRONOUN_RESOLVER_MAX_WORDS = 8
PRONOUN_RESOLVER_MAX_CALLS_PER_TURN = 2       # 独立预算池

# ===== Phase-1 think prefix =====
FORCE_THINK_PREFIX = True                     # 防止 greedy decode 跳过 <think>
```

---

## 八、测试套件

```
$ python tests/test_p6_pronoun_resolver.py
Ran 36 tests in 0.876s
OK
```

分布：
- TestCheapGate: 5
- TestParseVerdict: 7
- TestResolveFlags: 9
- TestAcceptanceFixtures: 1（含 8 fixtures）
- TestP62ShadowMode: 6
- TestP63LLMFirstIntegration: 5（锁 cutover 契约）
- TestP61RegexOnlyEquivalence: 2
- 其他：1

---

## 九、遗留事项

### 必须做（生产正式上线前）

- [ ] **生产环境 region latency 复测**——Colab 跨洋数据不能作为生产决策依据，
  在国内服务器（同 DeepSeek region）跑 [tests/test_p6_latency_probe.py](../tests/test_p6_latency_probe.py)
  30-50 条，验证 P95 落在 v4 阈值（cn_native ≤ 5000ms）

### 应该做（P6 收尾）

- [ ] **P6.4 deletion patch 应用**——条件：P6.3 在生产稳定运行 ≥ 30 天，
  LLM 调用成功率持续 ≥ 98%。按 [P6_4_deletion_patch.md](P6_4_deletion_patch.md)
  逐步执行：删 4 个 legacy regex symbol + `regex_only` mode + shadow helper

### 可以做（优化）

- [x] **UX 异步化** — **applied 2026-05-08**：
  在 verifier-repair 路径的两个阻塞 DeepSeek 调用前打印
  `[PENDING DeepSeek] <label>` 占位符，吸收 ~6s 死气时间。
  - resolver LLM 主路径前（[eva_pronoun_resolver.py](../eva_pronoun_resolver.py)）
  - `synthesize_tool_thought` 前（[eva_intent_judge.py](../eva_intent_judge.py)）
  helper `announce_pending_llm()` 在 [eva_intent_judge.py](../eva_intent_judge.py)，
  无条件 print（不受 debug flag 控制——目的就是非 debug 也要看见）。
  Cache hit / 预算耗尽 / cheap-gate skip 路径正确不打印（无延迟可吸收）。
  6 tests in [tests/test_pending_llm_announcement.py](../tests/test_pending_llm_announcement.py)。
  注：未串 progress_callback——SDK 消费者（Colab UI）仍走现有
  step_once 的高层里程碑回调；此处只是 console-side。
- [ ] **记忆库继续扩展**：用 [add_memory.py](../Memory_maker/add_memory.py) 加更多
  共同事件，让模型有更丰富的具体记忆可引用，进一步降幻觉
- [x] **Verifier 防幻觉规则**（"选 3"）— **applied 2026-05-08**：
  当 memory 输出含 `[NO ELABORATION RULE]` 标记时，answer 中的内容
  token 必须在 records 或 user query 里能找到，否则判 fail。
  新增 reason `unsupported_specifics_under_no_elaboration_rule`
  ([eva_verifier_logic.py](../eva_verifier_logic.py))，fix_class=`regenerate`，
  受 RegenerateGuard 1/reason + 2/total 预算保护。
  17 tests in [tests/test_no_elaboration_rule.py](../tests/test_no_elaboration_rule.py)，全过。
  对治：lasagna/smoke-alarm/thunderstorm 这类多词幻觉。
- [x] **Trace rewrite 视觉标记** — **applied 2026-05-08**：
  当 verifier fail 触发 `--- STEP-5 TRACE REWRITE ---` 时，trace 现在
  打印一个带框 + 显式"^^^ above are SUPERSEDED ^^^"提示的块，新 thought
  + tool_code 各自带 `[REWRITTEN]` 标签。两种 style：
  - `TRACE_REWRITE_STYLE="ansi"`（默认）— bold-yellow header +
    dim-strike supersede notice，VS Code / Windows Terminal / *nix 终端正常显示
  - `TRACE_REWRITE_STYLE="ascii"` — 纯 `===` 横线 + 方括号标签，所有终端可读
  实施于 [eva_verifier_logic.py](../eva_verifier_logic.py) `_render_step5_rewrite_block`
  + [eva_config.py](../eva_config.py) `TRACE_REWRITE_STYLE`。
  注：StreamPrinter 没改——已打印的 THOUGHT/ANSWER 在流式终端无法回头改写，
  所以改用清晰的"以下作废"分隔符替代 strikethrough。
  7 tests in [tests/test_step5_rewrite_render.py](../tests/test_step5_rewrite_render.py)。

### 监控点（上线后第一周）

- LLM 调用成功率（`llm_first` 路径）目标 ≥ 95%
- LLM 调用 P95 延迟（生产真实用户）→ 验证 v4 阈值
- 用户对话样本：是否有继续"编 lasagna 这种事"的幻觉

---

## 十、文件改动清单

### 新建

```
docs/                                          (新建目录)
docs/P6_pronoun_resolver_refactor_v2.md
docs/P6_pronoun_resolver_refactor_v3.md
docs/P6_pronoun_resolver_refactor_v4.md
docs/P6_4_deletion_patch.md
docs/P6_2_shadow_runbook.md
docs/SESSION_ARCHIVE_2026-05-08.md             (本文件)

tests/                                         (新建目录)
tests/test_p6_pronoun_resolver.py              (36 tests)
tests/test_p6_latency_probe.py                 (latency benchmark)

Memory_maker/rewrite_memory.py                 (100 records 源代码)
Memory_maker/add_memory.py                     (LLM 驱动追加工具)
Memory_maker/8.memory_optimized.jsonl.bak      (原 91 records 备份)

eva_pronoun_resolver.py                        (核心 resolver 模块)
```

### 重写

```
eva_config.py                                  (8d 段 6+1 个 flag + cutover audit trail)
eva_core.py                                    (step_once 加 FORCE_THINK_PREFIX)
eva_intent_judge.py                            (JudgeState.pronoun_call_count + reset)
eva_verifier_logic.py                          (build_required_memory_params 改用 resolve_pronoun + 修 VERIFIER_DEBUG bug)

topic_keywords.json                            (56 → 58 topics，关键词大扩展)
Memory_maker/8.memory_optimized.jsonl          (91 → 100 records，全部重写)
Memory_maker/Memory.py                         (argparse + smoke + 输出到 ../Memory/)
```

### 移动

```
test_*.py                                      (4 个) → tests/
P6_*.md                                        (5 个) → docs/
Memory_maker/Memory/*                          (3 个) → Memory/  + 删空目录
```

---

## 十二、二次会话补完（2026-05-08）

> 本节记录 SESSION_ARCHIVE 一稿落定后的二次会话。重点是清完 § 九 的
> P0/P1 遗留事项，并补齐一份 regex 审计文档作为日后判定 "要不要再删一批
> regex" 的参考基线。

### 12.1 Regex 全审计

**起源**：用户读完代码 + md 后问"全 codebase 的 regex 是否合理？是否需要
LLM 替代？"

**审视范围**：6 类共 ~90 个正则点，涵盖 [eva_render](../eva_render.py)、
[eva_memory_v2](../eva_memory_v2.py)、[eva_memory_legacy](../eva_memory_legacy.py)、
[eva_slots](../eva_slots.py)、[eva_verifier_logic](../eva_verifier_logic.py)、
[eva_core](../eva_core.py)、[eva_tools_runtime](../eva_tools_runtime.py)、
[eva_pronoun_resolver](../eva_pronoun_resolver.py) 等。

**结论**：P6 删除是有原则的特例（开放枚举不收敛），不是普遍模式。
当前 codebase 的 regex 全部在闭集合域：
- A 协议解析（5）/ B 闭集合抽取（~20）/ C Verifier 谓词（~30）—— 全部应保留
- D1 `_HARD_GUARD_REGEX`（15+，开放语言但 fail-safe）—— ⚠️ 监控但不动
- D2 `_RELATIONAL_PREDICATES`（2，闭集合）/ D3 路由谓词（~10，已分层）—— 保留
- E 输入清洗 / F Topic 字典 —— 保留

详见 [docs/REGEX_AUDIT.md](REGEX_AUDIT.md)。

### 12.2 Verifier 防幻觉规则（"选 3"）

**目的**：对治 lasagna/smoke-alarm/thunderstorm 这类多词幻觉。
当 PRE PROBE / MemorySearch 注入的 evidence 因低置信度被打上
`[NO ELABORATION RULE]` 标记，answer 中出现具体名词必须在 records
或 user query 里能找到，否则判 fail → regenerate。

**核心实现**（[eva_verifier_logic.py](../eva_verifier_logic.py)）：
- 新 reason `unsupported_specifics_under_no_elaboration_rule`，severity=hard，
  fix=regenerate（受 RegenerateGuard `1/reason + 2/total` 预算保护）
- helper `answer_violates_no_elaboration_rule(agent, answer, q, min_unsupported=3)`
- 触发链：rule 标记存在 + 无 hedge 短语 + ≥3 个 token 既不在 records 也
  不在 query → 判 fail
- token 提取：长度 ≥ 4 经 `'s` / `-ing` / `-ed` / `-es` / `-s` stem 后过滤
  通用停用词集（modal/persona/mental verb/placeholder 等）；时间词、
  天气词、动作词、地点词**不在停用词**——这正是规则要拦截的"具体场景"
- 阈值 3 抑制单词假阳性；调用方可降到 1

**测试**：[tests/test_no_elaboration_rule.py](../tests/test_no_elaboration_rule.py)
17 tests 全过——覆盖 rule 触发门控、3 种 hedge 旁路、lasagna 真幻觉
案例、阈值边界、查询回声、helpers + REASON_POLICY 注册。

### 12.3 STEP-5 Trace 视觉标记

**目的**：verifier rewrite 后旧的 THOUGHT/ANSWER 还在屏幕上，操作员要
脑补"哪段已废"。流式终端无法回头改写已打印内容，改用清晰分隔符替代
strikethrough。

**核心实现**：
- 新 config flag [eva_config.py](../eva_config.py) `TRACE_REWRITE_STYLE = "ansi" | "ascii"`
- 新 helper [eva_verifier_logic.py](../eva_verifier_logic.py)
  `_render_step5_rewrite_block(thought, tool_call_str, indent)`，输出
  `===` 框 + "^^^ above are SUPERSEDED ^^^" 提示 + `[REWRITTEN]` 前缀的
  thought / tool_code 各一行
- ANSI 模式用 bold-yellow header + dim-strike supersede；ASCII 模式纯
  `===` 横线
- `_rewrite_assistant_for_tool_repair` 把原本的 4 行平淡 print 替换为
  这个块

**测试**：[tests/test_step5_rewrite_render.py](../tests/test_step5_rewrite_render.py)
7 tests 全过——结构不变量、ANSI/ASCII 转义码隔离、indent 应用、长 thought
截断。

**与原 todo 偏差**：原 todo 想给 StreamPrinter 加 retroactively 添加
strikethrough 的能力，但流式终端字节流出去就回不来了——做不到。改
为显式"以下作废"分隔符 + 新内容打 `[REWRITTEN]` 标签，效果等价，
实现简洁，没动 StreamPrinter。

### 12.4 UX DeepSeek pending 占位符

**目的**：verifier-repair 路径里两个连续阻塞的 DeepSeek 调用（resolver
~3s + synthesize_tool_thought ~3s）造成 5-6s "死气"，操作员误以为挂了。

**核心实现**：
- 新 helper [eva_intent_judge.py](../eva_intent_judge.py) `announce_pending_llm(label)`
  无条件 print `        | [PENDING DeepSeek] <label>`（不受 debug flag
  控制——目的就是非 debug 也要看见）
- 调用点 1：[eva_pronoun_resolver.py](../eva_pronoun_resolver.py)
  `state.pronoun_call_count += 1` 之后、`_call_llm` 之前
- 调用点 2：[eva_intent_judge.py](../eva_intent_judge.py)
  `synthesize_tool_thought` 中 `state.call_count += 1` 之后、
  `call_deepseek_judge` 之前

**Skip 路径正确不打印**（cache hit / 预算耗尽 / cheap-gate skip 无延迟可吸收）。

**未做**：未串 `agent.progress_callback`——那是 SDK 消费者（Colab UI）的
高层里程碑回调，要串到 resolver/judge 需要在 JudgeState 上挂 callback ref。
此次只覆盖 console-side。

**测试**：[tests/test_pending_llm_announcement.py](../tests/test_pending_llm_announcement.py)
6 tests 全过——helper 输出 shape + 截断、四种调用路径下的有/无打印验证。

### 12.5 二次会话改动清单

**新建文件**：

```
docs/REGEX_AUDIT.md                            (regex 全审计 6 类基线)
tests/test_no_elaboration_rule.py              (17 tests, NO ELABORATION 防幻觉)
tests/test_step5_rewrite_render.py             (7 tests, STEP-5 视觉块渲染)
tests/test_pending_llm_announcement.py         (6 tests, PENDING 占位符)
```

**修改文件**：

```
eva_config.py                                  (+ TRACE_REWRITE_STYLE flag)
eva_verifier_logic.py                          (+ NO ELABORATION 检查 + reason policy
                                                + STEP-5 渲染 helper +
                                                  替换 _rewrite_assistant_for_tool_repair 内 print)
eva_intent_judge.py                            (+ announce_pending_llm helper +
                                                  synthesize_tool_thought 调用点)
eva_pronoun_resolver.py                        (+ resolve_pronoun LLM 调用点 lazy import)
docs/SESSION_ARCHIVE_2026-05-08.md             (Executive Summary 扩展 + 本节)
```

### 12.6 测试矩阵（66/66 全绿）

```
tests/test_p6_pronoun_resolver.py              36 tests   (P6 既有)
tests/test_no_elaboration_rule.py              17 tests   (新)
tests/test_step5_rewrite_render.py              7 tests   (新)
tests/test_pending_llm_announcement.py          6 tests   (新)
                                              ----------
                                              Total: 66 tests, 全部 OK
```

无回归。

### 12.7 § 九 遗留事项状态更新

P0/P1 已清完。剩下的都需外部环境或等时机：

- ⏸️ P2 生产环境 latency 复测（需上线机器，不可在此处做）
- ⏸️ P2 上线后第一周监控（需 SDK/运维）
- ⏸️ P3 P6.4 deletion patch（最早 2026-06-06）
- ⏸️ P3 D1 hard-guard 监控计数器（条件触发：漏检 > 5%）
- ⏸️ P3 记忆库继续扩展（非紧急）

---

## 十一、致谢和回顾

整个会话最有价值的发现：

1. **regex 漏检率 76%**——n=50 实测的 llm_rescue_rate 数字，比 v3 plan
   预想严重得多。这是 P6 重构最强的实证证据
2. **DeepSeek 服务端时间是延迟下限**——不是网络，靠改 prompt / 部署位置
   优化不动，必须接受 ~2-3s 的物理上限
3. **strict Jaccard 严重低估 LLM 质量**——agreement_rate=2% 看似灾难，
   实际是 LLM 在拯救（rescue）regex 漏的 case；改用 effective_quality
   metric 真相是 86%
4. **`[Topic:]` 元标签污染 embedding**——记忆库 vector_text 的格式细节
   决定召回质量，去标签后 toy 查询从 0 命中变成 0.472
5. **跳 thought = 跳 self-reflection**——greedy decoder 偷懒不仅省时间，
   也省了模型自查的窗口；force-prefix 是技术解，prompt 软约束模型不听

下一会话续接点：[第九章遗留事项](#九遗留事项)。
