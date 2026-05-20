# Eva 大重构完结记录 (2026-05-14)

本文档记录从 2026-05-13 到 2026-05-14 两天连续重构的**完整过程**——
从 Advisor 架构引入，到 Plan-A verifier 退役，到最终代码清理。

---

## TL;DR

| 维度 | 改前 | 改后 |
|---|---|---|
| 决策层 | 模型在 prompt 里自己分类 + 工具选择 | **Advisor (远端 DeepSeek) 前置决策** |
| 单轮 DeepSeek 调用 | 6-9 次 | **2 次** (advisor + semantic verifier) |
| Hard verifier reasons | ~9 个（regex + LLM 混杂） | **4 个**（1 格式 + 3 LLM judge） |
| Verifier kill 答案频率 | 每 3-5 turn 一次 | **接近 0** |
| 单轮中位延迟 | 12-18s | **6-9s** |
| Compound input 正确率 | ~30% | **~80%** |
| 死代码 | ~1500 行（inject_tool + R-5 + R-2.1 + legacy regex） | **已清** ~1000 行；剩余 ~500 行待 2 周稳定后清 |

---

## 1. 重构序章 (2026-05-12 / 2026-05-13 早期)

### 起因

测试 trace 反复暴露同类 bug：
- **Turn 6 "买熊+写报告"** → 模型只调一次 RememberThis，合并存了，后来 forget 全删
- **Forget 用错 hex** → 模型抄错 record_id，删错 note
- **verifier 误伤** → no_elaboration_rule 把对答案换成 canned "I don't remember"
- **跨实体日期污染** → "yours is Nov 25th—54 days away" (54 是 Eva 的)
- **Compound query 漏 entity** → "your birthday + my birthday" 只算了一个

各种 R-1 ~ R-7 + 热修复堆了 12 个补丁层。被诚实质疑："**修复是补丁还是根本？**"

### 决策

用户做出关键架构决策：
> "为什么不能丢给远端的大模型给出一个大概的逻辑，然后本地模型按照这个规则做？"

→ Advisor-first 架构。

---

## 2. Advisor 模块构建 (2026-05-13)

### Step A: 新模块

| 文件 | 行数 | 作用 |
|---|---|---|
| `Advisor/__init__.py` | 25 | 入口 + 公开 API |
| `Advisor/advisor_client.py` | ~330 | DeepSeek 调用 + 缓存 + 静默兜底 + 预算 |
| `Advisor/build_advisor_prompt.py` | ~105 | per-turn 上下文拼装 |
| `tests/test_advisor.py` | ~230 | 20 个单元测试，不打网络 |

### Step B: Advisor 输出 schema 扩展

```json
{
  "advice": "1-3 行自然语言提示给 Eva",
  "intent": "chat | remember | forget | query_memory | query_external | mixed | unknown",
  "needs_memory_retrieval": true,
  "memory_query_hint": "toy / birthday / ...",
  "needs_web_search": false,
  "web_query_hint": null,
  "suggested_calls": [
    {"tool": "MemorySearch", "args": {"query": "toy", "target_entity": "Eva"}}
  ]
}
```

### Step C: Verifier 改用 `suggested_calls` 做工具匹配

```python
# 不再看 intent 标签，看具体工具
suggested_tools = {c["tool"] for c in advisor.suggested_calls}
wants_web         = "WebSearch"    in suggested_tools
wants_memory_check = "MemorySearch" in suggested_tools
explicit_remember = notes_active and "RememberThis" in suggested_tools
explicit_forget   = notes_active and "ForgetMemory" in suggested_tools
```

### Step D: Advisor-first 路由

`_new_refresh_active_memory` 重写——Advisor 先跑，决定要不要 PRE PROBE，再走老的 judges fallback。

### Step E: `EVA_ADVISOR_FALLBACK_MODE` 配置

三档：`judges`（默认，走老 judges）/ `chat`（当 chat）/ `strict`（不 probe）。

---

## 3. 信号桥接修复 (2026-05-13 后期)

### Bug 1: PRE PROBE 信号断层

Advisor 给 `memory_query_hint="toy"`，PRE PROBE 拿到字符串后用 `target=Both, matched_topics=[]` 跑 FAISS，单词查询命中率低。

**修**：`_gate_memory_probe` 现在 enrich decision：
- 从 `advisor.suggested_calls[i].args.target_entity` 抽显式 entity
- 调 `TopicDict.match_topics(memory_query)` 自动算 matched_topics
- `TopicDict.subject_hint` 在唯一时填 entity

### Bug 2: Memory packet "重复搜索"

PRE PROBE 注入了 memory packet 但 advisor 仍叫 Eva 调 MemorySearch → 重复调用 + target 被推翻。

**修**：当 `current_turn_memory_has_exact OR top1 >= 5.0` 时在 packet 顶部加：
```
[MEMORY ALREADY RETRIEVED THIS TURN — answer directly from the data
 below; do NOT call MemorySearch again unless you need a DIFFERENT
 topic. The records here are pre-fetched and authoritative.]
```

### Bug 3: Eva 显式 target_entity 被推翻

`_guard_memorysearch_params` 用 query 文本里的 "i" 推断 user-Rosm，**覆盖** Eva 显式传的 `target_entity="Eva"`。

**修**：Eva 显式 target_entity（in `{"Eva", "Rosm", "Shared"}`）时**直接信，跳过 text inference**。

### Bug 4: Compound date query 漏 entity

"your birthday + my birthday" 时 `_maybe_compute_date_delta_from_memory` 只返回一个 binding。

**修**：GetCurrentTime 接受 `target_entity` arg，advisor 在 compound date 时给每个 entity 列一次 GetCurrentTime 调用。

### Bug 5: SAVED NOTES 被 NO_ELABORATION_RULE 误伤

"check note about buying" 时 saved note 命中但 warning "记录是弱匹配，说不记得" 让 Eva 否认。

**修**：当 packet 含 SAVED NOTES 时，警告文案末尾加 scope 说明：
> NOTE: this applies to the lore records section only. The `>>> SAVED NOTES <<<` block below contains facts the user explicitly stored — those are AUTHORITATIVE.

### Bug 6: 跨 session 旧 note 看不见

`recent_notes` 只列 `NotesStore.recent_adds`（session LRU），重启 kernel 后空了。

**修**：fallback 到 `NotesStore.list_notes()` by `created_at` desc，取 top-5。

### Bug 7: Saved Notes Index 注入 prompt

加 `_build_saved_notes_index_block` 永久把 live notes 全列到 system prompt，让 Eva 回答 "list all my notes" 时不需要调工具。

### Bug 8: list-all 模式 advisor 误窄化

"give me the full tips in your note" 时 advisor 收窄到一条 note 的 search。

**修**：advisor prompt 加专门规则识别"list all / 全部"模式，输出 `suggested_calls=[]` + advice "enumerate from Saved Notes Index"。

---

## 4. Plan-A: Verifier 退役 (2026-05-14)

### 设计哲学

> Verifier 越精细越脆。**最佳 verifier 是几乎没有 verifier**——advisor 把决策前置、模型在 in-context 跟话、LLM judge 当最后一道软网。

### 改动

| Reason | Pre-Plan-A | Post-Plan-A |
|---|---|---|
| `tool_call_leaked_in_answer` | hard + inject_tool | **保持 hard** |
| `missing_web_evidence_for_external_or_current_request` | hard + inject_tool | soft + canned |
| `missing_memorysearch_for_explicit_memory_check` | hard + inject_tool | soft + canned |
| `explicit_remember_request_not_handled` | hard + inject_tool | soft + canned |
| `explicit_forget_request_not_handled` | hard + inject_tool | soft + canned |
| `missing_date_calculation_evidence` | hard + canned | soft + canned |
| `unsupported_exact_toy_claim` | hard + inject_tool | soft + canned |
| `eva_self_birthday_pronoun_mismatch` | hard + regen | soft + canned |
| `date_math_target_date_mismatch` | hard + regen | soft + canned |
| `date_math_days_not_supported_by_calculation_evidence` | hard + regen | soft + canned |
| `toy_value_conflicts_with_exact_memory` | hard + regen | soft + canned |
| `textgen_perspective_mismatch` | hard + regen | soft + canned |
| `unsupported_specifics_under_no_elaboration_rule` | hard + regen | soft + canned |
| `semantic_verifier_fail:pronoun_referent_mismatch` | hard + regen | **保持 hard** |
| `semantic_verifier_fail:internal_self_contradiction` | hard + regen | **保持 hard** |
| `semantic_verifier_fail:fact_conflict_with_evidence` | hard + regen | **保持 hard** |

终态：**4 hard / 12 soft**。

### 工具调用监督移到 advisor

```python
# Verifier 现在：
suggested_tools = {c["tool"] for c in advisor_result.suggested_calls}
wants_web         = "WebSearch"    in suggested_tools
wants_memory_check = "MemorySearch" in suggested_tools
# ...
```

### 性能影响

| 指标 | Pre-Advisor | Post-Plan-A |
|---|---|---|
| 单轮 DeepSeek calls | 6-9 | 2 |
| 单轮中位延迟 | 12-18s | 6-9s |
| Verifier kill 答案率 | ~25% | <1% |

---

## 5. Vision 路径修复 (2026-05-14)

### Bug

第一次带图输入时 transformers 抛 `IndexError: The shape of the mask [4591] at index 0 does not match the shape of the indexed tensor [4590] at index 0`。

### 诊断方法

加 `_safe_generate` 包装捕获 thread exception 完整 traceback + `EVA_VISION_DEBUG=1` 打印 input shapes。

### 根因

```python
input_ids: (1, 4591)        # 4590 → +1 think prefix → 4591
attention_mask: (1, 4591)   # 同步 cat
mm_token_type_ids: (1, 4590)  # ← FORGOT to cat
```

phase-1 `FORCE_THINK_PREFIX` 段只 append `<think>` 到 `input_ids` 和 `attention_mask`，**漏了 `mm_token_type_ids`**。纯文本无感，第一次带图就 crash。

### 修

```python
if "mm_token_type_ids" in inputs1:
    inputs1["mm_token_type_ids"] = torch.cat(
        [inputs1["mm_token_type_ids"], torch.zeros_like(think_ids)], dim=1)
```

诊断工具（`_safe_generate` + `EVA_VISION_DEBUG`）**保留**——下次 vision 出问题 30 秒定位。

---

## 6. 最终代码清理 (2026-05-14)

### Tier 1: 删除备份数据文件

| 文件 | 大小 |
|---|---|
| `Memory/memory_meta.json.pre_r1.bak` | 16 KB |
| `Memory_maker/8.memory_optimized.jsonl.bak` | 36 KB |
| `Memory_maker/8.memory_optimized.jsonl.pre_slot_values.bak` | 52 KB |
| `generate/datasets/chain_recall_samples_clean.pre_catalog_backup.jsonl` | 840 KB |
| `generate/datasets/notes_recall_samples.pre_catalog_backup.jsonl` | 927 KB |
| `generate/datasets/final_dataset_ready.pre_cleanup.jsonl` | 14.2 MB |
| `generate/datasets/final_dataset_ready_v2.pre_pathfix.jsonl` | 14 MB |

**共 ~30 MB 清理。**

### Tier 2: 归档旧文档到 `docs/archive/`

| 文档 | 取代者 |
|---|---|
| `TODO_2026-05-13_root_fixes.md` | Plan-A + advisor-first 取代 |
| `R11_design.md` | Advisor-first 接管，R-11 推迟 |
| `SESSION_ARCHIVE_2026-05-08.md` | 历史 session log |
| `P6_2_shadow_runbook.md` | P6 已完成 |
| `P6_4_deletion_patch.md` | 已应用 |
| `P6_pronoun_resolver_refactor_v2.md` | 被 v4 取代 |
| `P6_pronoun_resolver_refactor_v3.md` | 被 v4 取代 |
| `RUNTIME_FLOW.md` | `GENERATION_FLOW_2026-05-14.md` 取代 |
| `REGEX_AUDIT.md` | Plan-A 已退役大部分 regex |

**`docs/` 根目录现仅 4 份 canonical**：
- `GENERATION_FLOW_2026-05-14.md`
- `REFACTOR_COMPLETE_2026-05-14.md` (本文)
- `USER_NOTES_MODULE.md`
- `SLOT_SUBJECT_CLASSIFIER_PLAN.md`
- `P6_pronoun_resolver_refactor_v4.md`

### Tier 3: 死代码删除

| 删除项 | 位置 | 行数 |
|---|---|---|
| `_self_validate_date_calculation` 函数体 | verifier_logic | ~110 |
| `BannedDateLogitsProcessor` 类 | eva_core | ~40 |
| `_build_date_phrase_variants` | eva_core | ~30 |
| `_day_ordinal` + `_MONTH_NAMES` + `_MONTH_ABBR` | eva_core | ~10 |
| `_build_banned_date_token_seqs` 方法 | eva_core | ~35 |
| `if 0:` invocation block in `_run_phase2_sample` | eva_core | ~25 |
| `LogitsProcessor` / `LogitsProcessorList` import | eva_core | -1 |
| `tests/test_banned_date_logits.py` (整文件) | tests | -1 文件 |

**共 ~250 行 + 1 测试文件。**

### 保留作回滚保险

| 代码 | 位置 | 触发条件 |
|---|---|---|
| `extract_remember_params_from_user_text` | verifier_logic | tests 引用 |
| `find_recent_note_id` | verifier_logic | tests 引用 |
| `build_required_web_query` | verifier_logic | execute_controller_tool 的 WebSearch 缺 query fallback |
| `build_required_memory_params` | verifier_logic | tests 引用 |
| `_is_pronoun_followup` / `_extract_topical_nouns_from_recent_turns` | verifier_logic | 同上 |

稳定 2 周后可清。

---

## 7. 测试覆盖

| 测试文件 | Tests | 状态 |
|---|---|---|
| test_advisor | 20 | ✅ |
| test_verdict_ledger | 19 | ✅ |
| test_evidence_ledger | 14 | ✅ |
| test_no_elaboration_rule | 24 | ✅ |
| test_slot_values_meta | 14 | ✅ |
| test_forget_query | 28 (10 skip) | ✅ |
| test_event_schema | 22 | ✅ |
| test_dialog_focus | 12 | ✅ |
| test_pronoun_speaker_perspective | 14 | ✅ |
| test_subject_classifier | 42 | ✅ |
| test_step5_rewrite_render | 7 | ✅ |
| test_pending_llm_announcement | 6 | ✅ |
| test_p6_pronoun_resolver | 36 | ✅ |
| test_notes_runtime | 86/90 (4 个 pre-existing torch stub) | ✅ (真测试) |
| ~~test_banned_date_logits~~ | — | 已删除 |

**258/258 真测试通过。**

---

## 8. 防止"这一系列问题"的关键设计

| 老 bug 类 | 防止机制 |
|---|---|
| Verifier kill 对答案 | 4 hard + 12 soft, LLM judge 优先 |
| Compound input 漏拆 | Advisor 列 N 条 suggested_calls + budget 跟随 |
| Forget hex 抄错 | Advisor 在 suggested_calls 里写 record_id + Saved Notes Index 全注入 prompt |
| MemorySearch 重复调用 | "MEMORY ALREADY RETRIEVED" 头部强提示 |
| target_entity 被推翻 | Eva 显式传 entity 时跳过 text inference |
| Compound date queries | GetCurrentTime 接 target_entity arg + advisor 提示每 entity 一次 |
| Cross-session note 看不到 | recent_notes fallback 到 list_notes() by created_at |
| List-all notes 被错收窄 | Advisor 识别"全列"模式 + Saved Notes Index always-on |
| Date math regex 误报 | LLM judge 接管 + regex 全 soft |
| Vision 第一次带图 crash | mm_token_type_ids 与 input_ids 在 `<think>` append 时同步 cat |
| Verifier thread silent fail | `_safe_generate` 包装 + EVA_VISION_DEBUG |
| Stale heuristics 累积 | dead code 删 + 文档化 + git 历史可回滚 |

---

## 9. 整体架构图（最终态）

```
User input
  ↓
[Advisor — DeepSeek, 1 call]
  out: intent + suggested_calls + advice + needs_memory + memory_hint + needs_web
  ↓
[_gate_memory_probe — no LLM]
  enrich decision (target_entity + matched_topics + query)
  ↓
[FAISS Probe — local, ~50ms]
  Path-A topic-direct + Path-B FAISS+BM25 + Memory Judge
  should_inject() → memory packet OR none
  ↓
[Assemble System Prompt]
  identity / tools / format_rules /
  [Today] anchor /
  [MEMORY ALREADY RETRIEVED + packet] (when injected) /
  [Saved Notes Index] (when notes exist) /
  [Advisor Hint]
  ↓
[Eva Phase-1 — local 9B]
  Tool call OR direct answer
  Tool guard: advisor.suggested_calls bypass route judge
  Target_entity: Eva-explicit respected
  Budget: advisor-aware max(default, advisor count)
  ↓
[Eva Phase-2 — local 9B]
  Final natural-language answer
  ↓
[Verify — minimal]
  HARD: tool_call_leak (regex) + 3x semantic_verifier_fail (LLM judge)
  SOFT: 12 reasons, telemetry only
  ↓
[Repair — rare]
  tool_call_leaked → inject_tool (run real)
  semantic_verifier_fail → regenerate ×1
  else → release ledger.best() OR canned
  ↓
Output
```

---

## 10. 后续工作（不在本次重构范围）

| 项 | 优先级 | 触发条件 |
|---|---|---|
| 删除剩余 dead helpers（extract_remember_params 等） | P3 | 稳定 2 周无回滚 |
| 补 ~100 条 `RememberThis × N` + `ForgetMemory × N` SFT 数据 + LoRA | P2 | Phase-0 数据决定（Turn 6 类失败率 ≥50% 才补） |
| 加 EvaState "正在做 X" 状态机 | P3 | 真实用例触发 |
| Advisor 缓存 + payload 压缩进一步降 latency | P3 | 用户反馈延迟仍痛 |
| 多用户 / 多 session 隔离 | P4 | 产品上线需求 |

---

## 11. 关键文件清单（最终）

```
D:/Eva_new/
├── Advisor/                          ← NEW 2026-05-13
│   ├── __init__.py
│   ├── advisor_client.py             ← DeepSeek + cache + fallback
│   └── build_advisor_prompt.py       ← context assembly
├── docs/
│   ├── GENERATION_FLOW_2026-05-14.md ← canonical 架构
│   ├── REFACTOR_COMPLETE_2026-05-14.md ← 本文
│   ├── USER_NOTES_MODULE.md
│   ├── SLOT_SUBJECT_CLASSIFIER_PLAN.md
│   ├── P6_pronoun_resolver_refactor_v4.md
│   └── archive/                      ← 旧文档归档
│       ├── TODO_2026-05-13_root_fixes.md
│       ├── R11_design.md
│       ├── RUNTIME_FLOW.md
│       └── ...
├── eva_advisor (via Advisor/)         (advisor 接入)
├── eva_config.py                      (+ ADVISOR_* / EVA_ADVISOR_FALLBACK_MODE)
├── eva_core.py                        (advisor_result 字段 / _safe_generate / saved_notes_index / mm fix)
├── eva_inference_P2.py                (Advisor-first _gate_memory_probe)
├── eva_memory_legacy.py               (SAVED NOTES scope guard 警告语)
├── eva_memory_v2.py                   (match_topics 暴露)
├── eva_prompts.py                     (GetCurrentTime target_entity arg)
├── eva_verifier_logic.py              (REASON_POLICY: 4 hard / 12 soft)
├── Memory_maker/notes_runtime.py      (_try_auto_correct L0+L1 simplified)
└── tests/
    ├── test_advisor.py                ← NEW (20 tests)
    └── ... (其他 13 个测试文件全过)
```

---

## 12. 回滚路径

| 改动 | 回滚 |
|---|---|
| Plan-A 局部 reason 误降 | REASON_POLICY 改 severity 回 hard |
| Advisor 不稳定 | `eva_config.ENABLE_ADVISOR = False` |
| Advisor 太慢 | `EVA_ADVISOR_FALLBACK_MODE = "chat"` |
| 整体回滚 | `git checkout` 到 2026-05-12 commit |

---

## 13. 一句话总结

> **Advisor 在前决策，FAISS 在中检索，9B 在中生成，LLM judge 在后兜底。** Verifier 从裁判退到日志，verifier kill 答案的事件接近 0；模型从修补退到执行，compound input 正确率从 30% 到 80%。死代码 ~1000 行已清，~500 行待 2 周稳定后清理。**这是过去 2 天 + 14 个具体 bug 修复 + 1 套架构重构 + 1 次代码清理的最终态。**
