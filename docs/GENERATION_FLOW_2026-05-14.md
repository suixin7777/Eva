# Eva 生成逻辑最终架构 (2026-05-14)

本文档定义经过 Plan-A 重构后 Eva 的完整推理流程。所有
post-2026-05-14 的代码改动应当对照本文档检查。

---

## 1. 核心设计原则

| 原则 | 含义 |
|---|---|
| **Advisor-first** | 决策前置——远端 DeepSeek (advisor) 在每轮开头分类意图 + 列出该调的工具 + 给 Eva 自然语言提示 |
| **Single source of truth** | advisor.suggested_calls 是 verifier / runtime / 用户 看的同一份契约 |
| **Verifier-as-monitor** | 仅 2 类 hard：格式安全 (tool_call_leak) + LLM judge (semantic verifier)。其余全 soft telemetry |
| **No redundant tool calls** | PRE PROBE 注入后 prompt 顶部强提示"已搜，别再搜" |
| **Trust explicit args** | Eva 显式传的 `target_entity` 不被 text inference 推翻 |

---

## 2. 单轮完整流程

```
┌──────────────────────────────────────────────────────────────────┐
│ User input                                                        │
└──────────────────────────────┬───────────────────────────────────┘
                               │
        ┌──────────────────────▼────────────────────────┐
        │ Step 0: ADVISOR (1 DeepSeek call, ~1-8s)      │
        │   in:  user_text + history (3 turns) +       │
        │        recent_notes + NotesStore.list_notes  │
        │   out: {advice, intent, needs_memory,         │
        │         memory_hint, needs_web, web_hint,     │
        │         suggested_calls}                      │
        │   fail: fallback to old judge flow (judges)   │
        └──────────────────────┬────────────────────────┘
                               │
        ┌──────────────────────▼────────────────────────┐
        │ Step 1: GATE MEMORY PROBE (no LLM)            │
        │   advisor.intent ∈ {chat, query_external}     │
        │     ∧ ¬needs_memory → skip probe              │
        │   else → build decision:                      │
        │     target_entity = advisor.suggested_calls   │
        │                   [MemorySearch].target_entity│
        │                   ∨ TopicDict.subject_hint    │
        │                   ∨ "Both"                    │
        │     matched_topics = TopicDict.match(hint)    │
        │     query = advisor.memory_query_hint         │
        │             ∨ user_text                       │
        └──────────────────────┬────────────────────────┘
                               │
        ┌──────────────────────▼────────────────────────┐
        │ Step 2: FAISS PROBE (no LLM, ~50ms)           │
        │   Path-A: pull records by matched_topics      │
        │   Path-B: FAISS+BM25 by query                 │
        │   Memory Judge: EXACT/RELATED labeling        │
        │   should_inject(): decide if results worth    │
        │                    showing the model          │
        └──────────────────────┬────────────────────────┘
                               │
        ┌──────────────────────▼────────────────────────┐
        │ Step 3: ASSEMBLE SYSTEM PROMPT                │
        │   identity / tools / format_rules /           │
        │   [Today] anchor /                            │
        │   [MEMORY ALREADY RETRIEVED] header +         │
        │     active_memory_context (when injected) /   │
        │   [Eva's Saved Notes Index]                   │
        │     (always when NotesStore has live notes) / │
        │   [Advisor Hint] block                        │
        └──────────────────────┬────────────────────────┘
                               │
        ┌──────────────────────▼────────────────────────┐
        │ Step 4: PHASE-1 GENERATION (local 9B)         │
        │   Decision: tool_code OR direct answer        │
        │   ReAct loop: max MAX_STEPS=8                 │
        │   Tool guard (_guard_tool_call):              │
        │     advisor.suggested_calls ⊇ {tool}          │
        │       → bypass route judge                    │
        │     else → route judge classify, guard route  │
        │   Eva-explicit target_entity respected        │
        └──────────────────────┬────────────────────────┘
                               │
        ┌──────────────────────▼────────────────────────┐
        │ Step 5: PHASE-2 GENERATION (local 9B)         │
        │   Generate final natural-language answer      │
        │   Sampling mode: direct / after_tool /        │
        │                  after_memory                 │
        └──────────────────────┬────────────────────────┘
                               │
        ┌──────────────────────▼────────────────────────┐
        │ Step 6: VERIFY (minimal)                      │
        │   HARD reasons (4):                           │
        │     - tool_call_leaked_in_answer (regex)      │
        │     - semantic_verifier_fail:                 │
        │       pronoun_referent_mismatch (LLM)         │
        │     - semantic_verifier_fail:                 │
        │       internal_self_contradiction (LLM)       │
        │     - semantic_verifier_fail:                 │
        │       fact_conflict_with_evidence (LLM)       │
        │   SOFT reasons (12): telemetry log only       │
        └──────────────────────┬────────────────────────┘
                               │
        ┌──────────────────────▼────────────────────────┐
        │ Step 7: REPAIR (rare)                         │
        │   tool_call_leaked → inject_tool (run real)   │
        │   semantic_verifier_fail → regenerate ×1      │
        │   fail → release ledger.best() OR canned      │
        └──────────────────────┬────────────────────────┘
                               │
        ┌──────────────────────▼────────────────────────┐
        │ Output to user                                │
        └──────────────────────────────────────────────┘
```

---

## 3. 每层的职责边界

### Advisor (远端 DeepSeek)
- **该做**：分类 intent，给具体工具建议，写 1-3 句给 Eva 的提示
- **不该做**：不写 tool call syntax；不替 Eva 生成最终回复

### Local 9B Model (Eva)
- **该做**：理解 user，按 advisor 提示选工具/调工具，生成 tsundere 回答
- **不该做**：不重复 advisor 已经做的意图分类工作

### Runtime (Python)
- **该做**：
  - PRE PROBE 把 advisor signal 转成完整 decision (target/topics/query)
  - 工具调用参数校验 + budget enforcement
  - Saved Notes Index 注入到 prompt
  - tool_call_leak 的确定性修复
- **不该做**：不做意图分类（advisor 的活）；不做语义判断（LLM judge 的活）

### Verifier (regex)
- **该做**：tool_call_leak 检测（纯格式）
- **不该做**：其他都不做（已退役）

### Semantic Verifier (LLM judge)
- **该做**：检测真矛盾、真幻觉、跨实体归因错
- **不该做**：不做意图分类；不做工具调用监督

---

## 4. 信号流转契约

| 信号 | 产生者 | 消费者 | 含义 |
|---|---|---|---|
| `advisor.intent` | DeepSeek | log + 调试 | 粗分类摘要 |
| `advisor.suggested_calls[].tool` | DeepSeek | runtime guard + verifier | **权威**——该调哪些工具 |
| `advisor.suggested_calls[].args.target_entity` | DeepSeek | gate_memory_probe | 显式实体路由 |
| `advisor.memory_query_hint` | DeepSeek | gate_memory_probe | FAISS 查询关键词 (1-4 词) |
| `advisor.advice` | DeepSeek | Eva prompt | 自然语言指引 |
| `active_memory_context` | runtime FAISS | Eva prompt | 已检索的事实 (顶部带 ALREADY RETRIEVED 提示) |
| `saved_notes_index` | runtime NotesStore | Eva prompt | 所有 live notes 摘要 |
| `tool_params.target_entity` | Eva (local) | runtime guard | 显式指定的搜索对象 (优先级 > 文本推断) |

---

## 5. Verifier 最终形态

### 4 个 HARD reason

```python
{
  "tool_call_leaked_in_answer":              # 格式：模型把 tool syntax 输出
      hard + inject_tool   (run real tool from leaked syntax)
  "semantic_verifier_fail:pronoun_referent_mismatch":
      hard + regenerate    (LLM 判断错归因)
  "semantic_verifier_fail:internal_self_contradiction":
      hard + regenerate    (LLM 判断真自相矛盾)
  "semantic_verifier_fail:fact_conflict_with_evidence":
      hard + regenerate    (LLM 判断真幻觉违背 evidence)
}
```

### 12 个 SOFT reason (telemetry only)

```python
{
  # Advisor 接管的工具调用监督 (4)
  "missing_web_evidence_for_external_or_current_request",
  "missing_memorysearch_for_explicit_memory_check",
  "explicit_remember_request_not_handled",
  "explicit_forget_request_not_handled",

  # Advisor + LLM judge 接管的内容验证 (5)
  "missing_date_calculation_evidence",
  "date_math_target_date_mismatch",
  "date_math_days_not_supported_by_calculation_evidence",
  "unsupported_specifics_under_no_elaboration_rule",
  "unsupported_exact_toy_claim",

  # LLM judge 接管的语义 (3，pre-Plan-A 时是 regex)
  "eva_self_birthday_pronoun_mismatch",
  "toy_value_conflicts_with_exact_memory",
  "textgen_perspective_mismatch",
}
```

---

## 6. 关键安全不变量

| Invariant | 检查点 | 说明 |
|---|---|---|
| NotesStore 写操作幂等 | NotesStore.add | 失败不污染 index |
| 软删保留 7 天 | NotesStore.tombstone | 误删可恢复 |
| Audit log 全保留 | NotesStore.audit_path | 所有 add/delete/compact 留痕 |
| tool_call leak 必修 | verifier hard | 防 raw syntax 漏给用户 |
| Advisor 失败不挂 | _run_advisor try/except | fallback_mode={judges,chat,strict} 三档兜底 |
| target_entity 显式优先 | _guard_memorysearch_params | Eva 写 target=X 不被推翻 |
| 算术 binding 单实体 | _maybe_compute_date_delta_from_memory | compound 需 GetCurrentTime × N |

---

## 7. 已删除的子系统

| 子系统 | 删除日期 | 原因 |
|---|---|---|
| `inject_tool` 分支 in verifier dispatch (5个) | 2026-05-14 | Plan-A: hard_reasons filter 之后只剩 tool_call_leak |
| `_self_validate_date_calculation` invocation in safe_fallback | 2026-05-14 | missing_date_calc 已 soft，永不进 safe_fallback |
| `ENABLE_LEGACY_SEMANTIC_REGEX` gated 3 个 regex 检查 | 2026-05-14 | LLM judge 接管 |
| R-2.1 P1/P2/P3 in _try_auto_correct | 2026-05-13 | Advisor 写正确 record_id 进 hint |
| `_R21_*` regex helpers + `_is_anaphoric_forget_intent` + `_infer_event_type_and_topic` | 2026-05-14 | 同上，对应的多层 fallback 不再需要 |
| BannedDateLogitsProcessor invocation 点 | 2026-05-13 | Advisor 写明 perspective，cross-entity 日期污染消失 |

---

## 8. 保留作回滚保险的 dead code

| 代码 | 位置 | 触发条件 |
|---|---|---|
| `extract_remember_params_from_user_text` | eva_verifier_logic L533 | 测试 source-level 检查 + 未来 explicit_remember 复活 |
| `find_recent_note_id` | eva_verifier_logic L609 | 同上 (explicit_forget) |
| `build_required_web_query` | eva_verifier_logic L1103 | 仍被 execute_controller_tool 的 WebSearch 缺 query fallback 引用 |
| `build_required_memory_params` | eva_verifier_logic L1325 | 测试 source-level 检查 |
| `BannedDateLogitsProcessor` class | eva_core L220 | 测试 + `if 0:` 守卫，不调用 |
| `_build_banned_date_token_seqs` | eva_core L1453 | 同上 |
| `_self_validate_date_calculation` 函数体 | eva_verifier_logic L1958 | 函数保留但不被调用 |

**清理时机**：稳定运行 2 周无回滚需求后可全删。

---

## 9. 回滚路径

| 改动 | 回滚方法 |
|---|---|
| 单个 reason 误降 soft | 在 REASON_POLICY 改 `severity: "hard"` |
| Advisor 整体失败 | `eva_config.ENABLE_ADVISOR = False` → 走老 judges 流 |
| Advisor 太慢 | `eva_config.EVA_ADVISOR_FALLBACK_MODE = "chat"` → 失败时直接 chat |
| 整个 advisor-first 架构回滚 | 把 `_new_refresh_active_memory` 里 advisor 部分注释掉，强制走 fallback 路径 |
| Plan-A 全部回滚 | 改 REASON_POLICY 把 5 个降级 reason 改回 hard + 复活 dead helpers |

---

## 10. 性能指标 (期待)

| 指标 | Pre-Advisor | Post-Plan-A |
|---|---|---|
| 单轮 DeepSeek 调用次数 | 6-9 | **2** (advisor + semantic verifier) |
| 单轮总延迟 (中位) | 12-18s | **6-9s** |
| Verifier kill 答案率 | 每 3-5 turn 一次 | **≤1%** (只剩真硬错) |
| Compound input 正确率 | ~30% (Turn 6 bug) | **~80%** (advisor 拆分 + N tool calls) |
| Forget 正确 record_id 率 | ~60% (hex 抄错) | **~95%** (advisor 写明 + saved notes index) |

---

## 11. 已知边界

1. **Advisor 超时**: 25s 内不返回 → fallback。最坏路径 = advisor 25s + Eva 生成 5s = 30s
2. **NotesStore > 10 live notes**: 索引 cap 在 10，超过的需要 MemorySearch
3. **Advisor 模型理解错**: rare，semantic verifier 兜底
4. **跨 session 的旧 note 引用**: 已修复（recent_notes fallback 到 list_notes）
5. **Compound date queries with > 2 entities**: 需要 GetCurrentTime × N，advisor prompt 已强调

---

## 12. 文件级别的代码量变化

| 文件 | Pre-Advisor | Post-Plan-A | 变化 |
|---|---|---|---|
| `eva_verifier_logic.py` | ~2650 | ~2480 | -170 (inject_tool 分支 + legacy regex) |
| `Memory_maker/notes_runtime.py` | ~1245 | ~1180 | -65 (R-2.1 helpers) |
| `Advisor/*.py` | 0 | ~560 | +560 (新模块) |
| 测试新增 | 0 | ~230 (test_advisor.py 20 tests) | +230 |
| **净** | | | **+555** (新增 advisor 模块) |

---

## 13. 一句话总结

> **Advisor 在前决策，FAISS 在中检索，9B 在中生成，LLM judge 在后兜底。**
> 每一层做它擅长的事，verifier 从裁判退到日志，模型从修补退到执行。
