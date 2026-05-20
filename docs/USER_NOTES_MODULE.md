# User Notes Module — Production

> Eva 的运行时记忆能力：模型可通过 `RememberThis` / `ForgetMemory`
> 工具向 `Notes/` 写入或软删用户告诉它的事实。该模块作为 Eva 的
> 基础能力**默认启用**（`eva_config.ENABLE_USER_NOTES = True`），
> 与人工 curated 的 `Memory/` lore corpus 构成双层记忆体系。
>
> 状态：**production**（自 2026-05-11 起）。早期作为 test-memory 沙箱
> 开发，经 P1-P5.3 七阶段验证后正式合入并去掉 "test" 命名。

---

## 〇、关联文档

- 总体架构与运行时流：[RUNTIME_FLOW.md](RUNTIME_FLOW.md)
- 待执行的 slot 系统重构（独立工作）：[SLOT_SUBJECT_CLASSIFIER_PLAN.md](SLOT_SUBJECT_CLASSIFIER_PLAN.md)

---

## 一、双层记忆架构

| 层 | 性质 | 位置 | 写入方式 | 读取方式 |
|---|---|---|---|---|
| **Lore corpus** | 人工 curated 的 Eva 设定与历史 | `Memory/` | offline (`Memory_maker/rewrite_memory.py` → `Memory.py`) | runtime via `MemorySearch` |
| **User notes** | 用户在对话中告诉 Eva 的运行时事实 | `Notes/` | runtime via `RememberThis` 工具 | runtime via `MemorySearch`，独立段渲染 |

两层在 prompt 里的视觉区分：

- Lore 命中：`Record N [Lore] [Subject: Eva] [Topic: Toy]: ...`
- Note 命中：渲染在独立段 `>>> SAVED NOTES <<<` 内，`Note N [Note #abc12345] [Subject: Rosm] [Topic: Pet]: ...`

模型只能用 `ForgetMemory(record_id="<8-char>")` 删除 `[Note #...]` 类记录；lore corpus 不可由模型删除。

---

## 二、命名一致性（产线统一）

模型看到的所有字符串围绕单一术语 **"Note"**：

| 出现位置 | 字符串 |
|---|---|
| 检索段头 | `>>> SAVED NOTES (pass record_id to ForgetMemory to delete) <<<` |
| 段尾 | `>>> END SAVED NOTES <<<` |
| 每条 tag | `[Note #abc12345]` |
| RememberThis 输出 | `[REMEMBERED] Stored as Note #abc12345 ...` |
| ForgetMemory 输出 | `[FORGOTTEN] Note #abc12345 tombstoned` |
| Prompt discipline 段头 | `# Note-taking tools` |

内部命名同步：

| 实体 | 名称 |
|---|---|
| 模块文件 | `Memory_maker/notes_runtime.py` |
| 类 | `NotesStore` |
| 沙箱目录 | `Notes/`（项目根） |
| 文件名 | `notes.index` / `notes.jsonl` / `notes_content.json` / `notes_meta.json` / `audit.log` |
| `memory_state` key | `notes_store` |
| 检索段标记常量 | `_NOTES_BLOCK_MARKER = ">>> SAVED NOTES"` |
| 启发式提取 | `extract_remember_params_from_user_text` |
| Verifier helper | `find_recent_note_id` / `current_turn_has_remember_evidence` / `current_turn_has_forget_evidence` |

---

## 三、配置开关

[eva_config.py](../eva_config.py)：

```python
ENABLE_USER_NOTES = True               # 默认启用
NOTES_DIR = "Notes"
REMEMBER_TOOL_MAX_CALLS_PER_TURN = 1   # 每轮 RememberThis 上限
FORGET_TOOL_MAX_CALLS_PER_TURN = 1     # 每轮 ForgetMemory 上限
```

`ENABLE_USER_NOTES = False` 完全关闭：跳过 store 实例化、prompt
appendix、verifier 注入路径、所有 dispatch。生产路径回退到纯 lore-only
行为。

---

## 四、生命周期 API

### `NotesStore`（[Memory_maker/notes_runtime.py](../Memory_maker/notes_runtime.py)）

| 方法 | 用途 |
|---|---|
| `add(vector_text, content, entity, topic, keywords, ...)` | 写入新 note，返回 8-char hex `note_id` |
| `tombstone(note_id, reason)` | 软删；FAISS 索引不动，meta 标记 `deleted=True` |
| `search(query, top_k=20)` | 检索 live notes（自动过滤 tombstoned） |
| `compact()` | 物理删 tombstoned，从 JSONL 重建索引 |
| `discard()` | 清空所有数据；audit.log 改名归档 |
| `status()` / `list_notes(include_deleted=False)` | 调试 |

**已删除**：`promote_to_jsonl()`（test-memory 时代用于"沙箱→主库"提升；
production 化后无意义）。

### 模型可见工具（[eva_prompts.py:TOOLS_OPTIMIZED_NOTES_APPENDIX](../eva_prompts.py)）

```python
def RememberThis(content: str, entity: str, topic: str, keywords: str): ...
def ForgetMemory(record_id: str, reason: str): ...
```

工具签名 + discipline 段以 prompt appendix 形式追加到
`TOOLS_OPTIMIZED`，**仅在 `ENABLE_USER_NOTES=True` 时可见**。

---

## 五、Verifier 兜底链路

模型不调工具时由 verifier 注入：

| Reason | 触发条件 | 修复 |
|---|---|---|
| `explicit_remember_request_not_handled` | 用户明说"记一下"等 + 当前轮无 `[REMEMBERED]` | 启发式提取 params + 注入 RememberThis |
| `explicit_forget_request_not_handled` | 用户明说"忘掉"等 + 当前轮无 `[FORGOTTEN]` | history 扫 `[Note #...]` → 失败则 live-store search → 注入 ForgetMemory |

判别由 [eva_intent_judge.py](../eva_intent_judge.py) 的双层（regex + LLM）：
`PROMPT_EXPLICIT_REMEMBER` / `PROMPT_EXPLICIT_FORGET`。

写意图压制读意图——`explicit_remember=True` 时跳过
`missing_memorysearch_for_explicit_memory_check`，避免对 "remember this:"
误判为读检查触发冗余 MemorySearch。

---

## 六、retrieval 拼接

[eva_memory_legacy.py](../eva_memory_legacy.py)：

```
run_memory_search(...)
  ↓
_collect_memory_records (lore-corpus FAISS+BM25+CrossEncoder rerank)
  ↓
_attach_slot_evidence_to_collection (slot 抽取，仅 lore)
  ↓
if memory_state["notes_store"]:
    _attach_user_notes(...)        ← 独立 cosine 过滤 + bucket
  ↓
_format_memory_records_block (双 section 渲染)
```

**关键设计**：notes 不与 lore 抢渲染 cap——独立 `>>> SAVED NOTES <<<`
段，过 `_NOTES_MIN_COSINE = 0.25` 即出。lore 的 `record_cap=1/3` 不影响
note 显隐。

---

## 七、操作命令（Colab）

```python
import eva_inference_P2 as eva
agent = eva.build_agent()                          # 默认启用 notes
sess = ChatSession(agent, user_name="Rosm")

# 启动行会输出：
# [notes] Store ready at Notes/ — live=N deleted=M session_id=...

# 模型驱动 add / forget（默认走自然语言）：
sess.send("Eva, remember this: I just adopted a cat named Peach.")
sess.send("What was the name of that cat?")
sess.send("Actually, forget about the cat.")

# 程序化清空（保留 audit log）：
agent.memory_state["notes_store"].discard()

# 周期性物理删 tombstoned：
agent.memory_state["notes_store"].compact()

# 调试：
agent.memory_state["notes_store"].status()
agent.memory_state["notes_store"].list_notes()
```

---

## 八、代价

每轮在 notes 启用时新增：

- ~2 LLM judge 调用（`EXPLICIT_REMEMBER` + `EXPLICIT_FORGET`）共享
  `LLM_JUDGE_MAX_CALLS_PER_TURN=6` budget pool
- regex 命中时短路不消耗 LLM budget
- 工具调用本身 0 LLM 调用（启发式提取，本地 I/O）
- mpnet encode 复用 agent 主 encoder，无双载

---

## 九、单测

`tests/test_notes_runtime.py` —— **90 用例覆盖**：

- `NotesStore` 增/删/检索/persist/reload/compact/discard
- `_attach_user_notes` 与 `_format_memory_records_block` 集成（独立段渲染）
- `execute_remember_this` / `execute_forget_memory` dispatch 错误路径
- 端到端：remember → search → forget → 不再召回
- `find_recent_note_id` 双阶段（history scan + live-store fallback）
- `current_turn_has_*` evidence 检测
- `extract_remember_params_from_user_text` 启发式（中英双语）
- slot extractor pet/animal name 否决
- `memory_block_has_notes` 警告压制

```powershell
D:/Anaconda/envs/py310/python.exe tests/test_notes_runtime.py
```

不依赖 mpnet（用 SHA256-seeded fake encoder）。

---

## 十、已知遗留 debt

无。

历史：曾有 `MEMORY_SLOT_FIELDS["full_name"]` alias 过宽问题（bare `"name"`
触发打地鼠式否决——P1.7.2 加 places、P5.3 加 pets）。已于 2026-05-11
通过引入
[`eva_subject_classifier.is_person_subject`](../eva_subject_classifier.py)
+ `SLOT_APPLICABLE_SUBJECTS` 表治本——slot 检测从单元判定升级到 `(slot,
subject)` 二元判定，旧否决正则已全部删除。详见
[SLOT_SUBJECT_CLASSIFIER_PLAN.md](SLOT_SUBJECT_CLASSIFIER_PLAN.md)。

---

## 十一、变更历史

- 2026-05-10 起草沙箱方案（test-memory 阶段）
- 2026-05-11 通过 Colab 真模型 16 轮 fixture 验证（P1-P5.3）
- 2026-05-11 **production 合并**：去 "test" 命名 → 统一 "Note"；
  `ENABLE_USER_NOTES=True` 默认开；删 `promote_to_jsonl`；wipe 旧
  `Memory_test/` 目录；rename `TestMemoryStore`→`NotesStore`、
  `Memory_maker/memory_test_runtime.py`→`notes_runtime.py`、
  `tests/test_memory_test_runtime.py`→`test_notes_runtime.py`
- 2026-05-11 **slot subject classifier 治本**：引入
  [`eva_subject_classifier.py`](../eva_subject_classifier.py) +
  `SLOT_APPLICABLE_SUBJECTS` 表；删除旧 `full_name_blocked` 否决正则
  （P1.7.2 places + P5.3 pets）。slot 检测升级为 `(slot, subject)`
  二元判定，加新测试 42 用例（含 28-query 黄金 fixture，中英双语）。
