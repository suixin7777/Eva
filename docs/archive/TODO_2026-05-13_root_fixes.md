# TODO — 根本修复路线图（2026-05-13 测试复盘衍生）

## 背景

2026-05-13 的 13 轮 inference 测试暴露了一批问题；当天提交了 7 项补丁
（见 git log `2026-05-13 fixes` 那一组）。补丁解决了表层症状，但其中
5 项本质是"再加一条 if / 再加一条正则"——下一种 lore 措辞、下一种用户
表达方式仍会复发。本文档记录每一项的**根本方案**，按 ROI 排序，作为
后续重构基线。

立场：现有补丁先留着保证回归不裂；根本方案逐项替换，替换后**主动撤补丁**，
让设计意图回到代码里，避免补丁层和根本层并存。

---

## 优先级总览

| # | 主题 | 当前补丁 | 根本方案 | 规模 | 推荐次序 |
|---|------|---------|---------|------|---------|
| R-1 | toy / slot 抽取 | 加正则动词族 | build-time slot extraction → `meta.slot_values` | 中 | 第 4 |
| R-2 | ForgetMemory record_id 幻觉 | 注入 Active Notes 到 prompt | `ForgetMemory(query=...)` 工具签名升级 | **小** | **第 2** |
| R-3 | fallback 释放 original vs regen | 改一行参数 | Verdict Ledger + best-of-N + disagreement | 中 | 第 5 |
| R-4 | verifier 不信 PRE PROBE | 加两条 skip | **Evidence Ledger 单一证据源** | 中大 | **第 1** |
| R-5 | 跨实体污染 | prompt SCOPE LOCK 文字 | logits processor / JSON-first structured output | 大 | 第 6 |
| R-6 | pronoun-followup target 漂移 | 读 `last_memory_target_entity` | `agent.dialog_focus` 统一 dialog state | 中 | 第 3 |
| R-7 | RememberThis 相对日期 | content 字符串改写 | RememberThis 增结构化字段 `event_date / event_type` | 中 | 第 7 |
| R-8 | topic_keywords.json 演化 | 单层字典 | 升级为 ledger 数据源 + LLM 反哺词典 | 中 | 与 R-4 同步 |

---

## R-1 build-time slot extraction

### 问题
`eva_slots._extract_toy_value_from_text` 用正则匹配 lore prose。一旦 lore 用
新动词形态（"has always been"、"belonged to her since"、"used to be"），
slot 抽取就漏。test Turn 8/10 暴露：PRE PROBE 把"Eva's favorite toy has
always been a cuddly bunny..."判为 EXACT，但 MemorySearch 跑同一条 record
返回 `toy: MISSING`，最终 Eva 否认自己有玩具。

### 当前补丁
`_extract_toy_value_from_text` 把动词扩展成 `(?:was|is|:|has\s+(?:always\s+)?been|had\s+been|used\s+to\s+be)`。

### 根本方案
slot value 在 **build-time** 写进 record meta，不在 inference-time 跑正则。

**改动点**：
1. `Memory_maker/rewrite_memory.py` 已经在离线阶段对每条 lore 跑 LLM。
   让那一步追加 schema 字段：

   ```jsonc
   {
     "vector_text": "...",
     "content": "...",
     "meta": {
       "entity": "Eva",
       "topic": "Toy",
       "slot_values": {           // ← 新增
         "toy": "cuddly bunny"
       }
     }
   }
   ```

2. `eva_slots._extract_slot_value_from_record` 改成读取顺序：
   ```python
   meta_slot = record.get("meta", {}).get("slot_values", {}).get(slot)
   if meta_slot: return meta_slot
   # 正则 fallback（保留，覆盖 user notes 等没跑过 build-time 的源）
   return _legacy_regex_extract(...)
   ```

3. 一次性迁移脚本：`generate/migrate_slot_values.py` 重跑全库一次，
   产出新版 jsonl。旧版作 `.bak`。

### 验证
- 现有 8 条 toy / birthday / age / full_name lore 全部从 `meta.slot_values`
  命中，正则路径零调用（在 `slot_values_extract_count` metric 上对账）。
- 加一条 prose 形如 "Her toy of choice has always been a tin soldier" 入库，
  确保 build-time 抽对，inference-time 不跑正则。

### 撤补丁
完成后 `eva_slots._extract_toy_value_from_text` 的扩展动词族可以撤掉，
保留极小的正则兜底（只用于 notes）。

---

## R-2 ForgetMemory 改成 query 形式 ★ 最高 ROI

### 问题
ForgetMemory 当前签名要求 `record_id="7a4a68da"`。模型必须把 8-char hex
正确抄回——test Turn 13 直接幻觉了 `70374433`，靠 verifier 兜底才修对。

### 当前补丁
`_build_system_prompt` 末尾注入 `[Active Notes — this session]` 列出
最近 5 条 note_id。模型抄概率提高，但仍依赖"抄对"。会话长了 prompt 膨胀。

### 根本方案
工具签名升级，让 runtime 而不是 model 负责对账。

**新签名**（向后兼容）：
```python
ForgetMemory(
    query: str = "",        # 自然语言描述要删的事
    record_id: str = "",    # 仍支持（确定性删法）
    topic: str = "",        # 可选过滤
    confirm: bool = True,   # 多候选时设 False 触发 disambiguation
)
```

**runtime 行为**：
1. 若 `record_id` 给了 → 直接 tombstone（不变）。
2. 若 `query` 给了 → `NotesStore.search(query, top_k=3)`：
   - 1 个候选 + score ≥ 阈值 → 直接 tombstone，返回 `[FORGOTTEN] ...`。
   - 多个候选 → 返回 `[FORGET DISAMBIGUATE] Found N notes matching query. Specify record_id or refine topic:` + 列出候选。
   - 0 个候选 → 返回 `[FORGET NOT FOUND] No live note matches query 'X'.`。
3. `topic` 给了 → 二次过滤。

**Tool prompt 改动**（`eva_prompts.TOOLS_OPTIMIZED_NOTES_APPENDIX`）：
- 把 ForgetMemory 的"必须给 record_id"改成"通常用 query，record_id 仅当显式列出时使用"。

### 验证
- Turn 13 场景："the meeting is canceled, forget it" → 模型调
  `ForgetMemory(query="meeting next monday")` → 直接命中唯一 note → 一次成功。
- 多候选 disambiguation：连续 RememberThis 三条 meeting 类 note，
  `ForgetMemory(query="the meeting")` 应返回 3 候选要求消岐。
- 老路径：模型仍写 `record_id="..."` 时维持原行为。

### 撤补丁
完成后 `_build_system_prompt` 里的 `[Active Notes — this session]` 注入
可以撤掉。

---

## R-3 Verdict Ledger + best-of-N + disagreement

### 问题
- Turn 5 original phase-2 `"55 days until my birthday"` 被 regex verifier 判
  `date_math_target_date_mismatch`（**误判**——55 实际是正确的）。
- 触发 regen → regen 输出 `"my birthday is way later"` 被 LLM-judge 判
  `pronoun_referent_mismatch`（**真错**）。
- Fallback 释放了"latest"的 regen，丢了原始正确的 55-days 信号。

### 当前补丁
fallback 调用点把 `phase2_answer=new_final_answer` 改成 `phase2_answer=final_answer`。
仍是二选一，没有比较能力。

### 根本方案
verifier 流水线升级为 **verdict ledger**：

```python
@dataclass
class Verdict:
    answer: str
    regex_issues: list[str]         # 来自 _verify_final_answer regex 检查
    semantic_issues: list[Issue]    # 来自 SemanticVerifier
    severity: int                   # 0=clean, 1=soft, 2=hard
    source_stage: str               # "original" | "regen_1" | "regen_2"

class VerdictLedger:
    candidates: list[Verdict]

    def best(self) -> Verdict:
        # severity 最低；同 severity 时偏好较早 source_stage
        ...

    def has_disagreement(self) -> bool:
        # regex 说原始 fail、LLM 说 regen fail → 双方互证不可靠
        ...
```

**Fallback 策略**：
- `best().severity == 0` → 直接释放。
- `best().severity == 1` → 释放 + 标 soft warning。
- `best().severity == 2` 且 `has_disagreement()` → 走 canned（双判都不可信）。
- `best().severity == 2` 一致认定 → 走 reason-specific canned。

### 验证
- Turn 5 复盘：original 进 ledger（severity=2 regex），regen 进 ledger
  （severity=2 semantic）。两路 reason 不同 → disagreement → 走 canned，
  而不是释放任何一边的坏答案。
- 新增单元测试 `test_verdict_ledger.py`，覆盖 disagreement / severity tiebreak。

### 撤补丁
完成后 `eva_core.py` 那段 `phase2_answer=final_answer` 的传参改回工程语义
自洽（每个 candidate 自带 source_stage 标签），不再依赖"传哪个"决定释放哪个。

---

## R-4 Evidence Ledger ★ 受益面最大

### 问题
verifier 检查"是否需要 MemorySearch"时只看 `_current_turn_has_memorysearch_evidence()`，
看不到 PRE PROBE 已经注入的同等证据。结果：
- Turn 10 PRE PROBE 已 inject EXACT toy lore，verifier 仍要求 MemorySearch，
  跑出来反而把同一条 record 判 RELATED，覆盖正确结论。
- Turn 11 "help me remember something" 是 setup 句，verifier 强行注入
  MemorySearch，把无关旧话题塞进回答。

### 当前补丁
verifier 加两条 skip：① `current_turn_memory_has_exact=True`；
② `_is_setup_remember_phrasing()` 命中。每加一种漏判场景就要再加一条 skip。

### 根本方案
所有证据落到**统一 ledger**，verifier 读 ledger 而不是 tool history。

```python
@dataclass
class Evidence:
    source: str        # "topic_dict" | "pre_probe" | "memory_search" | "remember_this" | "date_calc"
    subject: str       # "Eva" | "Rosm" | "Shared"
    slot: str = ""     # "toy" / "birthday" / "" (topic-only)
    topic: str = ""    # 来自 topic_keywords 或 record meta
    value: str = ""    # 抽出的具体 value（slot 有值时填）
    judge_tier: str    # "exact" | "related" | "topic" | "missing"
    confidence: float
    record_ref: str = ""  # 可追溯的 record_id / note_id

class TurnEvidenceLedger:
    items: list[Evidence]

    def covers(self, subject: str, slot: str = "", topic: str = "") -> bool:
        """verifier 用：本轮是否已有这条证据，不管来源是 PRE PROBE 还是 tool。"""
        ...

    def best_for(self, subject, slot) -> Evidence | None:
        """phase-2 用：取 judge_tier 最强的一条。"""
```

**写入点统一**：
- `MemoryModule.probe` 命中 topic → 写 1 条 `source="topic_dict"`。
- PRE PROBE EXACT 注入 → 写 1 条 `source="pre_probe" judge_tier="exact"`。
- MemorySearch tool 跑完 → 每条 returned record 写 1 条。
- RememberThis / ForgetMemory → 写 / tombstone 对应 evidence。
- DATE CALCULATION BINDING → 写 1 条 `source="date_calc" slot="birthday"`。

**verifier 改写**：
```python
# 旧：
if not agent._current_turn_has_memorysearch_evidence():
    reasons.append("missing_memorysearch_for_explicit_memory_check")

# 新：
asked_slots = extract_memory_slots(latest_user_text)
target = infer_target(latest_user_text)
missing = [s for s in asked_slots if not ledger.covers(target, slot=s)]
if missing and explicit_check:
    reasons.append("missing_memorysearch_for_explicit_memory_check")
```

### 验证
- Turn 10：PRE PROBE 写 EXACT evidence → ledger.covers(Eva, "toy") == True →
  verifier 不再注入 MemorySearch。
- Turn 11：没有 slot 被问 → ledger 无需 cover 任何东西 → 不触发。
- 加单元测试 `test_evidence_ledger.py`：模拟各 source 写入 + verifier 读取。

### 撤补丁
- 撤 `eva_verifier_logic._is_setup_remember_phrasing()` 全部代码。
- 撤 `missing_memorysearch_for_explicit_memory_check` 中的
  `already_has_pre_probe_exact` skip 条件。
- 这是这次 7 个补丁中**收益最广**的根本化——单点撬动 3 个补丁失效。

---

## R-5 跨实体污染 → constrained decoding / structured output

### 问题
Turn 5 用户问 "how many days until **your** birthday?"，DATE BINDING 把
`target_entity=Eva` 钉死，phase-2 仍输出 "Your birthday is November 25th,
and mine is July 7th, so 55 days"——Rosm 的日期混进了 Eva-only 回答。

### 当前补丁
DATE BINDING 加 `[ANSWER SCOPE]: do NOT mention any other person's date`。
prompt 文字约束，模型可以忽略，verifier 才能事后 catch。

### 根本方案 A（短期，logits processor）
phase-2 生成阶段挂一个 `LogitsProcessor`：当 binding 里
`target_entity` 单一时，把另一实体的 birthday token 序列（如
"November 25", "Nov. 25", "11/25"）做强负 bias（-inf 或 -10）。

实现：
- 离线把 lore 里出现的 birthday / date 字面值索引成
  `{entity: [token_sequences]}`。
- DATE BINDING 注入时连带把 `banned_token_sequences` 传给 generator。
- 现有 `_run_phase2_sample` 接收这个参数，构造 transformers
  `LogitsProcessorList`。

### 根本方案 B（长期，JSON-first phase-2）
当本轮有 DATE BINDING / SLOT EVIDENCE 时，phase-2 先输出 JSON：
```json
{"target_entity": "Eva", "slot": "birthday", "days_until": 55, "date_text": "July 7"}
```
再用 template 渲染成对话：
```
"{persona_opener}, my birthday is {date_text}—{days_until} days away~"
```
template 阶段不可能写出"Your birthday is November 25th"，因为它根本没那个 slot。

### 验证
- 方案 A：Turn 5 token-level 验证 "November" 在 logits 里被压到底部，
  最终 sample 不会带这词。
- 方案 B：phase-2 JSON 阶段加 schema validation，failed schema 自动 regen。

### 撤补丁
- 方案 A 完成：DATE BINDING 里的 `[ANSWER SCOPE]` 文字降为冗余兜底。
- 方案 B 完成：连 verifier 的 `date_math_target_date_mismatch` regex 都能撤。

### 推荐次序
先 A 后 B。A 改动小（1 个 LogitsProcessor 类 + generator 接线）；
B 改动大（phase-2 模式重构）。

---

## R-6 dialog_focus 统一 dialog state

### 问题
"上一轮的实体" 这个信息现在散落在 3 处：
- `agent.last_memory_target_entity`（PRE PROBE 写）
- pronoun resolver 内部推断（每次重新算）
- `_infer_memory_target_from_text` 默认 `default_target="Both"`（散文推断）

Turn 9 "really? check it" 的 antecedents 是 toy，上一轮 target=Eva，
但 pronoun 路径默认 Both → Eva-only 的 bunny lore 被降级 RELATED。

### 当前补丁
`build_required_memory_params` 在 pronoun-followup 时显式读
`agent.last_memory_target_entity`，覆盖 `infer` 的 Both 默认值。

### 根本方案
统一 dialog state 对象：

```python
@dataclass
class DialogFocus:
    entity: str = ""            # 当前焦点实体
    slot: str = ""              # 当前焦点 slot（如有）
    topic: str = ""             # 当前焦点 topic
    set_at_turn: int = -1       # 触发轮次（为 stale detection）
    source: str = ""            # "user_named" | "pronoun_inherit" | "pre_probe"

class ChatAgent:
    dialog_focus: DialogFocus
```

**update 时机**：
- 用户句中显式提名实体 ("Rosm's...", "Eva's...") → `source="user_named"`。
- PRE PROBE 命中 single-entity subject_hint → `source="pre_probe"`。
- pronoun-followup 解析成功 → 继承上一轮 focus，`source="pronoun_inherit"`。

**读取点**：
- pronoun resolver 输入参数加 `current_focus`。
- `_infer_memory_target_from_text` 的 default_target 改为
  `current_focus.entity or "Both"`。
- DATE CALCULATION BINDING 的 `bound_entity` 改为读 focus（而不是
  `last_memory_target_entity`）。

### 验证
- Turn 8 → focus={entity: "Eva", slot: "toy"}。
- Turn 9 "really? check it" pronoun-followup → 继承 Eva → MemorySearch
  target=Eva → bunny lore 仍 EXACT，不降级。
- 用户显式 "what about Rosm's toy?" → focus 改为 Rosm，覆盖继承。

### 撤补丁
完成后 `build_required_memory_params` 里的 `inherited_target` 读取逻辑
就变成"读 focus"，统一接口，删掉 `last_memory_target_entity` 的散点读取。

---

## R-7 RememberThis 结构化 event schema

### 问题
"meeting next Monday, remember it" → 存进 `content: "Master has a
meeting next Monday."`。"next Monday" 一周后语义就漂了；也没法做
"列出本周所有 meeting"、"自动清理过期 event" 等查询。

### 当前补丁
入库前 `_resolve_relative_dates(content)` 把相对日期改写成
"next Monday (2026-05-18)" 字面叠在散文里。日期在字符串里，但仍是字符串。

### 根本方案
RememberThis 工具签名引入可选结构化字段：

```python
RememberThis(
    content: str,                    # 散文（保留人类可读）
    entity: str, topic: str, keywords: str,
    # ↓ 新增可选结构化 slot
    event_date: str = "",            # ISO8601 YYYY-MM-DD
    event_time: str = "",            # HH:MM
    event_type: str = "",            # "meeting" / "birthday" / "appointment" / ...
    participants: list[str] = None,
    expires_at: str = "",            # 过期后可自动 tombstone
)
```

**phase-1 抽取**：模型 thought 阶段（或额外 LLM judge）从 content 抽
`event_date / event_type / participants`，调用时填上。phase-1 漏填时，
runtime fallback 跑 `_resolve_relative_dates` 兜底。

**NotesStore meta** 增字段，新查询 API：
```python
NotesStore.search_by_date(date_range=("2026-05-13", "2026-05-20"))
NotesStore.search_by_event_type("meeting")
NotesStore.expire_stale(now=datetime.now())  # 自动清理过期 event
```

### 验证
- Turn 12 复盘：模型应该调
  `RememberThis(content="Master has a meeting next Monday", event_date="2026-05-18", event_type="meeting", participants=["Rosm"])`。
- query API：`search_by_date(("2026-05-18", "2026-05-18"))` 应返回此 note。
- `expire_stale(now=2026-05-19)` 应 tombstone 该 note。

### 撤补丁
完成后 `_resolve_relative_dates` 降级为 fallback（仅当模型没填 event_date
时跑），核心日期信息走结构化字段。

---

## R-8 topic_keywords.json 演化

### 当前状态
- `topic_keywords.json` 58 个 topic，每 topic 一份 alias 列表 + 可选
  `subject_hint`。
- `eva_memory_v2.TopicDictionary._compile_patterns` 编译 word-boundary regex。
- PRE PROBE 第一道筛：alias 命中 → 输出 topic 名 + subject_hint → 走
  layered LLM 复核。

### 问题
- alias 是手工维护的，每出一种新表达（"my favorite plushie when I was
  a kid"）就要人工补 alias，否则 PRE PROBE 直接 miss。
- 命中结果是裸 topic 名，丢了 alias_hit / confidence 等信息，下游想做
  细粒度判断（如"alias 是核心词还是边缘词"）就拿不到。

### 与其他根本方案的协同
- **与 R-4 合并**：TopicDictionary 命中应直接写一条 Evidence 到 ledger：
  ```python
  Evidence(
      source="topic_dict",
      subject=topic_dict.subject_hint(topic) or "",
      topic=topic,
      judge_tier="topic",
      confidence=0.7 if alias_is_core else 0.5,
      record_ref="",
  )
  ```
  verifier 直接读 ledger，topic-only 命中也算 evidence 的一种 tier。

- **LLM 反哺词典**：SemanticVerifier / layered LLM 那一层目前只对单轮
  做接受/否决。可以加一个**离线 aggregation**：把 "LLM judge 说 relevant
  但 keyword 未命中" 的样本累积起来，每月产出 alias 补丁建议
  `topic_keywords.suggested.json`，人工审一遍 merge。

### 改动点
1. `TopicDictionary.match` 返回 `list[TopicMatch]` 而不是 `list[str]`：
   ```python
   @dataclass
   class TopicMatch:
       topic: str
       alias_hit: str         # 实际命中的 alias 文本
       subject_hint: str | None
       is_core_alias: bool    # 是否落在 topic 的"核心词"区
   ```
2. `topic_keywords.json` schema 演化：每个 topic 允许标 `core_aliases`：
   ```jsonc
   "Toy": {
     "core_aliases": ["toy", "toys", "plushie", "stuffed animal"],
     "extended_aliases": ["bunny", "teddy", "have a toy"]
   }
   ```
   旧 schema（list[str]）作 fallback 兼容。
3. `_audit_topic_dict_suggestions.py`：离线脚本，扫
   `Notes/audit.log` + verifier shadow log，产出补 alias 建议。

### 验证
- 跑历史 trace 回放：所有"alias miss 但 LLM judge True"的样本进
  suggestion list；人工抽样确认 ≥80% 是合理的扩展。
- R-4 ledger 跑通后，verifier 看到 topic-only evidence 也能合理决策
  （比如"topic 命中但 slot MISSING" 时不应该再要求 MemorySearch）。

---

## 总落地次序

```
第 1: R-4 Evidence Ledger              ← 单点撬动 3 个补丁
第 2: R-2 ForgetMemory query           ← 改动小、用户感知直接
第 3: R-6 dialog_focus                 ← 消除散落 dialog state
第 4: R-1 build-time slot extraction   ← 一次成本，长期收益
第 5: R-3 Verdict Ledger               ← 接 R-4 的 ledger 范式
第 6: R-5 logits processor (短期版)    ← R-5 长期 JSON 版排在后面
第 7: R-7 event schema                 ← 配合 R-2 共同形成 notes 结构化
第 8: R-8 topic_keywords 升级          ← 与 R-4 同步做（共享 evidence 写入点）
```

R-4 是底座，做完后 R-3 / R-6 / R-8 都能复用 ledger 范式。
R-2 与 R-4 完全独立，可以并行做。

---

## 补丁与根本方案对照表（撤补丁清单）

完成根本方案后必须撤的补丁，避免双层并存：

| 根本方案完成 | 撤掉的补丁文件:行 |
|------------|-----------------|
| R-1 | `eva_slots._extract_toy_value_from_text` 的扩展动词族 |
| R-2 | `eva_core._build_system_prompt` 的 `[Active Notes — this session]` 注入块 |
| R-3 | `eva_core.py` regen-failed 调 fallback 处的 `phase2_answer=final_answer` 改回参数语义自洽形式 |
| R-4 | `eva_verifier_logic._is_setup_remember_phrasing` + `_SETUP_REMEMBER_RE` + `_SETUP_REMEMBER_ZH_RE`；`missing_memorysearch_for_explicit_memory_check` 中的 `already_has_pre_probe_exact` skip |
| R-5 | `eva_core._compute_date_binding` 的 `[ANSWER SCOPE]` 文字段 |
| R-6 | `eva_verifier_logic.build_required_memory_params` 的 `inherited_target` 读取逻辑 |
| R-7 | `Memory_maker/notes_runtime._resolve_relative_dates` 在 `execute_remember_this` 主路径里的调用（保留为 fallback） |

---

## 元元层观察

这次复盘最大收获不是"修了 7 个 bug"，而是发现 7 个 bug 里有 4 个共享
同一个底层缺陷：**evidence 没有统一表示，verifier / phase-2 / PRE PROBE
各自维护一份对"本轮已知什么"的理解**。

R-4 Evidence Ledger 是这一层的修复。做完它，未来再加新的证据源
（webSearch / image_caption / external_api）只需要写一条 Evidence，
所有下游模块自动看见。

类似地，R-6 dialog_focus 是 dialog state 那一层的统一；R-3 Verdict
Ledger 是 verifier verdict 那一层的统一。三个 Ledger 合起来覆盖
"什么是真的 / 我们在谈谁 / 我们对答案多有信心" 这三类核心运行时知识。

下一次有新的 inference bug 时，第一个问题应该是："这是哪个 Ledger
看不见 / 看错了？"——而不是"哪条 if 没盖到"。
