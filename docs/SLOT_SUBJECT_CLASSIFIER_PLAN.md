# Slot Subject Classifier — Implemented

> 起草：2026-05-11。**实施：2026-05-11。状态：已交付。**
>
> 把 `MEMORY_SLOT_FIELDS` 的检测从"slot_field 单元判定"升级到
> **(slot_field, target_subject) 二元判定**，从架构上消除了
> `_detect_requested_slot_fields` 里的"过宽 alias + 否决列表"打地鼠
> 模式。

## 实施摘要（2026-05-11）

- 新增模块 [`eva_subject_classifier.py`](../eva_subject_classifier.py) ——
  三层架构（regex strict / regex loose / embedding NN），中英双语
- 新增 `SLOT_APPLICABLE_SUBJECTS` 表（[`eva_config.py`](../eva_config.py)）
- 改写 [`eva_memory_legacy.py:_detect_requested_slot_fields`](../eva_memory_legacy.py)
  + [`eva_slots.py:extract_memory_slots`](../eva_slots.py) 走 subject 检查；
  全部丢弃旧的 P1.7.2 + P5.3 否决正则
- Thread encoder 经 4 处调用站（`run_memory_search` 两次 / `MemoryModule` /
  `ChatAgent._extract_memory_slots`）
- 新测试 [`tests/test_subject_classifier.py`](../tests/test_subject_classifier.py)
  ：42 用例 + 28-query 黄金 fixture（覆盖 Person / NonPerson / 中文 / 边界）
- 旧 `tests/test_notes_runtime.py:TestSlotDetectionPetSuppression` 8 用例
  契约保留——测试输入由新机制成功拒绝

**回归**：`test_notes_runtime` 90/90 OK；`test_p6_pronoun_resolver` 36/36 OK；
`test_no_elaboration_rule` 17/17 OK；`test_step5_rewrite_render` 7/7 OK。

下文保留作为设计依据 + 未来扩展参考。

---

## 一、问题陈述

[eva_memory_legacy.py:_detect_requested_slot_fields](../eva_memory_legacy.py:1084)
当前用过宽 alias 触发 + 否决列表抑制：

```python
MEMORY_SLOT_FIELDS = {
    "full_name": ["full name", "real name", ..., "name", "called", "identity"],  # ← 过宽
    ...
}
```

bare `"name"` alias 会被任何含 `name` 的句子激活。补救：detector 里
加 `full_name_blocked` 否决（先后加过 place/venue 类，2026-05-11 又加
pet/animal 类——P5.3）。每出现一类新名词场景就得追一条否决——**打地
鼠**。

**根因**：slot 检测不知道 subject。"cat's name" / "your name" /
"museum name" 在 slot 层都激活同一个 `full_name`。`full_name` 这个
slot 概念只适用于 **Person**，不适用于 **Pet / Place / Object / ...**。
当前架构没表达这个约束。

---

## 二、目标方案：二元 (slot, subject)

### 2.1 Subject 分类器

引入一个新检测器：

```python
def detect_subject_class(query: str) -> Literal["Person", "Pet", "Place", "Object", "Event", "Unknown"]:
    """Classify what kind of entity the user's query refers to."""
```

实现路径（按成本递增）：

- **(a) 大 regex 列表**：每类一组 noun pattern。**成本低，但又是打地鼠**——
  rejected.
- **(b) Embedding 最近邻**：query embedding 对一组 prototype query 的 cosine。
  prototypes 例：`"What's your full name?"`(Person)、`"What's the cat called?"`(Pet)、
  `"Name of that museum?"`(Place)。Top-1 match 决定类。Encoder 复用 mpnet。
- **(c) DeepSeek judge**：`PROMPT_SUBJECT_CLASS`，三选一/六选一返回。准但贵
  （每轮 +1 LLM 调用）。

**推荐 (b)**——零 LLM 成本，准度足够，向量空间共用。
留 (c) 作为 (b) 低置信度时的兜底。

### 2.2 Slot ↔ Subject 兼容矩阵

```python
SLOT_APPLICABLE_SUBJECTS = {
    "full_name":  {"Person"},
    "birthday":   {"Person"},
    "age":        {"Person"},
    "toy":        {"Person"},   # toy 也是属于人的（Eva 的 toy）
    # 未来扩展：
    # "pet_name": {"Pet"},     ← 如果想给 pet 也建结构化 slot
    # "place_name": {"Place"},
}
```

### 2.3 检测逻辑替换

```python
def _detect_requested_slot_fields(text):
    subject = detect_subject_class(text)
    slots = []
    for slot, aliases in MEMORY_SLOT_FIELDS.items():
        if subject not in SLOT_APPLICABLE_SUBJECTS.get(slot, set()):
            continue
        if any(_phrase_matches_text(a, text) for a in [slot, *aliases]):
            slots.append(slot)
    return slots
```

`full_name_blocked` 整段否决逻辑可以**全部删除**——subject 分类器
代替了它的功能。

### 2.4 Alias 收紧（可选）

subject 分类器接管之后，bare `"name" / "called" / "identity"` 留在
alias 里也无害（subject 已限定 Person，不会误激活）。但更干净：

```python
"full_name": ["full name", "real name", "legal name", "complete name", "name"],
```

只保留 "name"（subject 限定为 Person 即不会过宽）；删掉 "called" /
"identity"（含义模糊，且属于 narrative 而非 slot 直接询问）。

---

## 三、风险与回归

### 3.1 主要风险

1. **Subject 分类器误判**：embedding 最近邻在边界 query 上可能选错。
   缓解：Top-2 都返；任一命中就触发 slot；置信度低时（cosine 差距 < 0.05）
   走 LLM judge 兜底。
2. **现有 working query 可能不再命中**：例 `"do you remember my birthday?"`
   这类长 query 在 mpnet 空间到 prototype `"what's my birthday?"` 的距离需
   要实测。
3. **Slot 抽取下游**：`_extract_slot_value_from_record` 消费 slot 名
   不消费 subject，**理论上无下游影响**——待验。

### 3.2 必须做的回归 fixture

新增一组黄金 query 集合，每条带期望 `(subject, slots)` 标签：

```python
GOLDEN_QUERIES = [
    ("what's your full name?",                 "Person", {"full_name"}),
    ("what's my birthday?",                    "Person", {"birthday"}),
    ("how old is Eva?",                        "Person", {"age"}),
    ("what's your favorite toy?",              "Person", {"toy"}),
    ("what was the name of that cat?",         "Pet",    set()),
    ("the cat's name?",                        "Pet",    set()),
    ("name of the museum we visited?",         "Place",  set()),
    ("what was that song called?",             "Object", set()),
    ("我家那只猫叫什么名字？",                  "Pet",    set()),
    ("你叫什么全名？",                          "Person", {"full_name"}),
    ("did we visit a pleasure ground?",        "Event",  set()),
    # ... 至少 30 条覆盖各 subject + slot 组合
]
```

回归门槛：≥ 95% 期望吻合，否则回退。

---

## 四、执行预算

| 阶段 | 内容 | 估时 |
|---|---|---|
| S1 | 写 detect_subject_class（embedding 最近邻 + LLM 兜底）+ 单测 | 1.5 h |
| S2 | 写 SLOT_APPLICABLE_SUBJECTS 表 + 替换 _detect_requested_slot_fields | 0.5 h |
| S3 | 收集 30+ golden query，标 ground truth | 0.5 h |
| S4 | 跑黄金 fixture，对比 slot 命中 | 0.5 h |
| S5 | 删除 full_name_blocked 整段否决 + P5.3 否决（清债务） | 0.3 h |
| S6 | 真模型 Colab 验证（同 test_memory 那种 16 轮 fixture） | 1 h |
| S7 | 文档 + memory 更新 | 0.3 h |

**总计 ~4.5 小时**（不算 Colab 实跑等待）。

---

## 五、触发条件

- 当再有第三类名词场景出现 missing-slot 误警告时（食物 / 颜色 /
  歌名等），就该启动这份方案。
- 或单独的优化窗口：把 P5.3 + P1.7.2 否决段一并清理，作为 slot 系统
  的小型重构。

---

## 六、与 test_memory 模块的关系

无直接耦合。test_memory 的 P5（`memory_block_has_test_records`）
解决的是"测试记录在场时压制误警告"，是**正交问题**——它防止有真实
test 记录被误警告盖过；这份方案防止 slot 检测过宽进而产生误警告。
两层兜底保留即可。

---

## 七、当前 stop-gap

[eva_memory_legacy.py:1094](../eva_memory_legacy.py:1094) 的 P5.3 否决段
作为**临时治标**保留，直到本方案落地。
[`docs/USER_NOTES_MODULE.md`](USER_NOTES_MODULE.md) § 十的"已知遗留 debt"
指向本文档。
