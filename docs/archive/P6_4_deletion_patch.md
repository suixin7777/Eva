# P6.4 Deletion Patch — 预制规范

> **状态**：未应用。当前默认 `PRONOUN_RESOLVER_MODE = "regex_only"`，
> P6.1 行为线上跑。本文件描述 P6.4 cutover 时**逐字**要做的删除动作。
>
> **应用前置条件**（v3 plan § 六 + § 七）：
>
>   - P6.3 已切换为 `MODE="llm_first"`、`SHADOW=False` 并在生产稳定 ≥ 30 天
>   - LLM 调用成功率持续 ≥ 98%
>   - 8 个 § 八 acceptance fixture 在生产 trace 中持续 100% 命中
>   - 没有任何外部代码（非测试）还在 `import` 本规范列出的待删 symbol
>
> **任一不满足，不要应用本 patch。** 当前的 P6.0–P6.3 实现已经能让"LLM 不可
> 用 → regex fallback"作为 safety net 工作；P6.4 的价值是清债，不是性能。

---

## 一、删除范围一览

| 文件 | 待删 symbol / 段落 | 行数（应用前） |
|---|---|---|
| `eva_verifier_logic.py` | `_PRONOUN_FOLLOWUP_PATTERNS` | 1 个常量 tuple |
| `eva_verifier_logic.py` | `_FOLLOWUP_NOUN_STOPWORDS` | 1 个 frozenset |
| `eva_verifier_logic.py` | `_is_pronoun_followup` | 1 个函数 |
| `eva_verifier_logic.py` | `_extract_topical_nouns_from_recent_turns` | 1 个函数 |
| `eva_pronoun_resolver.py` | `_regex_fallback` | 简化为 1 行 stub |
| `eva_pronoun_resolver.py` | `_shadow_trace` / `_jaccard` / `_normalise_for_overlap` | 3 个 helper |
| `eva_pronoun_resolver.py` | `resolve_pronoun()` 中 shadow 分支 | 一段 if 块 |
| `eva_pronoun_resolver.py` | `resolve_pronoun()` 中 `regex_only` 分支 | 一段 if 块 |
| `eva_config.py` | `PRONOUN_RESOLVER_SHADOW` flag + 注释 | 几行 |
| `eva_config.py` | MODE 注释中 `"regex_only"` 提及 | 几行 |
| `eva_config.py` | "P6.3 cutover checklist" 注释块 | 整块（已不需要） |
| `test_p6_pronoun_resolver.py` | `TestP62ShadowMode` 整类 | 6 个 test |
| `test_p6_pronoun_resolver.py` | `TestP61RegexOnlyEquivalence` 整类 | 2 个 test |
| `test_p6_pronoun_resolver.py` | `test_regex_only_mode_never_calls_llm` | 1 个 test |
| `test_p6_pronoun_resolver.py` | `test_jaccard_helper` | 已包含在 shadow 类 |

净删除量：约 250 行生产代码 + 200 行测试。

---

## 二、`eva_verifier_logic.py` 删除规范

### 2.1 删除 `_PRONOUN_FOLLOWUP_PATTERNS` 常量

定位锚：注释 `# P5 — pronoun-followup helpers (replaced by ...)` 上方。

**删除整段**（从模块注释 `# Stage 1.6 ...` 块到 `_extract_topical_nouns_from_recent_turns` 函数末尾，约从 line 590 到 line 756 — 应用前需用 grep 确认范围）：

```text
开始锚定: re.compile(
开始所在行的上方注释: "P5 — pronoun-followup antecedent extraction"
结束锚定: 函数 _extract_topical_nouns_from_recent_turns 的最后一行 `return out[:max_terms]`
```

具体来说，要全部移除的代码块：
- `_PRONOUN_FOLLOWUP_PATTERNS = ( ... )` 三个 regex 编译
- `_FOLLOWUP_NOUN_STOPWORDS = frozenset({ ... })`
- `def _is_pronoun_followup(q): ...`
- `def _extract_topical_nouns_from_recent_turns(turns, max_terms=6): ...`

### 2.2 验证 `re` 和 `VERIFIER_DEBUG` 的引用面

`_is_pronoun_followup` 用了 `re.findall` 和 `VERIFIER_DEBUG`。删除后需确认：

```bash
# 确认 re 仍被本文件其他位置使用（应至少剩 _extract_date_from_text 等）
grep -n "re\." D:/Eva_new/eva_verifier_logic.py | head -5

# 确认 VERIFIER_DEBUG lazy-import 仍被其他位置使用
grep -n "VERIFIER_DEBUG" D:/Eva_new/eva_verifier_logic.py
```

预期：两者均仍在使用，**不删 module-level imports**。

### 2.3 `build_required_memory_params` 内部清理

`build_required_memory_params` 的 docstring 和注释提及了被删 symbol。修订：

**old_string**：
```
P5 (2026-05-08): when the user text is a pronoun-only follow-up,
    enrich the query and keywords with antecedent nouns harvested
    from the most recent assistant turn.

    P6 (2026-05-08, P6.1): replaces the inline regex pipeline
    (_is_pronoun_followup + _extract_topical_nouns_from_recent_turns)
    with a single call to eva_pronoun_resolver.resolve_pronoun(). The
    resolver decides "is this a follow-up?" AND "what is the
    antecedent?" in one pass, with three execution stages internally
    (cheap gate / LLM / regex fallback).

    Mode flag PRONOUN_RESOLVER_MODE controls behaviour:
      - "regex_only" (P6.1 default): resolver delegates to the same
        regex helpers used pre-P6. Behaviour is bit-identical (the
        regex fallback intentionally keeps max_terms=6 to match the
        legacy keyword set).
      - "llm_first" (P6.3+): DeepSeek main path; regex fallback only
        on LLM failure during P6.0–P6.3.
      - "off": resolver short-circuits; falls through to the
        original cleaned query without antecedent augmentation.
```

**new_string**：
```
P6 (P6.4 final): pronoun-followup detection and antecedent
    extraction are delegated entirely to eva_pronoun_resolver.
    resolve_pronoun(). The resolver runs the LLM main path; if
    DeepSeek is unavailable, the turn falls through to pre-P5
    behaviour (no antecedent augmentation) — acceptable degradation
    since P5 was a recall optimisation, not a correctness
    requirement. The legacy regex helpers were deleted in P6.4
    after 30 days of stable LLM-first operation; see
    P6_4_deletion_patch.md for the audit trail.

    Mode flag PRONOUN_RESOLVER_MODE: "llm_first" | "off".
```

---

## 三、`eva_pronoun_resolver.py` 删除规范

### 3.1 删除 shadow helper 三件套

定位锚：comment block `# Trace helpers`。

**old_string**：
```python
def _normalise_for_overlap(s: str) -> str:
    """Light normalisation for Jaccard comparison. Case-fold + trim;
    no lemma / stemming yet (would require spaCy). The shadow trace
    can flag false-mismatches caused by plurals or capitalisation;
    if those dominate the diff log we'll add stemming, but pulling
    in a heavy NLP dep speculatively isn't justified.
    """
    return (s or "").strip().lower()


def _jaccard(a, b) -> float:
    """Jaccard similarity of two string sequences (case-fold normalised).

    Returns 1.0 when both lists are empty (vacuously equal — both
    paths agree there's no antecedent).
    """
    sa = {_normalise_for_overlap(x) for x in (a or []) if isinstance(x, str)}
    sa.discard("")
    sb = {_normalise_for_overlap(x) for x in (b or []) if isinstance(x, str)}
    sb.discard("")
    if not sa and not sb:
        return 1.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 1.0


def _shadow_trace(
    query: str,
    regex_v: PronounResolution,
    llm_v: Optional[PronounResolution],
) -> None:
    # ... entire function ...
```

**new_string**：（全部删除，留空）

### 3.2 简化 `_regex_fallback`

整个函数体替换为 stub —— P6.4 后没有 regex helpers 可调用：

**old_string**：从 `def _regex_fallback(query, recent_turns):` 起到 `reasoning="..."` 函数结尾整体。

**new_string**：
```python
# P6.4: regex fallback removed. Kept as a stub returning source="skip"
# so call sites in resolve_pronoun() don't need restructuring. Calls
# arrive here when the LLM main path fails; the turn degrades to
# pre-P5 behaviour (no antecedent augmentation). This is the
# intended P6.4 final state — see P6_pronoun_resolver_refactor_v3.md
# § 六 and P6_4_deletion_patch.md.
def _regex_fallback(query: str, recent_turns) -> PronounResolution:
    return PronounResolution(
        needs_resolution=False,
        antecedents=[],
        confidence=1.0,
        source="skip",
        reasoning="P6.4: regex helpers removed; degrade to pre-P5",
    )
```

### 3.3 删 `resolve_pronoun()` 中 shadow 分支

**old_string**：
```python
    # ----- P6.2 shadow mode -----
    # Orthogonal to MODE. When SHADOW=True and MODE="llm_first", run
    # ... entire shadow block ...
    from eva_config import PRONOUN_RESOLVER_SHADOW
    if PRONOUN_RESOLVER_MODE == "llm_first" and PRONOUN_RESOLVER_SHADOW:
        # ... 30+ lines ...
        return regex_v
```

**new_string**：（全部删除）

### 3.4 删 `resolve_pronoun()` 中 `regex_only` 分支

**old_string**：
```python
    # ----- regex_only mode short-circuit -----
    # P6.1 starts in this mode: wiring lands but behaviour is
    # bit-identical to pre-P6 (regex still drives every decision).
    if PRONOUN_RESOLVER_MODE == "regex_only":
        verdict = _regex_fallback(query, recent_turns)
        # ... ~15 lines ...
        return verdict
```

**new_string**：（全部删除）

### 3.5 修 `resolve_pronoun()` import

**old_string**：
```python
    from eva_config import (
        ENABLE_PRONOUN_RESOLVER,
        PRONOUN_RESOLVER_MODE,
        PRONOUN_RESOLVER_MIN_CONFIDENCE,
        PRONOUN_RESOLVER_DEBUG,
        PRONOUN_RESOLVER_MAX_WORDS,
        PRONOUN_RESOLVER_MAX_CALLS_PER_TURN,
    )
```

不变（这些 flag 都还在）。但 mode 检查需要改：

**old_string**：
```python
    if (not ENABLE_PRONOUN_RESOLVER) or PRONOUN_RESOLVER_MODE == "off":
```

**new_string**：（不变）—— 仍然只检查 `"off"`，因为合法值现在只有 `{"llm_first", "off"}`。

---

## 四、`eva_config.py` 删除规范

### 4.1 删 `PRONOUN_RESOLVER_SHADOW` flag

**old_string**：
```python
# P6.2 shadow mode. Orthogonal to PRONOUN_RESOLVER_MODE — when both
# this flag and MODE="llm_first" are set, the resolver runs BOTH the
# LLM and regex paths but adopts the REGEX verdict (preserving P6.1
# behaviour). The LLM verdict is logged via a [PRONOUN-SHADOW] trace
# line so operators can compare the two over a sample window before
# committing to P6.3 (regex → LLM main).
#
# Has no effect when MODE != "llm_first" — there's nothing to compare
# against in regex_only / off modes.
#
# Cost: every shadow turn consumes one LLM call from
# PRONOUN_RESOLVER_MAX_CALLS_PER_TURN. Disable when shadow analysis is
# done.
PRONOUN_RESOLVER_SHADOW = False
```

**new_string**：（全部删除）

### 4.2 删 P6.3 cutover checklist 注释块

**old_string**：从 `# P6.3 cutover checklist` 起到结束的 `# ---` 分隔符。

**new_string**：（全部删除）

### 4.3 修 MODE 注释

**old_string**：
```python
# Mode: "llm_first" | "regex_only" | "off".
# - "llm_first": LLM main path; regex only triggers on LLM failure
#                (during P6.0–P6.3). Becomes the only mode after P6.4.
# - "regex_only": skip LLM entirely; preserves pre-P6 behaviour.
#                 P6.1 starts here so the wiring change ships dark.
#                 Removed in P6.4.
# - "off": disable resolver completely; build_required_memory_params
#          falls through to its original cleaned-query branch.
PRONOUN_RESOLVER_MODE = "regex_only"
```

**new_string**：
```python
# Mode: "llm_first" | "off".
# - "llm_first": LLM main path. On LLM failure the turn degrades to
#                pre-P5 behaviour (no antecedent augmentation) — see
#                eva_pronoun_resolver._regex_fallback stub.
# - "off": disable resolver completely; build_required_memory_params
#          falls through to its original cleaned-query branch.
PRONOUN_RESOLVER_MODE = "llm_first"
```

注意：默认从 `"regex_only"` 改为 `"llm_first"` —— 这是 P6.4 production 配置。

### 4.4 修 8d 段块标题注释

**old_string**：
```python
# 8d. Pronoun resolver (P6, replaces _PRONOUN_FOLLOWUP_PATTERNS)
#
# Replaces the regex-based pronoun-followup detection in
# eva_verifier_logic.build_required_memory_params with an LLM-driven
# resolver. The LLM answers two questions in one call: "is this a
# pronoun follow-up?" AND "what is the antecedent?"; the regex layer
# cannot do the second question and required two heuristic helpers
# (_is_pronoun_followup + _extract_topical_nouns_from_recent_turns)
# wired in series — each compounding the other's errors.
```

**new_string**：
```python
# 8d. Pronoun resolver (P6 final state, post-P6.4)
#
# LLM-driven detection of pronoun follow-ups + antecedent extraction.
# Replaces the legacy two-helper regex pipeline that was deleted in
# P6.4 (see git history for _PRONOUN_FOLLOWUP_PATTERNS,
# _is_pronoun_followup, _extract_topical_nouns_from_recent_turns).
# Failure mode: when the LLM is unavailable, the turn degrades to
# pre-P5 behaviour (no antecedent augmentation in
# build_required_memory_params).
```

---

## 五、`test_p6_pronoun_resolver.py` 删除规范

### 5.1 删整个 `TestP62ShadowMode` 类

整类不再有意义（无 shadow 可测）。

### 5.2 删整个 `TestP61RegexOnlyEquivalence` 类

`regex_only` 模式被删除，等价性测试无法运行。

### 5.3 删 `test_regex_only_mode_never_calls_llm`

在 `TestResolveFlags` 类中。

### 5.4 修 `TestResolveFlags` 中其他测试的 `_snap` 字典

各测试 setUp 中 snapshot 了 `PRONOUN_RESOLVER_MODE`。修改默认期望从 `"regex_only"` 切换为 `"llm_first"` 的依赖处不需要改（每个测试自己 set 模式），但务必确认 tearDown 后默认值正常。

### 5.5 保留

- `TestCheapGate` 全保留
- `TestParseVerdict` 全保留
- `TestResolveFlags` 中的 `test_off_mode_short_circuits` / `test_disabled_flag_short_circuits` / `test_skip_when_no_trigger` / `test_llm_first_happy_path` / `test_low_confidence_demoted` / `test_llm_unavailable_falls_through_to_regex` / `test_budget_exhaustion_skips_llm` / `test_cache_hit_skips_second_call` / `test_reset_state_clears_pronoun_counter`
- `TestAcceptanceFixtures` 全保留
- `TestP63LLMFirstIntegration` 全保留 — 这恰恰是 P6.4 后的核心契约

应用后预期测试数：从 36 降到约 22。

### 5.6 注意 LLM 失败路径的预期变化

当前 `test_llm_unavailable_falls_through_to_regex` 断言：
```python
self.assertIn(v.source, ("regex", "skip"))
```

应用 patch 后，`_regex_fallback` 是 stub 永远返回 `source="skip"`，所以 `"regex"` 永远不会出现。可保持此断言（`"skip"` 仍属合法），或收紧为：
```python
self.assertEqual(v.source, "skip")
```

后者更精确。同样适用于 `test_budget_exhaustion_skips_llm`。

`test_llm_failure_falls_through_to_regex` 在 `TestP63LLMFirstIntegration` 中断言 `"music box" in params["keywords"]` —— 这个断言会**失败**，因为 P6.4 后 LLM 失败 = 没 antecedent。需改为：

**old_string**：
```python
    def test_llm_failure_falls_through_to_regex(self):
        """When LLM is unavailable, verifier_logic still gets a
        sensible MemorySearch params dict via the regex fallback.
        ..."""
        agent = self._stub_agent(["I have my music box."])
        with patch(
            "eva_pronoun_resolver._call_llm",
            return_value=None,  # LLM down
        ):
            params = self._build(agent, "really? Check it")
        # Regex fallback fired — keywords still contain antecedents.
        self.assertIn("music box", params["keywords"])
        # Budget consumed even on LLM failure (we tried).
        self.assertEqual(agent._llm_judge_state.pronoun_call_count, 1)
```

**new_string**：
```python
    def test_llm_failure_degrades_to_pre_p5(self):
        """P6.4 contract: when LLM is unavailable, the resolver
        returns source='skip' and build_required_memory_params
        proceeds without antecedent augmentation. This is the
        documented degradation — pre-P5 behaviour for that turn."""
        agent = self._stub_agent(["I have my music box."])
        with patch(
            "eva_pronoun_resolver._call_llm",
            return_value=None,  # LLM down
        ):
            params = self._build(agent, "really? Check it")
        # Antecedent NOT injected — degrade to pre-P5.
        self.assertNotIn("music box", params["keywords"])
        # Budget consumed even on LLM failure (we tried).
        self.assertEqual(agent._llm_judge_state.pronoun_call_count, 1)
```

并更新方法名引用（如果有）。

---

## 六、应用步骤

```bash
# 1. 创建独立分支
git checkout -b p6.4-deletion

# 2. 按 § 二 / § 三 / § 四 / § 五 顺序应用各处编辑
#    （建议逐文件提交，便于 review 和回滚）

# 3. 运行测试套件 — 应仍全绿
D:/Anaconda/envs/py310/python.exe tests/test_p6_pronoun_resolver.py

# 4. Smoke check
D:/Anaconda/envs/py310/python.exe -c "
from eva_pronoun_resolver import resolve_pronoun, PronounResolution
from eva_intent_judge import JudgeState
import inspect
# 确认 _regex_fallback 是 stub
import eva_pronoun_resolver as r
src = inspect.getsource(r._regex_fallback)
assert 'P6.4' in src, 'regex_fallback was not simplified'
assert '_is_pronoun_followup' not in src
print('OK: _regex_fallback is the P6.4 stub')

# 确认 4 个 legacy symbol 都被删
import eva_verifier_logic as v
for sym in ['_PRONOUN_FOLLOWUP_PATTERNS', '_FOLLOWUP_NOUN_STOPWORDS',
            '_is_pronoun_followup', '_extract_topical_nouns_from_recent_turns']:
    assert not hasattr(v, sym), f'{sym} still present'
print('OK: 4 legacy symbols deleted')
"

# 5. 部署到 staging，再次 smoke + 真实流量观察 24h

# 6. 合入 main，关闭 P6 epic
```

---

## 七、回滚

P6.4 是高风险变更（删除生产代码）。如果应用后出问题：

```bash
# 立即回滚整个 patch（最快路径）
git revert <p6.4 commit hash>

# 同时应急配置（不需要 deploy 新代码）：
# eva_config.py 改回 PRONOUN_RESOLVER_MODE = "off"
# 这会让 build_required_memory_params 走 cleaned-query 分支，
# 完全跳过 antecedent 增强 —— 等同于 pre-P5 行为，但不需要 git revert。
```

---

## 八、前置条件复查（应用前最后一次）

打勾后才执行：

- [ ] `MODE="llm_first"`、`SHADOW=False` 已在生产稳定运行 ≥ 30 天
- [ ] LLM 调用成功率 ≥ 98%（取最近 7 天滚动平均）
- [ ] 8 个 § 八 acceptance fixture 在生产 trace 中 30 天内 100% 命中
- [ ] 无外部依赖：`grep -r "_PRONOUN_FOLLOWUP_PATTERNS\|_is_pronoun_followup\|_extract_topical_nouns_from_recent_turns" --include="*.py" .` 仅匹配本规范涉及的文件
- [ ] 已在 staging 环境完整跑通 P6.4 patch 至少 24h，无回归
- [ ] 团队成员 review 通过

任一未打勾，**停止**，让 P6.3 再多跑一段时间。
