# P6.2 Shadow 测试 Colab Runbook（交互式版）

> **使用模式**：你直接和 Eva 聊天，每次跟进型查询都会触发 shadow 比对并被记录。
> 累积到 ≥ 200 条观测时，自动收尾、计算指标、保存数据、释放模型。
>
> **总流程**：Cell 1-4 设置（一次，~5 min）→ Cell 5 smoke 验线（10 秒）
> → Cell 6 装 hook（一次）→ **Cell 7 持续聊天**（自由时长）→ Cell 8 结算 + 释放。

---

## 〇、Colab 环境前置

| 项 | 要求 |
|---|---|
| Python | 3.10+ |
| GPU | A100 / V100 推荐；T4 也能跑 |
| Drive | Eva 工程文件已传到 Drive 某目录 |
| API key | `DEEPSEEK_API_KEY` 必填 |

下文假设工程目录在 `/content/drive/MyDrive/Eva_new`，按你实际路径替换。

---

## Cell 1 — 挂载 Drive + 切到工程目录

```python
from google.colab import drive
drive.mount('/content/drive')

import os, sys
EVA_DIR = '/content/drive/MyDrive/Eva_new'   # 改成你的实际路径
os.chdir(EVA_DIR)
if EVA_DIR not in sys.path:
    sys.path.insert(0, EVA_DIR)

for f in ['eva_inference_P2.py', 'eva_pronoun_resolver.py',
          'eva_config.py', 'topic_keywords.json']:
    assert os.path.exists(f), f'missing {f}'
print('working dir:', os.getcwd())
```

---

## Cell 2 — 安装依赖 + 设置 API key

```python
!pip install -q rank_bm25 faiss-cpu sentence-transformers tavily-python openai

# DeepSeek key——shadow 模式必需
from google.colab import userdata
try:
    os.environ['DEEPSEEK_API_KEY'] = userdata.get('DEEPSEEK_API_KEY')
except Exception:
    os.environ['DEEPSEEK_API_KEY'] = 'sk-...'  # fallback：填你的 key

assert os.environ.get('DEEPSEEK_API_KEY', '').startswith('sk-'), \
    'DEEPSEEK_API_KEY 没设好——shadow 模式跑不起来'
print('API keys configured')
```

---

## Cell 3 — 翻 shadow 配置

```python
import eva_config
eva_config.PRONOUN_RESOLVER_MODE = 'llm_first'
eva_config.PRONOUN_RESOLVER_SHADOW = True
eva_config.PRONOUN_RESOLVER_DEBUG = False

print('MODE   =', eva_config.PRONOUN_RESOLVER_MODE)
print('SHADOW =', eva_config.PRONOUN_RESOLVER_SHADOW)
print('budget/turn =', eva_config.PRONOUN_RESOLVER_MAX_CALLS_PER_TURN)
```

---

## Cell 4 — 构建 agent

```python
import eva_inference_P2 as eva
agent = eva.build_agent()    # 第一次调用 1-3 分钟
print('agent ready:', type(agent).__name__)
```

---

## Cell 5 — Smoke 验线（确认 trace 真的被发出）

聊天前先跑一次，**确认你的 wiring 工作**。这版**直接调 resolver**，
不经 ChatSession / verifier 链，是最可靠的 wiring 验证。耗时 ~3 秒。

> **不用 ChatSession 跑 smoke 的原因**：
>
>  1. `ChatSession.send(verbose=False)` 内部会 `redirect_stdout` 然后丢弃 buffer，
>     shadow trace 会被吞。
>  2. shadow trace 只在 verifier 触发 memory repair 路径时才发出——这条件不
>     是每个短跟进句都满足，smoke 0 条 trace 不能区分是 wiring 坏还是
>     verifier 没触发。直接调 resolver 绕开这两个干扰源。

```python
import io, contextlib
import eva_pronoun_resolver
from eva_intent_judge import JudgeState

# Sanity check：flag 真的翻了吗
assert eva_config.PRONOUN_RESOLVER_MODE == 'llm_first', \
    f"MODE={eva_config.PRONOUN_RESOLVER_MODE}, expected 'llm_first' — 重跑 Cell 3"
assert eva_config.PRONOUN_RESOLVER_SHADOW is True, \
    f"SHADOW={eva_config.PRONOUN_RESOLVER_SHADOW}, expected True — 重跑 Cell 3"

# 直接调 resolver——不经 agent / verifier
buf = io.StringIO()
state = JudgeState()
with contextlib.redirect_stdout(buf):
    v = eva_pronoun_resolver.resolve_pronoun(
        "really? Check it",
        [{"user": "", "assistant": "I have a music box on my shelf."}],
        state=state,
    )

text = buf.getvalue()
print(text)   # 看实际打印了什么
shadow_lines = [l for l in text.splitlines() if '[PRONOUN-SHADOW]' in l]
print(f'\n[smoke] shadow trace lines: {len(shadow_lines)}')
print(f'[smoke] verdict: source={v.source} needs={v.needs_resolution} '
      f'antecedents={v.antecedents}')
print(f'[smoke] LLM call count: {state.pronoun_call_count}')

assert len(shadow_lines) >= 1, (
    "shadow trace 没发出——可能原因:\n"
    "  1. SHADOW=False（重跑 Cell 3）\n"
    "  2. DEEPSEEK_API_KEY 失效（看上面 print 是否有 'LLM unavailable'）\n"
    "  3. cheap gate 把 query 过滤了（不应该——'really? Check it' 是 3 词带触发词）"
)
print('[smoke] PASS — resolver wiring works')
```

**期望输出**（latency 因网络浮动，但 `[PRONOUN-SHADOW]` 一定要出现一行）：

```
        | [PRONOUN-SHADOW] q='really? Check it' regex_needs=True llm_needs=True
        |                  regex_terms=['music box', ...] llm_ants=['music box']
        |                  agree=False overlap=0.20 llm_conf=0.95 latency_ms=512.3

[smoke] shadow trace lines: 1
[smoke] verdict: source=regex needs=True antecedents=[...]
[smoke] LLM call count: 1
[smoke] PASS — resolver wiring works
```

注意：
- `agree=False overlap=0.20` 是 **Jaccard 噪声**，不是真分歧。regex 输出多
  粒度（`['music box', 'box', 'music', ...]`），LLM 输出单短语
  （`['music box']`），交集 1 / 并集 5 = 0.20。但**两边都抓到了
  `music box`**——真信号一致。详见 Q2。
- `latency_ms` 第一次冷启动可能 1500-3000 ms。后续调用会降到 400-800 ms。如
  果持续 > 1500 ms，参见 Q3。

---

## Cell 6 — 装 live observation hook + 设目标样本量

这一步把 `_shadow_trace` 包装一层，每次它被调用时往：

1. 全局列表 `SHADOW_OBSERVATIONS`（给 Cell 8 算指标用）
2. log 文件（JSON-per-line 格式，事后人工抽查）

各 push 一条结构化记录。**完全不动 `sys.stdout`**——Colab 的 cell 显示
保持原样，agent 的 print 你能照常看见。重复跑也无副作用。

```python
import datetime, json, time
import eva_pronoun_resolver as _r

# ---- 全局观测累加器 ----
SHADOW_OBSERVATIONS = []          # list[dict] — 每次 shadow 比对一条
TARGET_OBS = 200                  # 命中后聊天循环自动退出
LOG_PATH = f'/content/shadow_trace_{datetime.datetime.now():%Y%m%d_%H%M%S}.log'

# ---- 打开 log 文件 (line-buffered)，hook 直接写入，不走 stdout ----
_log_fp = open(LOG_PATH, 'a', encoding='utf-8', buffering=1)
_log_fp.write(f'\n===== shadow run @ {datetime.datetime.now()} =====\n')

# ---- 安装 hook (幂等) ----
if not getattr(_r, '_shadow_trace_hooked', False):
    _orig_shadow_trace = _r._shadow_trace
    _orig_jaccard      = _r._jaccard

    def _hooked_shadow_trace(query, regex_v, llm_v, *, latency_ms=None):
        # 1. 计算 overlap / agree
        if llm_v is None:
            overlap = None; agree = None
        else:
            overlap = _orig_jaccard(regex_v.antecedents, llm_v.antecedents)
            agree   = (regex_v.needs_resolution == llm_v.needs_resolution
                       and overlap >= 0.5)
        # 2. 结构化记录
        obs = {
            'query':       query,
            'regex_needs': regex_v.needs_resolution,
            'regex_terms': list(regex_v.antecedents),
            'llm_needs':   (llm_v.needs_resolution if llm_v else None),
            'llm_ants':    (list(llm_v.antecedents) if llm_v else None),
            'llm_conf':    (llm_v.confidence if llm_v else None),
            'overlap':     overlap,
            'agree':       agree,
            'latency_ms':  latency_ms,
            'timestamp':   time.time(),
        }
        # 3. push 到内存数组 + 写 log（直接写文件，不经 stdout）
        SHADOW_OBSERVATIONS.append(obs)
        try:
            _log_fp.write(json.dumps(obs, ensure_ascii=False) + '\n')
            _log_fp.flush()
        except Exception:
            pass
        # 4. 仍调原 trace——屏显，让你聊天时能实时看到对照
        _orig_shadow_trace(query, regex_v, llm_v, latency_ms=latency_ms)

    _r._shadow_trace = _hooked_shadow_trace
    _r._shadow_trace_hooked = True
    print('[hook] installed')
else:
    print('[hook] already installed — skipping re-wrap')

print(f'log file:   {LOG_PATH}')
print(f'target:     {TARGET_OBS} observations')
print(f'collected:  {len(SHADOW_OBSERVATIONS)}')
print('[stdout untouched — Colab display preserved]')
```

**为什么这版比之前的 Tee 方案稳**：

| 旧 Tee 方案 | 新方案 |
|---|---|
| 替换 `sys.stdout` 为 Tee | 完全不动 `sys.stdout` |
| Cell 6 重跑会嵌套到上次的 Tee 上 | hook 函数幂等，重跑无副作用 |
| 旧版 Cell 6 用过 `sys.__stdout__` 就回不来 | 跟 sys.stdout 状态完全无关 |
| Cell 8 要小心还原 stdout | Cell 8 直接关 log 文件即可 |
| log 是 stdout 镜像（含杂乱 ANSI / 进度条） | log 是干净 JSON，Cell 8 直接 `json.loads` |

如果你之前已经跑过旧版 Cell 6 把 stdout 弄坏了，**重启 runtime 一次**
（Runtime → Restart runtime）然后跑 Cell 1-4 + 这个新 Cell 6，之后就稳了。

---

## Cell 7 — 持续聊天，自动收尾

这是核心 cell。在 Colab 的 `input()` prompt 里输入消息，每次 `Enter` 发送一条。
状态命令以冒号开头：

| 命令 | 作用 |
|---|---|
| `:status` | 当前观测数 / 目标 |
| `:metrics` | 不退出，预览当前指标 |
| `:reset` | 清空对话 history（不影响 SHADOW_OBSERVATIONS） |
| `:quit` / 空行 / Ctrl+C | 提前退出 |

到达 `TARGET_OBS` 时**自动**退出循环。

聊天技巧：要让 shadow 触发，多用**短跟进型**句子（"check it" / "really? do that" /
"hold on, look at them"）。完整长句会被 cheap gate 过滤掉，不消耗 LLM 也不产生
观测——这是设计，但意味着你只发长句聊一晚也凑不出 200 条。

```python
from eva_chat_colab import ChatSession   # Cell 5 不再 import 它，所以这里要补
from eva_intent_judge import JudgeState
from eva_pronoun_resolver import resolve_pronoun

sess = ChatSession(agent, user_name='Rosm')

# ★ 关键改动：每个用户输入主动 probe 一次 resolver，
# 不再被动等 verifier 修复路径触发。理由：
#  - Verifier 修复是窄路径，~5% 触发率，凑 200 条要聊 4000 turn 不现实
#  - Probe 是只读调用，不影响 Eva 实际行为
#  - 用 fresh JudgeState 避免和 agent 的预算池干扰
def _probe_resolver(user_text):
    try:
        recent = []
        if hasattr(agent, 'history_manager') and agent.history_manager:
            recent = agent.history_manager.recent_turns(n=2)
        # Cheap gate 会跳过长句 / 无触发词的——只有合格输入才真的发 LLM
        resolve_pronoun(user_text, recent, state=JudgeState())
    except Exception as e:
        print(f'[probe error: {e!r}]')

def _quick_metrics(obs):
    n = len(obs)
    if n == 0: return {'observations': 0}
    llm_ok = [o for o in obs if o['llm_needs'] is not None]
    avail = len(llm_ok) / n
    agreeable = [o for o in obs if o['agree'] is not None]
    agree_rate = (sum(1 for o in agreeable if o['agree']) / len(agreeable)
                  if agreeable else None)
    overlaps = [o['overlap'] for o in llm_ok if o['overlap'] is not None]
    mean_overlap = statistics.mean(overlaps) if overlaps else None
    lats = sorted(o['latency_ms'] for o in obs if o['latency_ms'] is not None)
    p95 = lats[max(0, int(round(0.95*(len(lats)-1))))] if lats else None
    return {
        'observations':         n,
        'llm_availability':     round(avail, 3),
        'agreement_rate':       (round(agree_rate, 3) if agree_rate is not None else None),
        'mean_jaccard_overlap': (round(mean_overlap, 3) if mean_overlap is not None else None),
        'latency_p95_ms':       (round(p95, 1) if p95 is not None else None),
    }

print(f'\n=== Chat with Eva (shadow collecting; target {TARGET_OBS}) ===')
print('  commands: :status :metrics :reset :quit\n')

try:
    while True:
        n = len(SHADOW_OBSERVATIONS)
        if n >= TARGET_OBS:
            print(f'\n[target reached: {n} observations] — exiting loop')
            break
        try:
            msg = input(f'[{n}/{TARGET_OBS}] You: ').strip()
        except EOFError:
            break

        if msg in ('', ':quit', ':q'):
            print('[exiting on user request]')
            break
        if msg == ':status':
            print(f'  observations={n} target={TARGET_OBS} '
                  f'remaining={TARGET_OBS - n}')
            continue
        if msg == ':metrics':
            print(json.dumps(_quick_metrics(SHADOW_OBSERVATIONS),
                             indent=2, ensure_ascii=False))
            continue
        if msg == ':reset':
            sess.reset()
            print('  [history reset; observations preserved]')
            continue

        # ★ 1. 先 probe resolver（采集 shadow 数据）
        _probe_resolver(msg)

        # 2. 再发给 Eva（正常对话）
        try:
            sess.send(msg)
        except KeyboardInterrupt:
            print('\n[KeyboardInterrupt during turn — exiting]')
            break
except KeyboardInterrupt:
    print('\n[KeyboardInterrupt — exiting loop]')

print(f'\n[final tally] {len(SHADOW_OBSERVATIONS)} observations collected')
```

---

## Cell 8 — 结算 + 落盘 + 释放模型

```python
import json, statistics, gc

# ---- 完整指标 ----
def compute_metrics(obs):
    n = len(obs)
    if n == 0:
        return {'observations': 0, 'note': 'no observations'}
    llm_ok = [o for o in obs if o['llm_needs'] is not None]
    avail = len(llm_ok) / n
    agreeable = [o for o in obs if o['agree'] is not None]
    agree_rate = (sum(1 for o in agreeable if o['agree']) / len(agreeable)
                  if agreeable else None)
    overlaps = [o['overlap'] for o in llm_ok if o['overlap'] is not None]
    mean_overlap = statistics.mean(overlaps) if overlaps else None
    lats = sorted(o['latency_ms'] for o in obs if o['latency_ms'] is not None)
    def pct(p):
        if not lats: return None
        k = max(0, min(len(lats)-1, int(round(p*(len(lats)-1)))))
        return lats[k]
    disagrees = [o for o in obs if o['agree'] is False][:10]
    return {
        'observations':         n,
        'llm_availability':     round(avail, 3),
        'agreement_rate':       (round(agree_rate, 3) if agree_rate is not None else None),
        'mean_jaccard_overlap': (round(mean_overlap, 3) if mean_overlap is not None else None),
        'latency_p50_ms':       (round(pct(0.50), 1) if lats else None),
        'latency_p95_ms':       (round(pct(0.95), 1) if lats else None),
        'latency_max_ms':       (round(lats[-1], 1) if lats else None),
        'disagreement_samples': disagrees,
    }

metrics = compute_metrics(SHADOW_OBSERVATIONS)

# ---- P6.3 verdict ----
THRESHOLDS = {
    'min_observations':     200,
    'llm_availability':     0.98,
    'agreement_rate':       0.95,
    'mean_jaccard_overlap': 0.50,
    'latency_p95_ms':       800,
}

fails = []
if metrics['observations'] < THRESHOLDS['min_observations']:
    fails.append(f"sample too small: {metrics['observations']} < {THRESHOLDS['min_observations']}")
for k in ('llm_availability', 'agreement_rate', 'mean_jaccard_overlap'):
    v = metrics.get(k)
    if v is None or v < THRESHOLDS[k]:
        fails.append(f"{k} {v} < {THRESHOLDS[k]}")
v = metrics.get('latency_p95_ms')
if v is None or v > THRESHOLDS['latency_p95_ms']:
    fails.append(f"latency_p95_ms {v} > {THRESHOLDS['latency_p95_ms']}")

print('=== METRICS ===')
print(json.dumps({k: v for k, v in metrics.items() if k != 'disagreement_samples'},
                 indent=2, ensure_ascii=False))
print('\n=== TOP DISAGREEMENTS (manual audit) ===')
for d in metrics.get('disagreement_samples', []):
    print(f"  q={d['query']!r}")
    print(f"    regex_needs={d['regex_needs']} llm_needs={d['llm_needs']}")
    print(f"    regex_terms={d['regex_terms']} llm_ants={d['llm_ants']}")
    print(f"    overlap={d['overlap']} llm_conf={d['llm_conf']} latency={d['latency_ms']}")

print('\n=== P6.3 VERDICT ===')
if fails:
    print('NOT READY for P6.3:')
    for f in fails: print(f'  • {f}')
else:
    print('READY for P6.3 cutover.')
    print('  Next: PRONOUN_RESOLVER_SHADOW = False (keep MODE="llm_first").')

# ---- 落盘到 Drive（容错版）----
import shutil, os, glob
out_dir = '/content/drive/MyDrive/Eva_new/shadow_logs'
os.makedirs(out_dir, exist_ok=True)

# 1. 当前 LOG_PATH（可能不存在——只是 log 镜像，不影响指标）
if os.path.exists(LOG_PATH):
    shutil.copy(LOG_PATH, out_dir)
    print(f'[saved] log: {os.path.join(out_dir, os.path.basename(LOG_PATH))}')
else:
    print(f'[skip]  log: {LOG_PATH} 不存在')

# 2. 把所有现存 log 一并拷走
for p in glob.glob('/content/shadow_trace_*.log'):
    try:
        shutil.copy(p, out_dir)
    except Exception as e:
        print(f'[err]   {p}: {e!r}')

# 3. metrics + observations——最重要的产物
metrics_path = os.path.join(out_dir,
    f'metrics_{datetime.datetime.now():%Y%m%d_%H%M%S}.json')
with open(metrics_path, 'w', encoding='utf-8') as f:
    json.dump({
        'metrics':       {k: v for k, v in metrics.items()
                          if k != 'disagreement_samples'},
        'verdict_fails': fails,
        'observations':  SHADOW_OBSERVATIONS,
    }, f, indent=2, ensure_ascii=False)
print(f'[saved] metrics: {metrics_path}')

# ---- 关 log 文件 + 释放模型 ----
# 新版 Cell 6 不动 sys.stdout，所以这里没有还原步骤——直接关 log 即可。
try:
    _log_fp.close()
except Exception:
    pass

del agent
gc.collect()
try:
    import torch
    torch.cuda.empty_cache()
    print('\n[cleanup] CUDA cache emptied')
except Exception:
    pass
print('[cleanup] agent released; safe to disconnect runtime')
```

---

## 九、典型问题排查

### Q1 Cell 5 smoke check 失败（0 条 trace）

| 检查 | 命令 |
|---|---|
| flag 翻了吗 | `print(eva_config.PRONOUN_RESOLVER_SHADOW)` 应输出 `True` |
| Mode 对吗 | `print(eva_config.PRONOUN_RESOLVER_MODE)` 应输出 `'llm_first'` |
| API key 设了吗 | `print(os.environ['DEEPSEEK_API_KEY'][:6])` |

任一失败 → 重跑 Cell 2 / 3。

### Q2 聊了很久但 `n` 不涨

最常见原因：你发的全是长句或不含指代触发词，cheap gate 过滤掉了。
**只有这些查询会触发 shadow**：

- 短查询（≤ 8 词）
- **且**包含 `it / that / them / those / this / these / him / her / his / hers / again / too` 任一

举例对比：

| 输入 | 是否触发 |
|---|---|
| `"check it"` | ✓ |
| `"really? Do it again"` | ✓ |
| `"can you tell me about your hobbies?"` | ✗ (无触发词) |
| `"please look up the music box in your records"` | ✗ (>8 词) |
| `"hold on, can you look at it for me"` | ✗ (10 词) |

为提高触发率，多用跟进句模式（"check it" / "look at them" / "do that again" 等）。

### Q3 `n` 涨了但 `latency_ms=None`

这一轮 LLM 调用被预算跳过（`PRONOUN_RESOLVER_MAX_CALLS_PER_TURN=2`）。罕见，
仅当一轮内 verifier 修复循环调用 ≥ 2 次 resolver 时发生。
临时把预算调到 4 即可：

```python
eva_config.PRONOUN_RESOLVER_MAX_CALLS_PER_TURN = 4
```

### Q4 想分多次 session 跑（中途休息）

`SHADOW_OBSERVATIONS` 是 module-level 列表，**只要不重启 Colab runtime 就保留**。
你可以中途退出 Cell 7、做其他事、再次进 Cell 7 接着聊。`n` 会从上次的位置继续累积。

如果想跨 runtime 持久化（关闭 Colab 后重连），在 Cell 8 之外手动 dump：

```python
import pickle
with open('/content/drive/MyDrive/Eva_new/shadow_logs/inprogress.pkl', 'wb') as f:
    pickle.dump(SHADOW_OBSERVATIONS, f)
```

下次进来读回：

```python
import pickle
with open('/content/drive/MyDrive/Eva_new/shadow_logs/inprogress.pkl', 'rb') as f:
    SHADOW_OBSERVATIONS.extend(pickle.load(f))   # extend，不要重新赋值
```

### Q5 Cell 7 聊天后 cell 输出区一片空白

如果你看到 `[0/200] You: <msg>` 后下一个 prompt 直接出来、中间什么都没显示，
说明你之前跑过**旧版 Cell 6**（用 Tee 替换 sys.stdout 那种）。Colab stdout
被指到了看不见的地方。

**修复**：Runtime → Restart runtime，然后只跑 Cell 1-4 + **新版 Cell 6**
（不动 sys.stdout 那个），之后聊天输出就正常了。新版 Cell 6 设计上避免了
这个问题——不替换 stdout，只 hook `_shadow_trace`。

**验证 log 没丢**：即使屏显没出来，hook 也直接写文件。重启前确认数据：

```python
!ls -la /content/shadow_trace_*.log
!tail -10 /content/shadow_trace_*.log    # 新版是 JSON-per-line
```

每行一条 `{"query": "...", "agree": false, ...}`。如果文件完全是空的，
那就是 agent 根本没跑——检查 Cell 4 是否成功 build_agent。

### Q6 想合并多次 metrics dump

用 Cell 8 的 verdict 单跑：

```python
import glob, json
all_obs = []
for p in sorted(glob.glob('/content/drive/MyDrive/Eva_new/shadow_logs/metrics_*.json')):
    with open(p, encoding='utf-8') as f:
        all_obs.extend(json.load(f).get('observations', []))
metrics = compute_metrics(all_obs)
print(json.dumps(metrics, indent=2, ensure_ascii=False))
```

---

## 十、产出物清单

跑完一次完整 shadow 评估，Drive 上应有：

```
shadow_logs/
  shadow_trace_YYYYMMDD_HHMMSS.log     原始 stdout（含所有 [PRONOUN-SHADOW] 三行块）
  metrics_YYYYMMDD_HHMMSS.json         结构化 metrics + verdict_fails + 全部 observations
```

PR / 评审附件（按 [v3 plan § 七](P6_pronoun_resolver_refactor_v3.md) 要求）：

1. metrics JSON 中 5 项硬指标的最终值
2. `verdict_fails` 列表（空 = ready，非空 = 哪条没过）
3. `disagreement_samples` 中 ≥ 10 条的人工标注（"真分歧"还是"格式噪声"）
4. 总样本量 ≥ 200 的截图证据

集齐 → 按 v3 plan § 六 决定是否进 P6.3 cutover。
