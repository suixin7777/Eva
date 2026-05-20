"""
eva_discord_format.py — 把 Eva 输出里的 LaTeX 数学符号转成 Discord 能显示的 Unicode。

Discord 不会渲染 LaTeX：

  $x_i$              → 直接显示成字面值 "$x_i$"，加上 _i 还会被当成斜体标记
  $\tau$             → 显示成 "$\tau$"，更糟
  $f(\cdot)$         → "$f(\cdot)$"

本模块的 format_for_discord(text) 把这些转成 Discord 友好的形式：

  $x_i$              → xᵢ
  $\tau$             → τ
  $f(\cdot)$         → f(·)
  $\mathbb{R}^n$     → ℝⁿ
  $\frac{a}{b}$      → (a)/(b)

策略：
  1. 去掉 $...$ 和 $$...$$ 包装
  2. LaTeX 命令 → Unicode 等价物（希腊字母、运算符、集合符号）
  3. 简单下标/上标 → Unicode 下标/上标（仅限可表达的字母数字）
  4. 不能表达的就保留原样（如 \frac → 文本 a/b 形式）
  5. 残留的 `\_` `\\` 之类清理掉

设计取舍：
  - 不追求 100% LaTeX 渲染。Discord 没有 KaTeX，目标是"读得懂"。
  - 优先 Unicode 单字符，因为 Discord markdown 不会处理它们。
  - 下划线主动转换：x_i 既是 LaTeX 下标又是 Discord 斜体起点，**必须转**。
"""

from __future__ import annotations

import re


# ============================================================
# 1. LaTeX 命令 → Unicode 单字符映射
# ============================================================

_GREEK_LOWER = {
    r"\alpha": "α", r"\beta": "β", r"\gamma": "γ", r"\delta": "δ",
    r"\epsilon": "ε", r"\varepsilon": "ε",
    r"\zeta": "ζ", r"\eta": "η",
    r"\theta": "θ", r"\vartheta": "ϑ",
    r"\iota": "ι", r"\kappa": "κ", r"\lambda": "λ", r"\mu": "μ",
    r"\nu": "ν", r"\xi": "ξ", r"\pi": "π", r"\varpi": "ϖ",
    r"\rho": "ρ", r"\varrho": "ϱ",
    r"\sigma": "σ", r"\varsigma": "ς",
    r"\tau": "τ", r"\upsilon": "υ",
    r"\phi": "φ", r"\varphi": "ϕ",
    r"\chi": "χ", r"\psi": "ψ", r"\omega": "ω",
}

_GREEK_UPPER = {
    r"\Gamma": "Γ", r"\Delta": "Δ", r"\Theta": "Θ",
    r"\Lambda": "Λ", r"\Xi": "Ξ", r"\Pi": "Π",
    r"\Sigma": "Σ", r"\Upsilon": "Υ", r"\Phi": "Φ",
    r"\Psi": "Ψ", r"\Omega": "Ω",
}

_MATH_OPS = {
    r"\cdot": "·", r"\times": "×", r"\div": "÷",
    r"\pm": "±", r"\mp": "∓",
    r"\infty": "∞", r"\partial": "∂", r"\nabla": "∇",
    r"\sum": "∑", r"\prod": "∏", r"\int": "∫",
    r"\leq": "≤", r"\le": "≤",
    r"\geq": "≥", r"\ge": "≥",
    r"\neq": "≠", r"\ne": "≠",
    r"\approx": "≈", r"\equiv": "≡", r"\sim": "∼", r"\simeq": "≃",
    r"\propto": "∝",
    r"\rightarrow": "→", r"\to": "→",
    r"\leftarrow": "←", r"\gets": "←",
    r"\Rightarrow": "⇒", r"\Leftarrow": "⇐",
    r"\leftrightarrow": "↔", r"\Leftrightarrow": "⇔",
    r"\mapsto": "↦",
    r"\in": "∈", r"\notin": "∉", r"\ni": "∋",
    r"\subset": "⊂", r"\supset": "⊃",
    r"\subseteq": "⊆", r"\supseteq": "⊇",
    r"\cup": "∪", r"\cap": "∩",
    r"\emptyset": "∅", r"\varnothing": "∅",
    r"\forall": "∀", r"\exists": "∃", r"\nexists": "∄",
    r"\neg": "¬", r"\lnot": "¬",
    r"\wedge": "∧", r"\land": "∧",
    r"\vee": "∨", r"\lor": "∨",
    r"\ldots": "…", r"\dots": "…", r"\cdots": "⋯", r"\vdots": "⋮", r"\ddots": "⋱",
    r"\circ": "∘", r"\bullet": "•",
    r"\ast": "∗", r"\star": "⋆",
    r"\angle": "∠",
    r"\perp": "⊥", r"\parallel": "∥",
    # 范数 / 尖括号 / 楼板 / 天花板
    r"\Vert": "‖", r"\vert": "|",
    r"\|": "‖",
    r"\langle": "⟨", r"\rangle": "⟩",
    r"\lfloor": "⌊", r"\rfloor": "⌋",
    r"\lceil": "⌈", r"\rceil": "⌉",
}

_BLACKBOARD = {
    r"\mathbb{R}": "ℝ", r"\mathbb{N}": "ℕ", r"\mathbb{Z}": "ℤ",
    r"\mathbb{Q}": "ℚ", r"\mathbb{C}": "ℂ", r"\mathbb{P}": "ℙ",
    r"\mathbb{E}": "𝔼", r"\mathbb{H}": "ℍ",
}

# 拼写出来的希腊字母名（模型直接输出 'theta' / 'phi'，没有反斜杠时用）。
# 只在明确的 math context 里替换（前面是 _ / ^ / { / \），避免误伤
# 'alpha test', 'pi crust', 'theta wave' 等英文短语。
_GREEK_WORD_TO_UNICODE = {
    # 小写
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ",
    "epsilon": "ε", "varepsilon": "ε",
    "zeta": "ζ", "eta": "η",
    "theta": "θ", "vartheta": "ϑ",
    "iota": "ι", "kappa": "κ", "lambda": "λ", "mu": "μ",
    "nu": "ν", "xi": "ξ",
    "pi": "π", "varpi": "ϖ",
    "rho": "ρ", "varrho": "ϱ",
    "sigma": "σ", "varsigma": "ς",
    "tau": "τ", "upsilon": "υ",
    "phi": "φ", "varphi": "ϕ",
    "chi": "χ", "psi": "ψ", "omega": "ω",
    # 大写（跟 Latin 字母不同形状的）
    "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ", "Lambda": "Λ",
    "Xi": "Ξ", "Pi": "Π", "Sigma": "Σ", "Upsilon": "Υ",
    "Phi": "Φ", "Psi": "Ψ", "Omega": "Ω",
}

# 合并所有，按长度降序处理（避免短键截掉长键的部分）
_ALL_REPLACEMENTS: dict[str, str] = {}
for d in (_BLACKBOARD, _GREEK_UPPER, _GREEK_LOWER, _MATH_OPS):
    _ALL_REPLACEMENTS.update(d)


# ============================================================
# 2. 下标/上标 Unicode 表
# ============================================================
_SUBSCRIPT_MAP = {
    "0":"₀","1":"₁","2":"₂","3":"₃","4":"₄",
    "5":"₅","6":"₆","7":"₇","8":"₈","9":"₉",
    "+":"₊","-":"₋","=":"₌","(":"₍",")":"₎",
    "a":"ₐ","e":"ₑ","h":"ₕ","i":"ᵢ","j":"ⱼ",
    "k":"ₖ","l":"ₗ","m":"ₘ","n":"ₙ","o":"ₒ",
    "p":"ₚ","r":"ᵣ","s":"ₛ","t":"ₜ","u":"ᵤ",
    "v":"ᵥ","x":"ₓ",
}

_SUPERSCRIPT_MAP = {
    "0":"⁰","1":"¹","2":"²","3":"³","4":"⁴",
    "5":"⁵","6":"⁶","7":"⁷","8":"⁸","9":"⁹",
    "+":"⁺","-":"⁻","=":"⁼","(":"⁽",")":"⁾",
    "a":"ᵃ","b":"ᵇ","c":"ᶜ","d":"ᵈ","e":"ᵉ",
    "f":"ᶠ","g":"ᵍ","h":"ʰ","i":"ⁱ","j":"ʲ",
    "k":"ᵏ","l":"ˡ","m":"ᵐ","n":"ⁿ","o":"ᵒ",
    "p":"ᵖ","r":"ʳ","s":"ˢ","t":"ᵗ","u":"ᵘ",
    "v":"ᵛ","w":"ʷ","x":"ˣ","y":"ʸ","z":"ᶻ",
    "A":"ᴬ","B":"ᴮ","D":"ᴰ","E":"ᴱ","G":"ᴳ",
    "H":"ᴴ","I":"ᴵ","J":"ᴶ","K":"ᴷ","L":"ᴸ",
    "M":"ᴹ","N":"ᴺ","O":"ᴼ","P":"ᴾ","R":"ᴿ",
    "T":"ᵀ","U":"ᵁ","V":"ⱽ","W":"ᵂ",
}


def _to_subscript(s: str) -> str | None:
    """转 Unicode 下标。任一字符无对应就返回 None（不部分转）。"""
    out = []
    for ch in s:
        if ch in _SUBSCRIPT_MAP:
            out.append(_SUBSCRIPT_MAP[ch])
        else:
            return None
    return "".join(out)


def _to_superscript(s: str) -> str | None:
    """转 Unicode 上标。任一字符无对应就返回 None。"""
    out = []
    for ch in s:
        if ch in _SUPERSCRIPT_MAP:
            out.append(_SUPERSCRIPT_MAP[ch])
        else:
            return None
    return "".join(out)


# ============================================================
# 3. 主转换函数
# ============================================================
# inline math $...$（不跨行；避免吞掉无关 $）
_INLINE_MATH_RE = re.compile(r"\$([^\$\n]+?)\$")
# block math $$...$$（可跨行）
_BLOCK_MATH_RE = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
# \( ... \) and \[ ... \]
_PAREN_MATH_RE = re.compile(r"\\\((.+?)\\\)", re.DOTALL)
_BRACKET_MATH_RE = re.compile(r"\\\[(.+?)\\\]", re.DOTALL)


def _replace_latex_commands(text: str) -> str:
    """逐个替换 LaTeX 命令为 Unicode。长键优先避免前缀冲突。"""
    for key in sorted(_ALL_REPLACEMENTS.keys(), key=len, reverse=True):
        text = text.replace(key, _ALL_REPLACEMENTS[key])
    return text


# ============================================================
# Accent / 重音附标命令：\hat{x} \tilde{z} \bar{X} \vec{v} \dot{x} ...
#
# Unicode 没有"带帽的 x"独立字符（除少数固定组合如 â），通用做法是
# 用 combining diacritical marks（组合附标）：base char 后紧跟一个
# 无宽度组合字符，渲染时显示成"base + 帽"。Discord 大多数字体支持。
#
# 例：
#   "z" + U+0303 (COMBINING TILDE)        → "z̃"
#   "x" + U+0302 (COMBINING CIRCUMFLEX)   → "x̂"
#   "X" + U+0304 (COMBINING MACRON)       → "X̄"
#   "v" + U+20D7 (COMBINING RIGHT ARROW ABOVE) → "v⃗"
#
# 多字符内容（如 \hat{abc}）下，附标只落在最后一个字符上，视觉怪。
# 我们只处理 单字符 / 单变量 的常见 case；多字符直接保留原样让 caller
# 知道。
# ============================================================
_ACCENT_TO_COMBINING = {
    "hat":       "̂",  # x̂
    "widehat":   "̂",
    "tilde":     "̃",  # x̃
    "widetilde": "̃",
    "bar":       "̄",  # x̄
    "overline":  "̄",
    "vec":       "⃗",  # x⃗
    "dot":       "̇",  # ẋ
    "ddot":      "̈",  # ẍ
    "check":     "̌",  # x̌
    "breve":     "̆",  # x̆
    "acute":     "́",  # x́
    "grave":     "̀",  # x̀
}


def _replace_accent_commands(text: str) -> str:
    """\\hat{x} → x̂, \\tilde{z} → z̃, etc. （只处理单字符 base）。

    覆盖两种语法：
      \\hat{x}     带大括号
      \\hat x      LaTeX 也允许无大括号 + 一个字母
    """
    for cmd, combining in _ACCENT_TO_COMBINING.items():
        # \cmd{X} — X 是单字符
        text = re.sub(
            r"\\" + cmd + r"\s*\{\s*([^\s{}])\s*\}",
            lambda m, c=combining: m.group(1) + c,
            text,
        )
        # \cmd X — 无大括号，X 是单字母
        text = re.sub(
            r"\\" + cmd + r"\s+([A-Za-z])\b",
            lambda m, c=combining: m.group(1) + c,
            text,
        )
    return text


def _replace_spelled_greek_in_math_context(text: str) -> str:
    """把 'theta' / 'phi' / ... 这类拼写出来的希腊字母名转 Unicode，**但只
    在明确的 math context 里**。

    两遍扫描：

    Pass A — 紧接前缀符（最常见情况）：
      `_<greek>`  下标（如 'f_theta')
      `^<greek>`  上标（如 'x^phi'）
      `{<greek>`  括号开头（如 'x_{theta}'）
      `\\<greek>` LaTeX 残余（如 '\\theta'，冗余但无害）

    Pass B — `_{...}` / `^{...}` 整块内部：
      整个下标/上标大括号块视为 math region，里面任意位置的希腊词都转。
      处理 'x_{j in nu}' / '^{theta + phi}' 等 'nu' 前面是空格的情况。

    Pass A 不动的安全例子（普通英文）：
      "alpha test"        → 不动
      "pi crust recipe"   → 不动
      "use lambda fn"     → 不动
    Pass A/B 转换的例子：
      "f_theta"           → "f_θ" (Pass A)
      "{xⱼ}_{j in nu}ᵢ"  → "{xⱼ}_{j in ν}ᵢ" (Pass B)
      "\\Sigma"           → "Σ" (Pass A)
    """
    # ----- Pass A: 紧接前缀符 -----
    # 长键优先（"vartheta" 在 "theta" 前先匹配，避免半截替换）
    sorted_keys = sorted(_GREEK_WORD_TO_UNICODE.keys(), key=len, reverse=True)
    for word in sorted_keys:
        uni = _GREEK_WORD_TO_UNICODE[word]
        pattern = r"(?<=[_^{\\])" + re.escape(word) + r"(?=[^A-Za-z0-9]|$)"
        text = re.sub(pattern, uni, text)

    # ----- Pass B: _{...} 和 ^{...} 整块内部 -----
    def _convert_inside_subbrace(match):
        opener = match.group(1)  # '_' or '^'
        inner = match.group(2)   # 大括号内的内容（不含嵌套 {/}）
        for w in sorted_keys:
            inner = re.sub(
                r"\b" + re.escape(w) + r"\b",
                _GREEK_WORD_TO_UNICODE[w],
                inner,
            )
        return f"{opener}{{{inner}}}"

    text = re.sub(r"([_^])\{([^{}]+)\}", _convert_inside_subbrace, text)

    # ----- Pass C: derivative notation -----
    # 处理 'dL/dphi'、'/ d theta'、'∂theta' 等情况 —— model 把希腊字母
    # 当变量名跟在 d / ∂ 后面（导数/偏导符号）。要求前面有 `/` 或者直接
    # 在 ∂ 后面（强 math 信号），避免误伤 "dpi"（dots per inch）这类
    # 普通缩写。
    for word in sorted_keys:
        uni = _GREEK_WORD_TO_UNICODE[word]
        # /d<greek> 或 /\s+d\s+<greek>（中间可有空格）
        text = re.sub(
            r"(/\s*d\s*)" + re.escape(word) + r"(?=[^A-Za-z0-9]|$)",
            lambda m, u=uni: m.group(1) + u,
            text,
        )
        # ∂<greek> 或 ∂\s+<greek>（Unicode 偏导符号 + 希腊词）
        text = re.sub(
            r"(∂\s*)" + re.escape(word) + r"(?=[^A-Za-z0-9]|$)",
            lambda m, u=uni: m.group(1) + u,
            text,
        )

    return text


def _replace_sub_super(text: str) -> str:
    """转下标 / 上标。仅当所有字符都能 Unicode 化时才转，否则保留原样。"""

    def sub_brace(m):
        r = _to_subscript(m.group(1))
        return r if r is not None else m.group(0)

    def sub_single(m):
        r = _to_subscript(m.group(1))
        return r if r is not None else m.group(0)

    def sup_brace(m):
        r = _to_superscript(m.group(1))
        return r if r is not None else m.group(0)

    def sup_single(m):
        r = _to_superscript(m.group(1))
        return r if r is not None else m.group(0)

    # 大括号包裹的下标 / 上标
    text = re.sub(r"_\{([^}]+)\}", sub_brace, text)
    text = re.sub(r"\^\{([^}]+)\}", sup_brace, text)
    # 单字符下标 / 上标 —— 要求后面不能再跟字母数字（避免把 F1_score 这种
    # 标识符里的 _s 错误转成 ₛ）。LaTeX 语义下 x_ij 也只会 subscript i，
    # 但实际模型输出里更常见的是 word_word 这种标识符，宁可保守。
    text = re.sub(r"_([A-Za-z0-9])(?![A-Za-z0-9])", sub_single, text)
    text = re.sub(r"\^([A-Za-z0-9])(?![A-Za-z0-9])", sup_single, text)
    return text


def _replace_frac(text: str) -> str:
    """\\frac{a}{b} → (a)/(b)。嵌套不处理。"""
    pattern = re.compile(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}")
    while True:
        new = pattern.sub(lambda m: f"({m.group(1)})/({m.group(2)})", text)
        if new == text:
            return new
        text = new


def _replace_sqrt(text: str) -> str:
    """\\sqrt{x} → √(x)。"""
    return re.sub(r"\\sqrt\s*\{([^{}]+)\}", lambda m: f"√({m.group(1)})", text)


def _escape_remaining_underscores_for_discord(text: str) -> str:
    """Discord 把 `_xxx_` 当斜体。已经转成 Unicode 下标的不会有 `_` 残留；
    但形如 'F1_score' 这种没被转的（多字符标识符）要给 `_` escape，
    避免被渲染成斜体。

    策略：保守只 escape "字母数字_字母数字" 这种容易被误判为 markdown 的位置。
    """
    # 把 word_word 形式的 _ 转义掉
    return re.sub(r"(?<=[A-Za-z0-9])_(?=[A-Za-z0-9])", r"\\_", text)


def _convert_double_pipe_to_norm(text: str) -> str:
    """把 ASCII `||X||` 转成 Unicode `‖X‖`（U+2016 Double Vertical Line）。

    背景（2026-05-16 实测）：Eva 输出数学范数时常用 ASCII `||x||` 表示
    L2 范数，但 **Discord 把 `||...||` 当 spoiler 标记**（点击才显示的
    隐藏块）。结果数学公式内容被遮住，只露出公式外的部分。

    解决：把短内容的 `||...||` 替换成 Unicode `‖...‖`，Discord 直接当
    普通字符渲染。

    保守限制：
      - 内容不超过 40 字符
      - 内容不能含换行
      - 内容不能含其他 `|`（避免嵌套混淆）
    这样真正的 Discord spoiler（一般是长句、跨行）不会被误改。
    """
    return re.sub(r"\|\|([^\n|]{1,40})\|\|", r"‖\1‖", text)


def format_for_discord(text: str) -> str:
    """Eva 输出 → Discord 友好文本的主入口。

    使用：
        from eva_discord_format import format_for_discord
        await message.reply(format_for_discord(eva_reply))
    """
    if not text:
        return text

    # 1. 拆掉 LaTeX math 包装（保留内容）
    text = _BLOCK_MATH_RE.sub(lambda m: m.group(1), text)
    text = _INLINE_MATH_RE.sub(lambda m: m.group(1), text)
    text = _PAREN_MATH_RE.sub(lambda m: m.group(1), text)
    text = _BRACKET_MATH_RE.sub(lambda m: m.group(1), text)

    # 2. \frac, \sqrt 这类带参数的命令优先处理（顺序：先 frac/sqrt 再单 token）
    text = _replace_frac(text)
    text = _replace_sqrt(text)

    # 3. LaTeX 命令 → Unicode 单字符（希腊字母、运算符）
    text = _replace_latex_commands(text)

    # 3a. Accent 重音附标：\hat{x} → x̂、\tilde{z} → z̃、\bar{X} → X̄ 等。
    #     必须在 _replace_sub_super 之前做，因为 '\hat{x}_t' 转换后变成
    #     'x̂_t'，下标处理再把 '_t' 转成 'ₜ' → 最终 'x̂ₜ'。
    text = _replace_accent_commands(text)

    # 3b. 拼写出来的希腊字母（'theta'/'phi'/...）在 math context 里 → Unicode。
    #     必须在 LaTeX 命令替换之后做（否则 '\theta' 里的 'theta' 会被错误地
    #     抓到再替换一次）；在下标 Unicode 化之前做（否则 '_theta' 里 'theta'
    #     变 'θ' 之后下标处理会跳过它）。
    text = _replace_spelled_greek_in_math_context(text)

    # 4. 下标 / 上标 → Unicode
    text = _replace_sub_super(text)

    # 5. 清理 LaTeX 残余符号
    # \\ 换行 → 实际换行
    text = text.replace(r"\\", "\n")
    # \left( \right) 之类没意义的命令，去掉
    text = re.sub(r"\\left", "", text)
    text = re.sub(r"\\right", "", text)
    # \, \; \: \! 等空格命令
    text = re.sub(r"\\[,;:!]", " ", text)
    # \text{...} → ...
    text = re.sub(r"\\text\s*\{([^{}]*)\}", lambda m: m.group(1), text)
    # \mathrm{...} → ...
    text = re.sub(r"\\mathrm\s*\{([^{}]*)\}", lambda m: m.group(1), text)
    # \mathbf{...} → **...**（Discord 加粗）
    text = re.sub(r"\\mathbf\s*\{([^{}]*)\}", lambda m: f"**{m.group(1)}**", text)

    # 6a. ASCII ||X|| → ‖X‖（Discord 里 ||...|| 是 spoiler 标记，会遮住内容）。
    # 必须在 underscore escape 之前做，避免 ||x_i|| 中的 `_` 被先 escape 影响匹配。
    text = _convert_double_pipe_to_norm(text)

    # 6b. 给 word_word 形式的 _ 转义，防止 Discord 渲染成斜体
    text = _escape_remaining_underscores_for_discord(text)

    return text
