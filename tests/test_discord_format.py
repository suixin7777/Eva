"""
test_discord_format.py — LaTeX → Discord 转换器的 offline 测试。

包含用户实际遇到的"对比学习幻灯片"翻译里的 LaTeX 表达式作为 golden case。

跑法（项目根目录下）：
    D:/Anaconda/envs/py310/python.exe tests/test_discord_format.py
"""

import os
import sys
import unittest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from eva_discord_format import format_for_discord


class StripDollarsTest(unittest.TestCase):
    """$...$ 包装必须去掉，否则 Discord 显示字面美元符号。"""

    def test_inline_math_dollars_removed(self):
        self.assertEqual(format_for_discord("score is $5$"), "score is 5")

    def test_block_math_dollars_removed(self):
        self.assertEqual(
            format_for_discord("eq: $$E = mc^2$$"),
            "eq: E = mc²",
        )

    def test_paren_math_removed(self):
        self.assertEqual(format_for_discord(r"value \(x\) here"), "value x here")

    def test_bracket_math_removed(self):
        self.assertEqual(format_for_discord(r"display \[x + y\]"), "display x + y")

    def test_unbalanced_dollar_left_alone(self):
        """一个孤立的 $（如 "cost is $5 USD"）不应被错误剥离。"""
        result = format_for_discord("cost is $5 USD only")
        # 一个 $ 找不到配对，应当原样保留
        self.assertIn("$5 USD", result)


class GreekLettersTest(unittest.TestCase):
    def test_tau(self):
        self.assertEqual(format_for_discord(r"$\tau$"), "τ")

    def test_alpha_beta(self):
        self.assertEqual(format_for_discord(r"$\alpha + \beta$"), "α + β")

    def test_upper_lambda(self):
        self.assertEqual(format_for_discord(r"$\Lambda$"), "Λ")

    def test_varepsilon_aliases(self):
        self.assertEqual(format_for_discord(r"$\varepsilon$"), "ε")


class MathOperatorsTest(unittest.TestCase):
    def test_cdot(self):
        self.assertEqual(format_for_discord(r"$f(\cdot)$"), "f(·)")

    def test_times(self):
        self.assertEqual(format_for_discord(r"$a \times b$"), "a × b")

    def test_infty(self):
        self.assertEqual(format_for_discord(r"$\infty$"), "∞")

    def test_leq_geq(self):
        self.assertEqual(format_for_discord(r"$x \leq y \geq z$"), "x ≤ y ≥ z")

    def test_in_notin(self):
        self.assertEqual(format_for_discord(r"$x \in A, y \notin B$"), "x ∈ A, y ∉ B")

    def test_arrow(self):
        self.assertEqual(format_for_discord(r"$a \to b$"), "a → b")


class SubscriptSuperscriptTest(unittest.TestCase):
    def test_single_subscript(self):
        # x_i → xᵢ
        self.assertEqual(format_for_discord(r"$x_i$"), "xᵢ")

    def test_compound_subscript_brace(self):
        # x_{ij} → xᵢⱼ
        self.assertEqual(format_for_discord(r"$x_{ij}$"), "xᵢⱼ")

    def test_digit_subscript(self):
        self.assertEqual(format_for_discord(r"$x_2$"), "x₂")

    def test_single_superscript(self):
        self.assertEqual(format_for_discord(r"$x^2$"), "x²")

    def test_compound_superscript(self):
        self.assertEqual(format_for_discord(r"$x^{10}$"), "x¹⁰")

    def test_unconvertible_subscript_kept_raw(self):
        """下标含没有 Unicode 对应的字符（如 'foo' 全字母），保持 LaTeX 原样。"""
        # _b 没有标准 Unicode 下标，应保留 _b
        result = format_for_discord(r"$x_b$")
        # 注意：单字符 _b 不可转，保留 "_b" 原样，可能会被 _escape 处理
        self.assertTrue("_b" in result or "\\_b" in result, f"got {result!r}")


class FracSqrtTest(unittest.TestCase):
    def test_frac(self):
        self.assertEqual(format_for_discord(r"$\frac{a}{b}$"), "(a)/(b)")

    def test_sqrt(self):
        self.assertEqual(format_for_discord(r"$\sqrt{x}$"), "√(x)")


class AccentCommandsTest(unittest.TestCase):
    """\\hat{x} \\tilde{z} \\bar{X} \\vec{v} 等重音附标命令。"""

    def test_hat_with_braces(self):
        self.assertEqual(format_for_discord(r"$\hat{x}$"), "x̂")

    def test_tilde_with_braces(self):
        # 用户实际遇到的 case：\tilde{z}
        result = format_for_discord(r"$\tilde{z}_t = z_t - c$")
        self.assertNotIn(r"\tilde", result)
        # 应该有 z̃ (z + combining tilde) 和下标 ₜ
        self.assertIn("̃", result)  # combining tilde
        self.assertIn("ₜ", result)        # subscript t

    def test_bar_with_braces(self):
        # \bar{X} → X̄
        self.assertEqual(format_for_discord(r"$\bar{X}$"), "X̄")

    def test_vec_with_braces(self):
        # \vec{v} → v⃗
        self.assertEqual(format_for_discord(r"$\vec{v}$"), "v⃗")

    def test_dot_with_braces(self):
        # \dot{x} → x + combining dot above (U+0307)
        # 注意：预合成的 'ẋ' (U+1E8B) 和分解的 'ẋ' 视觉一样、字节不同。
        # 我们不做 NFC 规范化，所以出来是分解形式。
        result = format_for_discord(r"$\dot{x}$")
        self.assertEqual(result, "ẋ")

    def test_no_braces_single_letter(self):
        # \hat x （无大括号）也覆盖
        result = format_for_discord(r"$\hat x$")
        self.assertIn("x̂", result)

    def test_widehat_alias(self):
        self.assertEqual(format_for_discord(r"$\widehat{x}$"), "x̂")

    def test_overline_alias(self):
        self.assertEqual(format_for_discord(r"$\overline{X}$"), "X̄")

    def test_realistic_dino_centering(self):
        """用户实际 case：DINO 的 centering 公式"""
        src = r"look at $\tilde{z}_t = z_t - c$ in the diagram"
        out = format_for_discord(src)
        self.assertNotIn(r"\tilde", out)
        self.assertNotIn("_t", out)  # 下标被吃掉
        self.assertIn("z̃", out)  # z + combining tilde

    def test_multichar_not_touched(self):
        # \hat{abc} (多字符) — 留原样不强转（附标加上去看着怪）
        src = r"$\hat{abc}$"
        out = format_for_discord(src)
        # 不应崩，但也不应该转成 abĉ 类似的怪样
        self.assertNotIn("̂", out)  # 不该出现 combining circumflex


class NormAndBracketTest(unittest.TestCase):
    """L2 范数 + 内积 + 楼板天花板符号的 LaTeX 兜底。"""

    def test_norm_double_pipe(self):
        # \| 是 LaTeX 范数符号
        self.assertEqual(format_for_discord(r"$\|x\|^2$"), "‖x‖²")

    def test_norm_vert(self):
        # 输入带空格，输出保留空格（LaTeX 转 Unicode 不动周围空白）
        self.assertEqual(format_for_discord(r"$\Vert x \Vert$"), "‖ x ‖")

    def test_inner_product(self):
        self.assertEqual(format_for_discord(r"$\langle x, y \rangle$"), "⟨ x, y ⟩")

    def test_floor_ceil(self):
        self.assertEqual(
            format_for_discord(r"$\lfloor x \rfloor + \lceil y \rceil$"),
            "⌊ x ⌋ + ⌈ y ⌉",
        )

    def test_norm_squared_realistic(self):
        # 真实场景：MAE loss 的范数平方
        src = r"L = \|x_{hat} - x\|^2"
        out = format_for_discord(src)
        self.assertIn("‖", out)
        self.assertIn("²", out)

    # === ASCII ||X|| → ‖X‖（Discord spoiler 防御）===
    def test_ascii_double_pipe_norm(self):
        # 2026-05-16 实测：模型用 ASCII ||x|| 表示 norm
        # Discord 误识别为 spoiler tag → 内容被遮 → 必须转 Unicode 范数
        result = format_for_discord("L = ||x - y||")
        self.assertNotIn("||", result)
        self.assertIn("‖", result)

    def test_ascii_double_pipe_mae_loss(self):
        # 用户实际遇到的 MAE loss
        result = format_for_discord("L = (1/|M|) Σᵢ∈M ||x̂ᵢ - xᵢ||²")
        self.assertNotIn("||", result)
        self.assertIn("‖x̂ᵢ - xᵢ‖", result)
        # 单 `|M|` 保留（绝对值/cardinality，单 pipe 不是 markdown）
        self.assertIn("|M|", result)

    def test_long_spoiler_preserved(self):
        # 真正的 Discord spoiler（长内容、跨意境）不该被误转
        result = format_for_discord("||This is a long sentence that's intentionally spoiler||")
        # 内容 > 40 字符 → 保留 ||
        self.assertIn("||", result)

    def test_multiline_spoiler_preserved(self):
        # 跨行 spoiler 也不动
        result = format_for_discord("||line one\nline two||")
        self.assertIn("||", result)

    def test_single_pipe_not_touched(self):
        # |M|（cardinality）单 pipe，Discord 不当 markdown，保留
        text = "size is |M| where M is the set"
        self.assertEqual(format_for_discord(text), text)


class BlackboardBoldTest(unittest.TestCase):
    def test_mathbb_R(self):
        self.assertEqual(format_for_discord(r"$\mathbb{R}^n$"), "ℝⁿ")

    def test_mathbb_Z(self):
        self.assertEqual(format_for_discord(r"$\mathbb{Z}_{\geq 0}$"),
                         "ℤ_{≥ 0}")  # 复杂下标无法 Unicode 化，保留原样


class UnderscoreEscapeTest(unittest.TestCase):
    """word_word 形式的 _ 会被 Discord 当斜体起点，要转义。"""

    def test_identifier_underscore_escaped(self):
        # F1_score → F1\_score（避免 Discord 渲染成斜体）
        result = format_for_discord("model F1_score is high")
        self.assertIn(r"F1\_score", result)

    def test_leading_underscore_not_escaped(self):
        # "_private" 开头的下划线不在 word 之间，保留
        result = format_for_discord("var _private")
        # 不应该看到 \_private
        self.assertNotIn(r"\_private", result)


class GoldenCaseSimCLRTest(unittest.TestCase):
    """用户实际遇到的 SimCLR 翻译里的 LaTeX 串端到端验证。"""

    def test_simclr_paragraph(self):
        src = (
            "An original image gets randomly augmented into two versions ($x_i, x_j$). "
            "Each goes through the same base encoder $f(\\cdot)$ to produce embeddings "
            "$h_i, h_j$. Those embeddings feed into a projection head $g(\\cdot)$ "
            "(two dense layers with ReLU), turning them into normalized vectors "
            "$z_i, z_j$ on a hypersphere."
        )
        out = format_for_discord(src)
        # 关键断言：不应该再含有 LaTeX 标记
        self.assertNotIn("$", out)
        self.assertNotIn(r"\cdot", out)
        # 关键符号应该被转换
        self.assertIn("·", out)        # \cdot
        self.assertIn("xᵢ", out)       # x_i
        self.assertIn("xⱼ", out)       # x_j
        self.assertIn("hᵢ", out)
        self.assertIn("hⱼ", out)
        self.assertIn("zᵢ", out)
        self.assertIn("zⱼ", out)

    def test_simclr_temperature(self):
        src = "Temperature $\\tau$: low makes softmax sharper."
        out = format_for_discord(src)
        self.assertEqual(
            out,
            "Temperature τ: low makes softmax sharper.",
        )


class IdempotenceTest(unittest.TestCase):
    """对已经 Unicode 化的文本再走一遍，应该等于自身。"""

    def test_idempotent(self):
        once = format_for_discord(r"$x_i + \tau \in \mathbb{R}$")
        twice = format_for_discord(once)
        self.assertEqual(once, twice)


class SpelledGreekInMathContextTest(unittest.TestCase):
    """模型直接输出 'theta' / 'phi' 等拼写名（不带反斜杠）的情况。
    应当只在 math context（_/^/{/\\ 之后）才转 Unicode，普通英文不动。
    """

    # === 应该转换（math context） ===
    def test_subscript_theta(self):
        self.assertEqual(format_for_discord("f_theta(x)"), "f_θ(x)")

    def test_subscript_phi(self):
        self.assertEqual(format_for_discord("g_phi(h)"), "g_φ(h)")

    def test_superscript_lambda(self):
        self.assertEqual(format_for_discord("x^lambda"), "x^λ")

    def test_inside_subscript_braces(self):
        # _{...} subscript 整块内部触发 Pass B 转换
        self.assertEqual(
            format_for_discord("x_{j in nu}"),
            "x_{j in ν}",
        )

    def test_inside_superscript_braces(self):
        self.assertEqual(
            format_for_discord("y^{alpha + beta}"),
            "y^{α + β}",
        )

    def test_standalone_braces_not_touched(self):
        # 没有 _/^ 前缀的 {...} 不算 math context，保留原样
        # 避免误伤 JSON / 普通文本里的 {key: nu}
        result = format_for_discord("config is {timeout: nu, retries: 3}")
        self.assertIn("nu", result)  # 不转

    def test_uppercase_sigma_with_backslash(self):
        # 已有 LaTeX 命令逻辑覆盖；这里再确认 spelled-out 兜底也工作
        self.assertEqual(format_for_discord("\\Sigma"), "Σ")

    def test_simclr_paper_passage(self):
        """用户实际遇到的案例（mae/contrastive learning 论文公式）。"""
        src = "h_i = f_theta({x_j}_{j in nu}_i) and x_hat_i = g_phi(h_i)"
        out = format_for_discord(src)
        # 关键 Greek 字母全部转换
        self.assertNotIn("theta", out)
        self.assertNotIn("phi", out)
        self.assertNotIn(" nu", out)  # 注意单独的 'nu' 应该转
        self.assertIn("θ", out)
        self.assertIn("φ", out)
        self.assertIn("ν", out)

    def test_derivative_format(self):
        # 'd theta' / 'dphi' 这类需要 _ 或 ^ 或 {} 才转
        # 'dphi' 没有前缀 → 不转（保守）
        # '_phi' 在 math context → 转
        self.assertEqual(format_for_discord("dL/d_phi"), "dL/d_φ")

    # === Pass C：导数符号 /d<greek> 和 ∂<greek> ===
    def test_derivative_slash_dphi(self):
        # dL/dphi → dL/dφ（最常见 case）
        self.assertEqual(format_for_discord("dL/dphi"), "dL/dφ")

    def test_derivative_slash_dtheta(self):
        self.assertEqual(format_for_discord("dL/dtheta"), "dL/dθ")

    def test_derivative_spaced_d(self):
        # '/ d theta' 中间带空格也应该转
        self.assertEqual(
            format_for_discord("chain rule: dL / d theta"),
            "chain rule: dL / d θ",
        )

    def test_partial_derivative_unicode(self):
        # ∂phi → ∂φ
        self.assertEqual(format_for_discord("∂L/∂phi"), "∂L/∂φ")

    def test_partial_with_space(self):
        # ∂ theta（带空格）也覆盖
        self.assertEqual(format_for_discord("∂ theta"), "∂ θ")

    def test_dpi_not_touched(self):
        # 'dpi' (dots per inch) 不该转 'dπ'
        # 因为前面没 / 也没 ∂，所以 Pass C 不命中
        text = "screen resolution is 300 dpi"
        self.assertEqual(format_for_discord(text), text)

    def test_dna_not_touched(self):
        # 'dna' 不在 Greek 词表里，也没风险
        text = "dna sequence analysis"
        self.assertEqual(format_for_discord(text), text)

    def test_full_simclr_passage(self):
        """用户实际 log 里的完整测试段落 —— 验证所有 case 都被吃下。"""
        src = (
            "h_i = f_theta({x_j}_{j in nu}_i), x_hat_i = g_phi(h_i). "
            "dL/dphi computes the gradient. "
            "dL/dtheta applies chain rule: dL/dh_i · d f_theta / d theta."
        )
        out = format_for_discord(src)
        # 关键：所有原始 spelled-out 形式都消失
        self.assertNotIn("theta", out)
        self.assertNotIn("phi", out)
        self.assertNotIn(" nu", out.replace(" nuclear", "x"))  # 排除可能的英文歧义
        # Unicode 替换全部到位
        self.assertIn("θ", out)
        self.assertIn("φ", out)
        self.assertIn("ν", out)

    # === 不应转换（普通英文） ===
    def test_english_word_not_touched(self):
        # 'alpha test' 是常见英文短语；'pi crust' 也是
        text = "this is the alpha test of the product"
        self.assertEqual(format_for_discord(text), text)

    def test_pi_crust(self):
        text = "make a pi crust for the dessert"
        self.assertEqual(format_for_discord(text), text)

    def test_lambda_function(self):
        # 'lambda function' 在编程里常用
        text = "use a lambda function here"
        self.assertEqual(format_for_discord(text), text)

    def test_omega_brand(self):
        text = "Omega 3 fish oil"
        self.assertEqual(format_for_discord(text), text)

    # === 边界 ===
    def test_partial_word_no_match(self):
        # 'thetaize' 不是希腊字母名，希腊词替换不应该 trigger。
        # 但 '_' 在 word_word 上下文里会被独立的 underscore-escape 逻辑转义，
        # 这是另一个保护层（防 Discord 渲染成斜体），不影响希腊词逻辑。
        result = format_for_discord("anti_thetaize is not real")
        # 关键断言：'theta' 没被替换成 θ（不是希腊字母）
        self.assertNotIn("θ", result)
        self.assertIn("thetaize", result)


class EdgeCasesTest(unittest.TestCase):
    def test_empty_string(self):
        self.assertEqual(format_for_discord(""), "")

    def test_no_latex(self):
        plain = "Just a regular sentence with no math!"
        self.assertEqual(format_for_discord(plain), plain)

    def test_double_backslash_newline(self):
        """LaTeX 里 \\\\ 是换行，应该转成实际换行符。"""
        result = format_for_discord(r"line1 \\ line2")
        self.assertIn("\n", result)

    def test_text_braces(self):
        # \text{hello} → hello
        self.assertEqual(format_for_discord(r"$\text{hello}$"), "hello")

    def test_mathbf_to_markdown_bold(self):
        # \mathbf{x} → **x** (Discord 粗体)
        result = format_for_discord(r"$\mathbf{x}$")
        self.assertEqual(result, "**x**")


if __name__ == "__main__":
    unittest.main(verbosity=2)
