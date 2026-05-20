"""
test_math_render.py — eva_math_render 的 splitter 单元测试。

只测纯逻辑（split_text_and_block_math），render_block_math_to_png 需要
matplotlib + GUI 后端，留给集成测试。

跑法（项目根目录）：
    D:/Anaconda/envs/py310/python.exe tests/test_math_render.py
"""

import os
import sys
import types
import unittest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# 不需要 stub matplotlib，因为 splitter 不触发其 import

from eva_math_render import split_text_and_block_math


class PureTextTest(unittest.TestCase):
    """没有 block math 的纯文本应该原样返回 [('text', ...)]。"""

    def test_empty_string(self):
        self.assertEqual(split_text_and_block_math(""), [])

    def test_plain_text(self):
        self.assertEqual(
            split_text_and_block_math("just hello"),
            [("text", "just hello")],
        )

    def test_inline_math_kept_in_text(self):
        # inline $...$ 不算 block，保留在 text 段
        text = "compute $f(x) = x^2$ to get the result"
        result = split_text_and_block_math(text)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], "text")
        self.assertIn("$f(x) = x^2$", result[0][1])


class DollarDollarBlockTest(unittest.TestCase):
    """$$...$$ display math 抽取。"""

    def test_single_block(self):
        result = split_text_and_block_math("intro $$x^2$$ outro")
        self.assertEqual(result, [
            ("text", "intro "),
            ("math", "x^2"),
            ("text", " outro"),
        ])

    def test_multiline_block(self):
        src = "Loss is\n$$\nL = \\sum_i x_i\n$$\nover all i."
        result = split_text_and_block_math(src)
        kinds = [k for k, _ in result]
        self.assertEqual(kinds, ["text", "math", "text"])
        # math content 应该被 strip
        math_content = [c for k, c in result if k == "math"][0]
        self.assertEqual(math_content, "L = \\sum_i x_i")

    def test_multiple_blocks(self):
        src = "first $$a$$ middle $$b$$ end"
        result = split_text_and_block_math(src)
        kinds = [k for k, _ in result]
        self.assertEqual(kinds, ["text", "math", "text", "math", "text"])
        maths = [c for k, c in result if k == "math"]
        self.assertEqual(maths, ["a", "b"])


class MarkdownMathCodeBlockTest(unittest.TestCase):
    """```math\n...\n``` 风格 block 抽取。"""

    def test_markdown_math(self):
        src = "see this:\n```math\n\\int_0^1 x dx\n```\nnice"
        result = split_text_and_block_math(src)
        kinds = [k for k, _ in result]
        self.assertEqual(kinds, ["text", "math", "text"])
        math_content = [c for k, c in result if k == "math"][0]
        self.assertEqual(math_content, "\\int_0^1 x dx")

    def test_markdown_math_no_trailing_newline(self):
        # ```math\n...``` 没结尾换行也要识别
        src = "```math\nx^2```"
        result = split_text_and_block_math(src)
        maths = [c for k, c in result if k == "math"]
        self.assertEqual(maths, ["x^2"])


class MixedScenarioTest(unittest.TestCase):
    """实战：Eva 输出可能 inline + block 混合。"""

    def test_inline_then_block(self):
        src = "We have $h_i = f(x)$ inline, and\n$$L = \\sum (y - \\hat{y})^2$$\nfor loss."
        result = split_text_and_block_math(src)
        kinds = [k for k, _ in result]
        # inline $...$ 留在 text 里，只抽 $$...$$
        self.assertEqual(kinds, ["text", "math", "text"])
        self.assertIn("$h_i = f(x)$", result[0][1])  # inline 保留
        self.assertIn("L = \\sum", result[1][1])    # block 抽走

    def test_only_block_math_in_message(self):
        src = "$$x^2 + y^2 = z^2$$"
        result = split_text_and_block_math(src)
        self.assertEqual(result, [("math", "x^2 + y^2 = z^2")])

    def test_empty_block_skipped(self):
        # $$  $$ 空块不应产生 ('math', '') 噪音
        src = "before $$  $$ after"
        result = split_text_and_block_math(src)
        # 期望：text 段直接相邻或合并；空 math 不进 list
        maths = [c for k, c in result if k == "math"]
        self.assertEqual(maths, [])


class ContractTest(unittest.TestCase):
    """元测试：函数 API 表面。"""

    def test_function_importable(self):
        from eva_math_render import split_text_and_block_math, render_block_math_to_png
        self.assertTrue(callable(split_text_and_block_math))
        self.assertTrue(callable(render_block_math_to_png))


if __name__ == "__main__":
    unittest.main(verbosity=2)
