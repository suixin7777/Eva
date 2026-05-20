"""
test_orphan_markdown_verifier.py — verifier 检测末尾 orphan ** 的契约测试。

只测纯 helper `_answer_has_orphan_bold_tail`，不触碰 verify_final_answer
（那个需要 agent 全套上下文）。

跑法（项目根目录）：
    D:/Anaconda/envs/py310/python.exe tests/test_orphan_markdown_verifier.py
"""

import os
import sys
import types
import unittest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# 标准 offline stub
if "torch" not in sys.modules:
    _torch_stub = types.ModuleType("torch")
    _backends = types.ModuleType("torch.backends")
    _cudnn = types.ModuleType("torch.backends.cudnn")
    _cudnn.benchmark = False
    _cudnn.deterministic = True
    _backends.cudnn = _cudnn
    _torch_stub.backends = _backends
    sys.modules["torch"] = _torch_stub
    sys.modules["torch.backends"] = _backends
    sys.modules["torch.backends.cudnn"] = _cudnn

# eva_verifier_logic 也会拉一堆 transformers 依赖。简单 stub。
if "transformers" not in sys.modules:
    _tf_stub = types.ModuleType("transformers")
    class _StoppingCriteriaStub:
        pass
    _tf_stub.StoppingCriteria = _StoppingCriteriaStub
    sys.modules["transformers"] = _tf_stub

if "PIL" not in sys.modules:
    _pil = types.ModuleType("PIL")
    class _ImageStub:
        @staticmethod
        def open(*a, **kw): return None
    _pil.Image = _ImageStub
    sys.modules["PIL"] = _pil
    sys.modules["PIL.Image"] = _ImageStub

from eva_verifier_logic import _answer_has_orphan_bold_tail


class OrphanBoldTailTest(unittest.TestCase):
    """触发场景：sampling collapse 留下未闭合 ** at 末尾。"""

    # ===== TRUE cases（应当触发 regenerate）=====

    def test_observed_case_tch(self):
        # 2026-05-15 11:43 真实截断
        text = "Tch! You keep asking, but I already told you: **"
        self.assertTrue(_answer_has_orphan_bold_tail(text))

    def test_observed_case_hmph(self):
        # 2026-05-15 12:04 真实截断
        text = "Hmph! Didn't I just say it? Pick the **"
        self.assertTrue(_answer_has_orphan_bold_tail(text))

    def test_orphan_with_trailing_space(self):
        text = "go with the **   "
        self.assertTrue(_answer_has_orphan_bold_tail(text))

    def test_orphan_with_trailing_punctuation(self):
        # 末尾如果有逗号/句号也算（标点不应阻止检测）
        text = "go with the **,"
        self.assertTrue(_answer_has_orphan_bold_tail(text))

    def test_balanced_pairs_then_orphan(self):
        # 前面成对 + 末尾孤儿
        text = "Use **commit logs** and also pick **"
        self.assertTrue(_answer_has_orphan_bold_tail(text))

    # ===== FALSE cases（合法 ** 用法，不应误伤）=====

    def test_no_bold_at_all(self):
        text = "Just a plain answer."
        self.assertFalse(_answer_has_orphan_bold_tail(text))

    def test_balanced_bold_in_middle(self):
        text = "Use **commit logs** as evidence."
        self.assertFalse(_answer_has_orphan_bold_tail(text))

    def test_math_double_star_inside(self):
        # 5**3 数学表达不该被误判
        text = "compute 5**3 is 125."
        self.assertFalse(_answer_has_orphan_bold_tail(text))

    def test_glob_pattern_inside(self):
        # **/*.py glob 模式不该被误判
        text = "use **/*.py for the pattern."
        self.assertFalse(_answer_has_orphan_bold_tail(text))

    def test_unmatched_in_middle_not_at_tail(self):
        # 中间未闭合但末尾是别的字符 → 不主动干预（streaming 中间态/未知用法）
        text = "I said **bold but forgot to close and kept typing."
        self.assertFalse(_answer_has_orphan_bold_tail(text))

    def test_empty_string(self):
        self.assertFalse(_answer_has_orphan_bold_tail(""))

    def test_none_value(self):
        self.assertFalse(_answer_has_orphan_bold_tail(None))

    def test_non_string_value(self):
        self.assertFalse(_answer_has_orphan_bold_tail(12345))


class RegisteredInReasonPolicyTest(unittest.TestCase):
    """元测试：orphan_markdown_in_answer 必须在 REASON_POLICY 里登记，
    否则会落到 DEFAULT_POLICY (canned_fallback) 而非 regenerate。"""

    def test_reason_registered_with_regenerate_fix(self):
        from eva_verifier_logic import REASON_POLICY
        self.assertIn("orphan_markdown_in_answer", REASON_POLICY)
        entry = REASON_POLICY["orphan_markdown_in_answer"]
        self.assertEqual(entry["fix"], "regenerate")
        self.assertEqual(entry["severity"], "hard")


if __name__ == "__main__":
    unittest.main(verbosity=2)
