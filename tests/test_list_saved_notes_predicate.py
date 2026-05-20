"""
test_list_saved_notes_predicate.py — `is_list_saved_notes_pattern` 契约测试。

这个 predicate 是 3 处分支决策的单一源（PRE PROBE skip / BINDING DIRECTIVE
注入 / Detailed Notes Index 注入），契约稳定至关重要。

跑法（项目根目录）：
    D:/Anaconda/envs/py310/python.exe tests/test_list_saved_notes_predicate.py
"""

import os
import sys
import types
import unittest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# 离线 stub
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

from Advisor.advisor_client import AdvisorResult, is_list_saved_notes_pattern


def _make(intent="query_memory", needs_mem=False, suggested=None, ok=True):
    """Build an AdvisorResult-like object for testing."""
    return AdvisorResult(
        ok=ok,
        advice="some advice",
        intent=intent,
        needs_memory_retrieval=needs_mem,
        suggested_calls=suggested if suggested is not None else [],
    )


class TruthyCasesTest(unittest.TestCase):
    """触发 LIST-SAVED-NOTES pattern 的唯一一组配置。"""

    def test_canonical_list_all_pattern(self):
        # intent=query_memory, needs_mem=False, suggested_calls=[]
        r = _make(intent="query_memory", needs_mem=False, suggested=[])
        self.assertTrue(is_list_saved_notes_pattern(r))

    def test_with_non_memorysearch_suggested(self):
        # suggested 里有别的工具（如 GetCurrentTime），但没 MemorySearch
        # → 仍然算 list-saved-notes pattern
        r = _make(suggested=[{"tool": "GetCurrentTime", "args": {}}])
        self.assertTrue(is_list_saved_notes_pattern(r))


class FalsyCasesTest(unittest.TestCase):
    """所有不应该触发的场景。"""

    def test_intent_chat(self):
        self.assertFalse(is_list_saved_notes_pattern(_make(intent="chat")))

    def test_intent_query_external(self):
        self.assertFalse(is_list_saved_notes_pattern(_make(intent="query_external")))

    def test_intent_remember(self):
        self.assertFalse(is_list_saved_notes_pattern(_make(intent="remember")))

    def test_intent_forget(self):
        self.assertFalse(is_list_saved_notes_pattern(_make(intent="forget")))

    def test_intent_unknown(self):
        self.assertFalse(is_list_saved_notes_pattern(_make(intent="unknown")))

    def test_needs_memory_true(self):
        # query_memory + needs_mem=True → 普通查询，需要 probe
        self.assertFalse(is_list_saved_notes_pattern(
            _make(intent="query_memory", needs_mem=True)
        ))

    def test_memorysearch_in_suggested(self):
        # query_memory + suggested 含 MemorySearch → advisor 想 fetch
        self.assertFalse(is_list_saved_notes_pattern(
            _make(suggested=[{"tool": "MemorySearch", "args": {"query": "x"}}])
        ))

    def test_mixed_suggested_with_memorysearch(self):
        # suggested 里既有 MemorySearch 又有别的 → 仍然 falsy
        self.assertFalse(is_list_saved_notes_pattern(
            _make(suggested=[
                {"tool": "MemorySearch", "args": {}},
                {"tool": "GetCurrentTime", "args": {}},
            ])
        ))


class DegenerateInputTest(unittest.TestCase):
    """非法输入都安全 fallback 到 False。"""

    def test_none(self):
        self.assertFalse(is_list_saved_notes_pattern(None))

    def test_ok_false(self):
        self.assertFalse(is_list_saved_notes_pattern(_make(ok=False)))

    def test_missing_attributes(self):
        # 一个空 namespace 当 result
        class Stub:
            pass
        self.assertFalse(is_list_saved_notes_pattern(Stub()))

    def test_suggested_with_non_dict_entries(self):
        # 列表里有 None / str 等非 dict（advisor JSON 解析时偶发），不应 crash
        r = _make(suggested=[None, "garbage", {"tool": "Foo"}])
        # 这种垃圾不含 MemorySearch → True
        self.assertTrue(is_list_saved_notes_pattern(r))

        r2 = _make(suggested=[None, {"tool": "MemorySearch"}, "garbage"])
        # 含 MemorySearch → False
        self.assertFalse(is_list_saved_notes_pattern(r2))


class ContractTest(unittest.TestCase):
    """元测试：predicate 还是从 Advisor.advisor_client 暴露。"""

    def test_importable_from_advisor_module(self):
        from Advisor.advisor_client import is_list_saved_notes_pattern as fn
        self.assertTrue(callable(fn))


if __name__ == "__main__":
    unittest.main(verbosity=2)
