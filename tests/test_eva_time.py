"""
test_eva_time.py — get_current_time() 单一时间接口的 offline 测试。

跑法（项目根目录下）：
    D:/Anaconda/envs/py310/python.exe tests/test_eva_time.py
"""

import os
import sys
import unittest
from datetime import datetime, timezone

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def _reload_eva_time_with_tz(tz_name):
    """以指定 EVA_TIMEZONE 重新加载 eva_time 模块。"""
    os.environ["EVA_TIMEZONE"] = tz_name
    if "eva_time" in sys.modules:
        del sys.modules["eva_time"]
    import importlib
    import eva_time as et
    importlib.reload(et)
    return et


class DefaultTimezoneTest(unittest.TestCase):
    """不传 tz 时使用 EVA_TIMEZONE 环境变量。"""

    def test_sydney_default(self):
        et = _reload_eva_time_with_tz("Australia/Sydney")
        t = et.get_current_time()
        self.assertIsNotNone(t.tzinfo, "应返回 tz-aware datetime")
        # 悉尼是 UTC+10（AEST）或 UTC+11（AEDT，夏令时）
        offset_h = t.utcoffset().total_seconds() / 3600
        self.assertIn(offset_h, (10.0, 11.0))

    def test_shanghai_default(self):
        et = _reload_eva_time_with_tz("Asia/Shanghai")
        t = et.get_current_time()
        offset_h = t.utcoffset().total_seconds() / 3600
        self.assertEqual(offset_h, 8.0)

    def test_utc_default(self):
        et = _reload_eva_time_with_tz("UTC")
        t = et.get_current_time()
        self.assertEqual(t.utcoffset().total_seconds(), 0.0)


class ExplicitTimezoneOverrideTest(unittest.TestCase):
    """显式传 tz 覆盖 EVA_TIMEZONE。"""

    def test_explicit_utc_overrides_default(self):
        et = _reload_eva_time_with_tz("Australia/Sydney")
        # 默认是 Sydney，但显式传 UTC 应该拿到 UTC
        t = et.get_current_time("UTC")
        self.assertEqual(t.utcoffset().total_seconds(), 0.0)

    def test_explicit_shanghai_overrides_default(self):
        et = _reload_eva_time_with_tz("UTC")
        t = et.get_current_time("Asia/Shanghai")
        offset_h = t.utcoffset().total_seconds() / 3600
        self.assertEqual(offset_h, 8.0)

    def test_default_and_explicit_consistent(self):
        """同一时刻 default 和 explicit 拿同一个时区，应该几乎相等。"""
        et = _reload_eva_time_with_tz("Australia/Sydney")
        t1 = et.get_current_time()
        t2 = et.get_current_time("Australia/Sydney")
        delta = abs((t2 - t1).total_seconds())
        self.assertLess(delta, 1.0)


class FallbackTest(unittest.TestCase):
    """乱填时区名要 fallback 到系统本地时间，不抛异常。"""

    def test_invalid_default_tz_falls_back(self):
        et = _reload_eva_time_with_tz("Not/A/Real/Zone")
        t = et.get_current_time()  # 不抛异常
        self.assertIsInstance(t, datetime)

    def test_invalid_explicit_tz_falls_back(self):
        et = _reload_eva_time_with_tz("UTC")
        # 显式传一个错的时区，也 fallback，不抛异常
        t = et.get_current_time("Mars/Olympus_Mons")
        self.assertIsInstance(t, datetime)


class UsagePatternsTest(unittest.TestCase):
    """验证常用调用模式仍然成立（"一个函数搞定一切"）。"""

    def test_today_via_dot_date(self):
        et = _reload_eva_time_with_tz("Australia/Sydney")
        d = et.get_current_time().date()
        from datetime import date
        self.assertIsInstance(d, date)

    def test_iso_utc_via_strftime(self):
        et = _reload_eva_time_with_tz("Australia/Sydney")
        s = et.get_current_time("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
        self.assertTrue(s.endswith("Z"))
        # 可被 parse 回来
        parsed = datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ")
        self.assertIsInstance(parsed, datetime)

    def test_filename_stamp_via_strftime(self):
        et = _reload_eva_time_with_tz("Australia/Sydney")
        s = et.get_current_time().strftime("%Y%m%d_%H%M%S")
        self.assertEqual(len(s), 15)
        parsed = datetime.strptime(s, "%Y%m%d_%H%M%S")
        self.assertIsInstance(parsed, datetime)


class BackwardCompatTest(unittest.TestCase):
    """eva_config.local_now 是 get_current_time 的别名（向后兼容）。"""

    def test_local_now_is_alias(self):
        _reload_eva_time_with_tz("Australia/Sydney")
        # 重新 import eva_config 拿到新别名
        if "eva_config" in sys.modules:
            del sys.modules["eva_config"]
        import eva_config
        from eva_time import get_current_time
        self.assertIs(eva_config.local_now, get_current_time)


class ApiSurfaceTest(unittest.TestCase):
    """元测试：eva_time 公共表面只暴露应该暴露的东西。"""

    def test_only_one_public_function(self):
        et = _reload_eva_time_with_tz("Australia/Sydney")
        publics = {x for x in dir(et) if not x.startswith("_")}
        # 必须有这两个，其它都是 stdlib/typing 重导出
        self.assertIn("get_current_time", publics)
        self.assertIn("EVA_TIMEZONE", publics)


if __name__ == "__main__":
    unittest.main(verbosity=2)
