"""
eva_time.py — Eva 的统一时间接口

只有一个公开函数：

    get_current_time(tz=None) -> datetime

不传参数就返回 EVA_TIMEZONE 环境变量配置的时区时间（默认 Australia/Sydney）；
传 IANA 时区名（"UTC" / "Asia/Shanghai" / ...）就返回该时区的时间。

切换 Eva 看到的"现在时间"，改 .env 里的 EVA_TIMEZONE 即可，不用动代码。

用法示例：

    from eva_time import get_current_time

    now      = get_current_time()                          # Eva 默认时区
    utc      = get_current_time("UTC")                     # canonical UTC（日志/跨时区）
    today    = get_current_time().date()                   # 今天的日期
    stamp    = get_current_time().strftime("%Y%m%d_%H%M%S")  # 文件名时间戳
    iso_utc  = get_current_time("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
"""

from __future__ import annotations

import os
from datetime import datetime

try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except ImportError:
    ZoneInfo = None  # type: ignore


# 默认时区。改 .env 里的 EVA_TIMEZONE 就能换，不用改代码。
EVA_TIMEZONE: str = os.environ.get("EVA_TIMEZONE", "Australia/Sydney")


def get_current_time(tz: str | None = None) -> datetime:
    """获取当前时间，时区可配置。

    Args:
        tz: IANA 时区名，比如 "UTC" / "Asia/Shanghai" / "Australia/Sydney"。
            None（默认）→ 使用 EVA_TIMEZONE 环境变量（默认 Australia/Sydney）。

    Returns:
        timezone-aware datetime。如果 zoneinfo 不可用或时区名解析失败，
        退化为系统本地时间（naive datetime），并打 warning 一次。
    """
    tz_name = tz if tz is not None else EVA_TIMEZONE

    if ZoneInfo is None:
        return datetime.now()

    try:
        return datetime.now(ZoneInfo(tz_name))
    except Exception as e:
        print(
            f"[eva_time] WARNING: failed to load timezone {tz_name!r} ({e}); "
            f"falling back to system local time."
        )
        return datetime.now()
