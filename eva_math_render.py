"""
eva_math_render.py — 把 Eva 输出里的 LaTeX block math 渲染成 PNG 图片附件。

设计动机
========
inline math（`f_θ(x)` / `∂L/∂φ`）走 Unicode post-processor 就够了——读起来
流畅、不打断聊天流。但**整段公式**（loss function、矩阵、嵌套分数）用 Unicode
拼会很丑：

    L = (1/|M|) Σ_{i∈M} ‖x̂ᵢ - xᵢ‖²       # 还行
    Σⱼ Πₖ (xⱼₖ - μₖ)² / σₖ²              # 开始难读
    \\begin{matrix} ... \\end{matrix}      # Unicode 无解

方案 F：模型在 `$$...$$` 或 ```math\n...\n``` 块里包公式，runtime 把这些
块抽出来用 matplotlib mathtext 渲成 PNG，作为附件随消息一起发。inline `$...$`
继续走 Unicode 转换。

API
===
    split_text_and_block_math(text) -> list[tuple[kind, content]]
        切分文本为 ('text', str) | ('math', latex_src) 交错列表。
        inline $...$ 不抽（留给 format_for_discord 转 Unicode）。

    render_block_math_to_png(latex_src) -> bytes
        渲染单个 block math 为 PNG bytes（白字 + Discord 深色背景）。
        失败时 raise RuntimeError，调用方应 fallback 到代码块。

依赖：matplotlib（mathtext 内置，不需要系统 LaTeX）。
"""

from __future__ import annotations

import io
import logging
import os
import re

log = logging.getLogger(__name__)

# Discord dark mode background（让公式图融入消息流而不是白底突兀）
DISCORD_DARK_BG = "#2B2D31"


# ============================================================
# Block math 切分
# ============================================================
# 匹配两种 block math 语法：
#   $$...$$        LaTeX display math
#   ```math\n...\n``` Markdown 风格代码块（语言标 math）
_BLOCK_MATH_RE = re.compile(
    r"\$\$(.+?)\$\$"
    r"|"
    r"```math\s*\n?(.+?)\n?```",
    re.DOTALL,
)


def split_text_and_block_math(text: str) -> list[tuple[str, str]]:
    """切分文本为交错的 ('text', s) / ('math', latex_src) 列表。

    inline `$...$` **不抽出**，留在 'text' 块里给 format_for_discord 转 Unicode。
    只抽 `$$...$$` 和 ```math``` 这类 display math。

    Returns:
        list of (kind, content) where kind ∈ {'text', 'math'}.
        'math' 的 content 是去掉 delimiter 的 raw LaTeX 源。
    """
    if not text:
        return []
    parts: list[tuple[str, str]] = []
    last = 0
    for m in _BLOCK_MATH_RE.finditer(text):
        if m.start() > last:
            seg = text[last:m.start()]
            if seg:
                parts.append(("text", seg))
        latex_src = (m.group(1) or m.group(2) or "").strip()
        if latex_src:
            parts.append(("math", latex_src))
        last = m.end()
    if last < len(text):
        seg = text[last:]
        if seg:
            parts.append(("text", seg))
    return parts


# ============================================================
# Matplotlib lazy loader
# ============================================================
# matplotlib import 慢（~1.5s 首次）。lazy 载入，第一次用 block math 才付代价。
# MPLBACKEND=Agg 通过 env var 设置，确保在 pyplot 第一次 import 之前生效。
os.environ.setdefault("MPLBACKEND", "Agg")
_plt = None
_mpl_load_attempted = False


def _get_plt():
    """惰性加载 matplotlib.pyplot；失败返回 None。"""
    global _plt, _mpl_load_attempted
    if _mpl_load_attempted:
        return _plt
    _mpl_load_attempted = True
    try:
        import matplotlib
        matplotlib.use("Agg")  # 双保险
        import matplotlib.pyplot as plt
        _plt = plt
        log.info("matplotlib loaded for math rendering")
    except Exception as e:
        log.warning(f"matplotlib unavailable: {e}; block math will fall back to code blocks")
        _plt = None
    return _plt


# ============================================================
# 渲染
# ============================================================
def render_block_math_to_png(
    latex_src: str,
    *,
    dpi: int = 220,
    fontsize: int = 18,
    text_color: str = "white",
    bg_color: str = DISCORD_DARK_BG,
) -> bytes:
    """渲染 LaTeX block math 为 PNG bytes。

    Args:
        latex_src: raw LaTeX source（不含 `$$`、不含 ` ```math `）。
        dpi: 输出 DPI，220 在 Discord 上既清晰又不过大。
        fontsize: 字号，18 是不刺眼但读得清的折中。
        text_color / bg_color: 适配 Discord 深色 UI。

    Raises:
        RuntimeError: matplotlib 不可用、源为空、或 mathtext 解析失败。

    Returns:
        PNG bytes（可直接喂 discord.File）。
    """
    plt = _get_plt()
    if plt is None:
        raise RuntimeError("matplotlib not available")

    src = (latex_src or "").strip().strip("$").strip()
    if not src:
        raise RuntimeError("empty math source")

    fig, ax = plt.subplots(figsize=(0.01, 0.01), dpi=dpi)
    try:
        ax.axis("off")
        # 用 $...$ 包起来交给 mpl mathtext 解析
        txt = ax.text(
            0, 0, f"${src}$",
            fontsize=fontsize,
            color=text_color,
            ha="left",
            va="bottom",
        )
        fig.canvas.draw()
        bbox = txt.get_window_extent(renderer=fig.canvas.get_renderer())
        # 留出小 padding 防止字符贴边
        fig.set_size_inches(
            max(bbox.width / dpi + 0.4, 0.6),
            max(bbox.height / dpi + 0.3, 0.5),
        )
        buf = io.BytesIO()
        fig.savefig(
            buf,
            format="png",
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.15,
            facecolor=bg_color,
        )
        buf.seek(0)
        return buf.getvalue()
    except Exception as e:
        # mpl mathtext 不认这段 LaTeX（比如 \begin{matrix}），让 caller fallback
        raise RuntimeError(f"mathtext render failed: {e}") from e
    finally:
        plt.close(fig)
