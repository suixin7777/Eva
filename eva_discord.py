"""
eva_discord.py — Discord 桥接层，把 Eva (Qwen3.5-VL-9B) 暴露为 Discord bot。

设计原则：
1. 一个进程只加载一次模型（build_agent 很慢），所有请求复用同一个 ChatAgent。
2. 串行推理：9B 模型不支持并发，用 asyncio.Lock 把多个用户的请求排队。
3. 视觉支持：Discord 图片附件 → 临时文件 → image_path 走 agent 的 vision tool。
4. 渐进式状态：progress_callback 复用旧版"临时状态 vs 永久内容"的视觉效果，
   工具产出（新闻片段）变成可保留的资料卡，思考态用 "💭" 前缀做覆盖式更新。
5. 配置全部走 .env / 环境变量，eva_config 已经自带 dotenv loader。

启动：
    python eva_discord.py
依赖（除项目原有）：
    discord.py>=2.3
    aiohttp (discord.py 自带)
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Optional

import discord

# 会话隔离逻辑抽到独立模块，让 tests/test_session_isolation.py 能 offline 测。
from eva_discord_sessions import (
    SessionStore,
    snapshot_agent_state,
    restore_agent_state,
    inject_resumed_turns,
)
# LaTeX → Unicode 转换器（Discord 不渲染 LaTeX，输出前过一遍）
from eva_discord_format import format_for_discord

# Block math 渲染（$$...$$ → PNG 附件；inline $...$ 仍走 Unicode 路径）
from eva_math_render import split_text_and_block_math, render_block_math_to_png

# ============================================================
#  开机恢复（Discord history resume）—— 默认关闭
#
#  曾经默认 4 轮，但实战发现弊大于利：
#    - 把过去的错误回答（如 sampling collapse 留下的"note's gone"）拉回
#      来，模型在 in-context history 中"看到自己"反复说错 → 继续犯同样
#      的错；
#    - 每次 /reset 后下一条消息又自动拉历史 → reset 形同虚设；
#    - Pod 任何重启都触发 → 干扰测试节奏；
#    - resume 的内容是 format_for_discord 之后的版本（Discord-rendered），
#      不是 raw eva_reply，会损失格式细节。
#
#  现在改成 env var 控制：
#    EVA_DISCORD_RESUME_TURNS=0   → 完全关闭 resume（默认，推荐）
#    EVA_DISCORD_RESUME_TURNS=4   → 行为同旧默认
#    EVA_DISCORD_RESUME_TURNS=N   → 任意 N≥1
#
#  长期对话需要延续性时再启用。开发/测试请保持 0。
# ============================================================
MAX_RESUMED_TURNS = int(os.environ.get("EVA_DISCORD_RESUME_TURNS", "0"))


# eva_config 的 dotenv loader 会自动把 .env 注入 os.environ，
# 但它只在 import eva_config 时触发。为了让 DISCORD_TOKEN 也走 .env，
# 我们手动让 eva_config 先加载，触发 _load_dotenv()。
import eva_config  # noqa: F401  必须先 import，触发 .env 注入


DISCORD_TOKEN = os.environ.get("DISCORD_TOKEN", "").strip()
if not DISCORD_TOKEN:
    print("[FATAL] DISCORD_TOKEN 未在 .env / 环境变量里设置，无法启动。")
    sys.exit(1)


# 仅在主线程做一次重型 import，否则每条消息都会 re-trigger torch 初始化告警
print("=" * 60)
print("  Eva Discord Bridge — booting model (this takes ~30s on cold start)")
print("=" * 60)
from eva_inference_P2 import build_agent  # noqa: E402

agent = build_agent()
print("=" * 60)
print("  Model ready. Connecting to Discord...")
print("=" * 60)


# ============================================================
#  Discord client setup
# ============================================================
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

# 9B 模型不支持并发推理 —— 用全局锁把所有用户请求排成队
inference_lock = asyncio.Lock()

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("eva_discord")


# ============================================================
#  多用户历史隔离
#
#  ChatAgent 内部把模型权重和"对话状态"绑在同一个对象上，所有 Discord
#  用户共享同一个 agent。直接把多人的 .run() 跨调用会让 history_manager
#  互相覆盖（eva_chat_colab.py:122 注释里有原话警告）。
#
#  做法：每个 Discord user_id 持有自己的 UserSession 快照；在推理前把
#  它写进 agent，在推理后再把 agent 的变化抽回 session。因为
#  inference_lock 已经把请求排成队，swap-in / swap-out 不会有竞争。
#
#  详细的 sticky 状态白名单 + LRU + snapshot/restore 实现都在
#  eva_discord_sessions.py 里（这样 tests 可以 offline 验证）。
# ============================================================
sessions = SessionStore()


# ============================================================
#  图片附件 helper：把 Discord CDN 上的图下到临时文件
# ============================================================
SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}


# ============================================================
#  开机恢复：从 Discord channel 历史还原最近 N 轮对话
#
#  场景：bot 重启后 in-memory session 全没了，但 Discord channel 里
#  之前的对话还在。第一次见到某用户时，反向扫一段历史，把 (user, assistant)
#  配对成 N 轮塞回 history_manager，让 Eva 能"接着聊"而不是从陌生人开始。
# ============================================================
SKIP_BOT_MSG_PREFIXES = ("💭", "✅", "🧹", "❌", "🛠️")
RESET_COMMAND_LITERALS = {"/reset", "!reset", "reset"}


async def _fetch_recent_discord_turns(
    channel: discord.abc.Messageable,
    target_user_id: int,
    bot_user_id: int,
    current_msg_id: int,
    max_turns: int = MAX_RESUMED_TURNS,
) -> list[tuple[str, str]]:
    """从 channel.history 抓最近 N 轮 (user, assistant) 对话，oldest-first。

    过滤：
      - 跳过当前触发本次推理的 message（user 还没收到回复）
      - 跳过 bot 自己的 emoji-前缀状态消息（💭 思考中、✅ Research Data 等）
      - 跳过 user 输入的 /reset 命令
      - 跳过空文本（纯附件等）
    """
    fetched: list[tuple[str, str]] = []
    try:
        # 取 max_turns * 8 条粗略覆盖（考虑被过滤掉的状态/重置等）
        async for msg in channel.history(limit=max_turns * 8):
            if msg.id == current_msg_id:
                continue
            text = (msg.content or "").strip()
            if not text:
                continue

            if msg.author.id == bot_user_id:
                if text.startswith(SKIP_BOT_MSG_PREFIXES):
                    continue
                fetched.append(("assistant", text))
            elif msg.author.id == target_user_id:
                if text.lower() in RESET_COMMAND_LITERALS:
                    continue
                # 剥 @bot mention 的占位
                cleaned = text.replace(f"<@{bot_user_id}>", "").strip()
                if not cleaned:
                    continue
                fetched.append(("user", cleaned))

            if len(fetched) >= max_turns * 4:
                break
    except Exception as e:
        log.warning(f"_fetch_recent_discord_turns failed: {e}")
        return []

    # channel.history 是 newest-first，反转成 oldest-first
    fetched.reverse()

    # 配对成 (user, assistant) 对：扫一遍，遇 user 暂存，遇 assistant 配对
    turns: list[tuple[str, str]] = []
    pending_user: Optional[str] = None
    for role, content in fetched:
        if role == "user":
            pending_user = content
        elif role == "assistant" and pending_user is not None:
            turns.append((pending_user, content))
            pending_user = None

    return turns[-max_turns:]


async def _download_first_image(message: discord.Message) -> Optional[str]:
    """如果消息带图片附件，下载到临时文件并返回路径。无附件返回 None。"""
    for att in message.attachments:
        ext = Path(att.filename).suffix.lower()
        if ext not in SUPPORTED_IMAGE_EXTS:
            continue
        tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
        tmp.close()
        try:
            await att.save(tmp.name)
            log.info(f"saved image attachment to {tmp.name}")
            return tmp.name
        except Exception as e:
            log.warning(f"image download failed: {e}")
            try:
                os.unlink(tmp.name)
            except OSError:
                pass
            return None
    return None


# ============================================================
#  progress_callback 工厂：把 agent 内部状态字符串转 Discord 消息编辑
#  复刻旧版 Web_Chat.py 的"临时状态覆盖、永久内容追加"视觉
# ============================================================
def _make_progress_callback(loop: asyncio.AbstractEventLoop,
                            channel: discord.abc.Messageable,
                            state: dict):
    """返回一个同步可调用对象，agent 在它的线程里会调用它。

    state 里的 keys：
        current_msg          : discord.Message — 当前正在 edit 的占位消息
        persistent_content   : str — 累积保留的永久内容（搜索结果等）
    """

    def progress_callback(status_text: str) -> None:
        async def _update():
            # 用 emoji 前缀做信号：💭/🛠️ 是临时态，其它是永久内容
            is_temp = status_text.startswith(("💭", "🛠️", "Eva is"))

            if is_temp:
                display = (
                    f"{state['persistent_content']}\n\n{status_text}"
                    if state["persistent_content"]
                    else status_text
                )
            else:
                if state["persistent_content"]:
                    state["persistent_content"] += f"\n\n{status_text}"
                else:
                    state["persistent_content"] = status_text
                display = state["persistent_content"]

            try:
                # Discord 单条 2000 char 上限，超长就开新消息接续
                if len(display) > 1950:
                    state["current_msg"] = await channel.send(status_text)
                    state["persistent_content"] = status_text
                else:
                    await state["current_msg"].edit(content=display)
            except discord.errors.HTTPException as e:
                log.warning(f"discord edit failed: {e}")

        # agent.run 跑在 to_thread 里，进度回调里 schedule 到主 loop
        asyncio.run_coroutine_threadsafe(_update(), loop)

    return progress_callback


# ============================================================
#  Discord 事件
# ============================================================
@client.event
async def on_ready():
    log.info(f"connected to Discord as {client.user}")
    log.info(f"inference queue depth: {inference_lock._waiters and len(inference_lock._waiters) or 0}")
    if MAX_RESUMED_TURNS > 0:
        log.info(
            f"discord history resume: ENABLED ({MAX_RESUMED_TURNS} turn(s)) — "
            f"first message after restart will pull previous chat from channel history"
        )
    else:
        log.info(
            "discord history resume: DISABLED — fresh session on every restart. "
            "Set EVA_DISCORD_RESUME_TURNS=N env var to enable."
        )


@client.event
async def on_message(message: discord.Message):
    if message.author == client.user:
        return

    is_dm = isinstance(message.channel, discord.DMChannel)
    is_mention = client.user in message.mentions

    # 只在 DM 或被 @ 时响应
    if not (is_dm or is_mention):
        return

    # 剥离 @机器人 的 mention，剩下的才是真正的问题
    user_text = message.content.replace(f"<@{client.user.id}>", "").strip()
    if not user_text and not message.attachments:
        return

    # DM 默认主人身份（解锁 dual-subject memory）；服务器频道用 Discord 昵称
    user_name = "Rosm" if is_dm else (message.author.display_name or message.author.name)
    user_id = message.author.id

    # /reset 命令：让用户主动清自己的会话历史
    if user_text.lower().strip() in ("/reset", "!reset", "reset"):
        wiped = sessions.reset(user_id)
        await message.reply(
            "🧹 Conversation history cleared. Eva will see you as a fresh stranger next turn."
            if wiped else
            "_(no history to clear — you don't have an active session)_"
        )
        return

    log.info(f"[{user_name}#{user_id}] -> {user_text!r} (attachments={len(message.attachments)})")

    image_path: Optional[str] = await _download_first_image(message)
    if image_path and not user_text:
        user_text = "Describe this image."

    status_msg = await message.reply("💭 _Eva is analyzing your request..._")
    state = {"current_msg": status_msg, "persistent_content": ""}

    loop = asyncio.get_running_loop()
    progress_cb = _make_progress_callback(loop, message.channel, state)

    try:
        async with inference_lock:
            # ----- 多用户隔离：swap-in 用户的对话状态 -----
            sess = sessions.get_or_create(user_id, user_name)
            is_first_turn_after_restart = (sess.state is None and sess.turn_count == 0)
            restore_agent_state(agent, sess.state)

            # ----- 开机恢复：bot 重启后第一次见到这个用户 → 拉历史还原上下文 -----
            # 默认禁用（MAX_RESUMED_TURNS=0），需要时设 EVA_DISCORD_RESUME_TURNS env var 启用。
            if (is_first_turn_after_restart
                    and client.user is not None
                    and MAX_RESUMED_TURNS > 0):
                try:
                    resumed = await _fetch_recent_discord_turns(
                        channel=message.channel,
                        target_user_id=user_id,
                        bot_user_id=client.user.id,
                        current_msg_id=message.id,
                        max_turns=MAX_RESUMED_TURNS,
                    )
                    if resumed:
                        # history_manager 已经在 restore 里建出来了，user_name 还需要同步
                        agent.history_manager.set_user_name(user_name)
                        injected = inject_resumed_turns(agent, resumed)
                        log.info(
                            f"resumed {injected} turn(s) from Discord for "
                            f"{user_name}#{user_id} "
                            f"(history={len(agent.history_manager.history)}, "
                            f"compressed_kv={len(agent.history_manager.compressed_kv)})"
                        )
                except Exception as e:
                    log.warning(
                        f"discord history resume failed for {user_name}#{user_id}: {e}"
                    )

            log.info(
                f"loaded session for {user_name}#{user_id} "
                f"(turn {sess.turn_count + 1}, "
                f"history={len(agent.history_manager.history)} turns)"
            )

            try:
                # 把同步推理扔到线程池，避免阻塞 Discord 心跳
                eva_reply = await asyncio.to_thread(
                    agent.run, user_text, user_name, image_path, progress_cb,
                )
            finally:
                # ----- 多用户隔离：swap-out 推理后的最新状态 -----
                # 即使推理抛异常，也尽量把已发生的状态变更存回去，避免
                # 用户的对话被"半截"。失败的 turn 会自然成为 history 里的
                # 一条记录（没有 final answer），下一轮 Eva 会看到并继续。
                try:
                    sess.state = snapshot_agent_state(agent)
                    sess.turn_count += 1
                    sessions.touch(user_id)
                except Exception as snap_err:
                    log.error(f"failed to snapshot session: {snap_err}")

        # 推理完成 —— 区分两种视觉效果
        if not state["persistent_content"]:
            # 没调工具，删掉占位气泡保持频道干净
            try:
                await state["current_msg"].delete()
            except discord.errors.HTTPException:
                pass
        else:
            # 调过工具，把占位气泡改写成永久"资料卡片"
            final_card = "✅ **Eva's Research Data:**\n\n" + state["persistent_content"]
            try:
                await state["current_msg"].edit(content=final_card[:1990])
            except discord.errors.HTTPException:
                pass

        # === 调试钩子：记录 agent.run 返回的原始内容，便于复盘截断/异常 ===
        log.info(f"[{user_name}] raw eva_reply ({len(eva_reply)} chars): {eva_reply!r}")

        # Plan F hybrid：先抽 block math（$$...$$、```math```），剩下的 text
        # 段走 format_for_discord 转 Unicode；block math 渲染成 PNG 附件。
        # 这样 inline $...$ 仍由 Unicode 路径处理（保聊天流畅），display
        # math 用图片专业渲染（保数学表达清晰）。
        await _send_with_math_rendering(message, eva_reply, user_name=user_name)

    except Exception as e:
        log.error(f"inference error: {e}")
        traceback.print_exc()
        try:
            await message.reply(f"❌ I hit an error while thinking: `{type(e).__name__}: {e}`")
        except discord.errors.HTTPException:
            pass
    finally:
        # 清理临时图片
        if image_path:
            try:
                os.unlink(image_path)
            except OSError:
                pass


async def _send_with_math_rendering(
    message: discord.Message,
    eva_reply: str,
    *,
    user_name: str = "Master",
) -> None:
    """Plan F hybrid sender：抽 block math → PNG 附件，其余文本走 Unicode 转换。

    - inline `$x$` / `\\tau` 等：走 format_for_discord，转 Unicode
    - block `$$...$$` / ```math\\n...\\n```：matplotlib 渲染成 PNG 作附件
    - 渲染失败：fallback 到 ``` 代码块 inline 显示（不影响文本完整性）

    Discord 限制：单条消息 ≤2000 字符，最多 10 个附件；超长走 chunk 分页。
    """
    parts = split_text_and_block_math(eva_reply or "")

    final_text_segs: list[str] = []
    files: list[discord.File] = []

    for kind, content in parts:
        if kind == "text":
            # inline 数学 + 自然语言 → Unicode 转换器
            final_text_segs.append(format_for_discord(content))
        else:  # kind == "math"
            if len(files) >= 10:
                # Discord 上限 10 附件，后续 block math 降级到代码块
                final_text_segs.append(f"\n```\n{content}\n```\n")
                continue
            try:
                # 在 thread 里跑 matplotlib（CPU-bound，不阻塞 event loop）
                png_bytes = await asyncio.to_thread(
                    render_block_math_to_png, content
                )
                fname = f"eq_{len(files)+1}.png"
                files.append(discord.File(io.BytesIO(png_bytes), filename=fname))
                final_text_segs.append(f"\n[📐 {fname}]\n")
            except Exception as e:
                log.warning(
                    f"[{user_name}] block math render failed "
                    f"({e}); falling back to code block. src={content!r}"
                )
                final_text_segs.append(f"\n```\n{content}\n```\n")

    final_text = "".join(final_text_segs).strip() or "_(empty answer)_"
    log.info(
        f"[{user_name}] <- reply sent ({len(final_text)} chars, "
        f"{len(files)} math image(s))"
    )

    # 没附件 → 复用现有 chunked 发送器
    if not files:
        await _send_chunked(message, final_text)
        return

    # 有附件：尽量把所有 file 跟首段文本一起发，剩下文本继续 chunk
    limit = 1990
    if len(final_text) <= limit:
        await message.reply(final_text, files=files)
        return

    # 长文本 + 附件：找一个合适的切点（优先换行）
    cut = final_text.rfind("\n", 0, limit)
    if cut < limit // 2:
        cut = limit
    first_chunk = final_text[:cut]
    remaining = final_text[cut:].lstrip("\n")

    await message.reply(first_chunk, files=files)

    # 剩余文本继续分页（无附件）
    if remaining:
        # 复用 _send_chunked，但它的入口是 reply 而非 channel.send；自己写一段
        while remaining:
            if len(remaining) <= limit:
                await message.channel.send(remaining)
                break
            cut = remaining.rfind("\n", 0, limit)
            if cut < limit // 2:
                cut = limit
            await message.channel.send(remaining[:cut])
            remaining = remaining[cut:].lstrip("\n")


async def _send_chunked(message: discord.Message, text: str, limit: int = 1990) -> None:
    """超过 2000 字符的 Eva 回答自动分页发送，保留代码块完整性优先。"""
    if not text:
        text = "_(empty answer)_"
    if len(text) <= limit:
        await message.reply(text)
        return

    # 优先在换行处断开，避免把代码块切两半
    chunks: list[str] = []
    remaining = text
    while remaining:
        if len(remaining) <= limit:
            chunks.append(remaining)
            break
        cut = remaining.rfind("\n", 0, limit)
        if cut < limit // 2:  # 找不到合适的换行就硬切
            cut = limit
        chunks.append(remaining[:cut])
        remaining = remaining[cut:].lstrip("\n")

    first = True
    for chunk in chunks:
        if first:
            await message.reply(chunk)
            first = False
        else:
            await message.channel.send(chunk)


# ============================================================
#  入口
# ============================================================
if __name__ == "__main__":
    try:
        client.run(DISCORD_TOKEN, log_handler=None)  # 沿用我们的 logging 配置
    except discord.errors.LoginFailure:
        print("[FATAL] DISCORD_TOKEN 无效，请检查 .env 里的 token 是否过期")
        sys.exit(2)
