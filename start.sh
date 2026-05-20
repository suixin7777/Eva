#!/usr/bin/env bash
# ============================================================
# start.sh — Eva Discord bot 容器入口
#
# 用途：
#   - 作为 RunPod Pod 的 "Container Start Command"
#   - 或者 systemd 服务 ExecStart
# 行为：
#   1. 跳到项目根
#   2. 把 HuggingFace cache 指到 network volume，避免 Pod 重启重下载
#   3. 确保 supervisord 安装
#   4. exec 启 supervisord 前台模式（容器主进程，不能 daemon）
# ============================================================
set -euo pipefail

PROJECT_DIR="${EVA_PROJECT_DIR:-/workspace/Eva_new}"
cd "$PROJECT_DIR"

# HF / torch 缓存全部塞到 network volume，避免冷启动重下 mpnet (420MB)
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export SENTENCE_TRANSFORMERS_HOME="${SENTENCE_TRANSFORMERS_HOME:-$HF_HOME/sentence-transformers}"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$SENTENCE_TRANSFORMERS_HOME"

# 日志目录（supervisor + python 自身的）落到 volume，重启后还能看
# 注意：必须用 `export VAR=value` 一行搞定；分开写 `export VAR` 不会把 value 带出去
export EVA_LOG_DIR="${EVA_LOG_DIR:-/workspace/eva_logs}"
mkdir -p "$EVA_LOG_DIR"

# 第一次启动可能没装 supervisor —— 自动补齐（幂等）
if ! command -v supervisord >/dev/null 2>&1; then
    echo "[start.sh] supervisord 未安装，正在 pip 安装..."
    pip install --quiet supervisor
fi

# 第一次启动可能也没装项目依赖
if [[ ! -f "$HF_HOME/.deps_installed" ]]; then
    echo "[start.sh] 首次启动，安装 requirements.txt..."
    pip install -r requirements.txt
    touch "$HF_HOME/.deps_installed"
fi

echo "[start.sh] 启动 supervisord（前台模式）"
# exec 让 supervisord 接管 PID 1，SIGTERM/SIGINT 能正确传到 python 进程
exec supervisord -n -c "$PROJECT_DIR/supervisord.conf"
