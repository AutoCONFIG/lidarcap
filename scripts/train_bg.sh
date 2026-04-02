#!/bin/bash
#
# 后台训练脚本 - 简单易用版
# 直接执行即可启动/恢复训练
#

set -e

SESSION_NAME="lidarcap_train"
CONFIG_DIR="config"

# 切换到项目根目录
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

# 检查tmux
if ! command -v tmux &> /dev/null; then
    echo "错误: tmux未安装，请先安装: sudo apt install tmux"
    exit 1
fi

# 检查会话是否存在
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "===================================================================="
    echo "正在连接到训练会话..."
    echo ""
    echo "【退出但不停止训练】按 Ctrl+B 然后按 D"
    echo "【停止训练】按 Ctrl+C"
    echo "===================================================================="
    echo ""
    sleep 1
    tmux attach-session -t "$SESSION_NAME"
    exit 0
fi

# 启动新训练
echo "===================================================================="
echo "启动后台训练"
echo ""
echo "【退出但不停止训练】按 Ctrl+B 然后按 D"
echo "【停止训练】按 Ctrl+C"
echo "【恢复会话】再次执行此脚本"
echo "===================================================================="
echo ""
sleep 1

tmux new-session -s "$SESSION_NAME" -c "$PROJECT_ROOT"
tmux send-keys -t "$SESSION_NAME" "cd $PROJECT_ROOT && python train.py --config-dir $CONFIG_DIR" Enter
tmux attach-session -t "$SESSION_NAME"
