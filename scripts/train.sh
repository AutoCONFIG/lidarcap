#!/bin/bash
#
# 训练STM-LiDARCap模型
# 配置从 config/runtime.yaml 读取
#

set -e

# 默认参数
CONFIG_DIR="config"

# 显示帮助
show_help() {
    cat << EOF
训练STM-LiDARCap模型

用法: ./scripts/train.sh [选项]

选项:
  -c, --config-dir DIR   配置文件目录 (默认: config)
  -h, --help             显示帮助

配置文件:
  所有训练参数从 config/runtime.yaml 读取，包括:
  - dataset: 数据集名称
  - gpu_id: GPU ID
  - resume: 恢复训练路径
  - output_dir: 输出目录
  - batch_size, num_epochs, lr 等

示例:
  # 使用默认配置
  ./scripts/train.sh

  # 使用自定义配置目录
  ./scripts/train.sh -c my_config

输出:
  模型保存在: output/run_YYYYMMDDHHMM/
EOF
}

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--config-dir)
            CONFIG_DIR="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "错误: 未知参数 $1"
            show_help
            exit 1
            ;;
    esac
done

# 检查配置目录
if [ ! -d "$CONFIG_DIR" ]; then
    echo "错误: 配置目录不存在: $CONFIG_DIR"
    exit 1
fi

# 切换到项目根目录
cd "$(dirname "$0")/.."

echo "===================================================================="
echo "开始训练"
echo "===================================================================="
echo "配置目录: $CONFIG_DIR"
echo "===================================================================="
echo ""

python train.py --config-dir "$CONFIG_DIR"
