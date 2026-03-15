#!/bin/bash

# =====================================================
# LiDARCap 一键环境安装脚本
# 使用 env 目录下的 environment.yml 和 requirements 文件
# =====================================================

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}    LiDARCap 一键环境安装脚本${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# 检查conda
if ! command -v conda &> /dev/null; then
    echo -e "${RED}错误: 未找到 conda${NC}"
    exit 1
fi

# 检查env目录
if [ ! -d "env" ]; then
    echo -e "${RED}错误: 未找到 env 目录${NC}"
    exit 1
fi

# 检查配置文件
for file in "env/environment.yml" "env/requirements_1.txt" "env/requirements_2.txt"; do
    if [ ! -f "$file" ]; then
        echo -e "${RED}错误: 未找到 $file${NC}"
        exit 1
    fi
done

ENV_NAME="lidarcap"

# 如果环境存在，询问是否删除
if conda env list | grep -q "^${ENV_NAME} "; then
    echo -e "${YELLOW}环境 '${ENV_NAME}' 已存在${NC}"
    read -p "是否删除并重新创建? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        conda env remove -n "${ENV_NAME}" -y
    else
        echo -e "${YELLOW}取消安装${NC}"
        exit 0
    fi
fi

# 步骤1: 创建conda环境
echo -e "${GREEN}步骤 1/3: 创建 Conda 环境${NC}"
conda env create -f env/environment.yml

# 步骤2: 激活环境
echo -e "${GREEN}步骤 2/3: 激活环境并安装 pip 依赖${NC}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${ENV_NAME}

# 升级pip
pip install --upgrade pip

# 步骤3: 安装pip依赖
echo -e "${GREEN}步骤 3/3: 安装 pip 依赖包${NC}"

echo -e "${YELLOW}安装 requirements_1.txt...${NC}"
pip install --no-build-isolation -r env/requirements_1.txt

echo -e "${YELLOW}安装 requirements_2.txt (本地库)...${NC}"
pip install --no-build-isolation -r env/requirements_2.txt

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}    安装完成！${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${YELLOW}激活环境命令: conda activate ${ENV_NAME}${NC}"
