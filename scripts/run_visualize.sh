#!/bin/bash
# 运行交互式投影可视化工具

cd /media/yun/de2a43ce-446c-4a62-99b3-8ddc6ea1ef87/lidarcap

# 检查预处理参数目录 (优先) 或文件
PARAMS_DIR="/media/yun/de2a43ce-446c-4a62-99b3-8ddc6ea1ef87/lidarhuman26M/projection_params"
PARAMS_FILE="/media/yun/de2a43ce-446c-4a62-99b3-8ddc6ea1ef87/lidarhuman26M/projection_params.json"

if [ -d "$PARAMS_DIR" ]; then
    echo "发现预处理参数目录，自动加载..."
    python tools/visualize_projection.py --data_root /media/yun/de2a43ce-446c-4a62-99b3-8ddc6ea1ef87/lidarhuman26M --params "$PARAMS_DIR"
elif [ -f "$PARAMS_FILE" ]; then
    echo "发现预处理参数文件，自动加载..."
    python tools/visualize_projection.py --data_root /media/yun/de2a43ce-446c-4a62-99b3-8ddc6ea1ef87/lidarhuman26M --params "$PARAMS_FILE"
else
    echo "未发现预处理参数，使用默认参数..."
    python tools/visualize_projection.py --data_root /media/yun/de2a43ce-446c-4a62-99b3-8ddc6ea1ef87/lidarhuman26M
fi
