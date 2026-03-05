# 工具脚本

本目录包含各种工具脚本，用于数据处理、缓存制作等辅助任务。

## 工具列表

### dataset_cache - 数据集缓存工具

从原始数据生成HDF5格式的训练集和测试集缓存文件。

**快速开始：**

```bash
# 使用默认配置
./tools/dataset_cache/create.sh

# 或使用根目录快捷方式
./create_cache.sh

# 自定义参数
./tools/dataset_cache/create.sh -d /path/to/data -o ./cache -s 16 -n 512
```

**详细文档：** 查看 [dataset_cache/README.md](dataset_cache/README.md)

---

## 添加新工具

在 `tools/` 目录下为每个工具创建独立子目录：

```
tools/
├── dataset_cache/     # 数据集缓存工具
│   ├── create.sh      # Shell脚本
│   ├── generate.py    # Python实现
│   └── README.md      # 工具文档
└── your_tool/         # 你的新工具
    ├── run.sh         # Shell脚本
    ├── main.py        # Python实现
    └── README.md       # 工具文档
```

**工具目录规范：**

1. 每个工具一个独立子目录
2. 提供Shell脚本作为主要入口
3. 提供Python脚本实现核心逻辑
4. 包含README.md说明文档
5. 文件命名简洁明了（如 create.sh, generate.py）

**Shell脚本规范：**

1. 添加清晰的注释说明
2. 支持命令行参数
3. 支持环境变量配置
4. 提供 `--help` 帮助信息
5. 显示执行进度和结果
6. 设置可执行权限 `chmod +x`
