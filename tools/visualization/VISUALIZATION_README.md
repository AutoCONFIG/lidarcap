# LiDARCap 数据集可视化工具

本目录包含用于可视化 LiDARCap 数据集的工具脚本。

## 🚀 快速启动（推荐）

在项目根目录使用 `visualize.sh` 脚本快速启动可视化：

```bash
# 可视化单帧
./visualize.sh -p data/0001.hdf5 -i 0001 -f 10

# 可视化序列
./visualize.sh -p data/0001.hdf5 -t sequence -s 0 -e 100

# 显示帮助
./visualize.sh -h
```

---

## 📝 Shell 脚本参数说明

`visualize.sh` 支持以下参数：

| 参数 | 简写 | 说明 | 默认值 |
|------|------|------|--------|
| `--path` | `-p` | HDF5数据集路径 | 必填 |
| `--id` | `-i` | 数据集ID | 自动推断 |
| `--frame` | `-f` | 单帧模式: 帧索引 | 0 |
| `--start` | `-s` | 序列模式: 起始帧 | 0 |
| `--end` | `-e` | 序列模式: 结束帧 | 全部 |
| `--mode` | `-m` | 可视化模式 | both |
| `--output` | `-o` | 输出目录 | ./visualization |
| `--tool` | `-t` | 工具类型: single/sequence | single |
| `--visualizer` | `-v` | 可视化库 | open3d |
| `--device` | `-d` | 计算设备 | cuda |
| `--help` | `-h` | 显示帮助 | - |

### Shell 脚本示例

```bash
# 单帧可视化
./visualize.sh -p data/0001.hdf5 -i 0001 -f 0

# 序列可视化（前100帧）
./visualize.sh -p data/0001.hdf5 -t sequence -s 0 -e 100 -o ./vis_output

# 使用Matplotlib（如果Open3D有问题）
./visualize.sh -p data/0001.hdf5 -f 0 -v matplotlib

# CPU模式（显存不足时）
./visualize.sh -p data/0001.hdf5 -t sequence -d cpu
```

---

## 📦 依赖安装

```bash
# 必需依赖
pip install matplotlib h5py numpy torch tqdm

# 可选依赖（推荐安装）
pip install open3d  # 更好的3D可视化体验
pip install opencv-python  # 用于生成视频
```

---

## 🔧 工具1: visualize_dataset.py

用于可视化单帧数据的工具。

### 基本用法

```bash
# 可视化单帧（显示窗口）
python visualize_dataset.py \
    --dataset_path /path/to/0001.hdf5 \
    --dataset_id 0001 \
    --frame_idx 0

# 保存为文件而不是显示
python visualize_dataset.py \
    --dataset_path /path/to/0001.hdf5 \
    --frame_idx 10 \
    --save_dir ./visual_output

# 只显示点云（不显示SMPL）
python visualize_dataset.py \
    --dataset_path /path/to/0001.hdf5 \
    --frame_idx 0 \
    --mode pointcloud

# 使用Matplotlib（如果Open3D有问题）
python visualize_dataset.py \
    --dataset_path /path/to/0001.hdf5 \
    --frame_idx 0 \
    --visualizer matplotlib
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--dataset_path` | HDF5数据集路径 | 必填 |
| `--dataset_id` | 数据集ID | 自动推断 |
| `--frame_idx` | 帧索引 | 0 |
| `--mode` | 可视化模式 [pointcloud/smlp/both] | both |
| `--save_dir` | 保存目录（不指定则显示窗口） | None |
| `--visualizer` | 可视化库 [open3d/matplotlib] | open3d |

---

## 🎬 工具2: visualize_sequence.py

用于可视化连续多帧序列，可生成视频。

### 基本用法

```bash
# 可视化连续100帧
python visualize_sequence.py \
    --dataset_path /path/to/0001.hdf5 \
    --dataset_id 0001 \
    --start_frame 0 \
    --end_frame 100 \
    --output_dir ./sequence_vis

# 可视化整个序列
python visualize_sequence.py \
    --dataset_path /path/to/0001.hdf5 \
    --dataset_id 0001 \
    --output_dir ./sequence_vis

# 对比预测结果和真值
python visualize_sequence.py \
    --dataset_path /path/to/0001.hdf5 \
    --dataset_id 0001 \
    --compare /path/to/pred_rotmats.npy \
    --output_dir ./comparison
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--dataset_path` | HDF5数据集路径 | 必填 |
| `--dataset_id` | 数据集ID | sequence |
| `--start_frame` | 起始帧 | 0 |
| `--end_frame` | 结束帧 | 全部 |
| `--output_dir` | 输出目录 | visualization |
| `--compare` | 预测结果文件(.npy) | None |
| `--device` | 计算设备 [cuda/cpu] | cuda |

---

## 📊 可视化输出说明

### 单帧可视化 (`visualize_dataset.py`)

1. **点云可视化**
   - 显示LiDAR扫描的人体点云
   - 按高度着色（Y轴）
   - 红色三角形表示关节位置

2. **SMPL网格可视化**
   - 显示SMPL模型生成的人体网格
   - 6890个顶点
   - 可以直观看到人体形状

### 序列可视化 (`visualize_sequence.py`)

生成3个子图：
- 左图：LiDAR点云
- 中图：骨架关节（带骨骼连接）
- 右图：SMPL人体网格

### 对比可视化

左右并排显示：
- 左图：Ground Truth（蓝色）
- 右图：Prediction（红色）

---

## 💡 使用建议

### 查看数据质量
```bash
# 随机查看几帧数据
for i in 0 50 100 150 200; do
    python visualize_dataset.py \
        --dataset_path /path/to/0001.hdf5 \
        --frame_idx $i \
        --save_dir ./quality_check
done
```

### 检查动作序列
```bash
# 可视化完整动作序列
python visualize_sequence.py \
    --dataset_path /path/to/0001.hdf5 \
    --start_frame 0 \
    --end_frame 200 \
    --output_dir ./action_vis
```

### 对比模型预测
```bash
# 先运行模型预测（保存为npy文件）
python train.py --eval --visual ...

# 然后对比
python visualize_sequence.py \
    --dataset_path /path/to/0001.hdf5 \
    --compare ./output/pred_rotmats.npy \
    --output_dir ./model_comparison
```

---

## 🔍 常见问题

### Q: Open3D安装失败怎么办？
A: 可以使用Matplotlib作为备选：
```bash
python visualize_dataset.py ... --visualizer matplotlib
```

### Q: 显存不足？
A: 使用CPU模式：
```bash
python visualize_sequence.py ... --device cpu
```

### Q: 生成的视频无法播放？
A: 确保安装了正确版本的OpenCV：
```bash
pip install opencv-python==4.5.5.64
```

### Q: 点云显示为空白？
A: 检查数据范围，可能是坐标值过大/过小导致视角问题。可以添加打印语句查看points的min/max值。

---

## 📁 输出文件结构

```
visualization/
├── frame_0000.png      # 单帧图像
├── frame_0001.png
├── ...
├── 0001_sequence.mp4   # 生成的视频（可选）
└── comparison/         # 对比结果
    ├── compare_0000.png
    └── ...
```

---

## 📚 相关文件

- 数据预处理：`datasets/preprocess/lidarcap.py`
- 数据集类：`datasets/lidarcap_dataset.py`
- 模型定义：`modules/regressor.py`
