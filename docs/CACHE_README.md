# 数据集缓存功能使用说明

**快速开始请查看: [QUICKSTART.md](QUICKSTART.md)**

这个文档介绍如何使用新的HDF5缓存功能来提高数据读取效率。

## 背景

原始数据集由大量小文件组成（JSON、PLY等），转换为HDF5后仍然需要打开多个小文件（每个数据集ID一个文件）。这种方式有以下问题：
- 打开大量小文件开销大
- 文件系统碎片化严重
- 传输速度慢

新的缓存功能将多个HDF5文件合并为单个大文件，支持直接从原始数据创建大文件，避免中间小文件步骤。

## 使用方法

### 1. 创建HDF5缓存文件

推荐使用Shell脚本：

```bash
# 使用默认配置
./tools/dataset_cache/create.sh

# 或使用根目录快捷方式
./create_cache.sh

# 指定数据目录
./tools/dataset_cache/create.sh -d /path/to/lidarhuman26M

# 调整参数
./tools/dataset_cache/create.sh -d /path/to/lidarhuman26M -s 16 -n 512
```

或者直接使用Python脚本：

```bash
python tools/dataset_cache/generate.py \
    --data-dir /path/to/lidarhuman26M \
    --output-dir ./cache
```

该脚本会：
- 自动读取 `train.txt` 和 `test.txt` 获取ID列表
- 从原始数据生成训练集和测试集的HDF5缓存文件
- 输出文件：`lidarcap_train.hdf5` 和 `lidarcap_test.hdf5`

可选参数（Shell脚本）：
- `-d, --data-dir`: 原始数据目录
- `-o, --output-dir`: 输出目录
- `-s, --seqlen`: 序列长度（默认16）
- `-n, --npoints`: 点云采样点数（默认512）
- `-c, --chunk-size`: 分块大小（默认1000）
- `--no-compress`: 禁用压缩（默认启用gzip压缩）

### 2. 验证缓存文件

```bash
# 使用Python验证
python -c "
import h5py
f = h5py.File('cache/lidarcap_train.hdf5', 'r')
print('数据集数量:', len(f['dataset_ids']))
print('总帧数:', len(f['pose']))
print('Keys:', list(f.keys()))
f.close()
"
```

## 在代码中使用缓存数据集

### 使用 `CachedLidarCapDataset` 类

```python
from datasets import CachedLidarCapDataset

# 基本使用
dataset = CachedLidarCapDataset({
    'dataset_path': '/path/to/merged_cache.hdf5',
    'seqlen': 16,
    'drop_first_n': 0,
})

# 使用 DataLoader
from torch.utils.data import DataLoader

loader = DataLoader(dataset, batch_size=4, shuffle=True)
for batch in loader:
    # 训练代码
    pass
```

### 预加载到内存

如果你的内存足够，可以启用预加载：

```python
dataset = CachedLidarCapDataset({
    'dataset_path': '/path/to/merged_cache.hdf5',
    'seqlen': 16,
    'preload': True,  # 预加载到内存
})
```

预加载后，所有数据会加载到内存中，数据读取速度最快，但需要足够的系统内存。

### 使用工具函数创建缓存

```python
from datasets import create_cache_dataset

# 自动合并目录下的所有HDF5文件
cache_path = create_cache_dataset(
    dataset_path='/path/to/hdf5/files',
    output_path='/path/to/output.hdf5',
    compress=True,
    chunk_size=1000
)
```

## 性能对比

| 方式 | 首次读取时间 | 随机访问 | 传输速度 |
|------|-------------|----------|---------|
| 多小HDF5文件 | 慢（需要打开多个文件） | 慢 | 慢 |
| 单个大HDF5文件 | 中等 | 快 | 快 |
| 内存预加载 | 一次加载后最快 | 最快 | - |

## 注意事项

1. **压缩 vs 速度**: 启用gzip压缩会减小文件大小（通常减少30-50%），但读取时需要解压缩，会有一定CPU开销。如果存储空间充足且追求极致读取速度，可以禁用压缩。

2. **分块大小**: chunk-size参数影响读取性能。对于按序列访问（16帧连续读取），建议使用较小的chunk-size（100-500）；对于随机访问，可以使用更大的chunk-size。

3. **内存使用**: 启用preload=True会将整个数据集加载到内存。确保你有足够的内存：
   - 粗略估算：总帧数 × 512点 × 3坐标 × 4字节 ≈ 每帧6KB数据
   - 1万帧约需要60MB基础数据，加上其他字段可能需要100-200MB

4. **文件锁定**: HDF5文件在打开时会被锁定。如果需要在多个进程间共享数据，建议使用多线程而非多进程，或者使用HDF5的SWMR（Single Writer Multiple Reader）模式。

## 示例工作流

### 场景1: 从原始数据创建缓存

```bash
# 直接从原始数据创建缓存文件
python tools/create_dataset_cache.py \
    --data-dir /data/lidarhuman26M \
    --output-dir ./cache

# 在训练中使用（修改配置）
# dataset_path: ./cache/lidarcap_train.hdf5
```

### 场景2: 使用预加载加速

```python
from datasets import CachedLidarCapDataset

# 使用缓存数据集
dataset = CachedLidarCapDataset({
    'dataset_path': './cache/lidarcap_train.hdf5',
    'seqlen': 16,
})
```

## 制作训练集和测试集缓存

项目提供了便捷脚本用于制作训练集和测试集的缓存文件：

```bash
# 从原始数据创建训练集和测试集缓存
python tools/create_dataset_cache.py \
    --data-dir /path/to/lidarhuman26M \
    --output-dir ./cache \
    --seqlen 16 \
    --npoints 512
```

该脚本会：
1. 自动读取 train.txt 和 test.txt 获取ID列表
2. 从原始数据生成训练集和测试集的HDF5缓存文件
3. 输出文件：lidarcap_train.hdf5 和 lidarcap_test.hdf5

### 更新配置文件

制作完成后，更新 [`base.yaml`](../base.yaml:60) 配置：

```yaml
TrainDataset:
  dataset_path: './cache/lidarcap_train.hdf5'  # 修改为缓存文件路径
  seqlen: 16
  # 其他配置保持不变...

TestDataset:
  dataset_path: './cache/lidarcap_test.hdf5'   # 修改为缓存文件路径
  seqlen: 16
  # 其他配置保持不变...
```

### 使用 CachedLidarCapDataset

在代码中使用 [`CachedLidarCapDataset`](datasets/lidarcap_dataset.py:558)：

```python
from datasets import CachedLidarCapDataset

# 训练集
train_dataset = CachedLidarCapDataset({
    'dataset_path': './cache/lidarcap_train.hdf5',
    'seqlen': 16,
    'use_aug': True,
    'preload': False  # 如果内存足够，可以设为True
})

# 测试集
test_dataset = CachedLidarCapDataset({
    'dataset_path': './cache/lidarcap_test.hdf5',
    'seqlen': 16,
    'use_aug': False
})
```
