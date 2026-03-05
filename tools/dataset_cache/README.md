# 数据集缓存制作工具

从原始数据生成HDF5格式的训练集和测试集缓存文件。

## 快速开始

### 使用Shell脚本（推荐）

```bash
# 使用默认配置
./tools/dataset_cache/create.sh

# 或使用根目录快捷方式
./create_cache.sh

# 指定数据目录
./tools/dataset_cache/create.sh -d /path/to/data

# 自定义参数
./tools/dataset_cache/create.sh -d /path/to/data -o ./mycache -s 16 -n 512

# 禁用压缩
./tools/dataset_cache/create.sh --no-compress
```

### 使用Python脚本

```bash
python3 tools/dataset_cache/generate.py \
    --data-dir /path/to/data \
    --output-dir ./cache \
    --seqlen 16 \
    --npoints 512 \
    --chunk-size 1000
```

## 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-d, --data-dir` | 原始数据目录 | `/mnt/66bc2970-1c9c-4d90-a36c-2c7ecd0299d6/datasets/lidarhuman26M` |
| `-o, --output-dir` | 输出目录 | `./cache` |
| `-s, --seqlen` | 序列长度 | 16 |
| `-n, --npoints` | 点云点数 | 512 |
| `-c, --chunk-size` | 分块大小 | 1000 |
| `--no-compress` | 禁用压缩 | 启用gzip压缩 |
| `-h, --help` | 显示帮助信息 | - |

## 环境变量

也可以通过环境变量设置配置：

```bash
DATA_DIR=/path/to/data OUTPUT_DIR=./cache ./tools/dataset_cache/create.sh
```

支持的变量：
- `DATA_DIR`: 原始数据目录
- `OUTPUT_DIR`: 输出目录
- `SEQLEN`: 序列长度
- `NPOINTS`: 点云点数
- `CHUNK_SIZE`: 分块大小
- `COMPRESS`: 压缩选项（空=禁用，`--compress`=启用）

## 输出文件

脚本会在输出目录下生成两个HDF5文件：
- `lidarcap_train.hdf5` - 训练集缓存
- `lidarcap_test.hdf5` - 测试集缓存

## 使用缓存

### 更新配置文件

编辑 `base.yaml`：

```yaml
TrainDataset:
  dataset_path: './cache/lidarcap_train.hdf5'
  seqlen: 16

TestDataset:
  dataset_path: './cache/lidarcap_test.hdf5'
  seqlen: 16
```

### 在代码中使用

```python
from datasets import CachedLidarCapDataset

train_dataset = CachedLidarCapDataset({
    'dataset_path': './cache/lidarcap_train.hdf5',
    'seqlen': 16,
})

test_dataset = CachedLidarCapDataset({
    'dataset_path': './cache/lidarcap_test.hdf5',
    'seqlen': 16,
})
```

## 注意事项

- 脚本需要CUDA环境，因为原始数据处理需要GPU加速
- 原始数据目录必须包含 `train.txt` 和 `test.txt` 文件
- HDF5文件使用gzip压缩（可禁用）以节省存储空间
- 分块大小影响读取性能，建议根据使用场景调整
