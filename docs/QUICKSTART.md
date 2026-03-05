# 数据集缓存使用指南

## 你的数据位置

数据路径：`/mnt/66bc2970-1c9c-4d90-a36c-2c7ecd0299d6/datasets/lidarhuman26M`

该目录包含：
- `train.txt` - 训练集ID列表
- `test.txt` - 测试集ID列表
- `labels/` - 原始数据文件夹

## 制作缓存

### 使用Shell脚本（推荐）

```bash
# 使用默认配置
./tools/dataset_cache/create.sh

# 或使用根目录快捷方式
./create_cache.sh

# 指定数据目录
./tools/dataset_cache/create.sh -d /mnt/66bc2970-1c9c-4d90-a36c-2c7ecd0299d6/datasets/lidarhuman26M

# 调整参数
./tools/dataset_cache/create.sh -d /mnt/66bc2970-1c9c-4d90-a36c-2c7ecd0299d6/datasets/lidarhuman26M -s 16 -n 512
```

更多选项请查看 `./tools/dataset_cache/create.sh --help` 或 `tools/dataset_cache/README.md`

### 使用Python脚本

```bash
python3 tools/dataset_cache/generate.py \
    --data-dir /mnt/66bc2970-1c9c-4d90-a36c-2c7ecd0299d6/datasets/lidarhuman26Mser \
    --output-dir ./cache
```

脚本会自动读取train.txt和test.txt，从原始数据制作对应的缓存文件。

## 更新配置

编辑 `base.yaml`，修改数据集路径：

```yaml
TrainDataset:
  dataset_path: './cache/lidarcap_train.hdf5'  # 改为缓存文件路径
  # 其他配置保持不变...

TestDataset:
  dataset_path: './cache/lidarcap_test.hdf5'   # 改为缓存文件路径
  # 其他配置保持不变...
```

## 使用缓存数据集训练

在代码中使用 [`CachedLidarCapDataset`](datasets/lidarcap_dataset.py:558):

```python
from datasets import CachedLidarCapDataset

# 训练集
train_dataset = CachedLidarCapDataset({
    'dataset_path': './cache/lidarcap_train.hdf5',
    'seqlen': 16,
    'use_aug': True,
    'preload': False  # 如果内存足够，可设为True
})

# 测试集
test_dataset = CachedLidarCapDataset({
    'dataset_path': './cache/lidarcap_test.hdf5',
    'seqlen': 16,
    'use_aug': False
})
```

## 可选参数

如果你想调整参数，可以添加以下选项：

```bash
python tools/create_dataset_cache.py \
    --data-dir /mnt/66bc2970-1c9c-4d90-a36c-2c7ecd0299d6/datasets/lidarhuman26M \
    --output-dir ./cache \
    --seqlen 16 \          # 序列长度，默认16
    --npoints 512 \        # 点云采样点数，默认512
    --chunk-size 1000       # 分块大小，默认1000
```

## 注意事项

- 脚本需要CUDA环境，因为原始数据需要GPU加速处理
- 输出文件为 `lidarcap_train.hdf5` 和 `lidarcap_test.hdf5`
