# 开发日志 - STM-LiDARCap科研创新项目

> 日志创建时间: 2026-03-28 18:18:14
> 日志格式版本: v1.0
> 计划文档: 科研创新改动计划.md v3.0

---

## 项目概述

**研究题目**: STM-LiDARCap: Spatio-Temporal Mamba for Efficient LiDAR Human Pose Estimation with Temporal Consistency Constraints

**核心任务**:
- T-001: 创建 modules/mamba_temporal.py (Mamba时序模块) ✅
- T-002: 修改 modules/regressor.py (集成Mamba模块) ✅
- T-003: 修改 modules/loss.py (添加时序一致性损失) ✅
- T-004: 新建 modules/multimodal_fusion.py (预留接口) ✅
- T-005: 更新 base.yaml (配置文件) ✅

---

## 进度汇总

| 指标 | 数值 |
|------|------|
| 总任务数 | 5 |
| 已完成 | 5 |
| 进行中 | 0 |
| 待开始 | 0 |
| 完成率 | 100% |
| 预计总工时 | 40小时 |
| 已用工时 | 8.5小时 |
| 剩余工时 | 0小时 |

---

## 详细任务记录

### T-001: 创建Mamba时序模块 ✅
**文件**: modules/mamba_temporal.py
**优先级**: P0 (立即实施)
**状态**: completed
**开始时间**: 2026-03-28 18:20:00
**结束时间**: 2026-03-28 18:25:00
**已完成工作量**: 100%
**剩余工作量**: 0%
**实际用时**: 5分钟
**备注**: 已创建MambaTemporal类,包含输入投影、2层Mamba、输出投影

### T-002: 修改regressor.py集成Mamba ✅
**文件**: modules/regressor.py
**优先级**: P0 (立即实施)
**状态**: completed
**开始时间**: 2026-03-28 18:25:00
**结束时间**: 2026-03-28 18:30:00
**已完成工作量**: 100%
**剩余工作量**: 0%
**实际用时**: 5分钟
**备注**: 已将RNN替换为MambaTemporal,添加导入语句

### T-003: 添加时序一致性损失 ✅
**文件**: modules/loss.py
**优先级**: P0 (立即实施)
**状态**: completed
**开始时间**: 2026-03-28 18:30:00
**结束时间**: 2026-03-28 18:40:00
**已完成工作量**: 100%
**剩余工作量**: 0%
**实际用时**: 10分钟
**备注**: 已添加TemporalConsistencyLoss类,集成到Loss类中

### T-004: 创建多模态融合预留接口 ✅
**文件**: modules/multimodal_fusion.py
**优先级**: P1 (后续扩展)
**状态**: completed
**开始时间**: 2026-03-28 18:40:00
**结束时间**: 2026-03-28 18:42:00
**已完成工作量**: 100%
**剩余工作量**: 0%
**实际用时**: 2分钟
**备注**: 已创建GCSPFusion预留接口类

### T-005: 更新配置文件 ✅
**文件**: base.yaml
**优先级**: P0 (立即实施)
**状态**: completed
**开始时间**: 2026-03-28 18:42:00
**结束时间**: 2026-03-28 18:45:00
**已完成工作量**: 100%
**剩余工作量**: 0%
**实际用时**: 3分钟
**备注**: 已添加MAMBA、TEMPORAL_LOSS、MULTIMODAL配置节

---

## 恢复点信息

**最新进度点**: 所有开发任务已完成
**下一步操作**: 进行训练验证和消融实验
**中断时间**: -

---

## 变更历史

| 时间 | 操作 | 详情 |
|------|------|------|
| 2026-03-28 18:18:14 | 创建日志 | 初始化开发日志 |
| 2026-03-28 18:25:00 | 完成T-001 | 创建modules/mamba_temporal.py |
| 2026-03-28 18:30:00 | 完成T-002 | 修改regressor.py集成Mamba |
| 2026-03-28 18:40:00 | 完成T-003 | 添加时序一致性损失到loss.py |
| 2026-03-28 18:42:00 | 完成T-004 | 创建多模态融合预留接口 |
| 2026-03-28 18:45:00 | 完成T-005 | 更新base.yaml配置文件 |
| 2026-03-28 18:45:00 | 完成开发 | 所有P0任务完成,项目可进入训练阶段 |

---

## 文件变更清单

| 文件 | 操作 | 行数变化 |
|------|------|---------|
| modules/mamba_temporal.py | 新建 | +48行 |
| modules/regressor.py | 修改 | +8行,-2行 |
| modules/loss.py | 修改 | +68行,-10行 |
| modules/multimodal_fusion.py | 新建 | +8行 |
| base.yaml | 修改 | +22行 |

**总代码变更**: +154行新增, -12行删除

---

## 性能优化记录 (2026-04-02)

### 优化背景
训练过程中发现GPU利用率不高，DataParallel效率低，需要优化训练性能。

### 优化项目

#### 1. DataParallel → DistributedDataParallel 迁移
**文件**: train.py
**问题**: nn.DataParallel 在每个 batch 都要复制模型、收集梯度，有大量开销
**解决方案**: 使用 DistributedDataParallel (DDP)
- 每个 GPU 独立进程，避免 GIL 瓶颈
- 梯度同步由 DDP 自动处理，效率更高
- 预期性能提升: 30-50%

#### 2. 混合精度训练兼容性修复
**文件**: modules/regressor.py
**问题**: pointnet2_ops CUDA 扩展不支持半精度 (float16)
**解决方案**: 添加 `DisableAutocast` 上下文管理器
- 在 PointNet2 CUDA 操作前后禁用 autocast
- 其他操作仍享受混合精度的加速和显存节省
- 兼容新旧版本 PyTorch

#### 3. h5py Pickle 错误修复
**文件**: datasets/lidarcap_dataset.py
**问题**: DDP 多进程需要 pickle Dataset 对象，但 HDF5 文件句柄不能序列化
**解决方案**: 预加载完成后关闭 HDF5 文件句柄，数据已在内存缓存中

#### 4. DDP device_id 警告修复
**文件**: train.py
**问题**: PyTorch 2.x 推荐显式指定每个进程使用的 GPU
**解决方案**: 在 `init_process_group` 中添加 `device_id` 参数，并兼容旧版本

#### 5. KNN 局部特征增强配置优化
**文件**: modules/regressor.py, config/train.yaml, modules/Transformer.py
**问题**: 
- `knn_layer=1` 导致大部分 Block 中的 knn_map、merge_map 参数不参与梯度计算
- DDP 报错 "parameters were not used in producing loss"

**解决方案**: 
- 将 `knn_layer` 从 1 改为 8
- Encoder 6 层全部使用 KNN 局部特征增强
- Decoder 8 层全部使用 KNN 局部特征增强
- 添加 PoinTr 配置节到 `config/train.yaml`，支持动态配置

**KNN 作用说明**:
- KNN (K-Nearest Neighbors) 局部特征增强
- 找每个点的 K=8 个最近邻
- 计算相对特征: (邻居特征 - 中心点特征)
- 聚合局部几何信息，增强模型对点云局部结构的感知

**配置参数**:
```yaml
PoinTr:
  trans_dim: 384             # Transformer 嵌入维度
  num_query: 224             # 查询点数量（粗点云点数）
  knn_layer: 8               # 使用 KNN 局部特征增强的层数
                            # 0: 不使用, 1-6: encoder前N层, >=7: 所有层
```

**显存开销说明**:
| knn_layer | 额外显存 | 计算开销 | 效果 |
|-----------|---------|---------|------|
| 1 | ~小 | ~小 | 基础局部感知 |
| 6 | ~中等 | ~中等 | Encoder 全部增强 |
| 8 | ~较大 | ~较大 | 全部增强 (当前配置) |

---

## 文件变更清单 (2026-04-02)

| 文件 | 操作 | 说明 |
|------|------|------|
| train.py | 重写 | DDP迁移、混合精度支持、分布式训练 |
| modules/regressor.py | 修改 | DisableAutocast、PoinTr配置读取 |
| modules/Transformer.py | 修改 | KNN参数始终参与计算 |
| datasets/lidarcap_dataset.py | 修改 | 预加载后关闭HDF5句柄 |
| config/train.yaml | 修改 | 添加PoinTr配置节、use_amp开启 |

---

## 变更历史 (2026-04-02)

| 时间 | 操作 | 详情 |
|------|------|------|
| 2026-04-02 14:00 | DDP迁移 | train.py 从 DataParallel 迁移到 DistributedDataParallel |
| 2026-04-02 15:30 | 混合精度修复 | 添加 DisableAutocast 处理旧CUDA扩展兼容性 |
| 2026-04-02 16:00 | Pickle修复 | 预加载后关闭 HDF5 文件句柄 |
| 2026-04-02 16:30 | device_id修复 | 添加 device_id 参数消除 DDP 警告 |
| 2026-04-02 22:00 | KNN优化 | knn_layer 从 1 改为 8，添加配置文件支持 |
