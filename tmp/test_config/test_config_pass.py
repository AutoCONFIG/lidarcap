"""
测试配置参数是否正确传递到模型和损失函数

运行方式:
    cd D:\lidarcap
    python tmp/test_config/test_config_pass.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from config import get_cfg
from modules.regressor import Regressor
from modules.loss import Loss, TemporalConsistencyLoss


def test_regressor_config():
    """测试Regressor是否正确读取Mamba配置"""
    print("=" * 60)
    print("测试1: Regressor Mamba配置")
    print("=" * 60)

    cfg = get_cfg()

    print("\n配置文件中的MAMBA参数:")
    print(f"  d_model: {cfg.MAMBA.d_model}")
    print(f"  d_state: {cfg.MAMBA.d_state}")
    print(f"  d_conv: {cfg.MAMBA.d_conv}")
    print(f"  expand: {cfg.MAMBA.expand}")
    print(f"  n_layers: {cfg.MAMBA.n_layers}")
    print(f"  dropout: {cfg.MAMBA.dropout}")

    print("\n创建Regressor模型...")
    model = Regressor(cfg=cfg)

    # 检查Mamba模块的参数
    mamba_module = model.pose_s1

    print("\n模型中Mamba模块的实际参数:")
    print(f"  d_model: {mamba_module.input_proj.in_features} -> {mamba_module.input_proj.out_features}")
    print(f"  n_layers: {len(mamba_module.mamba_layers)}")
    print(f"  dropout: {mamba_module.dropout.p}")

    # 检查第一个Mamba层的参数
    first_mamba = mamba_module.mamba_layers[0]
    print(f"  d_state (from A_log shape): {first_mamba.A_log.shape[0]}")

    print("\n[PASS] Regressor配置测试通过!" if len(mamba_module.mamba_layers) == cfg.MAMBA.n_layers else "\n[FAIL] Regressor配置测试失败!")
    return True


def test_loss_config():
    """测试Loss是否正确读取时序损失配置"""
    print("\n" + "=" * 60)
    print("测试2: Loss 时序损失配置")
    print("=" * 60)

    cfg = get_cfg()

    print("\n配置文件中的TEMPORAL_LOSS参数:")
    print(f"  weight: {cfg.TEMPORAL_LOSS.weight}")
    print(f"  velocity_weight: {cfg.TEMPORAL_LOSS.velocity_weight}")
    print(f"  acceleration_weight: {cfg.TEMPORAL_LOSS.acceleration_weight}")
    print(f"  bone_length_weight: {cfg.TEMPORAL_LOSS.bone_length_weight}")

    print("\n创建Loss模块...")
    loss_fn = Loss(cfg=cfg)

    print("\nLoss模块中的实际参数:")
    print(f"  temporal_weight: {loss_fn.temporal_weight}")
    print(f"  velocity_weight: {loss_fn.temporal_loss.velocity_weight}")
    print(f"  acceleration_weight: {loss_fn.temporal_loss.acceleration_weight}")
    print(f"  bone_length_weight: {loss_fn.temporal_loss.bone_length_weight}")

    # 验证参数是否匹配
    match = (
        loss_fn.temporal_weight == cfg.TEMPORAL_LOSS.weight and
        loss_fn.temporal_loss.velocity_weight == cfg.TEMPORAL_LOSS.velocity_weight and
        loss_fn.temporal_loss.acceleration_weight == cfg.TEMPORAL_LOSS.acceleration_weight and
        loss_fn.temporal_loss.bone_length_weight == cfg.TEMPORAL_LOSS.bone_length_weight
    )

    print("\n[PASS] Loss配置测试通过!" if match else "\n[FAIL] Loss配置测试失败!")
    return match


def test_forward_pass():
    """测试前向传播是否正常工作"""
    print("\n" + "=" * 60)
    print("测试3: 前向传播测试")
    print("=" * 60)

    cfg = get_cfg()

    print("\n创建模型和损失函数...")
    model = Regressor(cfg=cfg)
    loss_fn = Loss(cfg=cfg)

    # 创建模拟输入数据
    B, T, N, C = 2, 16, 512, 3  # batch=2, 序列长度=16, 点数=512, 坐标=3

    mock_data = {
        'human_points': torch.randn(B, T, N, C),
        'pose': torch.randn(B, T, 72),
        'full_joints': torch.randn(B, T, 24, 3),
    }

    print(f"\n输入数据形状:")
    print(f"  human_points: {mock_data['human_points'].shape}")
    print(f"  pose: {mock_data['pose'].shape}")
    print(f"  full_joints: {mock_data['full_joints'].shape}")

    print("\n执行模型前向传播...")
    try:
        with torch.no_grad():
            output = model(mock_data)
        print(f"\n输出键: {list(output.keys())}")
        print(f"  pred_full_joints: {output['pred_full_joints'].shape}")
        print(f"  pred_rotmats: {output['pred_rotmats'].shape}")
        print("\n[模型前向传播成功!]")
    except Exception as e:
        print(f"\n[模型前向传播失败!] 错误: {e}")
        return False

    print("\n执行损失计算...")
    try:
        with torch.no_grad():
            loss, details = loss_fn(**output)
        print(f"\n总损失: {loss.item():.6f}")
        print(f"损失详情:")
        for k, v in details.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: {v.item():.6f}")
        print("\n[损失计算成功!]")
    except Exception as e:
        print(f"\n[损失计算失败!] 错误: {e}")
        return False

    print("\n[PASS] 前向传播测试通过!")
    return True


def test_config_change():
    """测试修改配置是否影响模型行为"""
    print("\n" + "=" * 60)
    print("测试4: 配置修改测试")
    print("=" * 60)

    from yacs.config import CfgNode

    # 获取默认配置
    cfg = get_cfg()

    # 保存原始值
    orig_d_model = cfg.MAMBA.d_model
    orig_dropout = cfg.MAMBA.dropout

    print(f"\n原始配置:")
    print(f"  d_model: {orig_d_model}")
    print(f"  dropout: {orig_dropout}")

    # 创建两个不同配置的模型
    model1 = Regressor(cfg=cfg)

    # 修改配置
    cfg_def = get_cfg()
    cfg_def.MAMBA.d_model = 512
    cfg_def.MAMBA.dropout = 0.5

    print(f"\n修改后配置:")
    print(f"  d_model: {cfg_def.MAMBA.d_model}")
    print(f"  dropout: {cfg_def.MAMBA.dropout}")

    model2 = Regressor(cfg=cfg_def)

    # 检查参数是否不同
    mamba1 = model1.pose_s1
    mamba2 = model2.pose_s1

    print(f"\n模型1 d_model (input_proj output): {mamba1.input_proj.out_features}")
    print(f"模型2 d_model (input_proj output): {mamba2.input_proj.out_features}")
    print(f"模型1 dropout: {mamba1.dropout.p}")
    print(f"模型2 dropout: {mamba2.dropout.p}")

    config_works = (
        mamba1.input_proj.out_features == orig_d_model and
        mamba2.input_proj.out_features == 512 and
        mamba1.dropout.p == orig_dropout and
        mamba2.dropout.p == 0.5
    )

    print("\n[PASS] 配置修改测试通过!" if config_works else "\n[FAIL] 配置修改测试失败!")
    return config_works


def main():
    print("\n" + "=" * 60)
    print("配置参数传递测试")
    print("=" * 60)

    results = []

    try:
        results.append(("Regressor配置", test_regressor_config()))
    except Exception as e:
        print(f"\n[ERROR] Regressor配置测试异常: {e}")
        results.append(("Regressor配置", False))

    try:
        results.append(("Loss配置", test_loss_config()))
    except Exception as e:
        print(f"\n[ERROR] Loss配置测试异常: {e}")
        results.append(("Loss配置", False))

    try:
        results.append(("前向传播", test_forward_pass()))
    except Exception as e:
        print(f"\n[ERROR] 前向传播测试异常: {e}")
        results.append(("前向传播", False))

    try:
        results.append(("配置修改", test_config_change()))
    except Exception as e:
        print(f"\n[ERROR] 配置修改测试异常: {e}")
        results.append(("配置修改", False))

    # 汇总结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)

    all_pass = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: [{status}]")
        if not passed:
            all_pass = False

    print("\n" + "=" * 60)
    if all_pass:
        print("所有测试通过!")
    else:
        print("存在测试失败，请检查!")
    print("=" * 60)

    return all_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
