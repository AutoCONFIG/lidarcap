#!/usr/bin/env python3
"""
服务器版本 - 测试 HDF5 数据加载行为
验证数据是否真正加载到内存
"""
import h5py
import time
import psutil
import os
import gc
import subprocess

def get_file_cache_status(filepath):
    """使用 vmtouch 检查文件缓存状态"""
    try:
        result = subprocess.run(['vmtouch', filepath], capture_output=True, text=True, timeout=30)
        print(f"文件缓存状态:\n{result.stdout}")
        return True
    except FileNotFoundError:
        print("[警告] vmtouch 未安装，无法查看文件缓存状态")
        print("  安装: sudo apt-get install vmtouch")
        return False
    except Exception as e:
        print(f"[错误] 检查缓存状态失败: {e}")
        return False

def clear_file_cache(filepath):
    """从文件系统缓存中驱逐文件"""
    try:
        result = subprocess.run(['vmtouch', '-e', filepath], capture_output=True, text=True, timeout=60)
        print(f"已清除文件缓存:\n{result.stdout}")
        return True
    except Exception as e:
        print(f"[错误] 清除缓存失败: {e}")
        print("  手动执行: vmtouch -e " + filepath)
        return False

def test_loading(filepath):
    """测试数据加载"""
    process = psutil.Process(os.getpid())

    print("\n" + "=" * 70)
    print("开始测试数据加载")
    print("=" * 70)

    gc.collect()
    mem_start = process.memory_info().rss / 1024 / 1024 / 1024
    print(f"\n起始内存: {mem_start:.3f} GB")

    f = h5py.File(filepath, 'r')

    # 先显示文件结构
    print("\n文件结构:")
    total_bytes = 0
    for key in f.keys():
        d = f[key]
        if hasattr(d, 'shape') and len(d.shape) > 0 and d.shape[0] > 0:
            size_mb = d.nbytes / 1024 / 1024
            total_bytes += d.nbytes
            print(f"  {key}: {d.shape} ({size_mb:.1f} MB)")

    print(f"\n总数据量: {total_bytes / 1024**3:.2f} GB")

    # 加载所有数据
    cache = {}
    start = time.time()
    for key in f.keys():
        d = f[key]
        if hasattr(d, 'shape') and len(d.shape) > 0 and d.shape[0] > 0:
            cache[key] = d[:]
    load_time = time.time() - start

    mem_after_load = process.memory_info().rss / 1024 / 1024 / 1024
    print(f"\n[:] 操作耗时: {load_time:.2f} 秒")
    print(f"加载后内存: {mem_after_load:.3f} GB (增量: {mem_after_load - mem_start:.3f} GB)")

    # 强制访问每个页面
    print("\n强制访问每个内存页面...")
    start = time.time()
    for key, data in cache.items():
        if len(data.shape) > 0 and data.shape[0] > 0:
            # 沿第一个维度切片访问，确保页面加载
            chunk_size = max(1, min(1000, len(data) // 100))
            for i in range(0, len(data), chunk_size):
                _ = data[i:i+chunk_size].sum()
    access_time = time.time() - start
    print(f"访问全部页面耗时: {access_time:.2f} 秒")

    mem_final = process.memory_info().rss / 1024 / 1024 / 1024
    print(f"访问后内存: {mem_final:.3f} GB")

    f.close()

    # 分析结果
    print("\n" + "=" * 70)
    print("结果分析")
    print("=" * 70)

    speed_gbps = total_bytes / 1024**3 / load_time
    print(f"加载速度: {speed_gbps:.2f} GB/s ({speed_gbps * 1024:.0f} MB/s)")

    print("\n参考速度:")
    print(f"  机械硬盘 (180 MB/s): 预计 {total_bytes / 1024**3 / 0.18:.0f} 秒")
    print(f"  SATA SSD (500 MB/s): 预计 {total_bytes / 1024**3 / 0.5:.0f} 秒")
    print(f"  NVMe SSD (3000 MB/s): 预计 {total_bytes / 1024**3 / 3:.0f} 秒")
    print(f"  内存带宽 (~20 GB/s): 预计 {total_bytes / 1024**3 / 20:.1f} 秒")

    # 判断数据来源
    if load_time < total_bytes / 1024**3 / 0.5:  # 快于 SATA SSD
        print(f"\n[结论] 加载速度远快于硬盘，数据可能来自文件系统缓存")
        print("       这是正常现象 - Linux 会自动缓存频繁访问的文件")
    else:
        print(f"\n[结论] 加载速度符合硬盘读取特征")

    # 判断内存加载
    if abs(mem_after_load - mem_start - total_bytes / 1024**3) < 0.5:
        print(f"[验证] ✓ 内存增量 ≈ 数据大小，数据已加载到进程内存")
    else:
        print(f"[验证] ✗ 内存增量与数据大小不符，请检查")

    print("=" * 70)

    return cache

def main():
    # 服务器路径
    filepath = '/data2/kaiyun/.yun/lidarhuman26M/lidarcap_train.hdf5'

    print("=" * 70)
    print("HDF5 数据加载测试 (服务器版)")
    print("=" * 70)
    print(f"\n测试文件: {filepath}")

    if not os.path.exists(filepath):
        print(f"[错误] 文件不存在: {filepath}")
        print("\n请修改脚本中的 filepath 变量为正确的路径")
        return

    file_size = os.path.getsize(filepath) / 1024**3
    print(f"文件大小: {file_size:.2f} GB")

    # 检查缓存状态
    print("\n--- 步骤1: 检查文件缓存状态 ---")
    cache_exists = get_file_cache_status(filepath)

    if cache_exists:
        print("\n--- 步骤2: 选择操作 ---")
        print("  1. 直接测试加载 (数据可能已在缓存)")
        print("  2. 清除缓存后测试 (强制从硬盘读取)")
        print("  3. 退出")

        choice = input("\n请选择 [1/2/3]: ").strip()

        if choice == '2':
            print("\n--- 清除文件缓存 ---")
            clear_file_cache(filepath)
            print("\n--- 步骤3: 测试加载 ---")
            test_loading(filepath)
        elif choice == '1':
            print("\n--- 步骤3: 测试加载 ---")
            test_loading(filepath)
        else:
            print("退出")
    else:
        print("\n--- 直接测试加载 ---")
        test_loading(filepath)

if __name__ == '__main__':
    main()