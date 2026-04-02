#!/usr/bin/env python3
"""
检测 HDF5 文件是否在文件系统缓存中
需要在服务器上运行
"""
import h5py
import time
import psutil
import os
import gc
import subprocess

def get_file_cache_status(filepath):
    """使用 fincore 或 vmtouch 检查文件缓存状态"""
    try:
        # 检查是否安装了 vmtouch
        result = subprocess.run(['vmtouch', filepath], capture_output=True, text=True, timeout=10)
        print(f"文件缓存状态:\n{result.stdout}")
    except FileNotFoundError:
        print("提示: 安装 vmtouch 可以查看文件缓存状态")
        print("  sudo apt-get install vmtouch")
    except Exception as e:
        print(f"无法检查缓存状态: {e}")

def test_with_clear_cache(filepath):
    """清除缓存后测试加载速度"""
    print("=" * 70)
    print("测试：清除文件系统缓存后的加载速度")
    print("=" * 70)

    print("\n[重要] 需要先清除文件系统缓存:")
    print("  sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'")
    print("\n或者只清除这个文件:")
    print("  vmtouch -e " + filepath)

    print("\n按 Enter 键继续（清除缓存后）...")
    input()

    process = psutil.Process(os.getpid())
    mem_start = process.memory_info().rss / 1024 / 1024 / 1024
    print(f"\n起始内存: {mem_start:.3f} GB")

    f = h5py.File(filepath, 'r')

    # 加载所有数据
    total_size = 0
    start = time.time()
    for key in f.keys():
        d = f[key]
        if hasattr(d, 'shape') and len(d.shape) > 0 and d.shape[0] > 0:
            _ = d[:]
            total_size += d.nbytes
    load_time = time.time() - start
    f.close()

    mem_end = process.memory_info().rss / 1024 / 1024 / 1024
    print(f"\n加载耗时: {load_time:.2f} 秒")
    print(f"数据总量: {total_size / 1024**3:.2f} GB")
    print(f"加载后内存: {mem_end:.3f} GB")
    print(f"计算速度: {total_size / 1024**3 / load_time:.2f} GB/s")

    # 机械硬盘速度参考
    print(f"\n参考速度:")
    print(f"  机械硬盘 (180 MB/s): {total_size / 1024**3 / 0.180:.0f} 秒")
    print(f"  SATA SSD (500 MB/s): {total_size / 1024**3 / 0.500:.0f} 秒")
    print(f"  NVMe SSD (3000 MB/s): {total_size / 1024**3 / 3.0:.0f} 秒")

if __name__ == '__main__':
    # 服务器路径
    filepath = '/data2/kaiyun/.yun/lidarhuman26M/lidarcap_train.hdf5'

    if not os.path.exists(filepath):
        # 本地路径
        filepath = '/media/yun/de2a43ce-446c-4a62-99b3-8ddc6ea1ef87/lidarhuman26M/lidarcap_train.hdf5'

    print(f"测试文件: {filepath}")
    print(f"文件大小: {os.path.getsize(filepath) / 1024**3:.2f} GB")

    get_file_cache_status(filepath)
    test_with_clear_cache(filepath)