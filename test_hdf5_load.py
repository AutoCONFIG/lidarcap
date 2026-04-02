#!/usr/bin/env python3
"""
严格测试：强制访问数据的每个页面，确保数据真正加载到内存
"""
import h5py
import time
import psutil
import os
import gc
import numpy as np

def test_strict_loading():
    path = '/media/yun/de2a43ce-446c-4a62-99b3-8ddc6ea1ef87/lidarhuman26M/lidarcap_train.hdf5'

    process = psutil.Process(os.getpid())

    print("=" * 70)
    print("严格测试：验证数据是否真正加载到物理内存")
    print("=" * 70)

    gc.collect()
    mem_start = process.memory_info().rss / 1024 / 1024 / 1024
    print(f"\n起始内存: {mem_start:.3f} GB")

    f = h5py.File(path, 'r')

    # 测试 point_clouds
    print("\n--- 测试 point_clouds (1.83 GB) ---")
    start = time.time()
    point_clouds = f['point_clouds'][:]
    load_time = time.time() - start
    print(f"[:] 操作耗时: {load_time:.2f} 秒")

    mem_after_load = process.memory_info().rss / 1024 / 1024 / 1024
    print(f"加载后内存: {mem_after_load:.3f} GB (增量: {mem_after_load - mem_start:.3f} GB)")

    # 强制访问每个页面 - 通过计算每个切片的和
    print("\n强制访问每个内存页面...")
    start = time.time()
    # 每 4096 字节访问一次（页面大小），确保所有页面都被加载
    page_size = 512  # 沿第一个维度切片
    total_sum = 0
    for i in range(0, len(point_clouds), page_size):
        total_sum += point_clouds[i:i+page_size].sum()
    access_time = time.time() - start
    print(f"访问全部页面耗时: {access_time:.2f} 秒")
    print(f"数据总和: {total_sum:.6f}")

    mem_after_access = process.memory_info().rss / 1024 / 1024 / 1024
    print(f"访问后内存: {mem_after_access:.3f} GB (增量: {mem_after_access - mem_after_load:.3f} GB)")

    # 测试 images
    print("\n--- 测试 images (7.32 GB) ---")
    gc.collect()
    mem_before_images = process.memory_info().rss / 1024 / 1024 / 1024

    start = time.time()
    images = f['images'][:]
    load_time = time.time() - start
    print(f"[:] 操作耗时: {load_time:.2f} 秒")

    mem_after_images = process.memory_info().rss / 1024 / 1024 / 1024
    print(f"加载后内存: {mem_after_images:.3f} GB (增量: {mem_after_images - mem_before_images:.3f} GB)")

    print("\n强制访问每个内存页面...")
    start = time.time()
    total_sum = 0
    for i in range(0, len(images), 1000):
        total_sum += images[i:i+1000].sum()
    access_time = time.time() - start
    print(f"访问全部页面耗时: {access_time:.2f} 秒")

    mem_final = process.memory_info().rss / 1024 / 1024 / 1024
    print(f"访问后内存: {mem_final:.3f} GB")

    f.close()

    print("\n" + "=" * 70)
    print("总结:")
    print(f"  总数据量: ~9.4 GB")
    print(f"  最终内存: {mem_final:.3f} GB")
    print(f"  如果最终内存 ≈ 9.4 GB，说明数据确实加载到内存了")
    print("=" * 70)

if __name__ == '__main__':
    test_strict_loading()