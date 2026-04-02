#!/usr/bin/env python3
"""
测试HDF5数据加载的实际行为
验证[:]操作是否真的将数据加载到内存
"""
import h5py
import time
import psutil
import os
import gc

def test_hdf5_loading():
    path = '/media/yun/de2a43ce-446c-4a62-99b3-8ddc6ea1ef87/lidarhuman26M/lidarcap_train.hdf5'

    process = psutil.Process(os.getpid())

    print("=" * 60)
    print("测试: HDF5 [:] 操作是否真正加载数据到内存")
    print("=" * 60)

    # 清理内存
    gc.collect()

    mem_start = process.memory_info().rss / 1024 / 1024 / 1024
    print(f"\n起始内存: {mem_start:.3f} GB")

    # 打开文件
    f = h5py.File(path, 'r')

    # 获取数据集引用但不读取
    ds = f['point_clouds']
    mem_after_open = process.memory_info().rss / 1024 / 1024 / 1024
    print(f"打开文件后: {mem_after_open:.3f} GB (增量: {mem_after_open - mem_start:.3f} GB)")

    # 执行[:]操作
    print("\n执行 point_clouds[:] ...")
    start_time = time.time()
    data = ds[:]
    elapsed = time.time() - start_time
    print(f"耗时: {elapsed:.2f} 秒")

    mem_after_load = process.memory_info().rss / 1024 / 1024 / 1024
    print(f"加载后内存: {mem_after_load:.3f} GB (增量: {mem_after_load - mem_after_open:.3f} GB)")

    # 验证数据大小
    expected_size = data.nbytes / 1024 / 1024 / 1024
    print(f"数据理论大小: {expected_size:.3f} GB")

    # 实际访问数据以确保它真的在内存中
    print("\n验证数据是否真的在内存...")
    start_time = time.time()
    _ = data.sum()  # 强制访问所有数据
    elapsed_access = time.time() - start_time
    print(f"访问全部数据(sum)耗时: {elapsed_access:.2f} 秒")

    f.close()

    print("\n" + "=" * 60)
    print("结论:")
    if mem_after_load - mem_after_open > expected_size * 0.9:
        print("✓ 数据确实被加载到内存了")
    else:
        print("✗ 数据可能没有被完全加载到内存!")
    print("=" * 60)

if __name__ == '__main__':
    test_hdf5_loading()
