#!/usr/bin/env python3
"""
清理脚本 - 移除Python运行时缓存文件，净化代码便于归档

使用方法:
    python cleanup.py              # 清理当前目录
    python cleanup.py /path/to/dir # 清理指定目录

Python运行时防止生成缓存:
    方法1: 设置环境变量 PYTHONDONTWRITEBYTECODE=1
    方法2: 在代码开头添加 sys.dont_write_bytecode = True
    方法3: 使用 python -B script.py (-B 参数阻止生成.pyc)
"""

import os
import sys
import shutil
from pathlib import Path


# 需要清理的文件和目录模式
CLEANUP_PATTERNS = {
    # 目录模式
    'dirs': [
        '__pycache__',           # Python字节码缓存
        '.pytest_cache',         # pytest缓存
        '.mypy_cache',           # mypy类型检查缓存
        '.ipynb_checkpoints',    # Jupyter notebook缓存
        '.ruff_cache',           # Ruff linter缓存
        '.pytype',               # pytype类型检查缓存
        '.hypothesis',           # Hypothesis测试缓存
        '.coverage_cache',       # coverage缓存
        '*.egg-info',            # 包安装信息
        'build',                 # 构建目录
        'dist',                  # 分发目录
        '.eggs',                 #  eggs目录
        '*.dist-info',           # 分发信息
    ],
    # 文件模式
    'files': [
        '*.pyc',                 # Python字节码
        '*.pyo',                 # 优化后的字节码
        '*.pyd',                 # Windows动态链接库
        '*~',                    # 备份文件
        '.*.swp',                # vim交换文件
        '.DS_Store',             # macOS系统文件
        'Thumbs.db',             # Windows缩略图缓存
        '.coverage',             # coverage数据文件
        'coverage.xml',          # coverage XML报告
        '*.cover',               # coverage报告
        '.noseids',              # nosetests缓存
        '*.log',                 # 日志文件
        '*.prof',                # 性能分析文件
        '*.lprof',               # line_profiler输出
    ]
}


def disable_bytecode_generation():
    """
    禁止Python在当前运行时生成字节码缓存(.pyc文件)
    在脚本开头调用此函数可阻止后续导入产生__pycache__
    """
    sys.dont_write_bytecode = True
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'


def enable_bytecode_generation():
    """恢复Python字节码生成（默认行为）"""
    sys.dont_write_bytecode = False
    os.environ.pop('PYTHONDONTWRITEBYTECODE', None)


def matches_pattern(name: str, pattern: str) -> bool:
    """检查名称是否匹配模式（支持简单的通配符 * ）"""
    if '*' in pattern:
        parts = pattern.split('*')
        if name.startswith(parts[0]) and name.endswith(parts[-1]):
            if len(parts) == 2:
                return True
            middle = name[len(parts[0]):len(name)-len(parts[-1])]
            return parts[1] in middle
    return name == pattern


def should_remove_dir(dir_name: str) -> bool:
    """判断目录是否应该被删除"""
    for pattern in CLEANUP_PATTERNS['dirs']:
        if matches_pattern(dir_name, pattern):
            return True
    return False


def should_remove_file(file_name: str) -> bool:
    """判断文件是否应该被删除"""
    for pattern in CLEANUP_PATTERNS['files']:
        if matches_pattern(file_name, pattern):
            return True
    return False


def clean_directory(target_dir: Path, dry_run: bool = False) -> dict:
    """
    清理指定目录下的缓存文件
    
    Args:
        target_dir: 目标目录路径
        dry_run: 如果为True，只显示将要删除的内容而不实际删除
    
    Returns:
        统计信息字典
    """
    stats = {
        'dirs_removed': 0,
        'files_removed': 0,
        'bytes_freed': 0,
        'errors': [],
        'items': []  # 记录具体删除了什么
    }
    
    dirs_to_remove = []
    files_to_remove = []
    
    for root, dirs, files in os.walk(target_dir, topdown=False):
        root_path = Path(root)
        
        for dir_name in dirs:
            if should_remove_dir(dir_name):
                dir_path = root_path / dir_name
                try:
                    dir_size = sum(
                        f.stat().st_size 
                        for f in dir_path.rglob('*') 
                        if f.is_file()
                    )
                    dirs_to_remove.append((dir_path, dir_size))
                except (OSError, PermissionError) as e:
                    stats['errors'].append(f"无法访问目录 {dir_path}: {e}")
        
        for file_name in files:
            if should_remove_file(file_name):
                file_path = root_path / file_name
                try:
                    file_size = file_path.stat().st_size
                    files_to_remove.append((file_path, file_size))
                except (OSError, PermissionError) as e:
                    stats['errors'].append(f"无法访问文件 {file_path}: {e}")
    
    # 显示和删除
    for file_path, file_size in files_to_remove:
        rel_path = file_path.relative_to(target_dir)
        if dry_run:
            stats['items'].append(f"[将要删除文件] {rel_path}")
        else:
            try:
                file_path.unlink()
                stats['items'].append(f"[已删除文件] {rel_path}")
                stats['files_removed'] += 1
                stats['bytes_freed'] += file_size
            except (OSError, PermissionError) as e:
                stats['errors'].append(f"无法删除文件 {file_path}: {e}")
    
    for dir_path, dir_size in dirs_to_remove:
        rel_path = dir_path.relative_to(target_dir)
        if dry_run:
            stats['items'].append(f"[将要删除目录] {rel_path}")
        else:
            try:
                shutil.rmtree(dir_path)
                stats['items'].append(f"[已删除目录] {rel_path}")
                stats['dirs_removed'] += 1
                stats['bytes_freed'] += dir_size
            except (OSError, PermissionError) as e:
                stats['errors'].append(f"无法删除目录 {dir_path}: {e}")
    
    return stats


def clean(target_dir: str = '.', dry_run: bool = False, silent: bool = False) -> dict:
    """
    清理接口函数，可在其他Python代码中调用
    
    Args:
        target_dir: 要清理的目录路径（字符串或Path）
        dry_run: 预演模式，只显示不删除
        silent: 静默模式，不打印输出
    
    Returns:
        包含清理结果的字典
    
    示例:
        >>> from cleanup import clean
        >>> result = clean('/path/to/project')
        >>> print(f"删除了 {result['files_removed']} 个文件")
    """
    target = Path(target_dir).resolve()
    
    if not target.exists() or not target.is_dir():
        return {
            'success': False,
            'error': f"目录不存在或不是目录: {target}",
            'dirs_removed': 0,
            'files_removed': 0,
            'bytes_freed': 0
        }
    
    stats = clean_directory(target, dry_run=dry_run)
    stats['success'] = True
    stats['target_dir'] = str(target)
    
    if not silent:
        print_summary(stats, dry_run=dry_run)
    
    return stats


def format_size(size_bytes: int) -> str:
    """格式化文件大小显示"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} TB"


def print_summary(stats: dict, dry_run: bool = False):
    """打印清理摘要"""
    action = "[预演模式] 将要" if dry_run else "✅ 已"
    print(f"\n{'='*60}")
    print(f"📁 目标: {stats.get('target_dir', '未知')}")
    print(f"{'='*60}")
    print(f"{action}删除目录: {stats['dirs_removed']} 个")
    print(f"{action}删除文件: {stats['files_removed']} 个")
    print(f"{action}释放空间: {format_size(stats['bytes_freed'])}")
    
    if stats['errors']:
        print(f"\n⚠️  警告/错误 ({len(stats['errors'])}个):")
        for error in stats['errors'][:10]:
            print(f"  - {error}")
        if len(stats['errors']) > 10:
            print(f"  ... 还有 {len(stats['errors']) - 10} 个错误未显示")
    
    print('='*60)


def auto_clean_on_import(project_root: str = None):
    """
    在导入模块时自动清理项目缓存（推荐在项目入口脚本使用）
    
    用法 - 在你的主脚本开头添加:
        from cleanup import auto_clean_on_import, disable_bytecode_generation
        disable_bytecode_generation()  # 禁止后续生成.pyc
        auto_clean_on_import()          # 清理已有的缓存
    """
    if project_root is None:
        project_root = os.getcwd()
    
    print(f"🧹 自动清理项目缓存: {project_root}")
    stats = clean(project_root, dry_run=False, silent=True)
    
    if stats['dirs_removed'] > 0 or stats['files_removed'] > 0:
        print(f"   清理完成: {stats['files_removed']}个文件, {stats['dirs_removed']}个目录")
        print(f"   释放空间: {format_size(stats['bytes_freed'])}")
    else:
        print("   无需清理")
    
    return stats


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='清理Python项目中的缓存文件，净化代码便于归档',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
使用示例:
  %(prog)s                    # 清理当前目录
  %(prog)s /path/to/project   # 清理指定目录
  %(prog)s -n                 # 预演模式，只显示将要删除的内容
  %(prog)s -y                 # 自动确认，不提示
  
Python运行时阻止生成缓存:
  方法1: export PYTHONDONTWRITEBYTECODE=1
  方法2: python -B your_script.py
  方法3: 代码中添加 sys.dont_write_bytecode = True

作为模块导入使用:
  from cleanup import clean, disable_bytecode_generation
  disable_bytecode_generation()  # 禁止生成.pyc
  clean('/path/to/project')      # 清理缓存
        '''
    )
    
    parser.add_argument(
        'directory',
        nargs='?',
        default='.',
        help='要清理的目录路径 (默认: 当前目录)'
    )
    parser.add_argument(
        '-n', '--dry-run',
        action='store_true',
        help='预演模式：只显示将要删除的内容，不实际删除'
    )
    parser.add_argument(
        '-y', '--yes',
        action='store_true',
        help='自动确认，不提示用户确认'
    )
    parser.add_argument(
        '--no-bytecode',
        action='store_true',
        help='同时禁止当前Python进程生成字节码缓存'
    )
    
    args = parser.parse_args()
    
    # 可选：禁止生成字节码
    if args.no_bytecode:
        disable_bytecode_generation()
        print("🚫 已禁止生成字节码缓存(.pyc)")
    
    target_dir = Path(args.directory).resolve()
    
    if not target_dir.exists():
        print(f"❌ 错误: 目录不存在: {target_dir}")
        sys.exit(1)
    
    if not target_dir.is_dir():
        print(f"❌ 错误: 不是目录: {target_dir}")
        sys.exit(1)
    
    print(f"🔍 扫描目录: {target_dir}")
    
    # 预演模式
    stats = clean_directory(target_dir, dry_run=True)
    
    if stats['dirs_removed'] == 0 and stats['files_removed'] == 0:
        print("\n✅ 未发现需要清理的缓存文件")
        return
    
    print_summary(stats, dry_run=True)
    
    # 确认删除
    if not args.dry_run and not args.yes:
        try:
            response = input("\n⚠️  确认删除以上文件? (y/N): ").strip().lower()
            if response not in ('y', 'yes'):
                print("❌ 已取消操作")
                return
        except (EOFError, KeyboardInterrupt):
            print("\n❌ 已取消操作")
            return
        
        # 实际执行删除
        stats = clean_directory(target_dir, dry_run=False)
        print_summary(stats, dry_run=False)
        print("\n✅ 清理完成！代码已净化，可以归档。")
    elif args.dry_run:
        print("\n💡 提示: 去掉 -n 参数执行实际删除")


if __name__ == '__main__':
    main()
