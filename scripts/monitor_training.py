"""
监控 AI 模型训练进度
显示预计完成时间和当前状态
"""
import sys
import os
import time
from pathlib import Path
from datetime import datetime, timedelta
import psutil

project_root = Path(__file__).parent.parent

def find_training_process():
    """查找正在运行的训练进程"""
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
        try:
            cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
            if 'train-ai' in cmdline or 'build_dataset' in cmdline:
                return proc
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None

def estimate_completion(start_time, current_progress, total_items=1435):
    """估算完成时间"""
    elapsed = time.time() - start_time
    items_per_second = current_progress / elapsed if elapsed > 0 else 0
    
    if items_per_second > 0:
        remaining_items = total_items - current_progress
        remaining_seconds = remaining_items / items_per_second
        eta = datetime.now() + timedelta(seconds=remaining_seconds)
        return eta, items_per_second
    return None, 0

print("=" * 80)
print("AI 模型训练进度监控")
print("=" * 80)
print(f"检查时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# 检查模型文件
model_path = project_root / "models" / "alpha_lgbm.txt"
if model_path.exists():
    stat = model_path.stat()
    size_mb = stat.st_size / 1024 / 1024
    mtime = datetime.fromtimestamp(stat.st_mtime)
    print(f"[OK] 模型文件已存在")
    print(f"  大小：{size_mb:.2f} MB")
    print(f"  修改时间：{mtime}")
    
    # 尝试加载
    try:
        import lightgbm as lgb
        model = lgb.Booster(model_file=str(model_path))
        print(f"[OK] 模型验证成功！")
        print(f"  特征数：{model.num_feature()}")
        print(f"  树数量：{model.num_trees()}")
        print(f"\n训练完成！")
        sys.exit(0)
    except Exception as e:
        print(f"[WARN] 模型验证失败：{e}")
        print(f"  文件可能正在写入中...")
else:
    print(f"[WAITING] 模型文件尚未生成")

print()

# 检查训练进程
proc = find_training_process()
if proc:
    try:
        proc_info = proc.as_dict()
        print(f"[RUNNING] 发现训练进程")
        print(f"  PID: {proc_info['pid']}")
        print(f"  CPU 时间：{proc.cpu_percent()}%")
        print(f"  内存：{proc.memory_info().rss / 1024 / 1024:.1f} MB")
        print(f"  启动时间：{datetime.fromtimestamp(proc_info['create_time'])}")
        
        # 估算进度（基于文件数量）
        data_dir = project_root / "data"
        csv_files = list(data_dir.glob("*.csv"))
        total_files = len([f for f in csv_files if f.name not in ["stock-list.csv", "sh.000001.csv"]])
        
        print(f"\n  数据文件总数：{total_files}")
        print(f"  预计处理时间：~30-40 分钟")
        
    except Exception as e:
        print(f"[INFO] 无法获取进程详情：{e}")
else:
    print(f"[IDLE] 未发现训练进程")
    print(f"  训练可能已完成或尚未开始")

print()
print("=" * 80)
print("提示：训练完成后将自动进行功能验证")
print("=" * 80)
