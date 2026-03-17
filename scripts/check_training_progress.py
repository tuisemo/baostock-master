"""检查 AI 模型训练进度"""
import sys
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

model_path = project_root / "models" / "alpha_lgbm.txt"

print("=" * 80)
print("AI 模型训练进度检查")
print("=" * 80)
print(f"\n模型文件路径：{model_path}")

if model_path.exists():
    stat = model_path.stat()
    size_mb = stat.st_size / 1024 / 1024
    mtime = datetime.fromtimestamp(stat.st_mtime)
    
    print(f"[OK] 模型文件已存在")
    print(f"  文件大小：{size_mb:.2f} MB")
    print(f"  最后修改时间：{mtime}")
    
    # 验证模型是否完整
    try:
        import lightgbm as lgb
        print(f"\n尝试加载模型...")
        model = lgb.Booster(model_file=str(model_path))
        print(f"[OK] 模型加载成功")
        print(f"  特征数：{model.num_feature()}")
        print(f"  树的数量：{model.num_trees()}")
        print(f"\n[OK] 模型训练完成且验证通过！")
    except Exception as e:
        print(f"[WARN] 模型加载失败：{e}")
        print(f"  可能训练仍在进行中或文件损坏")
else:
    print(f"[WAITING] 模型文件尚不存在")
    print(f"  训练仍在进行中...")

print("\n" + "=" * 80)
