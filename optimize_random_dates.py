"""
全市场大范围参数优化迭代
- 覆盖 2022-2026 全时间段，等距抽取 20 个横截面验证日
- max_hold_days 固定为 5 天
- 使用 100 轮 Optuna trials 进行大规模搜索
"""
import os
import json
import time
import warnings
import numpy as np
from quant.app.validation_pipeline import ValidationPipeline
from quant.app.backtester import get_market_index
from quant.core.strategy_params import StrategyParams
from quant.infra.logger import logger

warnings.filterwarnings("ignore")

def main():
    import logging
    logger.setLevel(logging.WARNING)  # 保留 WARNING 方便看 Optuna 进度

    t0 = time.time()

    idx_df = get_market_index()
    if idx_df is None:
        print("Failed to load market index data.")
        return

    dates = idx_df.index
    # 覆盖全部可用数据区间: 2022 到 2026
    valid_dates = dates[(dates >= "2022-01-01") & (dates <= "2026-06-01")]
    valid_date_strs = [d.strftime("%Y-%m-%d") for d in valid_dates]

    if len(valid_date_strs) < 20:
        print(f"可用交易日数量不足 ({len(valid_date_strs)})，无法抽取 20 个截面。")
        return

    # 等距抽取 20 个验证日，全面覆盖牛熊震荡各阶段
    n_samples = 20
    indices = np.linspace(0, len(valid_date_strs) - 1, n_samples, dtype=int)
    sample_dates = [valid_date_strs[i] for i in indices]
    print(f"{'='*60}")
    print(f"全市场参数优化 - {n_samples} 个分散横截面验证日")
    print(f"{'='*60}")
    print(f"日期范围: {valid_date_strs[0]} ~ {valid_date_strs[-1]} (共 {len(valid_date_strs)} 个交易日)")
    print(f"抽样日期: {sample_dates}")
    print(f"约束: max_hold_days = 5")
    print(f"Optuna trials: 100")
    print(f"{'='*60}\n")

    pipeline = ValidationPipeline(validation_dates=sample_dates)

    fixed_params = {
        "max_hold_days": 5
    }

    # ===== 100 轮大范围搜索 =====
    opt_res = pipeline.optimize_for_real_trading(n_trials=100, fixed_params=fixed_params)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"优化完成！耗时 {elapsed/60:.1f} 分钟")
    print(f"{'='*60}")
    print(f"最高复合得分 (Composite Score): {opt_res['best_composite_score']:.4f}")

    best_p = opt_res['best_params']
    print(f"\n=== 最优参数组合 ===")
    for k, v in sorted(best_p.items()):
        print(f"  {k}: {v}")

    # 使用最优参数重新跑一遍验证，输出每天的详细盈亏
    print(f"\n{'='*60}")
    print(f"最优参数详细回测验证")
    print(f"{'='*60}")
    out_params = StrategyParams.from_dict({**StrategyParams().to_dict(), **best_p})
    res = pipeline.run_full_evaluation(out_params)
    print(f"复合得分: {res['composite_score']:.4f}")
    print(f"平均胜率: {res['avg_win_rate']:.2f}%")
    print(f"平均PnL: {res['avg_pnl']:.2f}%")
    print(f"回撤均值: {res['avg_dd']:.2f}%")

    print(f"\n各日期详情:")
    for d in res.get("detail", []):
        print(f"  {d['date']} | 交易笔数: {d['trades']:>4d} | 胜率: {d['win_rate']:>6.2f}% | PnL: {d['avg_pnl']:>+6.2f}% | 最大回撤: {d['max_drawdown']:>+7.2f}%")

    # 保存结果到 JSON
    output_file = "best_full_market_params.json"
    save_data = {
        "best_params": best_p,
        "best_composite_score": opt_res["best_composite_score"],
        "verification": {
            "composite_score": res["composite_score"],
            "avg_win_rate": res["avg_win_rate"],
            "avg_pnl": res["avg_pnl"],
            "avg_dd": res["avg_dd"],
        },
        "config": {
            "n_trials": 100,
            "n_sample_dates": n_samples,
            "date_range": f"{valid_date_strs[0]} ~ {valid_date_strs[-1]}",
            "max_hold_days": 5,
        }
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存至: {output_file}")

if __name__ == "__main__":
    main()
