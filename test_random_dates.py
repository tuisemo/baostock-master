import pandas as pd
import random
import json
import warnings
from quant.app.validation_pipeline import ValidationPipeline
from quant.app.backtester import get_market_index
from quant.core.strategy_params import StrategyParams
from quant.infra.logger import logger

warnings.filterwarnings("ignore")

def main():
    # 关闭多余的 logger 打印
    import logging
    logger.setLevel(logging.ERROR)

    # 1. 获取大盘交易日，截取 2022-2023 之间的交易日
    idx_df = get_market_index()
    if idx_df is None:
        print("Failed to load market index data.")
        return
        
    dates = idx_df.index
    valid_dates = dates[(dates >= "2022-01-01") & (dates <= "2024-01-01")]
    valid_date_strs = [d.strftime("%Y-%m-%d") for d in valid_dates]
    
    # 2. 随机抽取 5 个日期
    random.seed()
    sample_dates = random.sample(valid_date_strs, 5)
    sample_dates.sort()
    print(f"随机抽取的 5 个截面测试日期: {sample_dates}\n")
    print("正在执行验证并模拟持股 (处理可能较慢请稍候)....")
    
    # 3. 跑 ValidationPipeline 
    pipeline = ValidationPipeline(validation_dates=sample_dates)
    params = StrategyParams()
    
    res = pipeline.run_full_evaluation(params)
    
    # 4. 汇总输出
    print("\n" + "="*50)
    print("验证结果汇总 (Validation Summary):")
    print("="*50)
    print(f"复合得分 (Composite Score): {res['composite_score']:.4f}")
    print(f"平均胜率 (Avg Win Rate): {res['avg_win_rate']:.2f}%")
    print(f"平均单笔盈利 (Avg PnL): {res['avg_pnl']:.2f}%")
    print(f"最大回撤均值 (Avg Max Drawdown): {res['avg_dd']:.2f}%")
    
    print("\n各日期验证详情 (Details per date):")
    for d in res.get("detail", []):
        print(f" - 日期: {d['date']}")
        print(f"   买入标的数量: {d['trades']}")
        print(f"   该批次持仓胜率: {d['win_rate']:.2f}%")
        print(f"   该批次平均盈亏(PnL): {d['avg_pnl']:.2f}%")
        print(f"   该批次平均最大回撤: {d['max_drawdown']:.2f}%")

if __name__ == "__main__":
    main()
