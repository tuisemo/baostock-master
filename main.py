from __future__ import annotations

import argparse
import os
import random
import sys

from quant.infra.logger import logger


def cmd_update_list():
    from quant.data.stock_filter import update_stock_list
    update_stock_list()


def cmd_update_data():
    from quant.data.data_updater import update_history_data
    update_history_data()


def cmd_analyze():
    from quant.features.features.analyzer import analyze_all_stocks
    analyze_all_stocks()


def cmd_ui():
    try:
        from app import demo
        logger.info("启动可视化量化平台 (Web UI)...")
        demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
    except Exception as e:
        logger.error(f"启动 Web UI 失败: {e}")


def cmd_optimize(args: argparse.Namespace):
    from quant.infra.config import CONF
    from quant.app.optimizer import run_optimization, save_results, apply_best_params

    if args.rounds is not None:
        CONF.optimizer.max_rounds = args.rounds
    if args.samples is not None:
        CONF.optimizer.sample_count = args.samples
    if args.objective is not None:
        CONF.optimizer.objective = args.objective

    result = run_optimization()
    save_results(result)

    baseline = result["baseline_score"]
    best = result["best_score"]
    logger.info(f"基线得分:  {baseline:.6f}")
    logger.info(f"最优得分:  {best:.6f}")
    logger.info(f"提升幅度:  {best - baseline:+.6f}")

    answer = input("是否将最优参数写回 config.yaml? [y/N] ").strip().lower()
    if answer in ("y", "yes"):
        apply_best_params(result)
        logger.info("最优参数已应用到 config.yaml")
    else:
        logger.info("已跳过参数应用。")


def cmd_batch_test(args: argparse.Namespace):
    from quant.app.backtester import batch_backtest
    from quant.infra.config import CONF

    data_dir = CONF.history_data.data_dir
    all_files = [f for f in os.listdir(data_dir)
                 if f.endswith(".csv") and f != "stock-list.csv"]
    random.seed(args.seed)
    sample_files = random.sample(all_files, min(args.num, len(all_files)))
    codes = [f.replace(".csv", "") for f in sample_files]

    logger.info(f"批量回测: {len(codes)} 只股票 (seed={args.seed})")
    df = batch_backtest(codes)

    if df.empty:
        logger.info("无有效回测结果。")
        return

    print(f"\n{'Code':<14} {'Return%':>9} {'WinRate%':>9} {'MaxDD%':>9} {'Trades':>7} {'Sharpe':>8}")
    print("-" * 62)
    for _, row in df.iterrows():
        print(
            f"{row['code']:<14} {row['return_pct']:>9.2f} {row['win_rate']:>9.2f} "
            f"{row['max_drawdown']:>9.2f} {row['num_trades']:>7.0f} {row['sharpe']:>8.2f}"
        )

    print(f"\n--- 汇总统计 ({len(df)} 只有交易的股票) ---")
    print(f"平均收益率:    {df['return_pct'].mean():.2f}%")
    print(f"平均胜率:      {df['win_rate'].mean():.2f}%")
    print(f"平均最大回撤:  {df['max_drawdown'].mean():.2f}%")
    print(f"平均交易次数:  {df['num_trades'].mean():.1f}")
    print(f"平均夏普比率:  {df['sharpe'].mean():.2f}")


def cmd_train_ai():
    from quant.core.trainer import build_dataset, train_model
    from quant.infra.config import CONF
    from quant.core.strategy_params import StrategyParams

    logger.info("启动 AI 模型离线训练管道 (Phase 8)...")
    p = StrategyParams.from_app_config(CONF)
    data_dir = CONF.history_data.data_dir
    
    # 构建包含高阶特征的数据集
    df = build_dataset(data_dir, p, n_forward_days=5, target_atr_mult=2.0, stop_loss_atr_mult=1.5)
    
    if df.empty:
        logger.error("数据集为空，训练中止。")
        return
        
    # 训练模型并保存
    train_model(df, model_path="models/alpha_lgbm.txt")


def cmd_auto_pilot():
    import time
    from quant.infra.logger import logger
    
    logger.info("==================================================")
    logger.info("🚀 启动 Auto-Pilot 全自动量化演进流水线 🚀")
    logger.info("==================================================")
    
    logger.info("[Step 1/4] 更新股票池基底...")
    cmd_update_list()
    
    logger.info("[Step 2/4] 并发增量拉取最新 K 线数据...")
    cmd_update_data()
    
    logger.info("[Step 3/4] 启动 Optuna 每日自适应参数微调 (Fast Walk-Forward)...")
    from quant.infra.config import CONF
    from quant.app.optimizer import run_optimization, save_results, apply_best_params
    
    # Fast daily evolution
    CONF.optimizer.max_rounds = 3
    CONF.optimizer.sample_count = 100
    try:
        result = run_optimization()
        save_results(result)
        apply_best_params(result)
        logger.info("✅ 每日参数自适应微调完成。")
    except Exception as e:
        logger.error(f"❌ 自动优化失败，跳过参数应用: {e}")
        
    logger.info("[Step 4/4] 运行盘后多因子选股与预生成候选名单...")
    cmd_analyze()
    
    logger.info("==================================================")
    logger.info("🏁 Auto-Pilot 闭环运转全部完成！ 🏁")
    logger.info("您可以直接进入 Web UI 的第 2 个 Tab 查看今日最新入选名单 (Scan)。")
    logger.info("==================================================")


def _scan_single(args):
    code, target_date = args
    from quant.app.backtester import scan_today_signal
    try:
        return scan_today_signal(code, target_date=target_date)
    except Exception:
        return None

def cmd_scan_date(args: argparse.Namespace):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from quant.infra.config import CONF
    
    target_date = args.date
    logger.info(f"========== 历史信号回溯扫描: {target_date} ==========")
    
    data_dir = CONF.history_data.data_dir
    all_files = [f for f in os.listdir(data_dir) if f.endswith(".csv") and f != "stock-list.csv"]
    if not all_files:
        logger.error(f"没有找到历史数据。请先执行 update-data。")
        return
        
    codes = [f.replace(".csv", "") for f in all_files]
    
    results = []
    total = len(codes)
    logger.info(f"开始遍历 {total} 只股票寻找 {target_date} 满足条件的买入点...")
            
    max_workers = os.cpu_count() or 4
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_scan_single, (code, target_date)): code for code in codes}
        import tqdm
        for future in tqdm.tqdm(as_completed(futures), total=total, desc="扫描进度"):
            res = future.result()
            if res:
                results.append(res)
                
    if not results:
        logger.info(f"[OK] 历史扫描完成：在 {target_date} 未发现任何满足策略要求的买入标的。")
        return
        
    import pandas as pd
    df = pd.DataFrame(results)
    print(f"\n[OK] {target_date} 共发现 {len(results)} 只高胜率买入节点标的：")
    print(df.to_string(index=False))
    
    out_file = f"historical_scan_{target_date.replace('-', '')}.csv"
    df.to_csv(out_file, index=False, encoding="utf-8-sig")
    logger.info(f"详细结果已保存至: {out_file}")


def main():
    parser = argparse.ArgumentParser(description="Production-grade A-Share Quant System")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser(
        "update-list",
        help="Update the baseline stock list from baostock.",
    )
    subparsers.add_parser(
        "update-data",
        help="Incrementally update historical K-line data.",
    )
    subparsers.add_parser(
        "analyze",
        help="Run multi-factor stock selection analysis.",
    )
    subparsers.add_parser(
        "ui",
        help="Launch the interactive Web UI (Gradio).",
    )

    opt_parser = subparsers.add_parser(
        "optimize",
        help="Run multi-round iterative strategy optimization.",
    )
    opt_parser.add_argument("--rounds", type=int, default=None, help="Override max optimization rounds")
    opt_parser.add_argument("--samples", type=int, default=None, help="Override stock sample count per round")
    opt_parser.add_argument("--objective", type=str, default=None, help="Override objective: sharpe_adj|return|win_rate")

    bt_parser = subparsers.add_parser(
        "batch-test",
        help="Batch backtest current strategy on random sample.",
    )
    bt_parser.add_argument("--num", type=int, default=100, help="Number of stocks to sample")
    bt_parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    subparsers.add_parser(
        "train-ai",
        help="Train the LightGBM machine learning predictive model.",
    )
    
    subparsers.add_parser(
        "auto-pilot",
        help="Run the complete daily pipeline: Update -> Optimize -> Analyze.",
    )
    
    scan_parser = subparsers.add_parser(
        "scan-date",
        help="Run signal scan on a specific historical date to verify past recommendations.",
    )
    scan_parser.add_argument("--date", type=str, required=True, help="Target date mapping in YYYY-MM-DD format (e.g., 2024-05-10)")

    args = parser.parse_args()

    if args.command == "update-list":
        cmd_update_list()
    elif args.command == "update-data":
        cmd_update_data()
    elif args.command == "analyze":
        cmd_analyze()
    elif args.command == "ui":
        cmd_ui()
    elif args.command == "optimize":
        cmd_optimize(args)
    elif args.command == "batch-test":
        cmd_batch_test(args)
    elif args.command == "train-ai":
        cmd_train_ai()
    elif args.command == "auto-pilot":
        cmd_auto_pilot()
    elif args.command == "scan-date":
        cmd_scan_date(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Process interrupted by user.")
        sys.exit(0)
    except Exception as e:
        logger.exception("A fatal error occurred.")
        sys.exit(1)
