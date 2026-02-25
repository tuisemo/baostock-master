from __future__ import annotations

import argparse
import os
import random
import sys

from quant.logger import logger


def cmd_update_list():
    from quant.stock_filter import update_stock_list
    update_stock_list()


def cmd_update_data():
    from quant.data_updater import update_history_data
    update_history_data()


def cmd_analyze():
    from quant.analyzer import analyze_all_stocks
    analyze_all_stocks()


def cmd_ui():
    try:
        from app import demo
        logger.info("启动可视化量化平台 (Web UI)...")
        demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
    except Exception as e:
        logger.error(f"启动 Web UI 失败: {e}")


def cmd_optimize(args: argparse.Namespace):
    from quant.config import CONF
    from quant.optimizer import run_optimization, save_results, apply_best_params

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
    from quant.backtester import batch_backtest
    from quant.config import CONF

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
