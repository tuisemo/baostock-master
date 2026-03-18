#!/usr/bin/env python3
"""
配置差异检测脚本

对比 config.yaml 与 config.py dataclass 的参数值，
检测并报告所有不一致项。

使用方法:
    python scripts/validate_config.py [--verbose] [--json]
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json

import yaml


def load_yaml_config(config_path: str = "config.yaml") -> dict:
    """加载YAML配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_dataclass_defaults():
    """获取所有dataclass的默认值"""
    from quant.infra.config import AppConfig
    from quant.core.strategy_params import StrategyParams
    
    config = AppConfig()
    
    defaults = {
        "config.AppConfig": {
            "log.level": config.log.level,
            "log.file": config.log.file,
        },
        "config.FilterConfig": {
            "filter.keep_star_market": config.filter.keep_star_market,
            "filter.min_market_cap_billion": config.filter.min_market_cap_billion,
            "filter.min_turnover_amount_wan": config.filter.min_turnover_amount_wan,
            "filter.max_turnover_rate_pct": config.filter.max_turnover_rate_pct,
            "filter.min_turnover_rate_pct": config.filter.min_turnover_rate_pct,
            "filter.min_pe": config.filter.min_pe,
            "filter.max_pe": config.filter.max_pe,
            "filter.min_pb": config.filter.min_pb,
        },
        "config.AnalyzerConfig": {
            "analyzer.ma_short": config.analyzer.ma_short,
            "analyzer.ma_long": config.analyzer.ma_long,
            "analyzer.macd_fast": config.analyzer.macd_fast,
            "analyzer.macd_slow": config.analyzer.macd_slow,
            "analyzer.macd_signal": config.analyzer.macd_signal,
            "analyzer.rsi_length": config.analyzer.rsi_length,
            "analyzer.rsi_buy_threshold": config.analyzer.rsi_buy_threshold,
            "analyzer.rsi_sell_threshold": config.analyzer.rsi_sell_threshold,
            "analyzer.bbands_length": config.analyzer.bbands_length,
            "analyzer.bbands_std": config.analyzer.bbands_std,
            "analyzer.atr_length": config.analyzer.atr_length,
            "analyzer.atr_multiplier": config.analyzer.atr_multiplier,
            "analyzer.weights.trend": config.analyzer.weights.trend,
            "analyzer.weights.reversion": config.analyzer.weights.reversion,
            "analyzer.weights.volume": config.analyzer.weights.volume,
        },
        "config.StrategyConfig": {
            "strategy.vol_up_ratio": config.strategy.vol_up_ratio,
            "strategy.rsi_cooled_max": config.strategy.rsi_cooled_max,
            "strategy.pullback_ma_tolerance": config.strategy.pullback_ma_tolerance,
            "strategy.negative_bias_pct": config.strategy.negative_bias_pct,
            "strategy.rsi_oversold": config.strategy.rsi_oversold,
            "strategy.bbands_lower_bias": config.strategy.bbands_lower_bias,
            "strategy.rsi_oversold_extreme": config.strategy.rsi_oversold_extreme,
            "strategy.trail_atr_mult": config.strategy.trail_atr_mult,
            "strategy.take_profit_pct": config.strategy.take_profit_pct,
            "strategy.breakeven_trigger": config.strategy.breakeven_trigger,
            "strategy.breakeven_buffer": config.strategy.breakeven_buffer,
            "strategy.w_pullback_ma": config.strategy.w_pullback_ma,
            "strategy.w_macd_cross": config.strategy.w_macd_cross,
            "strategy.w_vol_up": config.strategy.w_vol_up,
            "strategy.w_rsi_rebound": config.strategy.w_rsi_rebound,
            "strategy.w_green_candle": config.strategy.w_green_candle,
            "strategy.ai_prob_threshold": config.strategy.ai_prob_threshold,
            "strategy.bear_market_ai_threshold": config.strategy.bear_market_ai_threshold,
            "strategy.position_size": config.strategy.position_size,
            "strategy.max_hold_days": config.strategy.max_hold_days,
            "strategy.max_hold_min_return": config.strategy.max_hold_min_return,
            "strategy.rsi_overbought_left": config.strategy.rsi_overbought_left,
            "strategy.rsi_overbought_right": config.strategy.rsi_overbought_right,
            "strategy.min_hold_days": config.strategy.min_hold_days,
            "strategy.signal_cooldown_days": config.strategy.signal_cooldown_days,
            "strategy.commission_pct": config.strategy.commission_pct,
            "strategy.slippage_pct": config.strategy.slippage_pct,
            "strategy.ai_forward_days": config.strategy.ai_forward_days,
            "strategy.ai_target_atr_mult": config.strategy.ai_target_atr_mult,
            "strategy.ai_stop_loss_atr_mult": config.strategy.ai_stop_loss_atr_mult,
            "strategy.min_expected_value_pct": config.strategy.min_expected_value_pct,
        },
        "config.OptimizerConfig": {
            "optimizer.sample_count": config.optimizer.sample_count,
            "optimizer.max_rounds": config.optimizer.max_rounds,
            "optimizer.convergence_threshold": config.optimizer.convergence_threshold,
            "optimizer.walk_forward_splits": config.optimizer.walk_forward_splits,
            "optimizer.train_ratio": config.optimizer.train_ratio,
            "optimizer.objective": config.optimizer.objective,
            "optimizer.regularization_strength": config.optimizer.regularization_strength,
            "optimizer.walk_forward_folds": config.optimizer.walk_forward_folds,
            "optimizer.results_dir": config.optimizer.results_dir,
        },
    }
    
    # StrategyParams
    params = StrategyParams()
    params_defaults = {
        "strategy_params.w_pullback_ma": params.w_pullback_ma,
        "strategy_params.w_macd_cross": params.w_macd_cross,
        "strategy_params.w_vol_up": params.w_vol_up,
        "strategy_params.w_rsi_rebound": params.w_rsi_rebound,
        "strategy_params.w_green_candle": params.w_green_candle,
        "strategy_params.ai_prob_threshold": params.ai_prob_threshold,
        "strategy_params.trail_atr_mult": params.trail_atr_mult,
        "strategy_params.take_profit_pct": params.take_profit_pct,
        "strategy_params.position_size": params.position_size,
    }
    defaults["config.StrategyParams"] = params_defaults
    
    return defaults


def flatten_yaml(yaml_data: dict, prefix: str = "") -> dict:
    """将嵌套的yaml数据展平为key路径"""
    result = {}
    for key, value in yaml_data.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(flatten_yaml(value, full_key))
        else:
            result[full_key] = value
    return result


def compare_configs(yaml_data: dict, dataclass_defaults: dict) -> list:
    """比较配置并返回差异列表"""
    yaml_flat = flatten_yaml(yaml_data)
    diffs = []
    
    for source, defaults in dataclass_defaults.items():
        for key, default_value in defaults.items():
            yaml_value = yaml_flat.get(key)
            
            # 跳过不存在于yaml中的配置（使用默认值）
            if key not in yaml_flat:
                continue
            
            # 比较值
            if yaml_value != default_value:
                diffs.append({
                    "key": key,
                    "source": source,
                    "yaml_value": yaml_value,
                    "dataclass_default": default_value,
                    "deviation": abs(yaml_value - default_value) / (abs(default_value) + 1e-8) if isinstance(default_value, (int, float)) else None
                })
    
    return diffs


def main():
    parser = argparse.ArgumentParser(description="配置差异检测脚本")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")
    parser.add_argument("--json", action="store_true", help="JSON格式输出")
    args = parser.parse_args()
    
    # 加载配置
    try:
        yaml_config = load_yaml_config()
        dataclass_defaults = get_dataclass_defaults()
    except Exception as e:
        print(f"ERROR: 加载配置失败: {e}", file=sys.stderr)
        return 1
    
    # 比较配置
    diffs = compare_configs(yaml_config, dataclass_defaults)
    
    if args.json:
        output = {
            "total_diffs": len(diffs),
            "diffs": diffs
        }
        print(json.dumps(output, indent=2, ensure_ascii=False))
        return 0 if len(diffs) == 0 else 1
    
    # 文本输出
    print("=" * 60)
    print("配置差异检测报告")
    print("=" * 60)
    print(f"配置文件: config.yaml")
    print(f"对比来源: config.py dataclass默认值")
    print(f"发现差异: {len(diffs)} 个")
    print("=" * 60)
    
    if diffs:
        print("\n详细差异:\n")
        for i, diff in enumerate(diffs, 1):
            print(f"[{i}] {diff['key']}")
            print(f"    来源: {diff['source']}")
            print(f"    YAML值: {diff['yaml_value']}")
            print(f"    dataclass默认值: {diff['dataclass_default']}")
            if diff['deviation'] is not None:
                print(f"    偏差: {diff['deviation']*100:.1f}%")
            print()
    else:
        print("\n配置一致，无差异!")
    
    print("=" * 60)
    
    return 0 if len(diffs) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
