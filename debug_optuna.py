import pandas as pd
from quant.backtester import batch_backtest
from quant.strategy_params import StrategyParams
from quant.optimizer import sample_stock_codes

def main():
    from quant.logger import logger
    logger.setLevel("DEBUG")
    
    codes = sample_stock_codes(5, seed=0)
    print("Codes:", codes)
    
    trial0_params = {'ma_short': 3, 'ma_long': 35, 'macd_fast': 10, 'macd_slow': 26, 'macd_signal': 13, 'rsi_length': 10, 'rsi_buy_threshold': 40, 'rsi_sell_threshold': 80, 'bbands_length': 30, 'bbands_std': 2.75, 'atr_length': 12, 'atr_multiplier': 2.0, 'weight_trend': 0.2, 'weight_reversion': 0.30000000000000004, 'weight_volume': 0.5, 'w_pullback_ma': 2.5, 'w_macd_cross': 2.0, 'w_vol_up': 1.5, 'w_rsi_rebound': 3.5, 'w_green_candle': 2.5, 'vol_up_ratio': 1.1, 'rsi_cooled_max': 50, 'pullback_ma_tolerance': 1.04, 'negative_bias_pct': 0.8300000000000001, 'rsi_oversold': 25, 'bbands_lower_bias': 1.03, 'rsi_oversold_extreme': 16, 'trail_atr_mult': 1.7000000000000002, 'take_profit_pct': 0.11, 'breakeven_trigger': 0.05500000000000001, 'breakeven_buffer': 1.002, 'ai_prob_threshold': 0.35}
    params = StrategyParams.from_dict(trial0_params)
    print("AI Gate:", params.ai_prob_threshold)
    
    try:
        from quant.backtester import run_backtest
        for code in codes:
            res = run_backtest(code, params, end_date="2025-09-19")
            if res is None:
                print(f"{code} -> run_backtest returned None")
            else:
                bt, stats = res
                print(f"{code} -> Trades:", stats['# Trades'])
    except Exception as e:
        import traceback
        traceback.print_exc()
        
if __name__ == "__main__":
    main()
