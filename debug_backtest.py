import pandas as pd
from quant.backtester import _resolve_params, _load_and_prepare, _build_column_names
from quant.config import CONF

def debug_signals(code):
    params = _resolve_params(None)
    df = _load_and_prepare(code, params)
    
    if df is None:
        print(f"Data for {code} could not be loaded or prepared.")
        return
        
    cols = _build_column_names(params)
    
    has_vol_slope = "vol_slope" in df.columns
    has_mom_div = "momentum_divergence" in df.columns
    
    counts = {
        'total_rows': 0,
        'ma_long_up': 0,
        'near_ma_long': 0,
        'green_candle': 0,
        'macd_gc': 0,
        'vol_up': 0,
        'rsi_cooled': 0,
        'mom_ok': 0,
        'pullback_signal': 0,
        
        'negative_bias': 0,
        'rsi_oversold': 0,
        'macd_turn_up': 0,
        'rebound_signal': 0
    }
    
    for i in range(2, len(df)):
        counts['total_rows'] += 1
        
        row_1 = df.iloc[i]
        row_2 = df.iloc[i-1]
        row_3 = df.iloc[i-2]
        
        price = row_1['Close']
        
        sma_l_1 = row_1[cols['sma_l']]
        sma_l_3 = row_3[cols['sma_l']]
        
        is_ma_long_up = sma_l_1 > sma_l_3
        if is_ma_long_up: counts['ma_long_up'] += 1
            
        near_ma_long = row_1['Low'] <= sma_l_1 * params.pullback_ma_tolerance
        if near_ma_long: counts['near_ma_long'] += 1
            
        is_green_candle = price > row_1['Open']
        if is_green_candle: counts['green_candle'] += 1
            
        if has_vol_slope and row_1['vol_slope'] > 0.1:
            vol_up = True
        else:
            vol_up = row_1['Volume'] > row_2['Volume'] * params.vol_up_ratio
        if vol_up: counts['vol_up'] += 1
            
        rsi_cooled = row_1[cols['rsi']] < params.rsi_cooled_max
        if rsi_cooled: counts['rsi_cooled'] += 1
            
        mom_ok = True
        if has_mom_div and row_1['momentum_divergence'] <= -0.02:
            mom_ok = False
        if mom_ok: counts['mom_ok'] += 1

        macd_golden_cross = (row_2[cols["macd_h"]] <= 0) and (row_1[cols["macd_h"]] > 0)
        if macd_golden_cross:
            counts['macd_gc'] += 1
            # Check what is missing
            missing = []
            if not is_ma_long_up: missing.append('ma_long_up')
            if not near_ma_long: missing.append('near_ma_long')
            if not is_green_candle: missing.append('green_candle')
            if not vol_up: missing.append('vol_up')
            if not rsi_cooled: missing.append('rsi_cooled')
            if not mom_ok: missing.append('mom_ok')
            print(f"MACD GC on {row_1.name}. Missing for Pullback: {missing}")

        negative_bias = price < row_1[cols['sma_s']] * params.negative_bias_pct
        rsi_oversold = row_1[cols['rsi']] < params.rsi_oversold
        macd_turn_up = row_1[cols['macd_h']] > row_2[cols['macd_h']]

        if negative_bias:
            counts['negative_bias'] += 1
            missing = []
            if not rsi_oversold: missing.append('rsi_oversold')
            if not is_green_candle: missing.append('green_candle')
            if not vol_up: missing.append('vol_up')
            if not macd_turn_up: missing.append('macd_turn_up')
            if not mom_ok: missing.append('mom_ok')
            print(f"Negative Bias on {row_1.name}. Missing for Rebound: {missing}")
            
    print(f"--- Signal Debug for {code} ---")
    for k, v in counts.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    debug_signals("sh.600010")
    debug_signals("sz.000006")
