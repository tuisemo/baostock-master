import baostock as bs
import pandas as pd
import requests
import os
import time
from tqdm import tqdm
from quant.config import CONF
from quant.logger import logger

def get_tencent_hq(codes: list) -> dict:
    """
    Fetch real-time quotes from Tencent Finance API.
    codes: list of expected tencent code format (e.g., sh600000)
    Returns a dict {code: {market_cap: float, turnover_amount: float, turnover_rate: float}}
    """
    if not codes:
        return {}
    
    url = f"http://qt.gtimg.cn/q={','.join(codes)}"
    try:
        resp = requests.get(url, timeout=5)
        text = resp.text
    except Exception as e:
        logger.error(f"Error fetching from Tencent API: {e}")
        return {}

    result = {}
    lines = text.strip().split(';')
    for line in lines:
        if not line.strip():
            continue
        parts = line.split('=')
        if len(parts) < 2:
            continue
        key = parts[0].strip().replace('v_', '')
        val_str = parts[1].strip().strip('"')
        vals = val_str.split('~')
        
        if len(vals) > 45:
            try:
                # 37: 成交额(万)
                # 38: 换手率(%)
                # 44: 流通市值(亿) 
                # 45: 总市值(亿)
                amount_wan = float(vals[37]) if vals[37] else 0.0
                turnover_rate = float(vals[38]) if vals[38] else 0.0
                circulating_mcap_yi = float(vals[44]) if vals[44] else 0.0
                
                result[key] = {
                    'amount_wan': amount_wan,
                    'turnover_rate': turnover_rate,
                    'circulating_mcap_yi': circulating_mcap_yi,
                }
            except ValueError:
                pass
    return result

def update_stock_list():
    """Fetch, filter, and save the active stock list."""
    logger.info("Starting stock list update process.")
    
    # 1. Login baostock
    lg = bs.login()
    if lg.error_code != '0':
        logger.error(f"Baostock login failed: {lg.error_msg}")
        return

    # 2. Fetch all stocks
    logger.info("Fetching all A-share combinations from baostock...")
    rs = bs.query_all_stock()
    if rs.error_code != '0':
        logger.error(f"Query all stock failed: {rs.error_msg}")
        bs.logout()
        return

    data_list = []
    while (rs.error_code == '0') and rs.next():
        data_list.append(rs.get_row_data())
    
    df = pd.DataFrame(data_list, columns=rs.fields)
    logger.info(f"Total entries from baostock: {len(df)}")
    
    # 3. Base Filtering
    # Remove BSE (bj.)
    filtered_df = df[~df['code'].str.startswith('bj.')]
    
    # Remove STAR if configured
    if not CONF.filter.keep_star_market:
        filtered_df = filtered_df[~filtered_df['code'].str.startswith('sh.68')]
        
    # Remove ST, *ST, and delisted stocks
    filtered_df = filtered_df[~filtered_df['code_name'].str.contains('ST', na=False, case=False)]
    filtered_df = filtered_df[~filtered_df['code_name'].str.contains('退', na=False)]
    
    logger.info(f"Count after basic rules (BSE, STAR, ST, Delisted): {len(filtered_df)}")

    # 4. Deep Filtering (Micro-cap, Zombie, Manipulated)
    filtered_df['tc_code'] = filtered_df['code'].apply(lambda x: x.replace('.', ''))
    tc_codes = filtered_df['tc_code'].tolist()
    
    logger.info("Fetching real-time market data from Tencent to filter micro-cap, zombie, and manipulated stocks...")
    batch_size = 60
    tc_data = {}
    for i in tqdm(range(0, len(tc_codes), batch_size), desc="Fetching HQ"):
        batch = tc_codes[i:i+batch_size]
        tc_data.update(get_tencent_hq(batch))
        time.sleep(0.1)

    final_list = []
    stats = {"micro_cap": 0, "zombie": 0, "manipulated": 0, "inactive": 0}
    
    for _, row in filtered_df.iterrows():
        hq = tc_data.get(row['tc_code'])
        if not hq:
            continue
            
        mcap = hq.get('circulating_mcap_yi', 0.0)
        amount = hq.get('amount_wan', 0.0)
        turnover = hq.get('turnover_rate', 0.0)
        
        if mcap < CONF.filter.min_market_cap_billion:
            stats["micro_cap"] += 1
            continue
            
        if amount < CONF.filter.min_turnover_amount_wan:
            stats["zombie"] += 1
            continue
            
        if turnover > CONF.filter.max_turnover_rate_pct:
            stats["manipulated"] += 1
            continue
            
        if turnover < CONF.filter.min_turnover_rate_pct:
            stats["inactive"] += 1
            continue
            
        final_list.append(row)

    logger.info(f"Filtered {stats['micro_cap']} micro-cap stocks (< {CONF.filter.min_market_cap_billion} 亿).")
    logger.info(f"Filtered {stats['zombie']} zombie stocks (< {CONF.filter.min_turnover_amount_wan} 万).")
    logger.info(f"Filtered {stats['manipulated']} potential manipulated stocks (turnover > {CONF.filter.max_turnover_rate_pct}%).")
    logger.info(f"Filtered {stats['inactive']} highly inactive stocks (turnover < {CONF.filter.min_turnover_rate_pct}%).")
    
    final_df = pd.DataFrame(final_list)
    logger.info(f"Final high-value stock count: {len(final_df)}")
    
    # 5. Save results
    os.makedirs(CONF.history_data.data_dir, exist_ok=True)
    out_path = os.path.join(CONF.history_data.data_dir, 'stock-list.csv')
    
    if not final_df.empty:
        final_df.drop(columns=['tc_code'], inplace=True, errors='ignore')
        final_df.to_csv(out_path, index=False, encoding='utf-8-sig')
        logger.info(f"Saved high-value stock list to {out_path}")
    else:
        logger.warning("No stocks passed the filters! Check your config threshold.")

    bs.logout()
