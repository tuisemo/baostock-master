import baostock as bs
import pandas as pd
import os
from datetime import datetime, timedelta
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from quant.config import CONF
from quant.logger import logger

def _process_single_stock(code: str, end_date_str: str, default_start_date: str) -> dict:
    # 必须在子进程内部重新登录 baostock
    lg = bs.login()
    if lg.error_code != '0':
        return {"code": code, "status": "error", "msg": lg.error_msg}

    out_file = os.path.join(CONF.history_data.data_dir, f"{code}.csv")
    rs_fields = "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST"
    
    # Determine start date for incremental update
    is_incremental = False
    if os.path.exists(out_file):
        try:
            existing_df = pd.read_csv(out_file)
            if not existing_df.empty and 'date' in existing_df.columns:
                last_date_str = str(existing_df['date'].max())
                last_date = datetime.strptime(last_date_str, "%Y-%m-%d")
                # Start from the next day
                start_date = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
                is_incremental = True
            else:
                start_date = default_start_date
        except Exception as e:
            start_date = default_start_date
    else:
        start_date = default_start_date

    # If we are already up to date
    if is_incremental and start_date > end_date_str:
        bs.logout()
        return {"code": code, "status": "skip"}

    rs = bs.query_history_k_data_plus(
        code,
        rs_fields,
        start_date=start_date,
        end_date=end_date_str,
        frequency="d",
        adjustflag="2"  # 前复权 (Forward-adjusted)
    )
    
    status_dict = {"code": code, "status": "error", "msg": rs.error_msg}
    
    if rs.error_code == '0':
        data_list = []
        while rs.error_code == '0' and rs.next():
            data_list.append(rs.get_row_data())
            
        if len(data_list) > 0:
            result_df = pd.DataFrame(data_list, columns=rs.fields)
            if is_incremental:
                # Append without header
                result_df.to_csv(out_file, mode='a', header=False, index=False, encoding="utf-8-sig")
                status_dict["status"] = "updated"
            else:
                result_df.to_csv(out_file, index=False, encoding="utf-8-sig")
                status_dict["status"] = "new"
        else:
            status_dict["status"] = "empty"
    
    bs.logout()
    return status_dict


def update_history_data():
    """Fetch and incrementally update historical K-line data for all stocks in the high-value list. (Parallelized)"""
    list_path = os.path.join(CONF.history_data.data_dir, "stock-list.csv")
    if not os.path.exists(list_path):
        logger.error(f"Stock list not found at {list_path}. Please run 'update-list' first.")
        return

    df_list = pd.read_csv(list_path)
    if df_list.empty:
        logger.warning("Stock list is empty. Nothing to update.")
        return

    codes = df_list['code'].tolist()
    if 'sh.000001' not in codes:
        codes.append('sh.000001')
    logger.info(f"Loaded {len(codes)} stocks to update historical data (including market index).")

    end_date = datetime.now()
    end_date_str = end_date.strftime("%Y-%m-%d")
    default_start_date = (end_date - timedelta(days=CONF.history_data.default_lookback_days)).strftime("%Y-%m-%d")

    updated_count = 0
    new_count = 0
    empty_count = 0
    error_count = 0
    skip_count = 0

    max_workers = os.cpu_count() or 4
    logger.info(f"Starting multi-process pool with {max_workers} workers...")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_process_single_stock, cd, end_date_str, default_start_date): cd for cd in codes}
        
        for future in tqdm(as_completed(futures), total=len(codes), desc="Updating History (Parallel)"):
            res = future.result()
            s = res["status"]
            if s == "updated":
                updated_count += 1
            elif s == "new":
                new_count += 1
            elif s == "empty":
                empty_count += 1
            elif s == "skip":
                skip_count += 1
            elif s == "error":
                error_count += 1
                logger.debug(f"Failed to fetch {res['code']}: {res.get('msg', 'Unknown Error')}")

    logger.info(f"Update summary: {new_count} new files created, {updated_count} files incrementally updated, {empty_count} API requests returned no new data, {skip_count} skipped (up-to-date), {error_count} errors.")
