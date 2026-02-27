import baostock as bs
import pandas as pd
import os
import time
from datetime import datetime, timedelta
from tqdm import tqdm
from quant.config import CONF
from quant.logger import logger


def _fetch_single_stock(code: str, end_date_str: str, default_start_date: str) -> dict:
    """
    在已登录的 baostock 会话中拉取单只股票的历史 K 线数据。
    调用方必须确保已 bs.login()。
    """
    out_file = os.path.join(CONF.history_data.data_dir, f"{code}.csv")
    rs_fields = "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST"

    # 判断是否增量更新
    is_incremental = False
    start_date = default_start_date
    if os.path.exists(out_file):
        try:
            existing_df = pd.read_csv(out_file)
            if not existing_df.empty and 'date' in existing_df.columns:
                last_date_str = str(existing_df['date'].max())
                last_date = datetime.strptime(last_date_str, "%Y-%m-%d")
                start_date = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
                is_incremental = True
        except Exception:
            pass

    # 已是最新数据则跳过
    if is_incremental and start_date > end_date_str:
        return {"code": code, "status": "skip"}

    rs = bs.query_history_k_data_plus(
        code,
        rs_fields,
        start_date=start_date,
        end_date=end_date_str,
        frequency="d",
        adjustflag="2"  # 前复权
    )

    if rs.error_code != '0':
        return {"code": code, "status": "error", "msg": f"API error_code={rs.error_code}, {rs.error_msg}"}

    data_list = []
    while rs.error_code == '0' and rs.next():
        data_list.append(rs.get_row_data())

    if len(data_list) > 0:
        result_df = pd.DataFrame(data_list, columns=rs.fields)
        if is_incremental:
            result_df.to_csv(out_file, mode='a', header=False, index=False, encoding="utf-8-sig")
            return {"code": code, "status": "updated"}
        else:
            result_df.to_csv(out_file, index=False, encoding="utf-8-sig")
            return {"code": code, "status": "new"}
    else:
        return {"code": code, "status": "empty"}


def _safe_session(action: str = "login") -> bool:
    """安全地执行 baostock login/logout，出错不抛异常。"""
    try:
        if action == "login":
            lg = bs.login()
            return lg.error_code == '0'
        else:
            bs.logout()
            return True
    except Exception:
        return False


def update_history_data():
    """
    拉取并增量更新股票池中所有标的的历史 K 线数据。

    策略：
      - baostock 底层使用单一全局 socket，不支持多线程/多进程并发。
      - 因此采用 **串行顺序** 拉取，保证连接稳定。
      - 对失败的股票做二次重试（重新登录后逐个重试）。
    """
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

    total = len(codes)
    logger.info(f"Loaded {total} stocks to update historical data (including market index).")

    end_date = datetime.now()
    end_date_str = end_date.strftime("%Y-%m-%d")
    default_start_date = (end_date - timedelta(days=CONF.history_data.default_lookback_days)).strftime("%Y-%m-%d")

    os.makedirs(CONF.history_data.data_dir, exist_ok=True)

    # ===== 第一轮：顺序拉取全量 =====
    if not _safe_session("login"):
        logger.error("Baostock 登录失败，无法更新历史数据。")
        return

    updated_count = 0
    new_count = 0
    empty_count = 0
    error_count = 0
    skip_count = 0
    failed_codes: list[str] = []

    logger.info("第一轮：串行拉取全量数据...")
    for i, code in enumerate(tqdm(codes, desc="拉取历史数据")):
        try:
            res = _fetch_single_stock(code, end_date_str, default_start_date)
        except Exception as e:
            res = {"code": code, "status": "error", "msg": str(e)}
            # 连接可能已损坏，尝试重新建立
            _safe_session("logout")
            time.sleep(1)
            if not _safe_session("login"):
                logger.error(f"重连失败，中止在第 {i}/{total} 只。")
                break

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
            failed_codes.append(code)
            logger.debug(f"拉取失败 [{code}]: {res.get('msg', 'Unknown')}")

    _safe_session("logout")

    logger.info(
        f"第一轮完成: {new_count} new, {updated_count} updated, "
        f"{empty_count} empty, {skip_count} skipped, {error_count} errors."
    )

    # ===== 第二轮：对失败的代码进行重试 =====
    if failed_codes:
        logger.info(f"第二轮：对 {len(failed_codes)} 只失败股票进行重试（间隔 500ms）...")
        time.sleep(2)  # 等待服务端释放连接

        if not _safe_session("login"):
            logger.error("第二轮重试登录失败，跳过。")
        else:
            retry_success = 0
            still_failed: list[str] = []

            for code in tqdm(failed_codes, desc="重试失败股票"):
                time.sleep(0.5)  # 每个请求间隔 500ms
                try:
                    res = _fetch_single_stock(code, end_date_str, default_start_date)
                    if res["status"] in ("new", "updated", "empty", "skip"):
                        retry_success += 1
                    else:
                        still_failed.append(code)
                        logger.warning(f"二次拉取仍失败 [{code}]: {res.get('msg', 'Unknown')}")
                except Exception as e:
                    still_failed.append(code)
                    logger.warning(f"二次拉取异常 [{code}]: {e}")
                    # 连接损坏时重连
                    _safe_session("logout")
                    time.sleep(1)
                    if not _safe_session("login"):
                        logger.error("重试期间重连失败，中止重试。")
                        still_failed.extend([c for c in failed_codes if c not in still_failed and c != code])
                        break

            _safe_session("logout")

            error_count = error_count - retry_success
            logger.info(
                f"第二轮重试完成: {retry_success}/{len(failed_codes)} 恢复成功。"
                f"最终剩余 {len(still_failed)} 只无法获取。"
            )

            if still_failed:
                logger.info(f"最终失败列表 (前 30 个): {still_failed[:30]}")

    # ===== 最终汇总 =====
    total_success = new_count + updated_count + retry_success if failed_codes else new_count + updated_count
    logger.info(
        f"===== 数据更新完毕 =====\n"
        f"  成功下载/更新: {total_success} 只\n"
        f"  无新数据 (empty): {empty_count} 只\n"
        f"  已是最新 (skip): {skip_count} 只\n"
        f"  最终失败: {error_count} 只"
    )
