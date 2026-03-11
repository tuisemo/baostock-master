import os
import pandas as pd
from datetime import datetime, timedelta

def verify_single_date(target_date: str):
    scan_file = f"historical_scan_{target_date.replace('-', '')}.csv"
    if not os.path.exists(scan_file):
        print(f"File {scan_file} not found. Ensure you ran `uv run python main.py scan-date --date {target_date}`")
        return

    df_scan = pd.read_csv(scan_file)
    if df_scan.empty:
        print(f"No signals found for {target_date}")
        return

    print(f"Found {len(df_scan)} buy signals on {target_date}.")
    print("-" * 60)
    
    data_dir = 'data'
    results = []
    
    for _, row in df_scan.iterrows():
        code = row['code']
        try:
            file_path = os.path.join(data_dir, f'{code}.csv')
            if not os.path.exists(file_path):
                continue
                
            df = pd.read_csv(file_path)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                
                # Find entry data (the target date)
                # target_date is the date the signal was emitted at close
                target_dt = pd.to_datetime(target_date)
                entry_data = df[df['date'] == target_dt]
                if len(entry_data) > 0:
                    entry_price = entry_data['close'].values[0]
                    entry_idx = df[df['date'] == target_dt].index[0]
                    
                    # Buy next day open technically, but let's just use close of the signal day as proxy or next day's open
                    # We will use next day's open if available, else signal's close
                    if entry_idx + 1 < len(df):
                        real_entry = df.iloc[entry_idx + 1]['open']
                    else:
                        real_entry = entry_price
                        
                    # Target exit 5 days later
                    if entry_idx + 5 < len(df):
                        exit_price = df.iloc[entry_idx + 5]['close']
                        return_pct = (exit_price - real_entry) / real_entry * 100
                        
                        status = 'PROFIT' if return_pct > 0 else 'LOSS'
                        results.append({
                            'code': code,
                            'entry_price': real_entry,
                            'exit_price': exit_price,
                            'return_pct': return_pct,
                            'status': status
                        })
                    else:
                        print(f"{code}: Not enough data for 5 days forward.")
        except Exception as e:
            print(f"Error checking {code}: {e}")
            
    if results:
        res_df = pd.DataFrame(results)
        print(res_df.to_string(index=False))
        
        win_rate = (res_df['status'] == 'PROFIT').sum() / len(res_df) * 100
        avg_ret = res_df['return_pct'].mean()
        
        print("-" * 60)
        print(f"Summary for {target_date}:")
        print(f"Total Trades: {len(res_df)}")
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Average Return: {avg_ret:.2f}%")
    else:
        print("No valid complete trades to check.")

if __name__ == "__main__":
    import sys
    d = "2025-01-15"
    if len(sys.argv) > 1:
        d = sys.argv[1]
    verify_single_date(d)
