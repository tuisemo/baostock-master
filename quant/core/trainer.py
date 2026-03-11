import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import datetime
# Time-series aware splitting (no random shuffle for temporal data)
from sklearn.metrics import classification_report, roc_auc_score
from quant.infra.config import CONF
from quant.infra.logger import logger
from quant.features.features.analyzer import calculate_indicators
from quant.core.strategy_params import StrategyParams
from quant.features.features import extract_features, create_targets, create_multi_class_targets
from quant.infra.cache_utils import DatasetCache, get_data_hash
from quant.infra.numba_accelerator import get_numba_status
from quant.core.ensemble_trainer import MultiModelEnsemble
from tqdm import tqdm
import time

# SHAP for model interpretability (optional)
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not available. Model interpretability features will be disabled.")

def focal_loss_lgb(y_pred, dataset):
    """
    Focal Loss implementation for LightGBM to handle class imbalance.
    Focal Loss = -alpha * (1 - p_t)^gamma * log(p_t)

    Args:
        y_pred: predicted probabilities
        dataset: LightGBM dataset

    Returns:
        gradients and hessians
    """
    y_true = dataset.get_label()
    y_pred = 1.0 / (1.0 + np.exp(-y_pred))  # sigmoid

    # Focal Loss parameters
    alpha = 0.75  # weight for positive class
    gamma = 2.0   # focusing parameter

    # Calculate p_t
    p_t = np.where(y_true == 1, y_pred, 1 - y_pred)

    # Calculate focal loss gradient and hessian
    gradient = alpha * (1 - p_t) ** gamma * (y_pred - y_true)
    hessian = alpha * (1 - p_t) ** gamma * (y_pred * (1 - y_pred))

    return gradient, hessian


def focal_loss_eval(y_pred, dataset):
    """
    Evaluation metric for Focal Loss.

    Args:
        y_pred: predicted probabilities
        dataset: LightGBM dataset

    Returns:
        metric name, value, is_higher_better
    """
    y_true = dataset.get_label()
    y_pred = 1.0 / (1.0 + np.exp(-y_pred))  # sigmoid

    # Calculate binary cross-entropy focal loss
    alpha = 0.75
    gamma = 2.0

    p_t = np.where(y_true == 1, y_pred, 1 - y_pred)
    focal_loss = -alpha * ((1 - p_t) ** gamma) * np.log(p_t + 1e-8)
    avg_loss = np.mean(focal_loss)

    return 'focal_loss', avg_loss, False  # lower is better


def balanced_temporal_split(X, y, train_ratio=0.8, min_class_ratio=0.3):
    """
    Implement balanced temporal split to ensure both classes are well-represented
    while maintaining temporal order.

    Args:
        X: Feature DataFrame
        y: Target Series
        train_ratio: Proportion of data for training
        min_class_ratio: Minimum ratio of minority class in each split

    Returns:
        X_train, y_train, X_test, y_test
    """
    n_samples = len(X)

    # Find natural split point
    split_idx = int(n_samples * train_ratio)

    # Check class distribution in train and test sets
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    train_pos_ratio = y_train.sum() / len(y_train) if len(y_train) > 0 else 0
    test_pos_ratio = y_test.sum() / len(y_test) if len(y_test) > 0 else 0

    logger.info(f"Initial split - Train pos ratio: {train_pos_ratio:.3f}, Test pos ratio: {test_pos_ratio:.3f}")

    # If both splits have reasonable class balance, use simple split
    if (train_pos_ratio >= min_class_ratio and test_pos_ratio >= min_class_ratio and
        train_pos_ratio <= 1 - min_class_ratio and test_pos_ratio <= 1 - min_class_ratio):
        X_train = X.iloc[:split_idx]
        y_train = y.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_test = y.iloc[split_idx:]
        logger.info("Using simple temporal split (classes well-balanced)")
        return X_train, y_train, X_test, y_test

    # Otherwise, use windowed sampling to balance classes
    logger.info("Using windowed balanced sampling for better class balance")

    # Divide data into windows and sample from each window
    window_size = 100  # days per window
    stride = 50  # overlap between windows

    train_samples_X = []
    train_samples_y = []
    test_samples_X = []
    test_samples_y = []

    for i in range(0, n_samples - window_size + 1, stride):
        window_start = i
        window_end = i + window_size

        # Split this window based on overall split ratio
        window_train_end = window_start + int(window_size * train_ratio)

        # Add to train set
        if window_train_end <= n_samples:
            window_X_train = X.iloc[window_start:window_train_end]
            window_y_train = y.iloc[window_start:window_train_end]
            train_samples_X.append(window_X_train)
            train_samples_y.append(window_y_train)

        # Add to test set
        if window_train_end < window_end and window_end <= n_samples:
            window_X_test = X.iloc[window_train_end:window_end]
            window_y_test = y.iloc[window_train_end:window_end]
            test_samples_X.append(window_X_test)
            test_samples_y.append(window_y_test)

    # Concatenate all samples
    if train_samples_X:
        X_train = pd.concat(train_samples_X, axis=0)
        y_train = pd.concat(train_samples_y, axis=0)
    else:
        X_train = X.iloc[:split_idx]
        y_train = y.iloc[:split_idx]

    if test_samples_X:
        X_test = pd.concat(test_samples_X, axis=0)
        y_test = pd.concat(test_samples_y, axis=0)
    else:
        X_test = X.iloc[split_idx:]
        y_test = y.iloc[split_idx:]

    # Re-sort by index to maintain temporal order within each split
    train_idx = X_train.index.argsort()
    X_train = X_train.iloc[train_idx]
    y_train = y_train.iloc[train_idx]

    test_idx = X_test.index.argsort()
    X_test = X_test.iloc[test_idx]
    y_test = y_test.iloc[test_idx]

    # Verify class balance
    new_train_pos_ratio = y_train.sum() / len(y_train) if len(y_train) > 0 else 0
    new_test_pos_ratio = y_test.sum() / len(y_test) if len(y_test) > 0 else 0

    logger.info(f"Balanced split - Train pos ratio: {new_train_pos_ratio:.3f}, Test pos ratio: {new_test_pos_ratio:.3f}")
    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    return X_train, y_train, X_test, y_test

def build_dataset(data_dir: str, p: StrategyParams, n_forward_days: int = 5, target_atr_mult: float = 2.0, stop_loss_atr_mult: float = 1.5) -> pd.DataFrame:
    """Read all csv files, compute indicators, extract features and targets, and concatenate into one large dataframe."""
    logger.info("Building dataset from historical data...")
    all_dfs = []
    
    files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    if not files:
        logger.error(f"No csv files found in {data_dir}")
        return pd.DataFrame()
        
    for f in tqdm(files, desc="Processing files"):
        if f == "stock-list.csv":
            continue
            
        file_path = os.path.join(data_dir, f)
        df = pd.read_csv(file_path)
        if df.empty or len(df) < 50:
            continue
            
        try:
            # 1. Base Indicators (requires raw lowercase columns like 'volume')
            df = calculate_indicators(df, p)
            
            # Clean and rename columns after indicator calculation
            rename_map = {
                "date": "Date", "open": "Open", "high": "High", "low": "Low", 
                "close": "Close", "volume": "Volume"
            }
            df.rename(columns=rename_map, inplace=True)
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"])
                df.set_index("Date", inplace=True)
                df.sort_index(inplace=True)
                
            # 2. Add derived features
            df = extract_features(df)
            # 3. Add truth Target Y (both binary and multi-class)
            df = create_targets(df, n_forward_days=n_forward_days, target_atr_mult=target_atr_mult, stop_loss_atr_mult=stop_loss_atr_mult)
            # 3.1 Add multi-class target
            from quant.features.features import create_multi_class_targets
            df = create_multi_class_targets(df, n_forward_days=n_forward_days, target_atr_mult=target_atr_mult, stop_loss_atr_mult=stop_loss_atr_mult)
        except Exception as e:
            logger.debug(f"Error processing {f}: {e}")
            continue
            
        feature_cols = [c for c in df.columns if c.startswith('feat_')]
        if not feature_cols:
            continue

        # Drop rows where Y is nan (the last n_forward_days)
        df = df.dropna(subset=['label_max_ret_5d'] + feature_cols)
        
        # Keep stock code for cross validation splits
        df['stock_code'] = f.replace(".csv", "")
        # Keep a sortable date column for chronological splitting
        if 'Date' not in df.columns and df.index.name == 'Date':
            df['_sort_date'] = df.index
        elif 'Date' in df.columns:
            df['_sort_date'] = pd.to_datetime(df['Date'])
        else:
            df['_sort_date'] = range(len(df))
        all_dfs.append(df)
        
    if not all_dfs:
        logger.error("Failed to build dataset. Valid data empty.")
        return pd.DataFrame()
        
    final_df = pd.concat(all_dfs, axis=0)
    logger.info(f"Dataset built successfully. Total rows: {len(final_df)}. Features: {len(feature_cols)}")
    return final_df


def build_dataset_with_cache(
    data_dir: str,
    p: StrategyParams,
    n_forward_days: int = 5,
    target_atr_mult: float = 2.0,
    stop_loss_atr_mult: float = 1.5,
    use_cache: bool = True,
    use_parallel: bool = True,
    n_workers: int = None,
    force_rebuild: bool = False
) -> pd.DataFrame:
    """
    带缓存和并行化支持的数据集构建函数

    Args:
        data_dir: 数据目录
        p: 策略参数
        n_forward_days: 向前看的天数
        target_atr_mult: 目标ATR倍数
        stop_loss_atr_mult: 止损ATR倍数
        use_cache: 是否使用缓存
        use_parallel: 是否使用并行化
        n_workers: 工作进程数，None 表示自动
        force_rebuild: 是否强制重建

    Returns:
        数据集 DataFrame
    """
    # 获取缓存键
    params_dict = {
        'n_forward_days': n_forward_days,
        'target_atr_mult': target_atr_mult,
        'stop_loss_atr_mult': stop_loss_atr_mult,
        'use_parallel': use_parallel,
        'n_workers': n_workers,
        'params': p.to_dict()
    }
    cache_key = get_data_hash(data_dir, params_dict)

    # 初始化缓存
    cache = DatasetCache(os.path.join(data_dir, "cache"))

    # 尝试从缓存加载
    if use_cache and not force_rebuild:
        cached_df = cache.load(cache_key)
        if cached_df is not None:
            feature_cols = [c for c in cached_df.columns if c.startswith('feat_')]
            logger.info(f"从缓存加载数据集: {len(cached_df)} 行, {len(feature_cols)} 个特征")
            return cached_df
        else:
            logger.info("缓存未命中，开始构建数据集...")

    # 正常构建数据集（支持并行化）
    start_time = time.time()
    if use_parallel:
        df = build_dataset_parallel(
            data_dir, p, n_forward_days,
            target_atr_mult, stop_loss_atr_mult, n_workers
        )
    else:
        df = build_dataset(data_dir, p, n_forward_days, target_atr_mult, stop_loss_atr_mult)
    build_time = time.time() - start_time

    # 保存缓存
    if use_cache and not df.empty:
        cache.save(df, cache_key)
        logger.info(f"数据集构建完成，耗时: {build_time:.2f}s，已缓存")

    return df


def _process_single_file(file_path: str, p: StrategyParams, n_forward_days: int,
                      target_atr_mult: float, stop_loss_atr_mult: float):
    """
    处理单个文件的函数，供并行化调用

    Args:
        file_path: CSV 文件路径
        p: 策略参数
        n_forward_days: 向前看的天数
        target_atr_mult: 目标 ATR 倍数
        stop_loss_atr_mult: 止损 ATR 倍数

    Returns:
        处理后的 DataFrame，如果失败则返回 None
    """
    import traceback

    try:
        df = pd.read_csv(file_path)
        if df.empty or len(df) < 50:
            return None

        # 1. Base Indicators
        df = calculate_indicators(df, p)

        # Clean and rename columns after indicator calculation
        rename_map = {
            "date": "Date", "open": "Open", "high": "High", "low": "Low",
            "close": "Close", "volume": "Volume"
        }
        df.rename(columns=rename_map, inplace=True)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df.set_index("Date", inplace=True)
            df.sort_index(inplace=True)

        # 2. Add derived features
        df = extract_features(df)

        # 3. Add truth Target Y (both binary and multi-class)
        df = create_targets(df, n_forward_days=n_forward_days,
                          target_atr_mult=target_atr_mult,
                          stop_loss_atr_mult=stop_loss_atr_mult)

        # 3.1 Add multi-class target
        df = create_multi_class_targets(df, n_forward_days=n_forward_days,
                                     target_atr_mult=target_atr_mult,
                                     stop_loss_atr_mult=stop_loss_atr_mult)

        feature_cols = [c for c in df.columns if c.startswith('feat_')]
        if not feature_cols:
            return None

        # Drop rows where Y is nan (the last n_forward_days)
        df = df.dropna(subset=['label_max_ret_5d'] + feature_cols)

        # Keep stock code for cross validation splits
        file_name = os.path.basename(file_path)
        df['stock_code'] = file_name.replace(".csv", "")

        # Keep a sortable date column for chronological splitting
        if 'Date' not in df.columns and df.index.name == 'Date':
            df['_sort_date'] = df.index
        elif 'Date' in df.columns:
            df['_sort_date'] = pd.to_datetime(df['Date'])
        else:
            df['_sort_date'] = range(len(df))

        return df
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        logger.error(traceback.format_exc())
        return None


def build_dataset_parallel(data_dir: str, p: StrategyParams, n_forward_days: int = 5,
                       target_atr_mult: float = 2.0, stop_loss_atr_mult: float = 1.5,
                       n_workers: int = None) -> pd.DataFrame:
    """
    并行化数据集构建函数

    Args:
        data_dir: 数据目录
        p: 策略参数
        n_forward_days: 向前看的天数
        target_atr_mult: 目标 ATR 倍数
        stop_loss_atr_mult: 止损 ATR 倍数
        n_workers: 工作进程数，None 表示自动

    Returns:
        合并后的数据集 DataFrame
    """
    from multiprocessing import Pool, cpu_count

    logger.info("Building dataset from historical data (parallel)...")
    files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]

    # 过滤掉 stock-list.csv
    files = [f for f in files if f != "stock-list.csv"]

    if not files:
        logger.error(f"No csv files found in {data_dir}")
        return pd.DataFrame()

    # 确定工作进程数
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)  # 保留一个核心给主进程
    else:
        n_workers = min(n_workers, len(files))

    logger.info(f"Using {n_workers} workers to process {len(files)} files")

    # 创建任务列表
    tasks = []
    for f in files:
        file_path = os.path.join(data_dir, f)
        tasks.append((file_path, p, n_forward_days, target_atr_mult, stop_loss_atr_mult))

    # 并行处理
    all_dfs = []
    with Pool(processes=n_workers) as pool:
        # 使用 starmap 并行调用 _process_single_file
        # 注意：Pool.imap 不能直接传递关键字参数，所以我们使用 starmap
        results = list(tqdm(
            pool.starmap(_process_single_file, tasks),
            total=len(tasks),
            desc="Processing files (parallel)"
        ))

        # 过滤掉 None 结果（处理失败的文件）
        all_dfs = [df for df in results if df is not None]

    if not all_dfs:
        logger.error("Failed to build dataset. Valid data empty.")
        return pd.DataFrame()

    # 合并所有 DataFrame
    final_df = pd.concat(all_dfs, axis=0)
    feature_cols = [c for c in final_df.columns if c.startswith('feat_')]
    logger.info(f"Dataset built successfully (parallel). Total rows: {len(final_df)}. Features: {len(feature_cols)}")

    return final_df


def train_model(df: pd.DataFrame, model_path: str = "models/alpha_lgbm.txt"):
    """Train a LightGBM classifier to predict label_max_ret_5d."""
    logger.info("Initializing LightGBM model training pipeline...")
    
    feature_cols = [c for c in df.columns if c.startswith('feat_')]
    target_col = 'label_max_ret_5d'
    
    if target_col not in df.columns or not feature_cols:
        logger.error("Target column or feature columns missing. Cannot train.")
        return None
        
    # ===== Chronological Split with Purging =====
    # Sort by date to prevent future data leaking into the training set
    if '_sort_date' in df.columns:
        df = df.sort_values('_sort_date')
    
    X = df[feature_cols]
    y = df[target_col].astype(int)
    
    # Class imbalance weight
    pos_count = y.sum()
    neg_count = len(y) - pos_count
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    logger.info(f"Target distribution: Positive {pos_count}, Negative {neg_count}. Ratio: 1:{scale_pos_weight:.2f}")

    # Time-series aware split: first 80% train, last 20% test
    # Use balanced temporal split to ensure both classes are well-represented
    split_idx = int(len(X) * 0.8)
    # Purging: remove n_forward_days rows around the split to prevent label leakage
    # Increased to 10 days (2x prediction window) to prevent lookahead bias
    purge_days = 10  # 2x default n_forward_days for stricter temporal isolation
    train_end = max(0, split_idx - purge_days)
    test_start = min(len(X), split_idx + purge_days)

    # Apply balanced temporal split
    X_train_raw, y_train_raw, X_test_raw, y_test_raw = balanced_temporal_split(
        X.iloc[:split_idx], y.iloc[:split_idx],
        train_ratio=0.8,
        min_class_ratio=0.3
    )

    # Apply purging to both train and test sets
    X_train = X_train_raw.iloc[:int(len(X_train_raw) * (train_end / split_idx))]
    y_train = y_train_raw.iloc[:int(len(y_train_raw) * (train_end / split_idx))]
    X_test = X_test_raw.iloc[int(len(X_test_raw) * (purge_days / len(X_test_raw))):]
    y_test = y_test_raw.iloc[int(len(y_test_raw) * (purge_days / len(y_test_raw))):]

    logger.info(f"Chronological split: Train {len(X_train)} rows, Test {len(X_test)} rows (purged {purge_days * 2} rows)")

    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

    # LightGBM 参数 - 使用标准 binary objective 避免保存/加载问题
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': 0.03,
        'num_leaves': 63,
        'max_depth': 8,
        'min_child_samples': 50,
        'feature_fraction': 0.85,
        'bagging_fraction': 0.85,
        'bagging_freq': 3,
        'lambda_l1': 0.05,
        'lambda_l2': 0.5,
        'scale_pos_weight': scale_pos_weight,
        'drop_rate': 0.1,
        'skip_drop': 0.5,
        'max_drop': 50,
        'verbose': -1
    }

    logger.info("Starting LightGBM training with standard binary objective...")
    gbm = lgb.train(
        params,
        lgb_train,
        num_boost_round=1000,
        valid_sets=[lgb_train, lgb_eval],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=50)
        ]
    )
    
    # Evaluate
    y_pred_prob = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    auc = roc_auc_score(y_test, y_pred_prob)
    logger.info(f"Model AUC on Test Set: {auc:.4f}")
    logger.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")

    # Feature Importance
    importance = gbm.feature_importance(importance_type='gain')
    feat_imp = pd.Series(importance, index=feature_cols).sort_values(ascending=False)
    logger.info(f"Top 10 Important Features:\n{feat_imp.head(10)}")

    # Feature Importance Visualization
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot top 20 features
        top_n = min(20, len(feat_imp))
        top_features = feat_imp.head(top_n)

        # Create horizontal bar chart
        y_pos = np.arange(top_n)
        ax.barh(y_pos, top_features.values, align='center', color='skyblue')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features.index, fontsize=10)
        ax.invert_yaxis()  # Labels read top-to-bottom
        ax.set_xlabel('Feature Importance (Gain)', fontsize=12)
        ax.set_title('Top 20 Important Features - LightGBM Model', fontsize=14, fontweight='bold')

        # Add grid
        ax.xaxis.grid(True, linestyle='--', alpha=0.7)

        # Adjust layout
        plt.tight_layout()

        # Save figure
        viz_path = os.path.join(os.path.dirname(model_path), 'feature_importance.png')
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        logger.info(f"Feature importance visualization saved to {viz_path}")

        plt.close()

        # Also save feature importance as CSV
        imp_csv_path = os.path.join(os.path.dirname(model_path), 'feature_importance.csv')
        feat_imp.to_csv(imp_csv_path)
        logger.info(f"Feature importance CSV saved to {imp_csv_path}")

    except ImportError:
        logger.warning("matplotlib not available, skipping feature importance visualization")
    except Exception as e:
        logger.warning(f"Failed to create feature importance visualization: {e}")

    # Save Model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    gbm.save_model(model_path)
    logger.info(f"Model saved to {model_path}")
    
    return gbm

def train_multi_class_model(df: pd.DataFrame, model_path: str = "models/alpha_lgbm_multiclass.txt"):
    """Train a LightGBM multi-class classifier to predict label_multi_class (0-3)."""
    logger.info("Initializing LightGBM multi-class model training pipeline...")
    
    feature_cols = [c for c in df.columns if c.startswith('feat_')]
    target_col = 'label_multi_class'
    
    if target_col not in df.columns or not feature_cols:
        logger.error("Multi-class target column or feature columns missing. Cannot train.")
        return None
        
    # Sort by date to prevent future data leaking
    if '_sort_date' in df.columns:
        df = df.sort_values('_sort_date')
    
    X = df[feature_cols]
    y = df[target_col].astype(int)
    
    # Filter out NaN targets
    valid_mask = ~y.isna()
    X = X[valid_mask]
    y = y[valid_mask]
    
    logger.info(f"Multi-class target distribution:\n{y.value_counts().sort_index()}")

    # Time-series aware split
    split_idx = int(len(X) * 0.8)
    # Increased to 10 days (2x prediction window) to prevent lookahead bias
    purge_days = 10
    train_end = max(0, split_idx - purge_days)
    test_start = min(len(X), split_idx + purge_days)

    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_test, y_test = X.iloc[test_start:], y.iloc[test_start:]
    logger.info(f"Chronological split: Train {len(X_train)} rows, Test {len(X_test)} rows (purged {purge_days * 2} rows)")
    
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

    # Multi-class parameters
    num_classes = len(y.unique())
    params = {
        'objective': 'multiclass',
        'num_class': num_classes,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': 7,
        'min_child_samples': 80,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'lambda_l1': 0.1,
        'lambda_l2': 1.0,
        'verbose': -1
    }

    logger.info("Starting LightGBM multi-class training...")
    gbm = lgb.train(
        params,
        lgb_train,
        num_boost_round=500,
        valid_sets=[lgb_train, lgb_eval],
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=50)]
    )
    
    # Evaluate
    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    y_pred_class = np.argmax(y_pred, axis=1)
    
    from sklearn.metrics import classification_report, confusion_matrix
    logger.info(f"Multi-class Classification Report:\n{classification_report(y_test, y_pred_class)}")
    logger.info(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred_class)}")

    # Feature Importance
    importance = gbm.feature_importance(importance_type='gain')
    feat_imp = pd.Series(importance, index=feature_cols).sort_values(ascending=False)
    logger.info(f"Top 10 Important Features:\n{feat_imp.head(10)}")
    
    # Save Model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    gbm.save_model(model_path)
    logger.info(f"Multi-class model saved to {model_path}")
    
    return gbm


def train_ensemble_model(
    df: pd.DataFrame,
    model_dir: str = "models/ensemble_v1",
    ensemble_method: str = "stacking",
    models: list = None
):
    """
    Train an ensemble model using LightGBM + XGBoost + CatBoost
    
    Args:
        df: Dataset with features and targets
        model_dir: Directory to save ensemble models
        ensemble_method: Ensemble method ('simple_avg', 'weighted_avg', 'stacking')
        models: List of models to use (default: ['lgb', 'xgb', 'cat'])
    
    Returns:
        MultiModelEnsemble: Trained ensemble model
    """
    logger.info("Initializing Ensemble Model training pipeline...")
    
    if models is None:
        models = ['lgb', 'xgb', 'cat']
    
    feature_cols = [c for c in df.columns if c.startswith('feat_')]
    target_col = 'label_max_ret_5d'
    
    if target_col not in df.columns or not feature_cols:
        logger.error("Target column or feature columns missing. Cannot train ensemble.")
        return None
    
    # Sort by date to prevent future data leaking
    if '_sort_date' in df.columns:
        df = df.sort_values('_sort_date')
    
    X = df[feature_cols]
    y = (df[target_col] > 0).astype(int)
    
    # Class imbalance weight
    pos_count = y.sum()
    neg_count = len(y) - pos_count
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    logger.info(f"Target distribution: Positive {pos_count}, Negative {neg_count}. Ratio: 1:{scale_pos_weight:.2f}")
    
    # Time-series aware split
    split_idx = int(len(X) * 0.8)
    purge_days = 10
    train_end = max(0, split_idx - purge_days)
    test_start = min(len(X), split_idx + purge_days)
    
    # Apply balanced temporal split
    X_train_raw, y_train_raw, X_test_raw, y_test_raw = balanced_temporal_split(
        X.iloc[:split_idx], y.iloc[:split_idx],
        train_ratio=0.8,
        min_class_ratio=0.3
    )
    
    # Apply purging
    X_train = X_train_raw.iloc[:int(len(X_train_raw) * (train_end / split_idx))]
    y_train = y_train_raw.iloc[:int(len(y_train_raw) * (train_end / split_idx))]
    X_test = X_test_raw.iloc[int(len(X_test_raw) * (purge_days / len(X_test_raw))):]
    y_test = y_test_raw.iloc[int(len(y_test_raw) * (purge_days / len(y_test_raw))):]
    
    logger.info(f"Chronological split: Train {len(X_train)} rows, Test {len(X_test)} rows (purged {purge_days * 2} rows)")
    
    # Initialize ensemble
    ensemble = MultiModelEnsemble(
        models=models,
        ensemble_method=ensemble_method,
        random_state=42
    )
    
    # Train ensemble
    logger.info(f"Training ensemble with {len(models)} models using {ensemble_method} strategy...")
    ensemble.fit(X_train, y_train, X_test, y_test)
    
    # Evaluate ensemble
    y_pred_prob = ensemble.predict_proba(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    auc = roc_auc_score(y_test, y_pred_prob)
    logger.info(f"Ensemble Model AUC on Test Set: {auc:.4f}")
    logger.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")
    
    # Get individual model performance
    if ensemble.model_weights:
        logger.info("Individual Model Performance:")
        for model_name, weight in ensemble.model_weights.items():
            logger.info(f"  {model_name}: AUC = {weight:.4f}")
    
    # Feature Importance (using best model)
    try:
        if ensemble.best_model_name:
            feat_imp = ensemble.get_feature_importance(ensemble.best_model_name)
            logger.info(f"Top 10 Important Features (from {ensemble.best_model_name}):\n{feat_imp.head(10)}")
            
            # Save feature importance
            os.makedirs(model_dir, exist_ok=True)
            imp_csv_path = os.path.join(model_dir, 'feature_importance.csv')
            feat_imp.to_csv(imp_csv_path)
            logger.info(f"Feature importance CSV saved to {imp_csv_path}")
    except Exception as e:
        logger.warning(f"Failed to get feature importance: {e}")
    
    # Save ensemble
    ensemble.save(model_dir)
    logger.info(f"Ensemble model saved to {model_dir}")
    
    # Save training metadata
    metadata = {
        'training_date': pd.Timestamp.now().isoformat(),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'test_auc': float(auc),
        'ensemble_method': ensemble_method,
        'models_used': models,
        'best_model': ensemble.best_model_name,
        'model_weights': {k: float(v) for k, v in ensemble.model_weights.items()}
    }
    metadata_path = os.path.join(model_dir, 'training_metadata.json')
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Training metadata saved to {metadata_path}")
    
    return ensemble


def calculate_shap_values(
    model,
    X: pd.DataFrame,
    feature_names: list = None,
    output_dir: str = None,
    model_name: str = "model"
) -> dict:
    """
    Calculate SHAP values for model interpretability
    
    Args:
        model: Trained model (LightGBM, XGBoost, or CatBoost)
        X: Feature DataFrame
        feature_names: List of feature names
        output_dir: Directory to save SHAP plots
        model_name: Name of the model for logging
    
    Returns:
        Dictionary containing SHAP values and summary statistics
    """
    if not SHAP_AVAILABLE:
        logger.warning("SHAP not available. Skipping SHAP analysis.")
        return {}
    
    if feature_names is None:
        feature_names = X.columns.tolist() if isinstance(X, pd.DataFrame) else [f"feature_{i}" for i in range(X.shape[1])]
    
    logger.info(f"Calculating SHAP values for {model_name}...")
    
    try:
        # Create SHAP explainer based on model type
        if isinstance(model, lgb.Booster):
            explainer = shap.TreeExplainer(model)
        elif hasattr(model, 'predict_proba'):
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.KernelExplainer(model.predict, X.iloc[:100] if len(X) > 100 else X)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(X)
        
        # Handle binary classification case
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use values for class 1
        
        # Calculate mean absolute SHAP values for feature importance
        mean_shap = np.abs(shap_values).mean(axis=0)
        shap_importance = pd.Series(mean_shap, index=feature_names).sort_values(ascending=False)
        
        logger.info(f"Top 10 SHAP Important Features ({model_name}):")
        logger.info(f"\n{shap_importance.head(10)}")
        
        result = {
            'shap_values': shap_values,
            'feature_importance': shap_importance,
            'explainer': explainer
        }
        
        # Save SHAP summary plots if output directory is specified
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            
            try:
                import matplotlib.pyplot as plt
                import matplotlib
                matplotlib.use('Agg')  # Use non-interactive backend
                
                # Summary plot (bar)
                plt.figure(figsize=(12, 8))
                shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'shap_summary_{model_name}.png'), dpi=300, bbox_inches='tight')
                plt.close()
                logger.info(f"SHAP summary plot saved to {output_dir}/shap_summary_{model_name}.png")
                
                # Feature importance (bar plot)
                plt.figure(figsize=(10, 6))
                shap_importance.head(20).plot(kind='barh')
                plt.title(f'Top 20 SHAP Feature Importance - {model_name}')
                plt.xlabel('Mean |SHAP Value|')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'shap_importance_{model_name}.png'), dpi=300, bbox_inches='tight')
                plt.close()
                
                # Save SHAP values to CSV
                shap_df = pd.DataFrame(shap_values, columns=feature_names)
                shap_df.to_csv(os.path.join(output_dir, f'shap_values_{model_name}.csv'), index=False)
                
                # Save feature importance ranking
                shap_importance.to_csv(os.path.join(output_dir, f'shap_feature_importance_{model_name}.csv'))
                
            except Exception as e:
                logger.warning(f"Failed to save SHAP plots: {e}")
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to calculate SHAP values: {e}")
        return {}


def analyze_ensemble_shap(
    ensemble: MultiModelEnsemble,
    X: pd.DataFrame,
    output_dir: str = "models/shap_analysis"
):
    """
    Analyze SHAP values for all models in an ensemble
    
    Args:
        ensemble: Trained MultiModelEnsemble
        X: Feature DataFrame for analysis
        output_dir: Directory to save SHAP analysis results
    """
    if not SHAP_AVAILABLE:
        logger.warning("SHAP not available. Skipping ensemble SHAP analysis.")
        return
    
    logger.info("Starting SHAP analysis for ensemble models...")
    os.makedirs(output_dir, exist_ok=True)
    
    all_importance = {}
    
    for model_name, model in ensemble.trained_models.items():
        try:
            shap_result = calculate_shap_values(
                model,
                X,
                feature_names=ensemble.feature_names,
                output_dir=output_dir,
                model_name=model_name
            )
            if shap_result:
                all_importance[model_name] = shap_result['feature_importance']
        except Exception as e:
            logger.warning(f"Failed to calculate SHAP for {model_name}: {e}")
    
    # Combine feature importance from all models
    if all_importance:
        combined_importance = pd.DataFrame(all_importance).fillna(0)
        combined_importance['average'] = combined_importance.mean(axis=1)
        combined_importance = combined_importance.sort_values('average', ascending=False)
        
        logger.info("Combined SHAP Feature Importance (across all models):")
        logger.info(f"\n{combined_importance.head(10)}")
        
        # Save combined importance
        combined_importance.to_csv(os.path.join(output_dir, 'shap_combined_importance.csv'))
        
        # Plot combined importance
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')
            
            plt.figure(figsize=(10, 8))
            combined_importance['average'].head(20).plot(kind='barh')
            plt.title('Top 20 Average SHAP Feature Importance (Ensemble)')
            plt.xlabel('Mean |SHAP Value|')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'shap_combined_importance.png'), dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Combined SHAP importance plot saved to {output_dir}/shap_combined_importance.png")
        except Exception as e:
            logger.warning(f"Failed to save combined SHAP plot: {e}")


class ModelPerformanceTracker:
    """
    Track model performance metrics over time
    Supports both binary and multi-class classification
    """
    
    def __init__(self, model_name: str = "model"):
        self.model_name = model_name
        self.metrics_history = []
        self.training_date = datetime.now().isoformat()
    
    def record_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray,
        model_name: str = None,
        dataset: str = "test"
    ):
        """
        Record performance metrics for a model
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            model_name: Optional model name override
            dataset: Dataset name (train/test/val)
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, confusion_matrix, log_loss
        )
        
        name = model_name or self.model_name
        
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'model_name': name,
            'dataset': dataset,
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, zero_division=0)),
            'f1_score': float(f1_score(y_true, y_pred, zero_division=0)),
            'auc': float(roc_auc_score(y_true, y_pred_proba)),
            'log_loss': float(log_loss(y_true, y_pred_proba + 1e-8)),
            'samples': int(len(y_true)),
            'positive_rate': float(y_true.mean()),
        }
        
        # Calculate confusion matrix components
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics['true_positives'] = int(tp)
            metrics['true_negatives'] = int(tn)
            metrics['false_positives'] = int(fp)
            metrics['false_negatives'] = int(fn)
            metrics['specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        
        self.metrics_history.append(metrics)
        
        logger.info(f"Model: {name}, Dataset: {dataset}, AUC: {metrics['auc']:.4f}, "
                   f"F1: {metrics['f1_score']:.4f}, Acc: {metrics['accuracy']:.4f}")
        
        return metrics
    
    def get_summary(self) -> pd.DataFrame:
        """Get summary of all recorded metrics"""
        return pd.DataFrame(self.metrics_history)
    
    def save(self, filepath: str):
        """Save metrics history to CSV"""
        df = self.get_summary()
        df.to_csv(filepath, index=False)
        logger.info(f"Performance metrics saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str):
        """Load metrics history from CSV"""
        instance = cls()
        df = pd.read_csv(filepath)
        instance.metrics_history = df.to_dict('records')
        return instance


class EnsembleLGBM:
    """Ensemble of LightGBM models with different parameters for better robustness."""
    
    def __init__(self, n_models=5, random_state=None):
        self.n_models = n_models
        self.random_state = random_state
        self.models = []
        self.feature_names = None
        
    def fit(self, X_train, y_train, X_eval=None, y_eval=None):
        """Train multiple LightGBM models with different hyperparameters."""
        logger.info(f"Training ensemble of {self.n_models} LightGBM models...")

        self.feature_names = X_train.columns.tolist()

        lgb_train = lgb.Dataset(X_train, y_train)

        if X_eval is not None and y_eval is not None:
            lgb_eval = lgb.Dataset(X_eval, y_eval, reference=lgb_train)
            eval_sets = [lgb_train, lgb_eval]
        else:
            eval_sets = [lgb_train]
        
        # Class distribution for imbalance handling
        pos_count = y_train.sum() if hasattr(y_train, 'sum') else 0
        neg_count = len(y_train) - pos_count
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
        
        for i in range(self.n_models):
            # Random hyperparameters for diversity
            model_params = {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'learning_rate': np.random.uniform(0.03, 0.1),
                'num_leaves': np.random.randint(20, 50),
                'max_depth': np.random.randint(5, 10),
                'min_child_samples': np.random.randint(50, 100),
                'feature_fraction': np.random.uniform(0.7, 0.9),
                'bagging_fraction': np.random.uniform(0.7, 0.9),
                'bagging_freq': 5,
                'lambda_l1': np.random.uniform(0.05, 0.2),
                'lambda_l2': np.random.uniform(0.5, 2.0),
                'scale_pos_weight': scale_pos_weight,
                'verbose': -1,
                'random_state': self.random_state + i if self.random_state else i
            }
            
            logger.info(f"Model {i+1}/{self.n_models}: learning_rate={model_params['learning_rate']:.4f}, "
                       f"num_leaves={model_params['num_leaves']}, max_depth={model_params['max_depth']}")
            
            model = lgb.train(
                model_params,
                eval_sets[0],
                num_boost_round=500,
                valid_sets=eval_sets,
                callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=100)]
            )
            
            self.models.append(model)
        
        logger.info("Ensemble training completed.")
        
    def predict_proba(self, X):
        """Predict probability using ensemble averaging."""
        if not self.models:
            raise ValueError("No models trained. Call fit() first.")
        
        predictions = []
        for model in self.models:
            pred = model.predict(X, num_iteration=model.best_iteration)
            predictions.append(pred)
        
        # Average predictions
        ensemble_pred = np.mean(predictions, axis=0)
        return ensemble_pred
    
    def predict(self, X, threshold=0.5):
        """Predict class labels using ensemble averaging."""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
    
    def save(self, directory):
        """Save all ensemble models to directory."""
        os.makedirs(directory, exist_ok=True)
        
        for i, model in enumerate(self.models):
            model_path = os.path.join(directory, f"model_{i}.txt")
            # 显式指定 num_iteration 参数，确保保存的是最优模型
            model.save_model(model_path, num_iteration=model.best_iteration)
        
        # Save metadata
        import json
        metadata = {
            'n_models': self.n_models,
            'random_state': self.random_state,
            'feature_names': self.feature_names
        }
        metadata_path = os.path.join(directory, 'ensemble_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Ensemble models saved to {directory}")
    
    @classmethod
    def load(cls, directory):
        """Load ensemble models from directory."""
        import json
        
        metadata_path = os.path.join(directory, 'ensemble_metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        ensemble = cls(n_models=metadata['n_models'], random_state=metadata['random_state'])
        ensemble.feature_names = metadata['feature_names']
        
        for i in range(ensemble.n_models):
            model_path = os.path.join(directory, f"model_{i}.txt")
            model = lgb.Booster(model_file=model_path)
            ensemble.models.append(model)
        
        logger.info(f"Ensemble models loaded from {directory}")
        return ensemble


def load_processed_dataset(
    data_dir: str = None,
    params: StrategyParams = None,
    n_forward_days: int = 5,
    target_atr_mult: float = 2.0,
    stop_loss_atr_mult: float = 1.5,
    use_cache: bool = True,
    use_parallel: bool = True,
    force_rebuild: bool = False
):
    """
    加载处理后的数据集
    如果数据集不存在，自动构建

    Args:
        data_dir: 数据目录
        params: 策略参数
        n_forward_days: 向前看的天数
        target_atr_mult: 目标ATR倍数
        stop_loss_atr_mult: 止损ATR倍数
        use_cache: 是否使用缓存
        use_parallel: 是否使用并行化
        force_rebuild: 是否强制重建

    Returns:
        tuple: (X_train, y_train, X_eval, y_eval)
    """
    # 设置默认值
    if data_dir is None:
        data_dir = CONF.history_data.data_dir
    if params is None:
        params = StrategyParams()

    logger.info("开始加载/构建数据集...")

    # 构建数据集
    df = build_dataset_with_cache(
        data_dir=data_dir,
        p=params,
        n_forward_days=n_forward_days,
        target_atr_mult=target_atr_mult,
        stop_loss_atr_mult=stop_loss_atr_mult,
        use_cache=use_cache,
        use_parallel=use_parallel,
        force_rebuild=force_rebuild
    )

    if df.empty:
        raise Exception("数据集构建失败，数据集为空")

    logger.info(f"数据集加载完成: {len(df)} 行")

    # 准备特征和标签
    feature_cols = [c for c in df.columns if c.startswith('feat_')]
    if not feature_cols:
        raise Exception("没有找到特征列")

    # 准备特征 X
    X = df[feature_cols].values

    # 准备标签 y（二分类）
    if 'label_max_ret_5d' in df.columns:
        y = (df['label_max_ret_5d'] > 0).astype(int).values
    else:
        raise Exception("没有找到标签列 'label_max_ret_5d'")

    # 时间序列切分（按日期分割）
    df = df.sort_index()
    split_idx = int(len(df) * 0.8)
    X_train, X_eval = X[:split_idx], X[split_idx:]
    y_train, y_eval = y[:split_idx], y[split_idx:]

    logger.info(f"数据集切分完成: 训练集 {len(X_train)} 样本, 验证集 {len(X_eval)} 样本")

    return X_train, y_train, X_eval, y_eval

