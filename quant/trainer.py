import os
import pandas as pd
import numpy as np
import lightgbm as lgb
# Time-series aware splitting (no random shuffle for temporal data)
from sklearn.metrics import classification_report, roc_auc_score
from quant.config import CONF
from quant.logger import logger
from quant.analyzer import calculate_indicators
from quant.strategy_params import StrategyParams
from quant.features import extract_features, create_targets, create_multi_class_targets
from quant.cache_utils import DatasetCache, get_data_hash
from quant.numba_accelerator import get_numba_status
from tqdm import tqdm
import time

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
            from quant.features import create_multi_class_targets
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
    split_idx = int(len(X) * 0.8)
    # Purging: remove n_forward_days rows around the split to prevent label leakage
    purge_days = 5  # matches default n_forward_days in create_targets
    train_end = max(0, split_idx - purge_days)
    test_start = min(len(X), split_idx + purge_days)
    
    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_test, y_test = X.iloc[test_start:], y.iloc[test_start:]
    logger.info(f"Chronological split: Train {len(X_train)} rows, Test {len(X_test)} rows (purged {purge_days * 2} rows)")
    
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

    # 优化后的 LightGBM 参数
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',

        # 学习率优化
        'learning_rate': 0.03,  # 降低学习率，增加迭代次数
        'num_boost_round': 1000,  # 增加最大迭代次数（传递给 lgb.train）

        # 树结构优化
        'num_leaves': 63,  # 增加叶子节点数，提升模型复杂度
        'max_depth': 8,    # 增加最大深度
        'min_child_samples': 50,  # 降低最小样本数要求

        # 采样优化
        'feature_fraction': 0.85,  # 增加特征采样比例
        'bagging_fraction': 0.85,  # 增加 bagging 采样比例
        'bagging_freq': 3,         # 更频繁的 bagging

        # 正则化优化
        'lambda_l1': 0.05,  # 降低 L1 正则化
        'lambda_l2': 0.5,   # 降低 L2 正则化

        # 类别不平衡处理
        'scale_pos_weight': scale_pos_weight,

        # 其他优化
        'drop_rate': 0.1,         # Dropout 防止过拟合
        'skip_drop': 0.5,         # Dropout 跳过概率
        'max_drop': 50,           # 最大 dropout 次数

        'verbose': -1
    }

    logger.info("Starting LightGBM training...")
    gbm = lgb.train(
        params,
        lgb_train,
        num_boost_round=1000,  # 增加最大迭代次数
        valid_sets=[lgb_train, lgb_eval],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),  # 增加早停轮次
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
    purge_days = 5
    train_end = max(0, split_idx - purge_days)
    test_start = min(len(X), split_idx + purge_days)
    
    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_test, y_test = X.iloc[test_start:], y.iloc[test_start:]
    logger.info(f"Chronological split: Train {len(X_train)} rows, Test {len(X_test)} rows")
    
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
            model.save_model(model_path)
        
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
