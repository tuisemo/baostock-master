import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from quant.config import CONF
from quant.logger import logger
from quant.analyzer import calculate_indicators
from quant.strategy_params import StrategyParams
from quant.features import extract_features, create_targets
from tqdm import tqdm

def build_dataset(data_dir: str, p: StrategyParams, n_forward_days: int = 5, target_pct: float = 0.05) -> pd.DataFrame:
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
            # 3. Add truth Target Y
            df = create_targets(df, n_forward_days=n_forward_days, target_pct=target_pct)
        except Exception as e:
            logger.debug(f"Error processing {f}: {e}")
            continue
            
        feature_cols = [c for c in df.columns if c.startswith('feat_')]
        if not feature_cols:
            continue

        # Drop rows where Y is nan (the last n_forward_days)
        df = df.dropna(subset=['label_max_ret_5d'] + feature_cols)
        
        # Optionally, keep stock code for cross validation splits
        df['stock_code'] = f.replace(".csv", "")
        all_dfs.append(df)
        
    if not all_dfs:
        logger.error("Failed to build dataset. Valid data empty.")
        return pd.DataFrame()
        
    final_df = pd.concat(all_dfs, axis=0)
    logger.info(f"Dataset built successfully. Total rows: {len(final_df)}. Features: {len(feature_cols)}")
    return final_df

def train_model(df: pd.DataFrame, model_path: str = "models/alpha_lgbm.txt"):
    """Train a LightGBM classifier to predict label_max_ret_5d."""
    logger.info("Initializing LightGBM model training pipeline...")
    
    feature_cols = [c for c in df.columns if c.startswith('feat_')]
    target_col = 'label_max_ret_5d'
    
    if target_col not in df.columns or not feature_cols:
        logger.error("Target column or feature columns missing. Cannot train.")
        return None
        
    X = df[feature_cols]
    y = df[target_col].astype(int)
    
    # Class imbalance weight
    pos_count = y.sum()
    neg_count = len(y) - pos_count
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    logger.info(f"Target distribution: Positive {pos_count}, Negative {neg_count}. Ratio: 1:{scale_pos_weight:.2f}")

    # Time series aware split (simplified as random for cross-sectional proof-of-concept)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'scale_pos_weight': scale_pos_weight, # Manage imbalanced data where >5% return is rare
        'verbose': -1
    }

    logger.info("Starting LightGBM training...")
    gbm = lgb.train(
        params,
        lgb_train,
        num_boost_round=500,
        valid_sets=[lgb_train, lgb_eval],
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=50)]
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
