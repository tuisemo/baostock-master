"""
多模型集成训练模块
实现 LightGBM + XGBoost + CatBoost 三模型集成，提升模型鲁棒性和预测能力
"""
import os
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from datetime import datetime

# LightGBM
try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

# XGBoost
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# CatBoost
try:
    from catboost import CatBoostClassifier, Pool
    CAT_AVAILABLE = True
except ImportError:
    CAT_AVAILABLE = False

# Scikit-learn
from sklearn.metrics import classification_report, roc_auc_score, log_loss
from sklearn.linear_model import LogisticRegression

from quant.infra.logger import logger
from quant.infra.config import CONF


class MultiModelEnsemble:
    """
    多模型集成系统
    支持 LightGBM, XGBoost, CatBoost 三种模型集成
    支持多种集成策略：简单平均、加权平均、Stacking
    """
    
    def __init__(
        self,
        models: List[str] = None,
        ensemble_method: str = "weighted_avg",
        random_state: int = 42
    ):
        """
        初始化多模型集成系统
        
        Args:
            models: 要使用的模型列表，如 ['lgb', 'xgb', 'cat']
            ensemble_method: 集成方法，可选 'simple_avg', 'weighted_avg', 'stacking'
            random_state: 随机种子
        """
        if models is None:
            models = ['lgb']  # 默认只使用 LightGBM（向后兼容）
        
        self.models = models
        self.ensemble_method = ensemble_method
        self.random_state = random_state
        self.trained_models = {}  # 存储训练好的模型
        self.model_weights = {}   # 存储模型权重
        self.stacking_model = None  # Stacking 元模型
        self.feature_names = None
        self.best_model_name = None  # 最佳模型名称
        
        # 验证模型可用性
        self._validate_models()
    
    def _validate_models(self):
        """验证所有指定的模型是否可用"""
        unavailable = []
        for model in self.models:
            if model == 'lgb' and not LGB_AVAILABLE:
                unavailable.append('lgb')
            elif model == 'xgb' and not XGB_AVAILABLE:
                unavailable.append('xgb')
            elif model == 'cat' and not CAT_AVAILABLE:
                unavailable.append('cat')
        
        if unavailable:
            logger.warning(f"以下模型不可用，将自动跳过: {unavailable}")
            self.models = [m for m in self.models if m not in unavailable]
    
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_eval: Optional[pd.DataFrame] = None,
        y_eval: Optional[pd.Series] = None
    ):
        """
        训练所有模型
        
        Args:
            X_train: 训练集特征
            y_train: 训练集标签
            X_eval: 验证集特征（可选）
            y_eval: 验证集标签（可选）
        """
        self.feature_names = X_train.columns.tolist()
        logger.info(f"训练多模型集成，使用模型: {self.models}, 集成方法: {self.ensemble_method}")
        
        # 训练每个基模型
        for model_name in self.models:
            logger.info(f"正在训练 {model_name} 模型...")
            try:
                if model_name == 'lgb':
                    model = self._train_lightgbm(X_train, y_train, X_eval, y_eval)
                elif model_name == 'xgb':
                    model = self._train_xgboost(X_train, y_train, X_eval, y_eval)
                elif model_name == 'cat':
                    model = self._train_catboost(X_train, y_train, X_eval, y_eval)
                else:
                    logger.warning(f"未知模型: {model_name}")
                    continue
                
                self.trained_models[model_name] = model
                
                # 评估模型性能
                if X_eval is not None and y_eval is not None:
                    score = self._evaluate_model(model, X_eval, y_eval)
                    logger.info(f"{model_name} 模型 AUC: {score:.4f}")
                    self.model_weights[model_name] = score
                else:
                    # 如果没有验证集，使用默认权重
                    self.model_weights[model_name] = 1.0
                    
            except Exception as e:
                logger.error(f"训练 {model_name} 模型失败: {e}")
        
        # 计算加权平均的权重
        if self.ensemble_method == 'weighted_avg':
            total_weight = sum(self.model_weights.values())
            for model_name in self.model_weights:
                self.model_weights[model_name] /= total_weight
            logger.info(f"模型权重: {self.model_weights}")
        
        # 训练 Stacking 元模型
        if self.ensemble_method == 'stacking' and len(self.models) > 1:
            self._train_stacking(X_train, y_train, X_eval, y_eval)
        
        # 选择最佳模型
        if self.model_weights:
            self.best_model_name = max(self.model_weights.items(), key=lambda x: x[1])[0]
            logger.info(f"最佳模型: {self.best_model_name} (AUC: {self.model_weights[self.best_model_name]:.4f})")
    
    def _train_lightgbm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_eval: Optional[pd.DataFrame],
        y_eval: Optional[pd.Series]
    ) -> lgb.Booster:
        """训练 LightGBM 模型"""
        lgb_train = lgb.Dataset(X_train, y_train)
        eval_sets = [lgb_train]
        
        if X_eval is not None and y_eval is not None:
            lgb_eval = lgb.Dataset(X_eval, y_eval, reference=lgb_train)
            eval_sets.append(lgb_eval)
        
        # 计算类别权重
        pos_count = y_train.sum()
        neg_count = len(y_train) - pos_count
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
        
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
            'verbose': -1,
            'random_state': self.random_state
        }
        
        model = lgb.train(
            params,
            lgb_train,
            num_boost_round=1000,
            valid_sets=eval_sets,
            callbacks=[
                lgb.early_stopping(stopping_rounds=100),
                lgb.log_evaluation(period=100)
            ]
        )
        
        return model
    
    def _train_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_eval: Optional[pd.DataFrame],
        y_eval: Optional[pd.Series]
    ) -> xgb.Booster:
        """训练 XGBoost 模型"""
        dtrain = xgb.DMatrix(X_train, label=y_train)
        
        evals = [(dtrain, 'train')]
        if X_eval is not None and y_eval is not None:
            deval = xgb.DMatrix(X_eval, label=y_eval)
            evals.append((deval, 'eval'))
        
        # 计算类别权重
        pos_count = y_train.sum()
        neg_count = len(y_train) - pos_count
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
        
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'learning_rate': 0.03,
            'max_depth': 8,
            'min_child_weight': 50,
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'lambda': 0.5,
            'alpha': 0.05,
            'scale_pos_weight': scale_pos_weight,
            'random_state': self.random_state,
            'verbosity': 0
        }
        
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            evals=evals,
            early_stopping_rounds=100,
            verbose_eval=False
        )
        
        return model
    
    def _train_catboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_eval: Optional[pd.DataFrame],
        y_eval: Optional[pd.Series]
    ) -> CatBoostClassifier:
        """训练 CatBoost 模型"""
        train_pool = Pool(X_train, y_train)
        
        eval_pool = None
        if X_eval is not None and y_eval is not None:
            eval_pool = Pool(X_eval, y_eval)
        
        model = CatBoostClassifier(
            iterations=1000,
            learning_rate=0.03,
            depth=8,
            l2_leaf_reg=3.0,
            random_strength=1.0,
            bagging_temperature=1.0,
            border_count=128,
            random_seed=self.random_state,
            verbose=False,
            eval_metric='AUC',
            early_stopping_rounds=100,
            auto_class_weights='Balanced'
        )
        
        model.fit(train_pool, eval_set=eval_pool)
        
        return model
    
    def _train_stacking(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_eval: Optional[pd.DataFrame],
        y_eval: Optional[pd.Series]
    ):
        """训练 Stacking 元模型"""
        logger.info("训练 Stacking 元模型...")
        
        # 生成基模型的预测结果
        meta_features = self._generate_meta_features(X_train)
        
        # 如果有验证集，也生成验证集的元特征
        if X_eval is not None:
            meta_features_eval = self._generate_meta_features(X_eval)
        else:
            meta_features_eval = None
        
        # 训练逻辑回归作为元模型
        self.stacking_model = LogisticRegression(
            random_state=self.random_state,
            max_iter=1000,
            C=1.0
        )
        
        if meta_features_eval is not None:
            self.stacking_model.fit(meta_features, y_train)
            # 验证元模型
            meta_pred = self.stacking_model.predict_proba(meta_features_eval)[:, 1]
            meta_auc = roc_auc_score(y_eval, meta_pred)
            logger.info(f"Stacking 元模型 AUC: {meta_auc:.4f}")
        else:
            # 如果没有验证集，使用简单的 K-Fold 验证
            from sklearn.model_selection import cross_val_score
            cv_scores = cross_val_score(self.stacking_model, meta_features, y_train, cv=5, scoring='roc_auc')
            logger.info(f"Stacking 元模型 CV AUC: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
            self.stacking_model.fit(meta_features, y_train)
    
    def _generate_meta_features(self, X: pd.DataFrame) -> np.ndarray:
        """生成元特征（基模型的预测结果）"""
        meta_features = []
        
        for model_name in self.models:
            model = self.trained_models[model_name]
            pred = self._predict_single_model(model, model_name, X)
            meta_features.append(pred)
        
        return np.column_stack(meta_features)
    
    def _evaluate_model(self, model, X: pd.DataFrame, y: pd.Series) -> float:
        """评估单个模型的性能"""
        if isinstance(model, lgb.Booster):
            pred = model.predict(X)
        elif isinstance(model, xgb.Booster):
            dtest = xgb.DMatrix(X)
            pred = model.predict(dtest)
        elif isinstance(model, CatBoostClassifier):
            pred = model.predict_proba(X)[:, 1]
        else:
            return 0.0
        
        return roc_auc_score(y, pred)
    
    def _predict_single_model(self, model, model_name: str, X: pd.DataFrame) -> np.ndarray:
        """使用单个模型预测"""
        if model_name == 'lgb' and isinstance(model, lgb.Booster):
            return model.predict(X)
        elif model_name == 'xgb' and isinstance(model, xgb.Booster):
            dtest = xgb.DMatrix(X)
            return model.predict(dtest)
        elif model_name == 'cat' and isinstance(model, CatBoostClassifier):
            return model.predict_proba(X)[:, 1]
        else:
            return np.zeros(len(X))
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        预测概率（使用集成策略）
        
        Args:
            X: 特征数据
            
        Returns:
            预测概率数组
        """
        if not self.trained_models:
            raise ValueError("模型尚未训练，请先调用 fit() 方法")
        
        if self.ensemble_method == 'simple_avg':
            # 简单平均
            predictions = []
            for model_name in self.models:
                model = self.trained_models[model_name]
                pred = self._predict_single_model(model, model_name, X)
                predictions.append(pred)
            return np.mean(predictions, axis=0)
        
        elif self.ensemble_method == 'weighted_avg':
            # 加权平均
            predictions = []
            for model_name in self.models:
                model = self.trained_models[model_name]
                pred = self._predict_single_model(model, model_name, X)
                weight = self.model_weights.get(model_name, 1.0)
                predictions.append(pred * weight)
            return np.sum(predictions, axis=0)
        
        elif self.ensemble_method == 'stacking':
            # Stacking
            meta_features = self._generate_meta_features(X)
            return self.stacking_model.predict_proba(meta_features)[:, 1]
        
        else:
            raise ValueError(f"未知的集成方法: {self.ensemble_method}")
    
    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """
        预测类别
        
        Args:
            X: 特征数据
            threshold: 分类阈值
            
        Returns:
            预测类别数组
        """
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
    
    def save(self, directory: str):
        """保存所有模型和元数据"""
        os.makedirs(directory, exist_ok=True)
        
        # 保存每个模型
        for model_name, model in self.trained_models.items():
            if model_name == 'lgb':
                model_path = os.path.join(directory, f"model_{model_name}.txt")
                model.save_model(model_path)
            elif model_name == 'xgb':
                model_path = os.path.join(directory, f"model_{model_name}.json")
                model.save_model(model_path)
            elif model_name == 'cat':
                model_path = os.path.join(directory, f"model_{model_name}.cbm")
                model.save_model(model_path)
        
        # 保存 Stacking 元模型
        if self.stacking_model is not None:
            import joblib
            meta_path = os.path.join(directory, "stacking_model.pkl")
            joblib.dump(self.stacking_model, meta_path)
        
        # 保存元数据
        metadata = {
            'models': self.models,
            'ensemble_method': self.ensemble_method,
            'random_state': self.random_state,
            'model_weights': {k: float(v) for k, v in self.model_weights.items()},
            'feature_names': self.feature_names,
            'best_model_name': self.best_model_name
        }
        metadata_path = os.path.join(directory, 'ensemble_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"多模型集成已保存至: {directory}")
    
    @classmethod
    def load(cls, directory: str) -> 'MultiModelEnsemble':
        """从目录加载多模型集成"""
        metadata_path = os.path.join(directory, 'ensemble_metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        ensemble = cls(
            models=metadata['models'],
            ensemble_method=metadata['ensemble_method'],
            random_state=metadata['random_state']
        )
        
        ensemble.feature_names = metadata['feature_names']
        ensemble.model_weights = {k: float(v) for k, v in metadata['model_weights'].items()}
        ensemble.best_model_name = metadata.get('best_model_name')
        
        # 加载每个模型
        for model_name in ensemble.models:
            if model_name == 'lgb':
                model_path = os.path.join(directory, f"model_{model_name}.txt")
                ensemble.trained_models[model_name] = lgb.Booster(model_file=model_path)
            elif model_name == 'xgb':
                model_path = os.path.join(directory, f"model_{model_name}.json")
                ensemble.trained_models[model_name] = xgb.Booster()
                ensemble.trained_models[model_name].load_model(model_path)
            elif model_name == 'cat':
                model_path = os.path.join(directory, f"model_{model_name}.cbm")
                ensemble.trained_models[model_name] = CatBoostClassifier()
                ensemble.trained_models[model_name].load_model(model_path)
        
        # 加载 Stacking 元模型
        stacking_path = os.path.join(directory, "stacking_model.pkl")
        if os.path.exists(stacking_path):
            import joblib
            ensemble.stacking_model = joblib.load(stacking_path)
        
        logger.info(f"多模型集成已从 {directory} 加载")
        return ensemble
    
    def get_feature_importance(self, model_name: str = None) -> pd.Series:
        """
        获取特征重要性
        
        Args:
            model_name: 模型名称，如果为 None 则返回最佳模型的特征重要性
            
        Returns:
            特征重要性 Series
        """
        if model_name is None:
            model_name = self.best_model_name
        
        if model_name not in self.trained_models:
            raise ValueError(f"模型 {model_name} 未训练")
        
        model = self.trained_models[model_name]
        
        if model_name == 'lgb':
            importance = model.feature_importance(importance_type='gain')
        elif model_name == 'xgb':
            importance = model.get_score(importance_type='gain')
            # 转换为数组，顺序为特征名
            importance = [importance.get(f, 0) for f in self.feature_names]
        elif model_name == 'cat':
            importance = model.get_feature_importance()
        
        return pd.Series(importance, index=self.feature_names).sort_values(ascending=False)
