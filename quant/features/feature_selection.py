"""
特征选择与降维模块
提供递归特征消除(RFE)、相关性分析和PCA降维功能
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Dict, Union
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import warnings

from quant.infra.logger import logger


class FeatureSelector:
    """
    特征选择器 - 支持多种特征选择方法
    """

    def __init__(self, method: str = 'rfe', n_features: Optional[int] = None,
                 correlation_threshold: float = 0.95, importance_threshold: float = 0.01):
        """
        初始化特征选择器

        Args:
            method: 选择方法 ('rfe', 'correlation', 'importance', 'pca')
            n_features: 目标特征数量 (用于RFE和PCA)
            correlation_threshold: 相关性阈值 (用于correlation方法)
            importance_threshold: 重要性阈值 (用于importance方法)
        """
        self.method = method
        self.n_features = n_features
        self.correlation_threshold = correlation_threshold
        self.importance_threshold = importance_threshold
        self.selected_features: List[str] = []
        self.scaler: Optional[StandardScaler] = None
        self.pca: Optional[PCA] = None
        self.selector: Optional[RFE] = None
        self.feature_importances_: Optional[pd.Series] = None
        self.explained_variance_ratio_: Optional[np.ndarray] = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'FeatureSelector':
        """
        拟合特征选择器

        Args:
            X: 特征DataFrame
            y: 目标变量 (用于监督学习方法)

        Returns:
            self
        """
        feature_cols = [c for c in X.columns if c.startswith('feat_')]
        X_feat = X[feature_cols].fillna(0)

        if self.method == 'rfe':
            self._fit_rfe(X_feat, y)
        elif self.method == 'correlation':
            self._fit_correlation(X_feat)
        elif self.method == 'importance':
            self._fit_importance(X_feat, y)
        elif self.method == 'pca':
            self._fit_pca(X_feat)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        转换数据，应用特征选择

        Args:
            X: 输入特征DataFrame

        Returns:
            选择后的特征DataFrame
        """
        if not self.selected_features and self.method != 'pca':
            raise ValueError("FeatureSelector must be fitted before transform")

        if self.method == 'pca':
            return self._transform_pca(X)
        else:
            # 保留非特征列
            non_feat_cols = [c for c in X.columns if not c.startswith('feat_')]
            result = X[non_feat_cols + self.selected_features].copy()
            return result

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        拟合并转换数据

        Args:
            X: 特征DataFrame
            y: 目标变量

        Returns:
            选择后的特征DataFrame
        """
        self.fit(X, y)
        return self.transform(X)

    def _fit_rfe(self, X: pd.DataFrame, y: Optional[pd.Series]):
        """递归特征消除"""
        if y is None:
            raise ValueError("RFE requires target variable y")

        n_samples = len(X)
        n_features_total = len(X.columns)

        # 自动确定目标特征数量
        if self.n_features is None:
            self.n_features = max(10, n_features_total // 3)

        # 确保n_features不超过实际特征数
        self.n_features = min(self.n_features, n_features_total)

        # 选择模型 - 根据目标类型选择分类或回归
        if y.nunique() <= 10:
            estimator = RandomForestClassifier(
                n_estimators=50,
                max_depth=10,
                min_samples_split=min(20, n_samples // 10),
                min_samples_leaf=min(10, n_samples // 20),
                random_state=42,
                n_jobs=-1
            )
        else:
            estimator = RandomForestRegressor(
                n_estimators=50,
                max_depth=10,
                min_samples_split=min(20, n_samples // 10),
                min_samples_leaf=min(10, n_samples // 20),
                random_state=42,
                n_jobs=-1
            )

        self.selector = RFE(
            estimator=estimator,
            n_features_to_select=self.n_features,
            step=0.1,  # 每次迭代移除10%的特征
            verbose=0
        )

        # 处理缺失值
        X_filled = X.fillna(0)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.selector.fit(X_filled, y)

        self.selected_features = X.columns[self.selector.support_].tolist()
        logger.info(f"RFE selected {len(self.selected_features)} features")

    def _fit_correlation(self, X: pd.DataFrame):
        """基于相关性的特征选择 - 移除高度相关的特征"""
        # 计算相关性矩阵
        corr_matrix = X.corr().abs()

        # 获取上三角矩阵（排除对角线）
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        # 找出高度相关的特征对
        high_corr_pairs = []
        for col in upper.columns:
            high_corr = upper[col][upper[col] > self.correlation_threshold]
            for idx, val in high_corr.items():
                high_corr_pairs.append((col, idx, val))

        # 按相关性排序
        high_corr_pairs.sort(key=lambda x: x[2], reverse=True)

        # 贪心算法：移除与最多其他特征相关的特征
        features_to_drop = set()
        all_features = set(X.columns)

        for feat1, feat2, corr in high_corr_pairs:
            if feat1 not in features_to_drop and feat2 not in features_to_drop:
                # 计算每个特征与其他特征的平均相关性
                feat1_corr = corr_matrix[feat1].mean()
                feat2_corr = corr_matrix[feat2].mean()

                # 移除平均相关性更高的特征
                if feat1_corr > feat2_corr:
                    features_to_drop.add(feat1)
                else:
                    features_to_drop.add(feat2)

        self.selected_features = list(all_features - features_to_drop)
        logger.info(f"Correlation filtering: kept {len(self.selected_features)} features, "
                   f"dropped {len(features_to_drop)} highly correlated features")

    def _fit_importance(self, X: pd.DataFrame, y: Optional[pd.Series]):
        """基于特征重要性的选择"""
        if y is None:
            raise ValueError("Importance-based selection requires target variable y")

        n_samples = len(X)

        # 选择模型
        if y.nunique() <= 10:
            estimator = RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_split=min(20, n_samples // 10),
                min_samples_leaf=min(10, n_samples // 20),
                random_state=42,
                n_jobs=-1
            )
        else:
            estimator = RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=min(20, n_samples // 10),
                min_samples_leaf=min(10, n_samples // 20),
                random_state=42,
                n_jobs=-1
            )

        X_filled = X.fillna(0)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            estimator.fit(X_filled, y)

        # 获取特征重要性
        self.feature_importances_ = pd.Series(
            estimator.feature_importances_,
            index=X.columns
        ).sort_values(ascending=False)

        # 选择重要特征
        if self.n_features is not None:
            self.selected_features = self.feature_importances_.head(self.n_features).index.tolist()
        else:
            # 基于重要性阈值
            total_importance = self.feature_importances_.sum()
            threshold = total_importance * self.importance_threshold
            self.selected_features = self.feature_importances_[
                self.feature_importances_ >= threshold
            ].index.tolist()

        logger.info(f"Importance-based selection: kept {len(self.selected_features)} features")

    def _fit_pca(self, X: pd.DataFrame):
        """PCA降维"""
        # 标准化
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X.fillna(0))

        # 自动确定组件数量
        if self.n_features is None:
            # 保留95%的方差
            self.n_features = min(X.shape[1], 20)

        self.pca = PCA(n_components=self.n_features, random_state=42)
        self.pca.fit(X_scaled)

        self.explained_variance_ratio_ = self.pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(self.explained_variance_ratio_)

        logger.info(f"PCA: {self.n_features} components explain {cumulative_variance[-1]:.2%} of variance")

    def _transform_pca(self, X: pd.DataFrame) -> pd.DataFrame:
        """应用PCA转换"""
        if self.pca is None or self.scaler is None:
            raise ValueError("PCA must be fitted before transform")

        feature_cols = [c for c in X.columns if c.startswith('feat_')]
        non_feat_cols = [c for c in X.columns if not c.startswith('feat_')]

        X_feat = X[feature_cols].fillna(0)
        X_scaled = self.scaler.transform(X_feat)
        X_pca = self.pca.transform(X_scaled)

        # 创建新的DataFrame
        pca_cols = [f'feat_pca_{i}' for i in range(X_pca.shape[1])]
        result = pd.DataFrame(X_pca, index=X.index, columns=pca_cols)

        # 添加非特征列
        for col in non_feat_cols:
            result[col] = X[col]

        return result

    def get_feature_importance_report(self) -> Optional[pd.DataFrame]:
        """
        获取特征重要性报告

        Returns:
            特征重要性DataFrame，如果不可用则返回None
        """
        if self.feature_importances_ is None:
            return None

        report = pd.DataFrame({
            'feature': self.feature_importances_.index,
            'importance': self.feature_importances_.values,
            'cumulative_importance': self.feature_importances_.cumsum().values,
            'rank': range(1, len(self.feature_importances_) + 1)
        })

        return report

    def get_correlation_matrix(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        获取特征相关性矩阵

        Args:
            X: 特征DataFrame

        Returns:
            相关性矩阵
        """
        feature_cols = [c for c in X.columns if c.startswith('feat_')]
        return X[feature_cols].corr()


def select_features_cv(X: pd.DataFrame, y: pd.Series,
                       methods: List[str] = ['rfe', 'correlation', 'importance'],
                       cv_folds: int = 3) -> Dict[str, any]:
    """
    交叉验证特征选择效果

    Args:
        X: 特征DataFrame
        y: 目标变量
        methods: 要测试的方法列表
        cv_folds: 交叉验证折数

    Returns:
        各方法的性能比较结果
    """
    results = {}

    for method in methods:
        logger.info(f"Evaluating feature selection method: {method}")

        try:
            selector = FeatureSelector(method=method)

            # 使用交叉验证评估
            from sklearn.model_selection import StratifiedKFold

            if y.nunique() <= 10:
                cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
                model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            else:
                from sklearn.model_selection import KFold
                cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
                model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)

            feature_cols = [c for c in X.columns if c.startswith('feat_')]

            scores = []
            for train_idx, val_idx in cv.split(X, y):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                # 只在训练集上拟合选择器
                selector.fit(X_train[feature_cols], y_train)
                X_train_sel = selector.transform(X_train[feature_cols])
                X_val_sel = selector.transform(X_val[feature_cols])

                # 训练模型并评估
                model.fit(X_train_sel.fillna(0), y_train)
                score = model.score(X_val_sel.fillna(0), y_val)
                scores.append(score)

            results[method] = {
                'cv_scores': scores,
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'n_features_selected': len(selector.selected_features) if method != 'pca' else selector.n_features
            }

        except Exception as e:
            logger.error(f"Error evaluating {method}: {e}")
            results[method] = {'error': str(e)}

    return results


def analyze_feature_correlation(X: pd.DataFrame,
                                threshold: float = 0.9) -> pd.DataFrame:
    """
    分析特征相关性并返回高度相关的特征对

    Args:
        X: 特征DataFrame
        threshold: 相关性阈值

    Returns:
        高度相关的特征对DataFrame
    """
    feature_cols = [c for c in X.columns if c.startswith('feat_')]
    corr_matrix = X[feature_cols].corr().abs()

    # 获取上三角
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    # 收集高相关对
    high_corr_list = []
    for col in upper.columns:
        high_corr = upper[col][upper[col] > threshold]
        for idx, val in high_corr.items():
            high_corr_list.append({
                'feature_1': col,
                'feature_2': idx,
                'correlation': val
            })

    return pd.DataFrame(high_corr_list).sort_values('correlation', ascending=False)


def get_optimal_feature_count(X: pd.DataFrame, y: pd.Series,
                              max_features: Optional[int] = None) -> int:
    """
    使用肘部法则确定最优特征数量

    Args:
        X: 特征DataFrame
        y: 目标变量
        max_features: 最大特征数量

    Returns:
        建议的特征数量
    """
    feature_cols = [c for c in X.columns if c.startswith('feat_')]
    X_feat = X[feature_cols].fillna(0)

    if max_features is None:
        max_features = min(len(feature_cols), 50)

    # 快速评估不同特征数量的性能
    scores = []
    feature_counts = list(range(5, min(max_features + 1, len(feature_cols) + 1), 5))

    if len(feature_counts) < 3:
        feature_counts = list(range(min(5, len(feature_cols)),
                                   min(max_features + 1, len(feature_cols) + 1),
                                   max(1, len(feature_cols) // 5)))

    for n_feat in feature_counts:
        selector = FeatureSelector(method='rfe', n_features=n_feat)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                selector.fit(X_feat, y)

                # 简单评估
                if y.nunique() <= 10:
                    model = RandomForestClassifier(n_estimators=30, random_state=42)
                else:
                    model = RandomForestRegressor(n_estimators=30, random_state=42)

                X_sel = selector.transform(X_feat)
                score = cross_val_score(model, X_sel.fillna(0), y, cv=3, scoring='accuracy' if y.nunique() <= 10 else 'r2')
                scores.append(score.mean())
        except Exception as e:
            logger.warning(f"Error with {n_feat} features: {e}")
            scores.append(0)

    if not scores:
        return min(15, len(feature_cols))

    # 使用肘部法则
    # 找到性能提升明显放缓的点
    if len(scores) >= 3:
        improvements = [scores[i+1] - scores[i] for i in range(len(scores)-1)]
        avg_improvement = np.mean(improvements)

        for i, imp in enumerate(improvements):
            if imp < avg_improvement * 0.3:  # 改进小于平均的30%
                return feature_counts[i + 1]

    # 默认返回中等数量
    return feature_counts[len(feature_counts) // 2]
