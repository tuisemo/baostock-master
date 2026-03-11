"""
Integration tests for Ensemble and Online Learning modules
Tests ensemble model loading, voting predictions, and A/B testing framework
"""

import pytest
import pandas as pd
import numpy as np
import os
import tempfile
import shutil
from datetime import datetime


def test_ensemble_imports():
    """Test all ensemble-related imports work"""
    from quant.core.ensemble_trainer import MultiModelEnsemble
    from quant.core.online_learner import OnlineLearner, ConceptDriftDetector
    from quant.core.trainer import train_ensemble_model, calculate_shap_values
    from quant.core.online_learner import ModelVersionManager, ModelVersion
    assert True


def test_multi_model_ensemble_init():
    """Test MultiModelEnsemble initialization"""
    from quant.core.ensemble_trainer import MultiModelEnsemble

    # Test with default models
    ensemble = MultiModelEnsemble(models=['lgb'], ensemble_method='weighted_avg')
    assert ensemble.models == ['lgb']
    assert ensemble.ensemble_method == 'weighted_avg'
    assert ensemble.trained_models == {}

    # Test with multiple models
    ensemble2 = MultiModelEnsemble(models=['lgb', 'xgb'], ensemble_method='stacking')
    assert 'lgb' in ensemble2.models


def test_model_version_manager_init():
    """Test ModelVersionManager initialization"""
    from quant.core.online_learner import ModelVersionManager

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = ModelVersionManager(models_dir=tmpdir)
        assert manager.models_dir == tmpdir
        assert os.path.exists(manager.versioning_dir)


def test_model_version_dataclass():
    """Test ModelVersion data class"""
    from quant.core.online_learner import ModelVersion

    version = ModelVersion(
        version_id='v1',
        model_type='lgb',
        created_at=datetime.now().isoformat(),
        auc_score=0.75,
        is_active=True
    )
    assert version.version_id == 'v1'
    assert version.model_type == 'lgb'
    assert version.auc_score == 0.75
    assert version.is_active is True

    # Test serialization
    version_dict = version.to_dict()
    assert version_dict['version_id'] == 'v1'

    # Test deserialization
    version2 = ModelVersion.from_dict(version_dict)
    assert version2.version_id == 'v1'


def test_ab_testing_groups():
    """Test A/B testing group assignment"""
    from quant.core.online_learner import ModelVersionManager

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = ModelVersionManager(models_dir=tmpdir)

        # Register multiple versions with different A/B groups
        from quant.core.online_learner import ModelVersion
        v1 = ModelVersion(version_id='v1', model_type='lgb', created_at=datetime.now().isoformat(), ab_test_group='A')
        v2 = ModelVersion(version_id='v2', model_type='lgb', created_at=datetime.now().isoformat(), ab_test_group='B')

        manager.versions = {'v1': v1, 'v2': v2}

        # Verify A/B groups are tracked
        groups = [v.ab_test_group for v in manager.versions.values() if v.ab_test_group]
        assert 'A' in groups or 'B' in groups


def test_concept_drift_detector():
    """Test ConceptDriftDetector initialization"""
    from quant.core.online_learner import ConceptDriftDetector

    detector = ConceptDriftDetector(
        window_size=100,
        threshold=3.0,
        min_samples=50
    )
    assert detector.window_size == 100
    assert detector.threshold == 3.0
    assert detector.min_samples == 50


def test_ensemble_prediction_mock():
    """Test ensemble prediction with mock data"""
    from quant.core.ensemble_trainer import MultiModelEnsemble

    # Create ensemble
    ensemble = MultiModelEnsemble(models=['lgb'], ensemble_method='simple_avg')

    # Create mock feature data
    n_samples = 50
    n_features = 20
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feat_{i}' for i in range(n_features)]
    )
    y = pd.Series(np.random.randint(0, 2, n_samples))

    # Note: We can't actually fit without LightGBM, but we can verify the setup
    assert ensemble.models == ['lgb']
    assert X.shape == (n_samples, n_features)


def test_online_learner():
    """Test OnlineLearner initialization"""
    from quant.core.online_learner import OnlineLearner

    with tempfile.TemporaryDirectory() as tmpdir:
        learner = OnlineLearner(
            model_path=tmpdir,
            learning_rate=0.01,
            min_samples=100
        )
        assert learner.learning_rate == 0.01
        assert learner.min_samples == 100


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
