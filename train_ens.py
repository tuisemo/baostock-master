from quant.trainer import train_ensemble_model, build_dataset_with_cache
from quant.strategy_params import StrategyParams
print('Building dataset...')
df = build_dataset_with_cache(data_dir='data', p=StrategyParams(), n_forward_days=5, use_cache=True, force_rebuild=False)
print('Dataset:', len(df), 'rows')
if len(df) > 0:
    print('Training ensemble (LightGBM+XGBoost+CatBoost)...')
    ens = train_ensemble_model(df, 'models/ensemble_v1', 'stacking', ['lgb', 'xgb', 'cat'])
    if ens:
        print('Best model:', ens.best_model_name)
        print('Model weights:', {k: round(v,4) for k,v in ens.model_weights.items()})
