"""Reference example for plot-grouped RF and baseline validation.

This is an independently prepared example of the validation design, NOT the
exact archived implementation used for the manuscript's numerical tables.
Requires an institutionally authorized, matched field–RS input table.

Required CSV columns: Station, F_alive, NDVI, EVI, SAVI, MSAVI,
NDMI, NBR, NBR2, NDWI, MNDWI.

Never train on the full dataset before constructing out-of-fold estimates.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

FEATURES = ['NDVI','EVI','SAVI','MSAVI','NDMI','NBR','NBR2','NDWI','MNDWI']


def evaluate(input_path, output_path):
    df = pd.read_csv(input_path)
    need = set(FEATURES + ['Station', 'F_alive'])
    missing = sorted(need - set(df.columns))
    if missing:
        raise ValueError(f'Missing columns: {missing}')
    d = df.dropna(subset=FEATURES + ['Station','F_alive']).copy()
    if d.Station.nunique() < 5:
        raise ValueError('At least five distinct plots are required for 5-fold GroupKFold.')
    X = d[FEATURES]
    y = d['F_alive'].astype(float).to_numpy()
    groups = d['Station']
    folds = GroupKFold(n_splits=5)
    outputs = []
    for name, make_model, cols in [
        ('RF_all9', lambda: RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1), FEATURES),
        ('linear_NDVI', lambda: LinearRegression(), ['NDVI']),
        ('ridge_NDVI', lambda: Ridge(alpha=1.0), ['NDVI']),
    ]:
        oof = np.full(len(d), np.nan)
        for train_idx, test_idx in folds.split(X, y, groups=groups):
            if set(groups.iloc[train_idx]) & set(groups.iloc[test_idx]):
                raise AssertionError('Plot group leakage')
            model = make_model()
            model.fit(X.iloc[train_idx][cols], y[train_idx])
            oof[test_idx] = model.predict(X.iloc[test_idx][cols])
        if not np.isfinite(oof).all():
            raise AssertionError('Incomplete OOF predictions')
        print(f'{name}: n={len(d)}, plots={groups.nunique()}, '
              f'R²={r2_score(y,oof):.3f}, RMSE={np.sqrt(mean_squared_error(y,oof)):.4f}, '
              f'MAE={mean_absolute_error(y,oof):.4f}')
        outputs.append(pd.DataFrame({'row_index':d.index, 'Station':groups.to_numpy(),
                                     'F_alive':y, 'model':name, 'oof_pred':oof}))
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(outputs, ignore_index=True).to_csv(path, index=False)
    print(f'Saved: {path}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    evaluate(args.input, args.output)
