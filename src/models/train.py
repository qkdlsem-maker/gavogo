"""models/train.py — joint 학습/평가 유틸. 위치: src/models/train.py
vehicle 기반 split(leakage 차단), multi-seed."""
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import xgboost as xgb


def split_by_vehicle(df, test_frac=0.2, seed=42):
    """데이터셋·차량 단위 split (같은 차량이 train/test 양쪽에 안 들어가게)."""
    key = df["dataset"].astype(str) + "_" + df["vehicle_id"].astype(str) if "dataset" in df else df["vehicle_id"].astype(str)
    vids = key.unique()
    rng = np.random.RandomState(seed)
    test_v = set(rng.choice(vids, int(len(vids) * test_frac), replace=False))
    m = key.isin(test_v)
    return df[~m].reset_index(drop=True), df[m].reset_index(drop=True)


def _xy(df, cols):
    return df[cols].replace([np.inf, -np.inf], np.nan), df["label"].astype(int)


def fit_xgb(Xtr, ytr, params=None, seed=42):
    base = dict(tree_method="hist", eval_metric="auc", n_estimators=500,
                max_depth=6, learning_rate=0.05, random_state=seed, n_jobs=-1)
    if params:
        base.update(params)
    m = xgb.XGBClassifier(**base)
    m.fit(Xtr, ytr, verbose=False)
    return m


def eval_auc(m, df, cols):
    X, y = _xy(df, cols)
    if y.nunique() < 2:
        return float("nan")
    return roc_auc_score(y, m.predict_proba(X)[:, 1])


def multiseed_auc(train_df, test_df, cols, seeds=(42, 0, 1, 7, 123)):
    Xtr, ytr = _xy(train_df, cols)
    aucs = []
    for s in seeds:
        m = fit_xgb(Xtr, ytr, seed=s)
        aucs.append(eval_auc(m, test_df, cols))
    return float(np.nanmean(aucs)), float(np.nanstd(aucs))
