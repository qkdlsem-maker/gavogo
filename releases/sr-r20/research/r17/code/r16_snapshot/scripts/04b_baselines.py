#!/usr/bin/env python
"""[4-1단계] baseline 비교: XGBoost vs LightGBM vs LSTM.
위치: scripts/04b_baselines.py
같은 joint 데이터·split·full피처로 공정 비교. in-domain 평균 + ETRI OOD.
실행: python scripts/04b_baselines.py
필요: pip install lightgbm torch"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config
from src.features.kinematic import FEATURE_COLS as KIN
from src.features.game_theory import NASH_FEATURE_COLS as GT
from src.models.train import split_by_vehicle, fit_xgb, eval_auc, _xy
from src.models.baselines import fit_lgbm, build_sequences, fit_lstm, lstm_auc

COLS = KIN + GT

def load(ds, h):
    df = pd.read_csv(config.PROCESSED_DIR / f"{ds}_gt_{h}s.csv"); df["dataset"] = ds
    return df

def auc_lgbm(m, df):
    X, y = _xy(df, COLS)
    return roc(y, m.predict_proba(X)[:, 1])

def roc(y, p):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, p) if y.nunique() > 1 else float("nan")

def main():
    rows = []
    for h in config.HORIZONS_SEC:
        trains_by_ds, tests = {}, {}
        for ds in config.TRAIN_DATASETS:
            tr, te = split_by_vehicle(load(ds, h), seed=config.RANDOM_STATE)
            trains_by_ds[ds] = tr; tests[ds] = te
        train = pd.concat(trains_by_ds.values(), ignore_index=True)
        ood = load(config.HOLDOUT_DATASET, h)
        cols = [c for c in COLS if c in train.columns]

        def indomain(fn):  # fn(df)->auc
            return float(np.nanmean([fn(tests[ds]) for ds in config.TRAIN_DATASETS]))

        # XGBoost
        mx = fit_xgb(*_xy(train, cols), seed=config.RANDOM_STATE)
        rows.append(dict(horizon=f"{h}s", model="XGBoost",
                         in_domain=round(indomain(lambda d: eval_auc(mx, d, cols)), 4),
                         ETRI_OOD=round(eval_auc(mx, ood, cols), 4)))
        # LightGBM
        ml = fit_lgbm(*_xy(train, cols), seed=config.RANDOM_STATE)
        rows.append(dict(horizon=f"{h}s", model="LightGBM",
                         in_domain=round(indomain(lambda d: roc(d["label"].astype(int), ml.predict_proba(d[cols].replace([np.inf,-np.inf],np.nan))[:,1])), 4),
                         ETRI_OOD=round(roc(ood["label"].astype(int), ml.predict_proba(ood[cols].replace([np.inf,-np.inf],np.nan))[:,1]), 4)))
        # LSTM (시퀀스 입력, GPU)
        cdir = config.INTERIM_DIR / "canonical"
        def seqs(df, ds):
            canon = pd.concat([pd.read_parquet(p) for p in (cdir/ds).glob("*.parquet")], ignore_index=True)
            return build_sequences(df, canon, K=10)
        Xtr_l, ytr_l = [], []
        for ds in config.TRAIN_DATASETS:
            X, y = seqs(trains_by_ds[ds], ds)
            Xtr_l.append(X); ytr_l.append(y)
        Xtr = np.concatenate(Xtr_l); ytr = np.concatenate(ytr_l)
        net, dev = fit_lstm(Xtr, ytr, seed=config.RANDOM_STATE)
        ind = []
        for ds in config.TRAIN_DATASETS:
            Xte, yte = seqs(tests[ds], ds); ind.append(lstm_auc(net, dev, Xte, yte))
        Xo, yo = seqs(ood, config.HOLDOUT_DATASET)
        rows.append(dict(horizon=f"{h}s", model="BiLSTM-Att",
                         in_domain=round(float(np.nanmean(ind)), 4),
                         ETRI_OOD=round(lstm_auc(net, dev, Xo, yo), 4)))
        for r in rows[-3:]:
            print(f"  [{h}s] {r['model']:9s} in-domain={r['in_domain']} ETRI_OOD={r['ETRI_OOD']}")

    pd.DataFrame(rows).to_csv(config.TABLES_DIR / "baseline_comparison.csv", index=False)
    print("\n저장: baseline_comparison.csv")

if __name__ == "__main__":
    main()
