#!/usr/bin/env python
"""[추가실험: TabTransformer] 최신 tabular transformer 비교.
위치: scripts/04c_tabtransformer.py
joint 데이터·동일 split로 XGBoost vs TabTransformer. in-domain + ETRI OOD.
GPU 1번 권장: CUDA_VISIBLE_DEVICES=1 python scripts/04c_tabtransformer.py
필요: pip install torch"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config
from src.features.kinematic import FEATURE_COLS as KIN
from src.features.game_theory import NASH_FEATURE_COLS as GT
from src.models.train import split_by_vehicle, fit_xgb, eval_auc, _xy
from src.models.baselines import fit_tabtransformer, tabt_auc

COLS = KIN + GT

def load(ds, h):
    df = pd.read_csv(config.PROCESSED_DIR / f"{ds}_gt_{h}s.csv"); df["dataset"] = ds
    return df

def roc(y, p):
    return roc_auc_score(y, p) if pd.Series(y).nunique() > 1 else float("nan")

def main():
    rows = []
    for h in config.HORIZONS_SEC:
        trains, tests = [], {}
        for ds in config.TRAIN_DATASETS:
            tr, te = split_by_vehicle(load(ds, h), seed=config.RANDOM_STATE)
            trains.append(tr); tests[ds] = te
        train = pd.concat(trains, ignore_index=True)
        ood = load(config.HOLDOUT_DATASET, h)
        cols = [c for c in COLS if c in train.columns]

        # XGBoost
        mx = fit_xgb(*_xy(train, cols), seed=config.RANDOM_STATE)
        xin = float(np.nanmean([eval_auc(mx, tests[ds], cols) for ds in config.TRAIN_DATASETS]))
        rows.append(dict(horizon=f"{h}s", model="XGBoost",
                         in_domain=round(xin,4), ETRI_OOD=round(eval_auc(mx, ood, cols),4)))

        # TabTransformer (표준화 후)
        sc = StandardScaler()
        Xtr = sc.fit_transform(train[cols].replace([np.inf,-np.inf],np.nan).fillna(0))
        ytr = train["label"].astype(int).values
        net, dev = fit_tabtransformer(Xtr, ytr, seed=config.RANDOM_STATE)
        ind = []
        for ds in config.TRAIN_DATASETS:
            Xte = sc.transform(tests[ds][cols].replace([np.inf,-np.inf],np.nan).fillna(0))
            ind.append(tabt_auc(net, dev, Xte, tests[ds]["label"].astype(int).values))
        Xo = sc.transform(ood[cols].replace([np.inf,-np.inf],np.nan).fillna(0))
        rows.append(dict(horizon=f"{h}s", model="TabTransformer",
                         in_domain=round(float(np.nanmean(ind)),4),
                         ETRI_OOD=round(tabt_auc(net, dev, Xo, ood["label"].astype(int).values),4)))
        for r in rows[-2:]:
            print(f"  [{h}s] {r['model']:15s} in-domain={r['in_domain']} ETRI_OOD={r['ETRI_OOD']}")

    pd.DataFrame(rows).to_csv(config.TABLES_DIR / "tabtransformer_comparison.csv", index=False)
    print("\n저장: tabtransformer_comparison.csv")

if __name__ == "__main__":
    main()
