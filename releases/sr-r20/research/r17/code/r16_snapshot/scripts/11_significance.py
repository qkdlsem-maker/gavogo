#!/usr/bin/env python
"""[추가실험: 통계검정] Bootstrap 95% CI + DeLong test. 위치: scripts/11_significance.py
모델 AUC 차이가 유의한지 검정. joint 학습, in-domain test 합쳐서.
- Bootstrap CI: XGBoost / LightGBM / TabTransformer(옵션) AUC의 95% 신뢰구간
- DeLong: XGBoost vs LightGBM AUC 차이 p-value
실행: python scripts/11_significance.py
필요: pip install scipy lightgbm"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config
from src.features.kinematic import FEATURE_COLS as KIN
from src.features.game_theory import NASH_FEATURE_COLS as GT
from src.models.train import split_by_vehicle, fit_xgb, _xy
from src.models.baselines import fit_lgbm

COLS = KIN + GT
H = 3
N_BOOT = 1000

def load(ds):
    df = pd.read_csv(config.PROCESSED_DIR / f"{ds}_gt_{H}s.csv"); df["dataset"] = ds
    return df

def boot_ci(y, p, n=N_BOOT, seed=42):
    rng = np.random.RandomState(seed); aucs = []
    y = np.asarray(y); p = np.asarray(p)
    for _ in range(n):
        idx = rng.randint(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2: continue
        aucs.append(roc_auc_score(y[idx], p[idx]))
    return np.mean(aucs), np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)

# DeLong (fast implementation)
def delong_p(y, p1, p2):
    from scipy import stats
    y = np.asarray(y)
    def midrank(x):
        s = np.argsort(x); x2 = x[s]; n = len(x); r = np.zeros(n)
        i = 0
        while i < n:
            j = i
            while j < n and x2[j] == x2[i]: j += 1
            r[i:j] = 0.5*(i+j-1)+1; i = j
        out = np.empty(n); out[s] = r; return out
    def auc_var(p):
        pos = p[y==1]; neg = p[y==0]; m, n = len(pos), len(neg)
        tx = midrank(np.r_[pos, neg])
        tp = midrank(pos); tn = midrank(neg)
        auc = (tx[:m].sum() - tp.sum())/(m*n) + 0.5  # approx
        v01 = (tx[:m] - tp)/n; v10 = 1 - (tx[m:] - tn)/m
        return auc, v01, v10, m, n
    a1, v01_1, v10_1, m, n = auc_var(p1)
    a2, v01_2, v10_2, _, _ = auc_var(p2)
    s01 = np.cov(np.vstack([v01_1, v01_2])); s10 = np.cov(np.vstack([v10_1, v10_2]))
    S = s01/m + s10/n
    var = S[0,0]+S[1,1]-2*S[0,1]
    if var <= 0: return a1, a2, 1.0
    z = (a1-a2)/np.sqrt(var)
    return a1, a2, 2*(1-stats.norm.cdf(abs(z)))

def main():
    trains, tests = [], []
    for ds in config.TRAIN_DATASETS:
        tr, te = split_by_vehicle(load(ds), seed=config.RANDOM_STATE)
        trains.append(tr); tests.append(te)
    train = pd.concat(trains, ignore_index=True)
    test = pd.concat(tests, ignore_index=True)
    cols = [c for c in COLS if c in train.columns]
    Xtr, ytr = _xy(train, cols)
    Xte = test[cols].replace([np.inf,-np.inf],np.nan); yte = test["label"].astype(int).values

    mx = fit_xgb(Xtr, ytr, seed=config.RANDOM_STATE); px = mx.predict_proba(Xte)[:,1]
    ml = fit_lgbm(Xtr.fillna(Xtr.median()), ytr, seed=config.RANDOM_STATE)
    pl = ml.predict_proba(Xte.fillna(Xtr.median()))[:,1]

    rows = []
    for name, p in [("XGBoost", px), ("LightGBM", pl)]:
        mean, lo, hi = boot_ci(yte, p)
        rows.append(dict(model=name, auc=round(mean,4), ci95_low=round(lo,4), ci95_high=round(hi,4)))
        print(f"  {name:9s} AUC={mean:.4f}  95% CI=[{lo:.4f}, {hi:.4f}]")
    pd.DataFrame(rows).to_csv(config.TABLES_DIR / "bootstrap_ci.csv", index=False)

    a1, a2, pval = delong_p(yte, px, pl)
    print(f"\n  DeLong XGBoost vs LightGBM: p={pval:.4f} ({'유의' if pval<0.05 else '유의차 없음'})")
    pd.DataFrame([dict(comparison="XGBoost_vs_LightGBM", p_value=round(pval,4),
                       significant=bool(pval<0.05))]).to_csv(config.TABLES_DIR / "delong.csv", index=False)
    print("\n저장: bootstrap_ci.csv, delong.csv")

if __name__ == "__main__":
    main()
