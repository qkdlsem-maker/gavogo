#!/usr/bin/env python
"""[추가실험: CatBoost + Latency + Calibration] scripts/12_extras.py"""
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config
from src.features.kinematic import FEATURE_COLS as KIN
from src.features.game_theory import NASH_FEATURE_COLS as GT
from src.models.train import split_by_vehicle, fit_xgb, eval_auc, _xy
from src.models.baselines import fit_lgbm, fit_catboost

COLS = KIN + GT
H = 3

def load(ds):
    df = pd.read_csv(config.PROCESSED_DIR / f"{ds}_gt_{H}s.csv"); df["dataset"] = ds
    return df

def roc(y, p):
    return roc_auc_score(y, p) if pd.Series(y).nunique() > 1 else float("nan")

def ece(y, p, bins=10):
    y = np.asarray(y); p = np.asarray(p)
    edges = np.linspace(0, 1, bins+1); e = 0.0
    accs, confs = [], []
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i+1])
        if m.sum() == 0:
            accs.append(np.nan); confs.append((edges[i]+edges[i+1])/2); continue
        acc = y[m].mean(); conf = p[m].mean()
        e += (m.sum()/len(p)) * abs(acc - conf)
        accs.append(acc); confs.append(conf)
    return e, np.array(accs), np.array(confs)

def main():
    trains, tests = [], {}
    for ds in config.TRAIN_DATASETS:
        tr, te = split_by_vehicle(load(ds), seed=config.RANDOM_STATE)
        trains.append(tr); tests[ds] = te
    train = pd.concat(trains, ignore_index=True)
    ood = load(config.HOLDOUT_DATASET)
    cols = [c for c in COLS if c in train.columns]
    Xtr, ytr = _xy(train, cols)
    test = pd.concat(tests.values(), ignore_index=True)
    Xte = test[cols].replace([np.inf,-np.inf],np.nan); yte = test["label"].astype(int).values

    models = {"XGBoost": fit_xgb(Xtr, ytr, seed=config.RANDOM_STATE),
              "LightGBM": fit_lgbm(Xtr.fillna(Xtr.median()), ytr, seed=config.RANDOM_STATE),
              "CatBoost": fit_catboost(Xtr.fillna(Xtr.median()), ytr, seed=config.RANDOM_STATE)}
    rows = []
    for name, m in models.items():
        ind = float(np.nanmean([eval_auc(m, tests[d], cols) if name=="XGBoost"
                    else roc(tests[d]["label"].astype(int), m.predict_proba(tests[d][cols].replace([np.inf,-np.inf],np.nan).fillna(Xtr.median()))[:,1])
                    for d in config.TRAIN_DATASETS]))
        Xo = ood[cols].replace([np.inf,-np.inf],np.nan)
        Xo = Xo if name=="XGBoost" else Xo.fillna(Xtr.median())
        oa = roc(ood["label"].astype(int), m.predict_proba(Xo)[:,1])
        rows.append(dict(model=name, in_domain=round(ind,4), ETRI_OOD=round(oa,4)))
        print(f"  {name:9s} in-domain={ind:.4f} ETRI_OOD={oa:.4f}")
    pd.DataFrame(rows).to_csv(config.TABLES_DIR / "catboost_comparison.csv", index=False)

    lat = []
    Xb = Xte.fillna(Xtr.median()).values[:5000]
    for name, m in models.items():
        t0 = time.perf_counter()
        for _ in range(3): m.predict_proba(Xb)
        ms = (time.perf_counter()-t0)/3/len(Xb)*1000
        lat.append(dict(model=name, ms_per_sample=round(ms,4)))
        print(f"  {name:9s} latency={ms:.4f} ms/sample")
    pd.DataFrame(lat).to_csv(config.TABLES_DIR / "latency.csv", index=False)

    # ---- 3) Calibration (3 models) ----
    cal_rows = []
    fig, ax = plt.subplots(figsize=(4.5,4.5))
    ax.plot([0,1],[0,1],'--',c='gray',label='Perfect')
    palette = {"XGBoost":"#2E6FB7","LightGBM":"#2CA02C","CatBoost":"#D62728"}
    for name, m in models.items():
        Xq = Xte if name == "XGBoost" else Xte.fillna(Xtr.median())
        p = m.predict_proba(Xq)[:,1]
        e, accs, confs = ece(yte, p)
        print(f"  {name:9s} ECE = {e:.4f}")
        ax.plot(confs, accs, 'o-', c=palette[name], label=f'{name} (ECE={e:.3f})')
        cal_rows.append(dict(model=name, ECE=round(e,4)))
    ax.set_xlabel('Confidence'); ax.set_ylabel('Accuracy'); ax.set_title('Reliability Diagram')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(config.FIGURES_DIR / "calibration.png", dpi=150); plt.close()
    pd.DataFrame(cal_rows).to_csv(config.TABLES_DIR / "calibration.csv", index=False)
    print("저장 완료")

if __name__ == "__main__":
    main()
