#!/usr/bin/env python3
"""34_coral.py — Feature-space CORAL as a 4th domain-invariant strategy.
Run from repo root: python3 scripts/34_coral.py
Loads {ds}_gt_3s.csv like 16_domain_invariant.py. Output: results/tables/34_coral.csv
"""
import sys, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config
from scipy import linalg
from sklearn.metrics import roc_auc_score
import xgboost as xgb

SOURCES = ["highD", "NGSIM", "MiTra"]
TARGETS = ["ETRI", "EMT", "uniD", "exiD"]
H = 3
META = {"dataset", "domain", "recording_id", "vehicle_id", "frame", "label", "event_type"}
SEEDS = [0, 1, 2]
OUT = Path("results/tables/34_coral.csv")

def load(ds):
    df = pd.read_csv(config.PROCESSED_DIR / f"{ds}_gt_{H}s.csv")
    df["__ds"] = ds
    return df

def coral_transform(Xs, Xt, eps=1e-3):
    Cs = np.cov(Xs, rowvar=False) + eps * np.eye(Xs.shape[1])
    Ct = np.cov(Xt, rowvar=False) + eps * np.eye(Xt.shape[1])
    A = linalg.fractional_matrix_power(Cs, -0.5).real @ linalg.fractional_matrix_power(Ct, 0.5).real
    mu_s, mu_t = Xs.mean(0), Xt.mean(0)
    return lambda X: (X - mu_s) @ A + mu_t

def main():
    src_df = pd.concat([load(d) for d in SOURCES], ignore_index=True)
    feats = [c for c in src_df.columns if c not in META and c != "__ds"
             and pd.api.types.is_numeric_dtype(src_df[c])]
    print(f"{len(feats)} features")
    Xs_raw = src_df[feats].to_numpy(float); ys = src_df["label"].to_numpy()
    med = np.nanmedian(Xs_raw, axis=0)
    Xs_raw = np.where(np.isnan(Xs_raw), med, Xs_raw)

    rows = []
    for tgt in TARGETS:
        t = load(tgt)
        Xt = t.reindex(columns=feats).to_numpy(float); yt = t["label"].to_numpy()
        Xt = np.where(np.isnan(Xt), med, Xt)
        f = coral_transform(Xs_raw, Xt)
        Xs = f(Xs_raw)
        for seed in SEEDS:
            m = xgb.XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, eval_metric="auc",
                random_state=seed, n_jobs=-1, tree_method="hist")
            m.fit(Xs, ys)
            auc = roc_auc_score(yt, m.predict_proba(Xt)[:, 1])
            rows.append({"target": tgt, "seed": seed, "auc": auc})
            print(f"{tgt} seed{seed}: AUC={auc:.3f}", flush=True)
    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(out.groupby("target").auc.agg(["mean", "std"]).round(3))

if __name__ == "__main__":
    main()
