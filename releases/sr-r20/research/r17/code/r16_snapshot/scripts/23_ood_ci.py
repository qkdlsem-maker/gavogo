#!/usr/bin/env python
"""[리뷰어 대응: OOD Bootstrap CI] 위치: scripts/23_ood_ci.py

Table IV(zero-shot OOD)의 각 target과 pooled OOD AUC에 대해
95% bootstrap 신뢰구간을 계산. "OOD 0.520 [0.511, 0.529]" 형태로 통계적 완성도↑.

기존 processed CSV(_gt_{h}s.csv)를 그대로 사용 — 재빌드 없음, 빠름.
학습: highD+NGSIM+MiTra 전체, 평가: ETRI/EMT/uniD/exiD zero-shot.
실행: python scripts/23_ood_ci.py
"""
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config
from src.features.kinematic import FEATURE_COLS as KIN
from src.features.game_theory import NASH_FEATURE_COLS as GT
from src.models.train import fit_xgb, _xy

COLS = KIN + GT
H = 3
N_BOOT = 2000
TARGETS = [config.HOLDOUT_DATASET] + config.OOD_DATASETS   # ETRI, EMT, uniD, exiD


def load(ds):
    df = pd.read_csv(config.PROCESSED_DIR / f"{ds}_gt_{H}s.csv")
    df["dataset"] = ds
    return df


def clean(df, cols):
    return df[cols].replace([np.inf, -np.inf], np.nan)


def boot_ci(y, p, n=N_BOOT, seed=42):
    rng = np.random.RandomState(seed)
    y = np.asarray(y); p = np.asarray(p)
    aucs = []
    for _ in range(n):
        idx = rng.randint(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y[idx], p[idx]))
    return (float(np.mean(aucs)),
            float(np.percentile(aucs, 2.5)),
            float(np.percentile(aucs, 97.5)))


def main():
    src = pd.concat([load(ds) for ds in config.TRAIN_DATASETS], ignore_index=True)
    tgt = {ds: load(ds) for ds in TARGETS}
    cols = [c for c in COLS if c in src.columns and all(c in tgt[d].columns for d in tgt)]

    m = fit_xgb(*_xy(src, cols), seed=config.RANDOM_STATE)

    rows = []
    ys_all, ps_all = [], []
    for ds in TARGETS:
        te = tgt[ds]
        y = te["label"].astype(int).values
        p = m.predict_proba(clean(te, cols))[:, 1]
        ys_all.append(y); ps_all.append(p)
        mean, lo, hi = boot_ci(y, p)
        rows.append(dict(target=ds, n=len(y), auc=round(mean, 4),
                         ci95_low=round(lo, 4), ci95_high=round(hi, 4)))
        print(f"  {ds:6s} AUC={mean:.4f}  95% CI=[{lo:.4f}, {hi:.4f}]  (n={len(y)})")

    # pooled OOD
    y_all = np.concatenate(ys_all); p_all = np.concatenate(ps_all)
    mean, lo, hi = boot_ci(y_all, p_all)
    rows.append(dict(target="POOLED", n=len(y_all), auc=round(mean, 4),
                     ci95_low=round(lo, 4), ci95_high=round(hi, 4)))
    print(f"  {'POOLED':6s} AUC={mean:.4f}  95% CI=[{lo:.4f}, {hi:.4f}]  (n={len(y_all)})")

    out = config.TABLES_DIR / "ood_ci.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
