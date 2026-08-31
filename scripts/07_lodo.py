#!/usr/bin/env python
"""[추가실험: LODO] Leave-One-Domain-Out 4-fold OOD.
위치: scripts/07_lodo.py
각 데이터셋을 한 번씩 테스트로 두고 나머지 3개로 학습 → zero-shot OOD.
실행: python scripts/07_lodo.py"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config
from src.features.kinematic import FEATURE_COLS as KIN
from src.features.game_theory import NASH_FEATURE_COLS as GT
from src.models.train import fit_xgb

COLS = KIN + GT
ALL = ["highD", "NGSIM", "MiTra", "ETRI"]

def load(ds, h):
    return pd.read_csv(config.PROCESSED_DIR / f"{ds}_gt_{h}s.csv")

def clean(df, c):
    return df[c].replace([np.inf, -np.inf], np.nan)

def main():
    rows = []
    for h in config.HORIZONS_SEC:
        data = {ds: load(ds, h) for ds in ALL}
        cols = [c for c in COLS if all(c in data[ds].columns for ds in ALL)]
        for test_ds in ALL:
            tr = pd.concat([data[d] for d in ALL if d != test_ds], ignore_index=True)
            te = data[test_ds]
            m = fit_xgb(clean(tr, cols), tr["label"].astype(int), seed=config.RANDOM_STATE)
            y = te["label"].astype(int)
            auc = roc_auc_score(y, m.predict_proba(clean(te, cols))[:, 1]) if y.nunique() > 1 else float("nan")
            rows.append(dict(horizon=f"{h}s", test_domain=test_ds, auc=round(auc, 4)))
            print(f"  [{h}s] test={test_ds:6s} (train=others)  OOD-AUC={auc:.4f}")

    res = pd.DataFrame(rows)
    res.to_csv(config.TABLES_DIR / "lodo.csv", index=False)
    print("\n── 도메인별 평균 OOD-AUC ──")
    for ds in ALL:
        print(f"  {ds}: {res[res.test_domain==ds].auc.mean():.4f}")
    print("\n저장: lodo.csv")

if __name__ == "__main__":
    main()
