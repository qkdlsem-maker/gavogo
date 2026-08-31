#!/usr/bin/env python
"""[5단계-uniD] cross-domain 적응: uniD OOD 개선.
학습: highD+NGSIM+MiTra, 타깃: uniD
실행: python scripts/05_domain_adapt_unid.py"""
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
from src.models.train import fit_xgb

COLS = KIN + GT


def load(ds, h):
    return pd.read_csv(config.PROCESSED_DIR / f"{ds}_gt_{h}s.csv")


def clean(df, cols):
    return df[cols].replace([np.inf, -np.inf], np.nan).fillna(df[cols].median())


def roc(y, p):
    return roc_auc_score(y, p) if pd.Series(y).nunique() > 1 else float("nan")


def main():
    rows = []
    for h in config.HORIZONS_SEC:
        src = pd.concat(
            [load(ds, h) for ds in ["highD", "NGSIM", "MiTra"]],
            ignore_index=True
        )
        unid = load("uniD", h)
        cols = [c for c in COLS if c in src.columns and c in unid.columns]

        Xs, ys = clean(src, cols), src["label"].astype(int)
        Xu, yu = clean(unid, cols), unid["label"].astype(int)

        # 1) zero-shot
        m = fit_xgb(Xs, ys, seed=config.RANDOM_STATE)
        zs = roc(yu, m.predict_proba(Xu)[:, 1])

        # 2) 표준화
        ss = StandardScaler().fit(Xs)
        su = StandardScaler().fit(Xu)
        ms = fit_xgb(ss.transform(Xs), ys, seed=config.RANDOM_STATE)
        std = roc(yu, ms.predict_proba(su.transform(Xu))[:, 1])

        # 3) fine-tune (uniD 60% 적응 → 40% 평가)
        idx = np.random.RandomState(42).permutation(len(Xu))
        cut = int(len(idx) * 0.6)
        tr_i, te_i = idx[:cut], idx[cut:]
        Xft = pd.concat([Xs, Xu.iloc[tr_i]])
        yft = np.concatenate([ys, yu.values[tr_i]])
        mft = fit_xgb(Xft, yft, seed=config.RANDOM_STATE)
        ft = roc(yu.values[te_i], mft.predict_proba(Xu.iloc[te_i])[:, 1])

        # 4) 표준화 + fine-tune
        Xs_s = ss.transform(Xs)
        Xu_s = su.transform(Xu)
        Xft2 = np.vstack([Xs_s, Xu_s[tr_i]])
        mft2 = fit_xgb(Xft2, yft, seed=config.RANDOM_STATE)
        ft2 = roc(yu.values[te_i], mft2.predict_proba(Xu_s[te_i])[:, 1])

        rows.append(dict(
            horizon=f"{h}s",
            zeroshot=round(zs, 4),
            std=round(std, 4),
            finetune=round(ft, 4),
            std_finetune=round(ft2, 4),
        ))
        print(f"  [{h}s] zeroshot={zs:.4f} std={std:.4f} finetune={ft:.4f} std+ft={ft2:.4f}")

    out = pd.DataFrame(rows)
    out.to_csv(config.TABLES_DIR / "domain_adapt_unid.csv", index=False)
    print("\n저장: domain_adapt_unid.csv")


if __name__ == "__main__":
    main()
