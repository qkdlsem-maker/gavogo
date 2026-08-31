#!/usr/bin/env python
"""[T-IV #4] Leave-One-Country-Out (LOCO) + Leave-One-RoadType-Out.

국가 매핑
  Germany : highD, uniD, exiD
  USA     : NGSIM
  Italy   : MiTra, EMT
  Korea   : ETRI

도로유형 매핑 (논문 핵심 주장: 국가보다 도로유형이 전이를 지배한다)
  Highway : highD, NGSIM, MiTra, ETRI, exiD
  Urban   : EMT, uniD

각 fold: 해당 그룹 전체를 제외하고 학습 → 제외 그룹 전체에서 zero-shot 평가.
multi-seed 평균±std.

실행: python scripts/15_loco.py
출력: results/tables/loco.csv, loco_roadtype.csv
"""
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
SEEDS = [42, 0, 1, 7, 123]

COUNTRY = {
    "Germany": ["highD", "uniD", "exiD"],
    "USA":     ["NGSIM"],
    "Italy":   ["MiTra", "EMT"],
    "Korea":   ["ETRI"],
}
ROADTYPE = {
    "Highway": ["highD", "NGSIM", "MiTra", "ETRI", "exiD"],
    "Urban":   ["EMT", "uniD"],
}
ALL = sorted({d for v in COUNTRY.values() for d in v})


def load(ds, h):
    df = pd.read_csv(config.PROCESSED_DIR / f"{ds}_gt_{h}s.csv")
    df["dataset"] = ds
    return df


def clean(df, cols, med=None):
    X = df[cols].replace([np.inf, -np.inf], np.nan)
    if med is None:
        med = X.median()
    return X.fillna(med), med


def roc(y, p):
    y = np.asarray(y)
    return roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")


def run(groups, tag, fname):
    rows = []
    for h in config.HORIZONS_SEC:
        data = {ds: load(ds, h) for ds in ALL}
        cols = [c for c in COLS if all(c in data[d].columns for d in ALL)]
        for gname, members in groups.items():
            tr_ds = [d for d in ALL if d not in members]
            if not tr_ds:
                continue
            tr = pd.concat([data[d] for d in tr_ds], ignore_index=True)
            te = pd.concat([data[d] for d in members], ignore_index=True)
            Xtr, med = clean(tr, cols)
            ytr = tr["label"].astype(int).values
            Xte, _ = clean(te, cols, med=med)
            yte = te["label"].astype(int).values

            aucs = []
            for s in SEEDS:
                m = fit_xgb(Xtr, ytr, seed=s)
                aucs.append(roc(yte, m.predict_proba(Xte)[:, 1]))
            mu, sd = float(np.nanmean(aucs)), float(np.nanstd(aucs))
            rows.append(dict(horizon=f"{h}s", held_out=gname,
                             members="+".join(members),
                             n_train=len(tr), n_test=len(te),
                             auc=round(mu, 4), auc_std=round(sd, 4)))
            print(f"  [{h}s] {tag}={gname:8s} ({'+'.join(members)}) "
                  f"→ OOD-AUC={mu:.4f} ± {sd:.4f}")

    out = pd.DataFrame(rows)
    out.to_csv(config.TABLES_DIR / fname, index=False)
    print(f"\n저장: {fname}\n")
    return out


def main():
    print("── Leave-One-Country-Out ──")
    c = run(COUNTRY, "country", "loco.csv")
    print("── Leave-One-RoadType-Out ──")
    r = run(ROADTYPE, "roadtype", "loco_roadtype.csv")

    print("\n── 요약 (horizon 평균) ──")
    print("[Country]")
    print(c.groupby("held_out").auc.mean().round(4).to_string())
    print("[RoadType]")
    print(r.groupby("held_out").auc.mean().round(4).to_string())


if __name__ == "__main__":
    main()
