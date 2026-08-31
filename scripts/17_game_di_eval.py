#!/usr/bin/env python
"""[T-IV #2] 게임이론 재정식화 평가: 절대단위 payoff vs 무차원 payoff.

질문: 게임피처의 도메인 leak(payoff_lc_max domain-AUC 0.907)은
      (a) 게임이론 자체가 도메인 의존적이기 때문인가?
      (b) 아니면 절대단위(m, m/s) 기반 payoff 설계 때문인가?

측정
  1) domain-discriminability: 각 게임피처가 source/target을 구별하는 AUC
     GT_abs (기존) vs GT_di (무차원) 비교
  2) 예측 성능: KIN / KIN+GT_abs / KIN+GT_di  (in-domain & zero-shot OOD)

실행: python scripts/17_game_di_eval.py
출력: results/tables/game_di_domain_auc.csv, game_di_perf.csv
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

SRC = ["highD", "NGSIM", "MiTra"]
TGT = ["ETRI", "EMT", "uniD", "exiD"]
ALL = SRC + TGT
SEEDS = [42, 0, 1]
H = 3


def roc(y, p):
    y = np.asarray(y)
    return roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")


def gkey(df):
    return (df["dataset"].astype(str) + "|" + df["recording_id"].astype(str)
            + "|" + df["vehicle_id"].astype(str)).values


def gsplit(keys, seed, frac=0.3):
    u = np.unique(keys)
    rng = np.random.RandomState(seed); rng.shuffle(u)
    te = set(u[:max(1, int(len(u) * frac))])
    m = np.array([k in te for k in keys])
    return ~m, m


def clean(df, cols, med=None):
    X = df[cols].replace([np.inf, -np.inf], np.nan)
    if med is None:
        med = X.median()
    return X.fillna(med), med


def load(ds, kind):
    tag = "gt" if kind == "abs" else "gtdi"
    df = pd.read_csv(config.PROCESSED_DIR / f"{ds}_{tag}_{H}s.csv")
    df["dataset"] = ds
    return df


def domain_auc(data, cols):
    s = pd.concat([data[d] for d in SRC], ignore_index=True)
    t = pd.concat([data[d] for d in TGT], ignore_index=True)
    rows = []
    for c in cols:
        a = s[c].replace([np.inf, -np.inf], np.nan).dropna()
        b = t[c].replace([np.inf, -np.inf], np.nan).dropna()
        if len(a) < 50 or len(b) < 50:
            continue
        n = min(len(a), len(b), 20000)
        y = np.r_[np.zeros(n), np.ones(n)]
        p = np.r_[a.sample(n, random_state=42).values, b.sample(n, random_state=42).values]
        au = roc(y, p)
        rows.append((c, round(float(max(au, 1 - au)), 4)))
    return dict(rows)


def evaluate(data, cols, tag):
    src = pd.concat([data[d] for d in SRC], ignore_index=True)
    keys = gkey(src)
    Xs, med = clean(src, cols)
    ys = src["label"].astype(int).values
    ind, ood = [], {t: [] for t in TGT}
    for s in SEEDS:
        tr, te = gsplit(keys, s)
        m = fit_xgb(Xs[tr], ys[tr], seed=s)
        ind.append(roc(ys[te], m.predict_proba(Xs[te])[:, 1]))
        for t in TGT:
            Xt, _ = clean(data[t], cols, med=med)
            ood[t].append(roc(data[t]["label"].astype(int).values,
                              m.predict_proba(Xt)[:, 1]))
    r = dict(featset=tag, n_feats=len(cols),
             in_domain=round(float(np.nanmean(ind)), 4))
    for t in TGT:
        r[f"OOD_{t}"] = round(float(np.nanmean(ood[t])), 4)
    r["OOD_mean"] = round(float(np.nanmean([r[f"OOD_{t}"] for t in TGT])), 4)
    print(f"  {tag:14s} feats={len(cols):2d} in={r['in_domain']:.4f} OOD={r['OOD_mean']:.4f}")
    return r


def main():
    d_abs = {d: load(d, "abs") for d in ALL}
    d_di = {d: load(d, "di") for d in ALL}

    print("── (1) 게임피처 domain-discriminability: 절대단위 vs 무차원 ──")
    a = domain_auc(d_abs, GT)
    b = domain_auc(d_di, GT)
    rows = []
    for c in GT:
        if c in a and c in b:
            rows.append(dict(feature=c, GT_abs=a[c], GT_di=b[c],
                             delta=round(b[c] - a[c], 4)))
    dd = pd.DataFrame(rows).sort_values("GT_abs", ascending=False)
    dd.to_csv(config.TABLES_DIR / "game_di_domain_auc.csv", index=False)
    print(dd.to_string(index=False))
    print(f"\n  평균 domain-AUC:  절대단위={dd.GT_abs.mean():.4f}  "
          f"무차원={dd.GT_di.mean():.4f}  (Δ={dd.GT_di.mean()-dd.GT_abs.mean():+.4f})\n")

    print("── (2) 예측 성능 ──")
    kin = [c for c in KIN if all(c in d_abs[d].columns for d in ALL)]
    perf = [
        evaluate(d_abs, kin, "KIN only"),
        evaluate(d_abs, kin + GT, "KIN+GT_abs"),
        evaluate(d_di, kin + GT, "KIN+GT_di"),
        evaluate(d_di, GT, "GT_di only"),
    ]
    out = pd.DataFrame(perf)
    out.to_csv(config.TABLES_DIR / "game_di_perf.csv", index=False)
    print("\n" + out.to_string(index=False))
    print("\n저장: game_di_domain_auc.csv, game_di_perf.csv")


if __name__ == "__main__":
    main()
