#!/usr/bin/env python
"""[T-IV Q4] 딥러닝 포함 전체 baseline 공정 비교.

공정성 조건 (리뷰 Q4 대응)
  - 동일 데이터 (within-vehicle 샘플링, purity=0)
  - 동일 그룹 분할 (recording|vehicle 단위, 동일 seed)
  - 동일 피처 (tree/TabTransformer/MLP: 48개 tabular)
  - BiLSTM만 시퀀스 입력(과거 K프레임 운동학) — 구조상 불가피, 명시
  - 동일 seed 집합, multi-seed 평균±std
  - 학습 하이퍼파라미터를 표로 명시 (아래 HPARAMS)

실행: CUDA_VISIBLE_DEVICES=0 python scripts/19_baselines_all.py
출력: results/tables/baselines_all.csv, hyperparams.csv
"""
import sys, time
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
from src.models.baselines import (fit_lgbm, fit_catboost, fit_tabtransformer, tabt_auc,
                                  build_sequences, fit_lstm, lstm_auc)

COLS = KIN + GT
SRC = ["highD", "NGSIM", "MiTra"]
TGT = ["ETRI", "EMT", "uniD", "exiD"]
SEEDS = [42, 0, 1]
H = 3

HPARAMS = [
    dict(model="XGBoost", params="n_estimators=500, max_depth=6, lr=0.05, tree_method=hist"),
    dict(model="LightGBM", params="n_estimators=500, max_depth=6, lr=0.05, num_leaves=31"),
    dict(model="CatBoost", params="iterations=500, depth=6, lr=0.05"),
    dict(model="TabTransformer", params="d=32, layers=2, heads=4, Adam lr=1e-3, wd=1e-5, epochs=30, bs=512"),
    dict(model="BiLSTM-Att", params="hidden=64, bidir, attention, Adam lr=1e-3, epochs=30, bs=512, K=10 frames"),
]


def load(ds):
    df = pd.read_csv(config.PROCESSED_DIR / f"{ds}_gt_{H}s.csv")
    df["dataset"] = ds
    return df


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


def roc(y, p):
    y = np.asarray(y)
    return roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")


def canon_of(ds):
    cdir = config.INTERIM_DIR / "canonical" / ds
    return pd.concat([pd.read_parquet(p) for p in sorted(cdir.glob("*.parquet"))],
                     ignore_index=True)


def main():
    data = {d: load(d) for d in SRC + TGT}
    cols = [c for c in COLS if all(c in data[d].columns for d in SRC + TGT)]
    src = pd.concat([data[d] for d in SRC], ignore_index=True)
    keys = gkey(src)
    X, med = clean(src, cols)
    y = src["label"].astype(int).values
    print(f"source rows={len(src)} feats={len(cols)}")

    # OOD 행렬 미리 준비
    Xt = {t: clean(data[t], cols, med=med)[0] for t in TGT}
    yt = {t: data[t]["label"].astype(int).values for t in TGT}

    # BiLSTM용 시퀀스 (canonical 필요)
    print("BiLSTM 시퀀스 생성 중...")
    canons = {d: canon_of(d) for d in SRC + TGT}
    seq_src = {d: build_sequences(data[d], canons[d], K=10) for d in SRC}
    seq_tgt = {t: build_sequences(data[t], canons[t], K=10) for t in TGT}

    res = {m["model"]: dict(ind=[], **{t: [] for t in TGT}) for m in HPARAMS}
    lat = {}

    for s in SEEDS:
        tr, te = gsplit(keys, s)
        print(f"\n── seed {s}  (train {tr.sum()} / test {te.sum()})")

        # ── Tree 계열
        for name, fn in [("XGBoost", fit_xgb), ("LightGBM", fit_lgbm),
                         ("CatBoost", fit_catboost)]:
            m = fn(X[tr], y[tr], seed=s)
            res[name]["ind"].append(roc(y[te], m.predict_proba(X[te])[:, 1]))
            for t in TGT:
                res[name][t].append(roc(yt[t], m.predict_proba(Xt[t])[:, 1]))
            if s == SEEDS[0]:
                Xb = X.values[:1]
                t0 = time.perf_counter()
                for _ in range(100):
                    m.predict_proba(Xb)
                lat[name] = (time.perf_counter() - t0) / 100 * 1000
            print(f"   {name:14s} in={res[name]['ind'][-1]:.4f}")

        # ── TabTransformer (동일 피처, 표준화)
        sc = StandardScaler().fit(X[tr])
        net, dev = fit_tabtransformer(sc.transform(X[tr]), y[tr], seed=s)
        res["TabTransformer"]["ind"].append(
            tabt_auc(net, dev, sc.transform(X[te]), y[te]))
        for t in TGT:
            res["TabTransformer"][t].append(
                tabt_auc(net, dev, sc.transform(Xt[t]), yt[t]))
        print(f"   {'TabTransformer':14s} in={res['TabTransformer']['ind'][-1]:.4f}")

        # ── BiLSTM (시퀀스 입력)
        Xs_all = np.concatenate([seq_src[d][0] for d in SRC])
        ys_all = np.concatenate([seq_src[d][1] for d in SRC])
        net2, dev2 = fit_lstm(Xs_all[tr], ys_all[tr], seed=s)
        res["BiLSTM-Att"]["ind"].append(
            lstm_auc(net2, dev2, Xs_all[te], ys_all[te]))
        for t in TGT:
            res["BiLSTM-Att"][t].append(
                lstm_auc(net2, dev2, seq_tgt[t][0], seq_tgt[t][1]))
        print(f"   {'BiLSTM-Att':14s} in={res['BiLSTM-Att']['ind'][-1]:.4f}")

    rows = []
    for m in HPARAMS:
        n = m["model"]
        r = dict(model=n,
                 in_domain=round(float(np.nanmean(res[n]["ind"])), 4),
                 in_domain_std=round(float(np.nanstd(res[n]["ind"])), 4))
        for t in TGT:
            r[f"OOD_{t}"] = round(float(np.nanmean(res[n][t])), 4)
        r["OOD_mean"] = round(float(np.nanmean([r[f"OOD_{t}"] for t in TGT])), 4)
        r["latency_ms_b1"] = round(lat.get(n, np.nan), 4) if n in lat else np.nan
        rows.append(r)

    out = pd.DataFrame(rows)
    out.to_csv(config.TABLES_DIR / "baselines_all.csv", index=False)
    pd.DataFrame(HPARAMS).to_csv(config.TABLES_DIR / "hyperparams.csv", index=False)
    print("\n" + out.to_string(index=False))
    print("\n저장: baselines_all.csv, hyperparams.csv")


if __name__ == "__main__":
    main()
