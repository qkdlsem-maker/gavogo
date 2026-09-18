#!/usr/bin/env python
"""[진단 A] 전체 7개 데이터셋의 group_label_purity + in-domain AUC 측정.

목적: exiD에서 발견된 '차량=라벨' 붕괴가 학습 데이터셋(highD/NGSIM/MiTra)에도
      존재하는지 확인. highD 0.93 AUC가 진짜 신호인지 artifact인지 판정.

핵심 지표
  group_label_purity : 차량(rec,veh)의 라벨이 100% 한쪽으로 결정되는 비율
                       → 1.0에 가까울수록 'intent 예측'이 아니라 '차량 분류'
  mixed_group_auc    : 라벨이 섞인 차량(positive/negative 둘 다 가진 차량)만으로
                       평가한 AUC. 이게 진짜 'when' 성능. full과 격차가 크면 붕괴.

실행: python scripts/diag_purity_all.py
출력: results/tables/diag_purity_all.csv
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
ALL = ["highD", "NGSIM", "MiTra", "ETRI", "EMT", "uniD", "exiD"]
SEEDS = [42, 0, 1]
TEST_FRAC = 0.30


def load(ds, h):
    return pd.read_csv(config.PROCESSED_DIR / f"{ds}_gt_{h}s.csv")


def gkey(df):
    return (df["recording_id"].astype(str) + "|" + df["vehicle_id"].astype(str)).values


def gsplit(keys, seed, frac=TEST_FRAC):
    u = np.unique(keys)
    rng = np.random.RandomState(seed)
    rng.shuffle(u)
    te = set(u[:max(1, int(len(u) * frac))])
    m = np.array([k in te for k in keys])
    return ~m, m


def clean(df, cols):
    X = df[cols].replace([np.inf, -np.inf], np.nan)
    return X.fillna(X.median())


def roc(y, p):
    y = np.asarray(y)
    return roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")


def main():
    rows = []
    for h in config.HORIZONS_SEC:
        for ds in ALL:
            try:
                df = load(ds, h)
            except FileNotFoundError:
                continue
            cols = [c for c in COLS if c in df.columns]
            y = df["label"].astype(int).values
            keys = gkey(df)

            # purity
            g = pd.DataFrame({"g": keys, "y": y})
            per = g.groupby("g").y.mean()
            pure_mask = (per == 0) | (per == 1)
            purity = float(pure_mask.mean())
            mixed_groups = set(per[~pure_mask].index)
            n_mixed_rows = int(np.isin(keys, list(mixed_groups)).sum()) if mixed_groups else 0

            X = clean(df, cols)
            full_a, mixed_a = [], []
            for s in SEEDS:
                tr, te = gsplit(keys, s)
                if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
                    continue
                m = fit_xgb(X[tr], y[tr], seed=s)
                p = m.predict_proba(X[te])[:, 1]
                full_a.append(roc(y[te], p))
                # 라벨 섞인 차량만으로 평가 = 진짜 'when' 성능
                mm = np.isin(keys[te], list(mixed_groups))
                if mm.sum() > 20 and len(np.unique(y[te][mm])) > 1:
                    mixed_a.append(roc(y[te][mm], p[mm]))

            fa = float(np.nanmean(full_a)) if full_a else np.nan
            ma = float(np.nanmean(mixed_a)) if mixed_a else np.nan
            gap = fa - ma if not (np.isnan(fa) or np.isnan(ma)) else np.nan

            rows.append(dict(dataset=ds, horizon=h,
                             n_rows=len(df), n_groups=len(per),
                             purity=round(purity, 4),
                             n_mixed_groups=len(mixed_groups),
                             n_mixed_rows=n_mixed_rows,
                             auc_full=round(fa, 4) if not np.isnan(fa) else np.nan,
                             auc_mixed_only=round(ma, 4) if not np.isnan(ma) else np.nan,
                             gap=round(gap, 4) if not np.isnan(gap) else np.nan))
            print(f"  [{h}s] {ds:6s} purity={purity:.3f}  "
                  f"AUC_full={fa:.4f}  AUC_mixed={ma if not np.isnan(ma) else float('nan'):.4f}  "
                  f"gap={gap if not np.isnan(gap) else float('nan'):+.4f}  "
                  f"(mixed groups {len(mixed_groups)}/{len(per)})")

    out = pd.DataFrame(rows)
    out.to_csv(config.TABLES_DIR / "diag_purity_all.csv", index=False)
    print("\n" + "=" * 70)
    print(out.to_string(index=False))
    print("\n판정 기준:")
    print("  purity > 0.85          → 'when'이 아니라 'who' 문제로 붕괴")
    print("  gap (full-mixed) > 0.15 → full AUC가 차량분류로 부풀려짐")
    print("\n저장: diag_purity_all.csv")


if __name__ == "__main__":
    main()
