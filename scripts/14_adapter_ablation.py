#!/usr/bin/env python
"""[T-IV #10] Adapter Ablation — Adapter가 핵심 기여임을 직접 증명.

변형 (data-level ablation, 02_build_features_v2.py의 플래그로 생성)
  V1 Raw             : 이웃 재구성 X, 도로좌표 X, 횡부호 정규화 X
  V2 +NeighborRecon  : 이웃 재구성 O
  V3 +Normalization  : + 도로좌표(중심선 복원) + 횡부호 정규화
  V4 Full GAVOGO     : V3 + 게임이론 피처 14개

각 변형마다: in-domain(그룹분할) + zero-shot OOD(4개 타깃) 평가.

선행: (아래를 7개 데이터셋 전부에 대해 실행)
  python scripts/02_build_features_v2.py --dataset $D --suffix _V1 \
         --no_neighbors --no_roadframe --no_latsign
  python scripts/02_build_features_v2.py --dataset $D --suffix _V2 \
         --no_roadframe --no_latsign
  python scripts/02_build_features_v2.py --dataset $D --suffix _V3
  python scripts/03_build_game_features.py --dataset $D          # V4용 (접미사 없음)

실행: python scripts/14_adapter_ablation.py
출력: results/tables/adapter_ablation.csv
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
SEEDS = [42, 0, 1]
H = 3

# (suffix, 게임피처 사용여부, 표시명)
VARIANTS = [
    ("_V1", False, "V1 Raw (no adapter)"),
    ("_V2", False, "V2 + Neighbor reconstruction"),
    ("_V3", False, "V3 + Normalization (road frame + lat sign)"),
    ("",    True,  "V4 Full GAVOGO (+ game theory)"),
]


def load(ds, suffix, use_game):
    name = f"{ds}_gt_{H}s.csv" if use_game else f"{ds}{suffix}_{H}s.csv"
    p = config.PROCESSED_DIR / name
    if not p.exists():
        return None
    df = pd.read_csv(p)
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


def main():
    rows = []
    for suffix, use_game, label in VARIANTS:
        data = {ds: load(ds, suffix, use_game) for ds in SRC + TGT}
        if any(v is None for v in data.values()):
            miss = [k for k, v in data.items() if v is None]
            print(f"[skip] {label}: 파일 없음 {miss}")
            continue

        base = KIN + GT if use_game else KIN
        cols = [c for c in base if all(c in data[d].columns for d in SRC + TGT)]

        src = pd.concat([data[d] for d in SRC], ignore_index=True)
        keys = gkey(src)
        Xs_all, med = clean(src, cols)
        ys_all = src["label"].astype(int).values

        ind, ood = [], {t: [] for t in TGT}
        for seed in SEEDS:
            tr, te = gsplit(keys, seed)
            m = fit_xgb(Xs_all[tr], ys_all[tr], seed=seed)
            ind.append(roc(ys_all[te], m.predict_proba(Xs_all[te])[:, 1]))
            for t in TGT:
                Xt, _ = clean(data[t], cols, med=med)
                yt = data[t]["label"].astype(int).values
                ood[t].append(roc(yt, m.predict_proba(Xt)[:, 1]))

        r = dict(variant=label, n_feats=len(cols),
                 in_domain=round(float(np.nanmean(ind)), 4),
                 in_domain_std=round(float(np.nanstd(ind)), 4))
        for t in TGT:
            r[f"OOD_{t}"] = round(float(np.nanmean(ood[t])), 4)
        r["OOD_mean"] = round(float(np.nanmean([r[f"OOD_{t}"] for t in TGT])), 4)
        rows.append(r)
        print(f"  {label:44s} feats={len(cols):2d} "
              f"in={r['in_domain']:.4f} OOD={r['OOD_mean']:.4f}")

    out = pd.DataFrame(rows)
    out.to_csv(config.TABLES_DIR / "adapter_ablation.csv", index=False)
    print("\n" + out.to_string(index=False))
    print("\n저장: adapter_ablation.csv")


if __name__ == "__main__":
    main()
