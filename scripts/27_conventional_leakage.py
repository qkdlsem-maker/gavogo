#!/usr/bin/env python
"""[Leakage 대비 — 수정판] 관습 vs within-vehicle identity 라벨 누수.
위치: scripts/27_conventional_leakage.py  (기존 파일 덮어쓰기)

수정: 이전 판은 target-encoding fold 를 '차량 단위'로 나눠 test 차량이 train 에
전혀 없어 인코딩이 항상 global mean → AUC 무조건 0.5(측정 오류).
누수는 (관습 cross-vehicle 샘플링)+(순진한 row-level 분할) 결합에서 발생하므로
vehicle-id 를 row-level 5-fold out-of-fold 로 target-encoding 해 AUC 를 잰다.
  - 관습: 차량 대부분 순수 pos/neg → 같은 차량이 train/test 양쪽 → AUC≈purity(1.0 근처)
  - within: 차량 pos/neg 혼재(purity=0) → AUC≈0.5
실행: python scripts/27_conventional_leakage.py
출력: results/tables/identity_leakage_contrast.csv
"""
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config
from src.features.kinematic import build_samples, balance

ALL = config.TRAIN_DATASETS + [config.HOLDOUT_DATASET] + config.OOD_DATASETS
H = 3


def rowlevel_id_auc(s, n_folds=5, seed=42):
    s = s.reset_index(drop=True)
    vids = (s["recording_id"].astype(str) + "|" + s["vehicle_id"].astype(str)).values
    y = s["label"].astype(int).values
    if len(np.unique(y)) < 2:
        return float("nan")
    rng = np.random.RandomState(seed)
    fold = rng.randint(0, n_folds, len(s))          # row 단위 무작위 분할
    score = np.empty(len(s)); glob = y.mean()
    for f in range(n_folds):
        tr = fold != f; te = fold == f
        enc = pd.DataFrame({"v": vids[tr], "y": y[tr]}).groupby("v")["y"].mean()
        score[te] = [enc.get(v, glob) for v in vids[te]]
    return roc_auc_score(y, score)


def purity(s):
    k = s["recording_id"].astype(str) + "|" + s["vehicle_id"].astype(str)
    per = s.assign(g=k).groupby("g").label.mean()
    return float(((per == 0) | (per == 1)).mean())


def build_conventional(ds):
    cdir = config.INTERIM_DIR / "canonical" / ds
    events = pd.read_csv(config.INTERIM_DIR / f"events_{ds}.csv")
    files = sorted(cdir.glob("*.parquet"))
    fps = config.FPS[ds]; hf = H * fps
    parts = []
    for cf in files:
        canon = pd.read_parquet(cf, columns=["recording_id", "vehicle_id", "frame", "lane_id"])
        s = build_samples(canon, events, hf, random_state=config.RANDOM_STATE)
        if len(s):
            parts.append(balance(s, 1.0, config.RANDOM_STATE))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def load_within(ds):
    return pd.read_csv(config.PROCESSED_DIR / f"{ds}_gt_{H}s.csv")


def main():
    rows = []
    for ds in ALL:
        conv = build_conventional(ds)
        wv = load_within(ds)
        r = dict(dataset=ds,
                 conv_purity=round(purity(conv), 4) if len(conv) else float("nan"),
                 conv_id_auc=round(rowlevel_id_auc(conv), 4) if len(conv) else float("nan"),
                 within_purity=round(purity(wv), 4),
                 within_id_auc=round(rowlevel_id_auc(wv), 4))
        rows.append(r)
        print(f"  {ds:6s} | conv: purity={r['conv_purity']} ID-AUC={r['conv_id_auc']} "
              f"| within: purity={r['within_purity']} ID-AUC={r['within_id_auc']}")
    pd.DataFrame(rows).to_csv(config.TABLES_DIR / "identity_leakage_contrast.csv", index=False)
    print("\n저장: identity_leakage_contrast.csv")
    print("기대: conv ID-AUC 높음(purity 따라 0.9+), within ID-AUC ≈ 0.5.")


if __name__ == "__main__":
    main()
