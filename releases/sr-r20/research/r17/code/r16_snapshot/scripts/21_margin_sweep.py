#!/usr/bin/env python
"""[리뷰어 대응: Margin Sensitivity] 위치: scripts/21_margin_sweep.py

within-vehicle negative sampling의 유일한 하이퍼파라미터인 margin_sec를
{1,2,3,5}s로 바꿔가며 재빌드 → purity / in-domain AUC / OOD(4-target) AUC 측정.

목적: 리뷰어의 "Why 2 s?" 공격 봉쇄.
    → "benchmark is robust to the margin choice" 문장을 뒷받침하는 Table 하나 생성.

주의: canonical parquet + events(01단계 산출물)가 있어야 함.
      실제 processed CSV는 건드리지 않고 전부 in-memory로 처리한다.
      기본 3s horizon만. exiD(93 recording)가 있어 margin당 수 분 소요될 수 있음.

실행: python scripts/21_margin_sweep.py
      python scripts/21_margin_sweep.py --margins 1 2 3 5 --horizon 3
"""
import argparse, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config
from src.features.kinematic import build_features, FEATURE_COLS as KIN
from src.features.game_theory import compute_game_features, NASH_FEATURE_COLS as GT
from src.features.roadframe import add_road_frame, estimate_lateral_sign
from src.features.sampling import (build_samples_within, balance_within,
                                   report_purity)
from src.models.train import split_by_vehicle, fit_xgb, eval_auc, _xy

COLS = KIN + GT
RECONSTRUCTED = {"NGSIM", "ETRI", "EMT", "uniD"}


def dataset_lateral_sign(files, events, fps, max_files=15):
    score = 0.0
    for cf in files[:max_files]:
        canon = pd.read_parquet(cf, columns=["recording_id", "vehicle_id",
                                             "frame", "x", "y", "lane_id"])
        s, n = estimate_lateral_sign(add_road_frame(canon), events, fps=fps)
        score += s * n
    return 1.0 if score >= 0 else -1.0


def build_ds_at_margin(ds, margin_sec, hsec):
    """단일 데이터셋을 주어진 margin으로 within-vehicle 재빌드 (게임피처 포함)."""
    cdir = config.INTERIM_DIR / "canonical" / ds
    events = pd.read_csv(config.INTERIM_DIR / f"events_{ds}.csv")
    files = sorted(cdir.glob("*.parquet"))
    assert files, f"canonical 없음: {cdir}"
    fps = config.FPS[ds]
    H = hsec * fps
    lat_sign = dataset_lateral_sign(files, events, fps)

    parts = []
    for cf in files:
        canon = pd.read_parquet(cf)
        s = build_samples_within(canon, events, H, fps=fps,
                                 margin_sec=margin_sec, neg_per_pos=3,
                                 require_both=True,
                                 random_state=config.RANDOM_STATE)
        if len(s) == 0:
            continue
        s = balance_within(s, 1.0, config.RANDOM_STATE)
        if len(s) == 0:
            continue
        feat = build_features(s, canon, fps=fps, lat_sign=lat_sign,
                              use_road_frame=True)
        feat = compute_game_features(feat, canon)   # +게임피처 (03단계와 동일)
        parts.append(feat)

    if not parts:
        return None
    out = pd.concat(parts, ignore_index=True)
    out["dataset"] = ds
    return out


def clean(df, cols):
    return df[cols].replace([np.inf, -np.inf], np.nan)


def roc(y, p):
    y = pd.Series(y)
    return roc_auc_score(y, p) if y.nunique() > 1 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--margins", type=float, nargs="+", default=[1, 2, 3, 5])
    ap.add_argument("--horizon", type=int, default=3)
    a = ap.parse_args()
    h = a.horizon

    all_ds = config.TRAIN_DATASETS + [config.HOLDOUT_DATASET] + config.OOD_DATASETS
    # config.OOD_DATASETS = [EMT, uniD, exiD]; HOLDOUT = ETRI
    targets = [config.HOLDOUT_DATASET] + config.OOD_DATASETS   # 4-target OOD

    rows = []
    for margin in a.margins:
        print(f"\n{'='*60}\n margin = {margin} s  (horizon {h}s)\n{'='*60}")
        data = {}
        purities = []
        for ds in tqdm(all_ds, desc=f"build m={margin}"):
            d = build_ds_at_margin(ds, margin, h)
            if d is None:
                print(f"  [{ds}] 샘플 없음 — skip")
                continue
            data[ds] = d
            rep = report_purity(d)
            purities.append(rep["group_label_purity"])
            print(f"  [{ds:6s}] rows={rep['n_rows']:6d} veh={rep['n_groups']:5d} "
                  f"purity={rep['group_label_purity']:.3f}")

        # 공통 피처 컬럼
        cols = [c for c in COLS if all(c in data[ds].columns for ds in data)]

        # in-domain: highD+NGSIM+MiTra, group split
        trains, tests = [], {}
        for ds in config.TRAIN_DATASETS:
            tr, te = split_by_vehicle(data[ds], seed=config.RANDOM_STATE)
            trains.append(tr); tests[ds] = te
        train = pd.concat(trains, ignore_index=True)
        m = fit_xgb(*_xy(train, cols), seed=config.RANDOM_STATE)
        indom = np.mean([eval_auc(m, tests[ds], cols) for ds in config.TRAIN_DATASETS])

        # zero-shot OOD: 4 targets
        ood = {}
        for ds in targets:
            if ds not in data:
                ood[ds] = float("nan"); continue
            te = data[ds]
            ood[ds] = roc(te["label"].astype(int),
                          m.predict_proba(clean(te, cols))[:, 1])
        ood_mean = np.nanmean(list(ood.values()))

        row = dict(margin=margin,
                   purity=round(float(np.mean(purities)), 4),
                   in_domain=round(float(indom), 4),
                   OOD_mean=round(float(ood_mean), 4))
        for ds in targets:
            row[f"OOD_{ds}"] = round(float(ood[ds]), 4)
        rows.append(row)
        print(f"  → in-domain={row['in_domain']}  OOD_mean={row['OOD_mean']}")

    res = pd.DataFrame(rows)
    out = config.TABLES_DIR / "margin_sweep.csv"
    res.to_csv(out, index=False)
    print(f"\n저장: {out}")
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
