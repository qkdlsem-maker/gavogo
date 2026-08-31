#!/usr/bin/env python
"""[2단계] canonical + 이벤트 → 피처.
within-vehicle 샘플링 + Adapter 구성요소 on/off (Adapter ablation 지원).

Adapter 구성요소
  --no_neighbors   : 이웃 재구성(compute_side_neighbors) 결과를 버림
                     → NGSIM/ETRI/EMT/uniD는 좌우 이웃이 사라짐 (gap/VOS 소실)
  --no_roadframe   : 차선 중심선 복원 없이 전역 y를 횡방향으로 사용
  --no_latsign     : 횡방향 부호 정규화(lane_id 방향 정렬) 생략

논문 Adapter ablation 조합
  V1 Raw            : --no_neighbors --no_roadframe --no_latsign --suffix _V1
  V2 +NeighborRecon :                --no_roadframe --no_latsign --suffix _V2
  V3 +Normalization :                                             --suffix _V3
  V4 Full GAVOGO    : V3 + 03_build_game_features.py

실행: python scripts/02_build_features_v2.py --dataset highD
"""
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config
from src.features.kinematic import build_features, FEATURE_COLS
from src.features.roadframe import add_road_frame, estimate_lateral_sign
from src.features.sampling import build_samples_within, balance_within, report_purity

META = ["recording_id", "vehicle_id", "frame", "label"]

# 원본에 좌우 이웃이 없어 Adapter가 재구성해준 데이터셋
RECONSTRUCTED = {"NGSIM", "ETRI", "EMT", "uniD"}
SIDE_ID_COLS = ["left_preceding_id", "left_alongside_id", "left_following_id",
                "right_preceding_id", "right_alongside_id", "right_following_id"]


def dataset_lateral_sign(files, events, fps, max_files=15):
    score = 0.0
    for cf in files[:max_files]:
        canon = pd.read_parquet(cf, columns=["recording_id", "vehicle_id",
                                             "frame", "x", "y", "lane_id"])
        s, n = estimate_lateral_sign(add_road_frame(canon), events, fps=fps)
        score += s * n
    sign = 1.0 if score >= 0 else -1.0
    print(f"  [lateral sign] score={score:+.0f} → sign={sign:+.0f}")
    return sign


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--neg_per_pos", type=int, default=3)
    ap.add_argument("--margin_sec", type=float, default=2.0)
    ap.add_argument("--balance", type=float, default=1.0)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--no_neighbors", action="store_true")
    ap.add_argument("--no_roadframe", action="store_true")
    ap.add_argument("--no_latsign", action="store_true")
    a = ap.parse_args()

    cdir = config.INTERIM_DIR / "canonical" / a.dataset
    events = pd.read_csv(config.INTERIM_DIR / f"events_{a.dataset}.csv")
    files = sorted(cdir.glob("*.parquet"))
    assert files, f"canonical 없음: {cdir} (01 먼저)"

    fps = config.FPS[a.dataset]
    use_rf = not a.no_roadframe
    if a.no_latsign:
        lat_sign = 1.0
        print("  [lateral sign] 생략(ablation) → sign=+1")
    else:
        lat_sign = dataset_lateral_sign(files, events, fps)

    drop_nb = a.no_neighbors and (a.dataset in RECONSTRUCTED)
    if a.no_neighbors:
        print(f"  [neighbors] 재구성 이웃 제거 "
              f"({'적용' if drop_nb else '해당없음 — 원본이 이웃 제공'})")

    for hsec in config.HORIZONS_SEC:
        H = hsec * fps
        parts = []
        for cf in tqdm(files, desc=f"{a.dataset}{a.suffix} {hsec}s"):
            canon = pd.read_parquet(cf)
            if drop_nb:
                for c in SIDE_ID_COLS:
                    if c in canon.columns:
                        canon[c] = 0
            s = build_samples_within(
                canon, events, H, fps=fps,
                margin_sec=a.margin_sec, neg_per_pos=a.neg_per_pos,
                require_both=True, random_state=config.RANDOM_STATE)
            if len(s) == 0:
                continue
            s = balance_within(s, a.balance, config.RANDOM_STATE)
            if len(s) == 0:
                continue
            parts.append(build_features(s, canon, fps=fps, lat_sign=lat_sign,
                                        use_road_frame=use_rf))

        if not parts:
            print(f"  [{hsec}s] 샘플 없음")
            continue

        ds = pd.concat(parts, ignore_index=True)
        rep = report_purity(ds)
        keep = META + [c for c in FEATURE_COLS if c in ds.columns]
        out = config.PROCESSED_DIR / f"{a.dataset}{a.suffix}_{hsec}s.csv"
        ds[keep].to_csv(out, index=False)
        flag = "OK" if rep["group_label_purity"] == 0.0 else "<<< purity != 0 !!"
        print(f"  [{hsec}s] rows={rep['n_rows']} veh={rep['n_groups']} "
              f"pos={rep['pos_rate']:.3f} purity={rep['group_label_purity']:.3f} {flag} "
              f"→ {out.name}")


if __name__ == "__main__":
    main()
