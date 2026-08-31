#!/usr/bin/env python
"""[2단계] canonical + 이벤트 → 피처 데이터셋 (horizon별).
실행: python scripts/02_build_features.py --dataset highD|NGSIM|MiTra|ETRI|uniD|exiD|EMT"""
import argparse, sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config
from src.features.kinematic import build_samples, balance, build_features, FEATURE_COLS

META = ["recording_id", "vehicle_id", "frame", "label"]


def parquet_path_str(rid):
    """recording_id가 str이면 그대로, int면 02d 포맷."""
    try:
        return f"{int(rid):02d}.parquet"
    except (ValueError, TypeError):
        return f"{rid}.parquet"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--balance", type=float, default=1.0)
    a = ap.parse_args()

    cdir = config.INTERIM_DIR / "canonical" / a.dataset
    events = pd.read_csv(config.INTERIM_DIR / f"events_{a.dataset}.csv")
    files = sorted(cdir.glob("*.parquet"))
    assert files, f"canonical 없음: {cdir} (01 먼저)"

    fps = config.FPS[a.dataset]
    for hsec in config.HORIZONS_SEC:
        H = hsec * fps
        parts = []
        for cf in tqdm(files, desc=f"{hsec}s"):
            canon = pd.read_parquet(cf)
            s = build_samples(canon, events, H, random_state=config.RANDOM_STATE)
            if len(s) == 0:
                continue
            s = balance(s, a.balance, config.RANDOM_STATE)
            parts.append(build_features(s, canon))

        ds = pd.concat(parts, ignore_index=True)

        # 전체 합산 후 최종 balance (exiD처럼 대용량 데이터셋 불균형 방지)
        pos = ds[ds.label == 1]
        neg = ds[ds.label == 0]
        n = min(len(pos), len(neg))
        ds = pd.concat([
            pos.sample(n=n, random_state=config.RANDOM_STATE),
            neg.sample(n=n, random_state=config.RANDOM_STATE),
        ], ignore_index=True)

        keep = META + [c for c in FEATURE_COLS if c in ds.columns]
        ds[keep].to_csv(config.PROCESSED_DIR / f"{a.dataset}_{hsec}s.csv", index=False)
        print(f"  [{hsec}s] {len(ds)} 샘플 "
              f"(LC={int(ds.label.sum())}, LK={int((ds.label==0).sum())}) "
              f"→ {a.dataset}_{hsec}s.csv")


if __name__ == "__main__":
    main()
