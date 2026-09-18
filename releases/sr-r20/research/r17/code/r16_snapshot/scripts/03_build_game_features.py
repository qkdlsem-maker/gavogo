#!/usr/bin/env python
"""[3단계] 피처에 게임이론 14개 추가.
실행: python scripts/03_build_game_features.py --dataset highD|NGSIM|MiTra|ETRI|EMT
출력: data/processed/{ds}_gt_{3,5,7}s.csv"""
import argparse, sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config
from src.features.game_theory import compute_game_features, NASH_FEATURE_COLS


def parquet_path(cdir, rid):
    """recording_id가 int면 02d 포맷, str이면 그대로 파일명."""
    try:
        return cdir / f"{int(rid):02d}.parquet"
    except (ValueError, TypeError):
        return cdir / f"{rid}.parquet"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    a = ap.parse_args()

    cdir = config.INTERIM_DIR / "canonical" / a.dataset
    for hsec in config.HORIZONS_SEC:
        fp = config.PROCESSED_DIR / f"{a.dataset}_{hsec}s.csv"
        if not fp.exists():
            print(f"  [skip] {fp.name}"); continue
        feat = pd.read_csv(fp)
        parts = []
        for rid, g in tqdm(feat.groupby("recording_id"), desc=f"{hsec}s"):
            canon = pd.read_parquet(parquet_path(cdir, rid))
            parts.append(compute_game_features(g, canon))
        m = pd.concat(parts, ignore_index=True)
        m.to_csv(config.PROCESSED_DIR / f"{a.dataset}_gt_{hsec}s.csv", index=False)
        print(f"  [{hsec}s] {len(m)} 샘플 +게임{len(NASH_FEATURE_COLS)} → {a.dataset}_gt_{hsec}s.csv")


if __name__ == "__main__":
    main()
