#!/usr/bin/env python
"""[3b] 무차원 게임이론 피처 생성 + Nash solver 진단.

solver가 순수/혼합/퇴화(fallback) 균형 중 무엇을 반환했는지 비율을 집계한다.
→ "균형이 자주 퇴화하는 것 아니냐"는 리뷰 질문에 데이터로 답하기 위함.

실행: python scripts/03b_build_game_di.py --dataset highD
출력: data/processed/{ds}_gtdi_{3,5,7}s.csv
      results/tables/nash_solver_stats.csv  (누적)
"""
import argparse, sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config
from src.features.game_theory_di import compute_game_features, NASH_FEATURE_COLS


def parquet_path(cdir, rid):
    try:
        return cdir / f"{int(rid):02d}.parquet"
    except (ValueError, TypeError):
        return cdir / f"{rid}.parquet"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    a = ap.parse_args()
    cdir = config.INTERIM_DIR / "canonical" / a.dataset

    stat_rows = []
    for hsec in config.HORIZONS_SEC:
        fp = config.PROCESSED_DIR / f"{a.dataset}_{hsec}s.csv"
        if not fp.exists():
            print(f"  [skip] {fp.name}"); continue
        feat = pd.read_csv(fp)
        stats = {"pure": 0, "mixed": 0, "fallback": 0}
        parts = []
        for rid, g in tqdm(feat.groupby("recording_id"), desc=f"{a.dataset} {hsec}s"):
            canon = pd.read_parquet(parquet_path(cdir, rid))
            parts.append(compute_game_features(g, canon, stats=stats))
        m = pd.concat(parts, ignore_index=True)
        out = config.PROCESSED_DIR / f"{a.dataset}_gtdi_{hsec}s.csv"
        m.to_csv(out, index=False)

        tot = sum(stats.values()) or 1
        pct = {k: 100.0 * v / tot for k, v in stats.items()}
        print(f"  [{hsec}s] {len(m)} 샘플 +게임{len(NASH_FEATURE_COLS)} → {out.name}")
        print(f"          Nash solver: pure={pct['pure']:.1f}%  "
              f"mixed={pct['mixed']:.1f}%  fallback={pct['fallback']:.1f}%")
        stat_rows.append(dict(dataset=a.dataset, horizon=hsec, n_games=tot,
                              pure_pct=round(pct["pure"], 2),
                              mixed_pct=round(pct["mixed"], 2),
                              fallback_pct=round(pct["fallback"], 2)))

    sp = config.TABLES_DIR / "nash_solver_stats.csv"
    df = pd.DataFrame(stat_rows)
    if sp.exists():
        old = pd.read_csv(sp)
        old = old[old.dataset != a.dataset]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(sp, index=False)
    print(f"\n저장: nash_solver_stats.csv")


if __name__ == "__main__":
    main()
