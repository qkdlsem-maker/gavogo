#!/usr/bin/env python
"""[리뷰어 W7: 도로유형 교란 분리] 위치: scripts/24_roadtype.py

리뷰어 지적: Train은 전부 Highway, Target에 Urban(EMT,uniD)이 섞여 있음.
  → zero-shot 실패가 '누수 제거' 때문인지 'Highway vs Urban 근본 차이' 때문인지 분리 안 됨.

두 분석으로 교란을 분리:
  (A) highway-train zero-shot 을 target 도로유형별로 묶어서 보고.
      exiD 는 Highway 라, Highway→Highway 인데도 chance 면
      "도로유형이 원인"이라는 대안설명이 기각된다.
  (B) Highway 데이터셋끼리만 LODO (highD/NGSIM/MiTra/exiD 중 하나씩 hold-out).
      도로유형을 완전히 고정한 순수 Highway↔Highway cross-domain.
      이것도 chance 근처면 → 실패는 도로유형이 아니라 도메인갭.

기존 processed CSV(_gt_{h}s.csv) 사용, 재빌드 없음.
실행: python scripts/24_roadtype.py

주의: ROAD_TYPE 딕셔너리 값(특히 ETRI)이 맞는지 확인하고 필요시 수정할 것.
"""
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config
from src.features.kinematic import FEATURE_COLS as KIN
from src.features.game_theory import NASH_FEATURE_COLS as GT
from src.models.train import fit_xgb

COLS = KIN + GT
H = 3

# ⚠️ 확인 필요: ETRI 가 고속도로면 highway, 도심이면 urban 으로 고쳐라.
ROAD_TYPE = {
    "highD": "highway", "NGSIM": "highway", "MiTra": "highway",
    "exiD":  "highway",              # highway + ramp
    "ETRI":  "highway",              # <<< 확인
    "EMT":   "urban", "uniD": "urban",
}
HIGHWAY = [d for d, t in ROAD_TYPE.items() if t == "highway"]


def load(ds):
    df = pd.read_csv(config.PROCESSED_DIR / f"{ds}_gt_{H}s.csv"); df["dataset"] = ds
    return df


def clean(df, cols):
    return df[cols].replace([np.inf, -np.inf], np.nan)


def roc(y, p):
    y = pd.Series(y)
    return roc_auc_score(y, p) if y.nunique() > 1 else float("nan")


def main():
    print("ROAD_TYPE:", ROAD_TYPE, "\n")

    # ── (A) highway-train zero-shot, target 도로유형별 ──────────────────
    src = pd.concat([load(d) for d in config.TRAIN_DATASETS], ignore_index=True)
    targets = [config.HOLDOUT_DATASET] + config.OOD_DATASETS   # ETRI,EMT,uniD,exiD
    tgt = {d: load(d) for d in targets}
    cols = [c for c in COLS if c in src.columns and all(c in tgt[d].columns for d in tgt)]
    m = fit_xgb(clean(src, cols), src["label"].astype(int), seed=config.RANDOM_STATE)

    rowsA = []
    for d in targets:
        te = tgt[d]
        auc = roc(te["label"].astype(int), m.predict_proba(clean(te, cols))[:, 1])
        rowsA.append(dict(target=d, road_type=ROAD_TYPE[d], n=len(te), auc=round(auc, 4)))
        print(f"  (A) {d:6s} [{ROAD_TYPE[d]:7s}] AUC={auc:.4f}  (n={len(te)})")
    A = pd.DataFrame(rowsA)
    print("\n  ── target 도로유형별 평균 (sample-weighted) ──")
    for rt in ["highway", "urban"]:
        sub = A[A.road_type == rt]
        if len(sub):
            w = np.average(sub.auc, weights=sub.n)
            print(f"    {rt:7s}: mean AUC={w:.4f}  ({', '.join(sub.target)})")
    A.to_csv(config.TABLES_DIR / "roadtype_zeroshot.csv", index=False)

    # ── (B) Highway-only LODO (도로유형 고정) ──────────────────────────
    print(f"\n  ── (B) Highway-only LODO  (domains: {HIGHWAY}) ──")
    data = {d: load(d) for d in HIGHWAY}
    cols2 = [c for c in COLS if all(c in data[d].columns for d in HIGHWAY)]
    rowsB = []
    for test_ds in HIGHWAY:
        tr = pd.concat([data[d] for d in HIGHWAY if d != test_ds], ignore_index=True)
        te = data[test_ds]
        mm = fit_xgb(clean(tr, cols2), tr["label"].astype(int), seed=config.RANDOM_STATE)
        auc = roc(te["label"].astype(int), mm.predict_proba(clean(te, cols2))[:, 1])
        rowsB.append(dict(test_domain=test_ds, n=len(te), auc=round(auc, 4)))
        print(f"    test={test_ds:6s} (train=other highway)  OOD-AUC={auc:.4f}")
    B = pd.DataFrame(rowsB)
    print(f"\n    Highway↔Highway 평균 OOD-AUC = {np.average(B.auc, weights=B.n):.4f}")
    B.to_csv(config.TABLES_DIR / "roadtype_highway_lodo.csv", index=False)

    print("\n저장: roadtype_zeroshot.csv, roadtype_highway_lodo.csv")
    print("\n해석 가이드:")
    print("  - Highway→Highway(exiD, LODO)도 chance면 → 실패는 도로유형이 아니라 도메인갭.")
    print("  - Highway는 되는데 Urban만 실패면 → 도로유형 교란이 실재 → 결론 범위 제한 필요.")


if __name__ == "__main__":
    main()
