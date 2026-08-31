#!/usr/bin/env python
"""[진단] exiD/uniD fine-tune AUC 폭등 원인 규명 — shortcut feature 탐지.

가설: 라벨이 track-level 속성(lane_id 등)으로 새고 있다.
      → 모델이 '차선변경 의도'가 아니라 '램프차량 vs 본선차량'을 분류 중.

검사 항목:
  (1) 단일 feature AUC 랭킹 (타깃 도메인 내부, 그룹 분할)
  (2) lane_id 분포: label=0 vs label=1
  (3) feature 제거 실험: full / -lane_id / -lane_id-vos / kinematic-only
  (4) vehicle-level 판별성: 차량당 label 평균 → 차량 단위로 라벨이 결정되는가

실행: python scripts/diag_shortcut.py
출력: results/tables/diag_shortcut_{single,ablate,lane}.csv
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
TARGETS = ["exiD", "uniD", "ETRI", "EMT"]
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


def in_domain_auc(df, cols, seed):
    """타깃 도메인 '내부'에서만 학습/평가 (fine-tune 상황을 극단적으로 모사)."""
    X = clean(df, cols)
    y = df["label"].astype(int).values
    tr, te = gsplit(gkey(df), seed)
    if len(np.unique(y[te])) < 2 or len(np.unique(y[tr])) < 2:
        return float("nan")
    m = fit_xgb(X[tr], y[tr], seed=seed)
    return roc(y[te], m.predict_proba(X[te])[:, 1])


def main():
    h = 7  # 가장 이상한 horizon (exiD 0.98)
    single_rows, ablate_rows, lane_rows, vlv_rows = [], [], [], []

    for tds in TARGETS:
        try:
            df = load(tds, h)
        except FileNotFoundError:
            continue
        cols = [c for c in COLS if c in df.columns]
        y = df["label"].astype(int).values
        print(f"\n{'='*60}\n{tds} | {h}s | rows={len(df)} pos={y.sum()} "
              f"groups={len(np.unique(gkey(df)))}")

        # ── (4) vehicle-level 판별성: 차량이 label을 결정하는가?
        g = pd.DataFrame({"g": gkey(df), "y": y})
        per = g.groupby("g").y.agg(["mean", "size"])
        pure = float(((per["mean"] == 0) | (per["mean"] == 1)).mean())
        print(f"  [vehicle purity] label이 차량단위로 100% 결정되는 차량 비율 = {pure:.1%}")
        print(f"  (1.0에 가까울수록 '차량 분류' 문제로 붕괴됨 → intent 예측 아님)")
        vlv_rows.append(dict(target=tds, horizon=h, group_label_purity=round(pure, 4),
                             n_groups=len(per), pos_rate=round(float(y.mean()), 4)))

        # ── (2) lane_id 분포
        if "lane_id" in df.columns:
            l0 = df.loc[y == 0, "lane_id"]
            l1 = df.loc[y == 1, "lane_id"]
            print(f"  [lane_id] label=0 mean={l0.mean():.2f} vals={sorted(l0.unique())[:8]}")
            print(f"  [lane_id] label=1 mean={l1.mean():.2f} vals={sorted(l1.unique())[:8]}")
            lane_rows.append(dict(target=tds, horizon=h,
                                  lane_mean_neg=round(float(l0.mean()), 3),
                                  lane_mean_pos=round(float(l1.mean()), 3),
                                  lane_auc_single=round(float(np.nanmean(
                                      [in_domain_auc(df, ["lane_id"], s) for s in SEEDS])), 4)))

        # ── (1) 단일 feature AUC 랭킹
        print("  [single-feature in-domain AUC] top 8:")
        sing = []
        for c in cols:
            a = float(np.nanmean([in_domain_auc(df, [c], s) for s in SEEDS]))
            sing.append((c, a))
        sing.sort(key=lambda t: -abs(t[1] - 0.5))
        for c, a in sing[:8]:
            flag = "  <<< SHORTCUT 의심" if a > 0.80 else ""
            print(f"      {c:24s} {a:.4f}{flag}")
        for c, a in sing:
            single_rows.append(dict(target=tds, horizon=h, feature=c, auc=round(a, 4)))

        # ── (3) feature 제거 실험
        variants = {
            "full":            cols,
            "-lane_id":        [c for c in cols if c != "lane_id"],
            "-lane -vos":      [c for c in cols if c != "lane_id" and not c.startswith("vos")],
            "kinematic_only":  [c for c in cols if c in KIN],
            "-game":           [c for c in cols if c not in GT],
        }
        # 단일 shortcut 상위 3개 제거 버전 추가
        top3 = [c for c, a in sing[:3]]
        variants["-top3_single"] = [c for c in cols if c not in top3]

        print("  [ablation: 타깃 내부 학습 AUC]")
        for name, cs in variants.items():
            if len(cs) == 0:
                continue
            a = float(np.nanmean([in_domain_auc(df, cs, s) for s in SEEDS]))
            print(f"      {name:16s} ({len(cs):2d} feats)  AUC={a:.4f}")
            ablate_rows.append(dict(target=tds, horizon=h, variant=name,
                                    n_feats=len(cs), auc=round(a, 4)))

    T = config.TABLES_DIR
    pd.DataFrame(single_rows).to_csv(T / "diag_shortcut_single.csv", index=False)
    pd.DataFrame(ablate_rows).to_csv(T / "diag_shortcut_ablate.csv", index=False)
    pd.DataFrame(lane_rows).to_csv(T / "diag_shortcut_lane.csv", index=False)
    pd.DataFrame(vlv_rows).to_csv(T / "diag_shortcut_purity.csv", index=False)
    print("\n저장: diag_shortcut_{single,ablate,lane,purity}.csv")


if __name__ == "__main__":
    main()
