#!/usr/bin/env python
"""[T-IV] 도메인 불변 정규화 (Domain-Invariant Normalization).

문제
----
Adapter가 in-domain은 올리지만 zero-shot OOD는 오히려 낮춘다(0.62→0.52).
정규화된 lateral/gap 피처가 in-domain에서 너무 강력해서 모델이 거기에 의존하는데,
그 피처들의 '스케일·분포'가 도메인마다 달라 전이 시 무너지기 때문이다.

접근
----
(A) 도메인 판별력 진단: 각 피처로 'source vs target'을 분류했을 때의 AUC.
    1.0에 가까울수록 그 피처는 도메인을 그대로 알려주는 = 전이를 망치는 피처.
    (논문 Figure 감. "무엇이 전이를 막는가"를 정량화)

(B) 무차원(dimensionless) 피처셋 DI:
    - 시간 gap      : gap / ego_speed        (거리 m → 시간 s)
    - heading 근사  : lat_speed / ego_speed  (무차원 각도)
    - 차선 정규화   : lat_offset / lane_halfwidth (이미 lat_offset_norm)
    - 상대속도 비율 : ego_speed / domain_median_speed
    - 차선 토폴로지 : lane_id → lane_pos(0~1), is_leftmost, is_rightmost
    - 절대 스케일 피처(원 gap, 원 speed, dhw, lane_id 등) 전면 제외

(C) 도메인별 분위수 정규화(rank normalization):
    각 도메인 안에서 피처를 경험분위수로 변환. 타깃의 '라벨 없는' 통계만 쓰므로
    unsupervised domain adaptation으로 정당하다.

비교: full(기존 48) / DI / DI+rank / full+rank  →  in-domain & zero-shot OOD

실행: python scripts/16_domain_invariant.py
출력: results/tables/domain_discriminability.csv, domain_invariant.csv
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

FULL = KIN + GT
SRC = ["highD", "NGSIM", "MiTra"]
TGT = ["ETRI", "EMT", "uniD", "exiD"]
ALL = SRC + TGT
SEEDS = [42, 0, 1]
H = 3
EPS = 1e-3


# ── 로딩 ─────────────────────────────────────────────────────────────
def load(ds, h=H):
    df = pd.read_csv(config.PROCESSED_DIR / f"{ds}_gt_{h}s.csv")
    df["dataset"] = ds
    return df


def roc(y, p):
    y = np.asarray(y)
    return roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")


def gkey(df):
    return (df["dataset"].astype(str) + "|" + df["recording_id"].astype(str)
            + "|" + df["vehicle_id"].astype(str)).values


def gsplit(keys, seed, frac=0.3):
    u = np.unique(keys)
    rng = np.random.RandomState(seed); rng.shuffle(u)
    te = set(u[:max(1, int(len(u) * frac))])
    m = np.array([k in te for k in keys])
    return ~m, m


# ── (B) 무차원 피처 생성 ─────────────────────────────────────────────
DI_COLS = [
    # 시간 단위 (거리/속도)
    "t_gap_lf", "t_gap_lb", "t_gap_rf", "t_gap_rb", "thw", "ttc",
    # 무차원
    "heading_approx", "heading_std5", "heading_max5",
    "lat_offset_norm", "lat_drift_norm",
    "speed_ratio_domain", "speed_ratio", "acc_norm",
    "speed_slope_norm", "ttc_slope_norm",
    # 토폴로지 (차선 코딩 비의존)
    "lane_pos", "is_leftmost", "is_rightmost", "n_lanes_norm",
    # 상대 비교 (좌우 비대칭 — 스케일 무관)
    "vos_ratio", "gap_asym",
]


def make_di(df):
    """무차원 피처 생성. df는 단일 도메인(dataset) 단위로 들어와야 한다."""
    d = df.copy()
    v = d["ego_speed"].abs().clip(lower=1.0)          # 0 나눗셈 방지
    vmed = float(np.nanmedian(d["ego_speed"].abs())) or 1.0

    # 시간 gap (m → s)
    for src_c, dst in [("gap_left_front", "t_gap_lf"), ("gap_left_back", "t_gap_lb"),
                       ("gap_right_front", "t_gap_rf"), ("gap_right_back", "t_gap_rb")]:
        d[dst] = (d[src_c] / v).clip(-30, 30)

    # heading 근사 (무차원): 횡속/종속
    d["heading_approx"] = (d["lat_speed"] / v).clip(-1, 1)
    d["heading_std5"]   = (d["lat_speed_std5"] / v).clip(0, 1)
    d["heading_max5"]   = (d["abs_lat_speed_max5"] / v).clip(0, 1)
    d["lat_drift_norm"] = (d["lat_drift"] / v).clip(-1, 1)

    # 속도/가속 정규화 (도메인 중앙값 기준)
    d["speed_ratio_domain"] = d["ego_speed"] / vmed
    d["acc_norm"]           = d["ego_acc"] / (vmed + EPS)
    d["speed_slope_norm"]   = d["ego_speed_slope5"] / (vmed + EPS)
    d["ttc_slope_norm"]     = d["ttc_slope5"] / (np.nanmedian(np.abs(d["ttc_slope5"])) + EPS)

    # 차선 토폴로지 (데이터셋별 lane_id 코딩에 의존하지 않음)
    lmin = float(d["lane_id"].min())
    lmax = float(d["lane_id"].max())
    span = max(lmax - lmin, 1.0)
    d["lane_pos"]     = (d["lane_id"] - lmin) / span          # 0~1
    d["is_leftmost"]  = (d["lane_id"] == lmin).astype(float)
    d["is_rightmost"] = (d["lane_id"] == lmax).astype(float)
    d["n_lanes_norm"] = span / 6.0                            # 도로 규모

    # 좌우 비대칭 (스케일 무관 비율)
    L = d["vos_left"].clip(lower=0).fillna(0)
    R = d["vos_right"].clip(lower=0).fillna(0)
    d["vos_ratio"] = (L - R) / (L + R + EPS)                  # -1 ~ 1
    d["gap_asym"]  = (d["gap_left_front"].fillna(0) - d["gap_right_front"].fillna(0)) \
                     / (d["gap_left_front"].abs().fillna(0)
                        + d["gap_right_front"].abs().fillna(0) + EPS)

    for c in DI_COLS:
        if c in d.columns:
            d[c] = d[c].replace([np.inf, -np.inf], np.nan)
    return d


# ── (C) 도메인별 분위수 정규화 ───────────────────────────────────────
def rank_normalize(d, cols):
    """도메인 내부에서 각 피처를 경험분위수(0~1)로 변환. 라벨 미사용."""
    out = d.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].rank(pct=True, na_option="keep")
    return out


# ── 평가 ─────────────────────────────────────────────────────────────
def clean(df, cols, med=None):
    X = df[cols].replace([np.inf, -np.inf], np.nan)
    if med is None:
        med = X.median()
    return X.fillna(med), med


def evaluate(data, cols, tag):
    src = pd.concat([data[d] for d in SRC], ignore_index=True)
    keys = gkey(src)
    Xs, med = clean(src, cols)
    ys = src["label"].astype(int).values

    ind, ood = [], {t: [] for t in TGT}
    for s in SEEDS:
        tr, te = gsplit(keys, s)
        m = fit_xgb(Xs[tr], ys[tr], seed=s)
        ind.append(roc(ys[te], m.predict_proba(Xs[te])[:, 1]))
        for t in TGT:
            Xt, _ = clean(data[t], cols, med=med)
            yt = data[t]["label"].astype(int).values
            ood[t].append(roc(yt, m.predict_proba(Xt)[:, 1]))

    r = dict(featset=tag, n_feats=len(cols),
             in_domain=round(float(np.nanmean(ind)), 4))
    for t in TGT:
        r[f"OOD_{t}"] = round(float(np.nanmean(ood[t])), 4)
    r["OOD_mean"] = round(float(np.nanmean([r[f"OOD_{t}"] for t in TGT])), 4)
    print(f"  {tag:14s} feats={len(cols):2d}  in={r['in_domain']:.4f}  "
          f"OOD={r['OOD_mean']:.4f}  " +
          " ".join(f"{t}={r[f'OOD_{t}']:.3f}" for t in TGT))
    return r


def main():
    raw = {d: load(d) for d in ALL}
    di  = {d: make_di(raw[d]) for d in ALL}

    # ── (A) 도메인 판별력 진단: source vs target 분류 AUC (피처 단독)
    print("── (A) 도메인 판별력: 각 피처가 source/target을 얼마나 구별하는가 ──")
    src_all = pd.concat([di[d] for d in SRC], ignore_index=True)
    tgt_all = pd.concat([di[d] for d in TGT], ignore_index=True)
    dd_rows = []
    cand = [c for c in FULL + DI_COLS
            if c in src_all.columns and c in tgt_all.columns]
    for c in cand:
        a = src_all[c].replace([np.inf, -np.inf], np.nan).dropna()
        b = tgt_all[c].replace([np.inf, -np.inf], np.nan).dropna()
        if len(a) < 50 or len(b) < 50:
            continue
        n = min(len(a), len(b), 20000)
        y = np.r_[np.zeros(n), np.ones(n)]
        p = np.r_[a.sample(n, random_state=42).values,
                  b.sample(n, random_state=42).values]
        auc = roc(y, p)
        auc = max(auc, 1 - auc)               # 방향 무관
        dd_rows.append(dict(feature=c, domain_auc=round(float(auc), 4),
                            is_DI=c in DI_COLS))
    dd = pd.DataFrame(dd_rows).sort_values("domain_auc", ascending=False)
    dd.to_csv(config.TABLES_DIR / "domain_discriminability.csv", index=False)
    print(dd.head(12).to_string(index=False))
    print(f"\n  기존 피처 평균 domain-AUC = {dd[~dd.is_DI].domain_auc.mean():.4f}")
    print(f"  DI   피처 평균 domain-AUC = {dd[dd.is_DI].domain_auc.mean():.4f}")
    print("  (1.0에 가까울수록 그 피처만 보고도 도메인을 알 수 있다 = 전이 방해)\n")

    # ── (B)(C) 피처셋별 성능
    print("── 피처셋별 in-domain / zero-shot OOD ──")
    full_cols = [c for c in FULL if all(c in raw[d].columns for d in ALL)]
    di_cols   = [c for c in DI_COLS if all(c in di[d].columns for d in ALL)]

    rows = [
        evaluate(raw, full_cols, "full"),
        evaluate(di,  di_cols,   "DI"),
    ]
    # rank 정규화 버전 (도메인별)
    raw_r = {d: rank_normalize(raw[d], full_cols) for d in ALL}
    di_r  = {d: rank_normalize(di[d],  di_cols)   for d in ALL}
    rows.append(evaluate(raw_r, full_cols, "full+rank"))
    rows.append(evaluate(di_r,  di_cols,   "DI+rank"))

    # DI + 게임피처
    gt = [c for c in GT if all(c in raw[d].columns for d in ALL)]
    di_gt = {d: pd.concat([di[d], raw[d][gt]], axis=1) for d in ALL}
    di_gt = {d: df.loc[:, ~df.columns.duplicated()] for d, df in di_gt.items()}
    rows.append(evaluate(di_gt, di_cols + gt, "DI+game"))
    di_gt_r = {d: rank_normalize(di_gt[d], di_cols + gt) for d in ALL}
    rows.append(evaluate(di_gt_r, di_cols + gt, "DI+game+rank"))

    out = pd.DataFrame(rows)
    out.to_csv(config.TABLES_DIR / "domain_invariant.csv", index=False)
    print("\n" + out.to_string(index=False))
    print("\n저장: domain_discriminability.csv, domain_invariant.csv")


if __name__ == "__main__":
    main()
