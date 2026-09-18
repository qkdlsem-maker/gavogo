#!/usr/bin/env python
"""[리뷰어 대응: Leakage 정량화] 위치: scripts/26_leakage_quant.py

purity 하나로는 "그래서 leakage가 실제로 얼마나 학습 가능한가?"에 답이 약함.
두 개의 직접 증거를 숫자로 만든다.

(A) Domain classifier (도메인 분리도)
    48개 피처로 "이 샘플이 어느 데이터셋인가"를 맞히는 다중분류기.
    group split(vehicle 단위). 높은 정확도/AUC = 도메인이 자명하게 분리됨
    → label 분류기가 도메인 고유 단서에 붙는 게 자연스러움 → zero-shot 실패의 기전.
    → H-divergence 계열의 model-based 근거. bootstrap CI 포함.

(B) Vehicle-identity 라벨 누수 (target-encoding AUC)
    vehicle_id 만으로 label 을 얼마나 맞힐 수 있나?
    out-of-fold group-mean-label 인코딩의 단일 스코어 AUC.
    - within-vehicle(_gt) 데이터: ≈0.5 여야 함 (purity=0 을 'learnable AUC'로 재확인).
    - 관습적(cross-vehicle) 데이터가 있으면(dataset_{h}s.csv 등) ≈1.0 로 대비.
      → "관습 프로토콜에서는 정체성만으로 label 이 거의 결정됨" 을 직접 증명.

실행: python scripts/26_leakage_quant.py
출력: results/tables/leakage_domain.csv, leakage_identity.csv
"""
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config
from src.features.kinematic import FEATURE_COLS as KIN
from src.features.game_theory import NASH_FEATURE_COLS as GT

COLS = KIN + GT
H = 3
ALL = config.TRAIN_DATASETS + [config.HOLDOUT_DATASET] + config.OOD_DATASETS
PER = 4000            # domain classifier: 데이터셋당 최대 표본 (불균형 방지)
N_BOOT = 2000


def load_gt(ds):
    df = pd.read_csv(config.PROCESSED_DIR / f"{ds}_gt_{H}s.csv"); df["dataset"] = ds
    return df


def clean(df, cols):
    return df[cols].replace([np.inf, -np.inf], np.nan)


def gkeys(df):
    return (df["dataset"].astype(str) + "|" + df["recording_id"].astype(str)
            + "|" + df["vehicle_id"].astype(str)).values


def group_split(keys, seed=42, frac=0.3):
    u = np.unique(keys); rng = np.random.RandomState(seed); rng.shuffle(u)
    te = set(u[:max(1, int(len(u) * frac))])
    m = np.array([k in te for k in keys])
    return ~m, m


def boot_ci(fn, y, p, n=N_BOOT, seed=42):
    rng = np.random.RandomState(seed); vals = []
    y = np.asarray(y)
    for _ in range(n):
        idx = rng.randint(0, len(y), len(y))
        try:
            vals.append(fn(y[idx], p[idx]))
        except Exception:
            continue
    return float(np.mean(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# ═══════════════════ (A) Domain classifier ═══════════════════
def domain_classifier():
    import xgboost as xgb
    parts = []
    for i, ds in enumerate(ALL):
        d = load_gt(ds)
        d = d.sample(min(PER, len(d)), random_state=config.RANDOM_STATE).copy()
        d["dom"] = i
        parts.append(d)
    D = pd.concat(parts, ignore_index=True)
    cols = [c for c in COLS if c in D.columns]
    keys = gkeys(D)
    tr, te = group_split(keys, seed=config.RANDOM_STATE)
    X = clean(D, cols).fillna(clean(D, cols).median())
    yd = D["dom"].values

    clf = xgb.XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
                            tree_method="hist", num_class=len(ALL),
                            objective="multi:softprob", n_jobs=-1,
                            random_state=config.RANDOM_STATE, verbosity=0)
    clf.fit(X[tr], yd[tr])
    proba = clf.predict_proba(X[te])
    pred = proba.argmax(1)
    acc = accuracy_score(yd[te], pred)
    chance = 1.0 / len(ALL)

    rows = []
    for i, ds in enumerate(ALL):
        ybin = (yd[te] == i).astype(int)
        auc = roc_auc_score(ybin, proba[:, i]) if ybin.sum() > 0 else float("nan")
        rows.append(dict(dataset=ds, ovr_auc=round(auc, 4)))
        print(f"  (A) {ds:6s} one-vs-rest AUC = {auc:.4f}")
    # accuracy bootstrap CI
    corr = (pred == yd[te]).astype(int)
    m, lo, hi = boot_ci(lambda a, b: b.mean(), corr, corr)
    print(f"\n  (A) domain accuracy = {acc:.4f}  95% CI [{lo:.4f}, {hi:.4f}]  (chance={chance:.4f})")
    rows.append(dict(dataset="__accuracy__", ovr_auc=round(acc, 4)))
    rows.append(dict(dataset="__acc_ci_low__", ovr_auc=round(lo, 4)))
    rows.append(dict(dataset="__acc_ci_high__", ovr_auc=round(hi, 4)))
    rows.append(dict(dataset="__chance__", ovr_auc=round(chance, 4)))
    pd.DataFrame(rows).to_csv(config.TABLES_DIR / "leakage_domain.csv", index=False)


# ═══════════════════ (B) Vehicle-identity 라벨 누수 ═══════════════════
def oof_identity_auc(df, n_folds=5, seed=42):
    """out-of-fold group-mean-label 인코딩(vehicle 단위)의 단일 스코어 AUC."""
    df = df.copy()
    df["vkey"] = df["recording_id"].astype(str) + "|" + df["vehicle_id"].astype(str)
    y = df["label"].astype(int).values
    vids = df["vkey"].values
    uniq = np.unique(vids)
    rng = np.random.RandomState(seed); rng.shuffle(uniq)
    fold = {v: i % n_folds for i, v in enumerate(uniq)}
    fold_of = np.array([fold[v] for v in vids])
    score = np.full(len(df), np.nan)
    glob = y.mean()
    for f in range(n_folds):
        tr = fold_of != f; te = fold_of == f
        # 학습 fold에서 vehicle별 평균 label
        tmp = pd.DataFrame({"v": vids[tr], "y": y[tr]})
        enc = tmp.groupby("v")["y"].mean()
        score[te] = [enc.get(v, glob) for v in vids[te]]
    if len(np.unique(y)) < 2:
        return float("nan")
    return roc_auc_score(y, score)


def identity_leakage():
    rows = []
    for ds in ALL:
        wv = oof_identity_auc(load_gt(ds))
        # 관습적(cross-vehicle) 파일이 있으면 대비 계산
        conv_auc = float("nan")
        for cand in [config.PROCESSED_DIR / f"{ds}_{H}s.csv",
                     config.PROCESSED_DIR / f"dataset_{H}s.csv"]:
            if cand.exists() and ds != "dataset":
                try:
                    cdf = pd.read_csv(cand)
                    if {"recording_id", "vehicle_id", "label"}.issubset(cdf.columns):
                        conv_auc = oof_identity_auc(cdf); break
                except Exception:
                    pass
        rows.append(dict(dataset=ds, within_vehicle_id_auc=round(wv, 4),
                         conventional_id_auc=(round(conv_auc, 4) if conv_auc == conv_auc else "n/a")))
        print(f"  (B) {ds:6s} within-vehicle ID-AUC = {wv:.4f}"
              + (f"   | conventional ID-AUC = {conv_auc:.4f}" if conv_auc == conv_auc else ""))
    pd.DataFrame(rows).to_csv(config.TABLES_DIR / "leakage_identity.csv", index=False)


def main():
    print("═══ (A) Domain classifier (separability) ═══")
    domain_classifier()
    print("\n═══ (B) Vehicle-identity label leakage ═══")
    identity_leakage()
    print("\n저장: leakage_domain.csv, leakage_identity.csv")
    print("\n해석:")
    print("  (A) 도메인 accuracy가 chance보다 훨씬 높으면 → 도메인 자명 분리 → zero-shot 실패 기전.")
    print("  (B) within-vehicle ID-AUC≈0.5 → 정체성으로 label 못 맞힘(누수 제거 확인).")
    print("      conventional ID-AUC가 1.0 근처면 → 관습 프로토콜의 누수를 직접 정량화.")


if __name__ == "__main__":
    main()
