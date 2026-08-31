#!/usr/bin/env python
"""[T-IV] 잔여 리뷰 항목 일괄 처리.

#5 통계검정   : Wilcoxon signed-rank, Friedman, Cohen's d  (모델/피처셋 비교)
#6 Failure    : FP/FN 각 20개 유형 분류 (dense/missing neighbor/merge/ramp/low-speed)
#7 SHAP       : 데이터셋별 feature importance 랭킹 비교
#8 Calibration: in-domain + OOD 에서 ECE / Brier
#9 Runtime    : feature 수 / 학습샘플 수 / batch 크기에 따른 추론 지연 scaling

실행: python scripts/18_tiv_remaining.py
필요: pip install shap lightgbm catboost scipy
출력: results/tables/{stat_tests,failure_analysis,shap_by_dataset,
                     calibration_ood,runtime_scaling}.csv
      results/figures/{shap_by_dataset,calibration_ood,runtime_scaling}.png
"""
import sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as st
from sklearn.metrics import roc_auc_score, brier_score_loss

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config
from src.features.kinematic import FEATURE_COLS as KIN
from src.features.game_theory import NASH_FEATURE_COLS as GT
from src.models.train import fit_xgb

COLS = KIN + GT
SRC = ["highD", "NGSIM", "MiTra"]
TGT = ["ETRI", "EMT", "uniD", "exiD"]
ALL = SRC + TGT
SEEDS = [42, 0, 1, 7, 123, 2024, 5, 11, 77, 999]
H = 3
T = config.TABLES_DIR
F = config.FIGURES_DIR


# ── 공통 ─────────────────────────────────────────────────────────────
def load(ds, h=H):
    df = pd.read_csv(config.PROCESSED_DIR / f"{ds}_gt_{h}s.csv")
    df["dataset"] = ds
    return df


def gkey(df):
    return (df["dataset"].astype(str) + "|" + df["recording_id"].astype(str)
            + "|" + df["vehicle_id"].astype(str)).values


def gsplit(keys, seed, frac=0.3):
    u = np.unique(keys)
    rng = np.random.RandomState(seed); rng.shuffle(u)
    te = set(u[:max(1, int(len(u) * frac))])
    m = np.array([k in te for k in keys])
    return ~m, m


def clean(df, cols, med=None):
    X = df[cols].replace([np.inf, -np.inf], np.nan)
    if med is None:
        med = X.median()
    return X.fillna(med), med


def roc(y, p):
    y = np.asarray(y)
    return roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")


def ece(y, p, bins=10):
    y, p = np.asarray(y), np.asarray(p)
    edges = np.linspace(0, 1, bins + 1)
    e, accs, confs = 0.0, [], []
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1])
        if m.sum() == 0:
            accs.append(np.nan); confs.append((edges[i] + edges[i + 1]) / 2); continue
        a, c = y[m].mean(), p[m].mean()
        e += (m.sum() / len(p)) * abs(a - c)
        accs.append(a); confs.append(c)
    return float(e), np.array(accs), np.array(confs)


# ═══════════════════════════════════════════════════════════════════
# #5 통계 검정
# ═══════════════════════════════════════════════════════════════════
def cohens_d(a, b):
    a, b = np.asarray(a), np.asarray(b)
    na, nb = len(a), len(b)
    s = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / s) if s > 0 else 0.0


def stat_tests():
    print("\n═══ #5 통계 검정 (Wilcoxon / Friedman / Cohen's d) ═══")
    from src.models.baselines import fit_lgbm, fit_catboost

    src = pd.concat([load(d) for d in SRC], ignore_index=True)
    cols = [c for c in COLS if c in src.columns]
    keys = gkey(src)
    X, med = clean(src, cols)
    y = src["label"].astype(int).values

    # 조건별 × seed별 AUC 행렬 (paired)
    conds = {
        "XGBoost":  lambda Xtr, ytr, s: fit_xgb(Xtr, ytr, seed=s),
        "LightGBM": lambda Xtr, ytr, s: fit_lgbm(Xtr, ytr, seed=s),
        "CatBoost": lambda Xtr, ytr, s: fit_catboost(Xtr, ytr, seed=s),
    }
    A = {k: [] for k in conds}
    # 피처셋 비교(게임이론 기여)도 같은 틀로
    B = {"full": [], "no_game": []}
    kin = [c for c in KIN if c in src.columns]

    for s in SEEDS:
        tr, te = gsplit(keys, s)
        for name, fn in conds.items():
            m = fn(X[tr], y[tr], s)
            A[name].append(roc(y[te], m.predict_proba(X[te])[:, 1]))
        mf = fit_xgb(X[tr], y[tr], seed=s)
        B["full"].append(roc(y[te], mf.predict_proba(X[te])[:, 1]))
        Xk = X[kin]
        mk = fit_xgb(Xk[tr], y[tr], seed=s)
        B["no_game"].append(roc(y[te], mk.predict_proba(Xk[te])[:, 1]))

    rows = []
    # Friedman (3개 모델, seed = block)
    fr = st.friedmanchisquare(*[A[k] for k in conds])
    rows.append(dict(test="Friedman (XGB/LGBM/Cat)", statistic=round(fr.statistic, 4),
                     p_value=round(fr.pvalue, 4), effect_size=np.nan,
                     significant=bool(fr.pvalue < 0.05)))
    print(f"  Friedman χ²={fr.statistic:.3f} p={fr.pvalue:.4f}")

    # Wilcoxon pairwise + Cohen's d
    names = list(conds)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = A[names[i]], A[names[j]]
            w = st.wilcoxon(a, b)
            d = cohens_d(a, b)
            rows.append(dict(test=f"Wilcoxon {names[i]} vs {names[j]}",
                             statistic=round(float(w.statistic), 4),
                             p_value=round(float(w.pvalue), 4),
                             effect_size=round(d, 4),
                             significant=bool(w.pvalue < 0.05)))
            print(f"  Wilcoxon {names[i]:9s} vs {names[j]:9s} "
                  f"p={w.pvalue:.4f}  d={d:+.3f}  "
                  f"({np.mean(a):.4f} vs {np.mean(b):.4f})")

    # 게임이론 기여 검정
    w = st.wilcoxon(B["full"], B["no_game"])
    d = cohens_d(B["full"], B["no_game"])
    rows.append(dict(test="Wilcoxon full vs no_game (game theory)",
                     statistic=round(float(w.statistic), 4),
                     p_value=round(float(w.pvalue), 4),
                     effect_size=round(d, 4),
                     significant=bool(w.pvalue < 0.05)))
    print(f"  Wilcoxon full vs no_game  p={w.pvalue:.4f}  d={d:+.3f}  "
          f"({np.mean(B['full']):.4f} vs {np.mean(B['no_game']):.4f})")

    pd.DataFrame(rows).to_csv(T / "stat_tests.csv", index=False)
    pd.DataFrame({**A, **B}).to_csv(T / "stat_tests_raw_auc.csv", index=False)
    print("  저장: stat_tests.csv, stat_tests_raw_auc.csv")


# ═══════════════════════════════════════════════════════════════════
# #6 Failure analysis
# ═══════════════════════════════════════════════════════════════════
def failure_analysis(n_each=20):
    print("\n═══ #6 Failure Analysis (FP/FN 유형 분류) ═══")
    src = pd.concat([load(d) for d in SRC], ignore_index=True)
    cols = [c for c in COLS if c in src.columns]
    keys = gkey(src)
    X, med = clean(src, cols)
    y = src["label"].astype(int).values
    tr, te = gsplit(keys, config.RANDOM_STATE)
    m = fit_xgb(X[tr], y[tr], seed=config.RANDOM_STATE)
    p = m.predict_proba(X[te])[:, 1]

    d = src[te].copy().reset_index(drop=True)
    d["pred"] = p
    d["y"] = y[te]

    # 오류 유형 규칙 (해석 가능한 상황 태그)
    def tag(r):
        tags = []
        nb = sum(int(pd.isna(r.get(c)) or r.get(c) == 0)
                 for c in ["gap_left_front", "gap_left_back",
                           "gap_right_front", "gap_right_back"])
        if nb >= 3:
            tags.append("missing_neighbor")
        if (r.get("vos_left", 0) + r.get("vos_right", 0)) < 20:
            tags.append("dense_traffic")
        if r.get("ttc", 999) < 3:
            tags.append("critical_ttc")
        if r.get("ego_speed", 30) < 8:
            tags.append("low_speed")
        if abs(r.get("lat_offset", 0) or 0) > 1.2:
            tags.append("off_center")
        if abs(r.get("lat_speed", 0) or 0) < 0.05:
            tags.append("no_lateral_cue")
        return "|".join(tags) if tags else "none"

    fp = d[(d.y == 0)].nlargest(n_each, "pred").copy()
    fn = d[(d.y == 1)].nsmallest(n_each, "pred").copy()
    fp["error"] = "FP"; fn["error"] = "FN"
    err = pd.concat([fp, fn], ignore_index=True)
    err["tags"] = err.apply(tag, axis=1)

    keep = ["error", "dataset", "recording_id", "vehicle_id", "frame", "y", "pred",
            "ego_speed", "ttc", "lat_offset", "lat_speed",
            "vos_left", "vos_right", "tags"]
    keep = [c for c in keep if c in err.columns]
    err[keep].round(3).to_csv(T / "failure_analysis.csv", index=False)

    # 유형별 집계
    rows = []
    for e in ["FP", "FN"]:
        sub = err[err.error == e]
        cnt = {}
        for t in sub.tags:
            for x in t.split("|"):
                cnt[x] = cnt.get(x, 0) + 1
        for k, v in sorted(cnt.items(), key=lambda z: -z[1]):
            rows.append(dict(error=e, category=k, count=v,
                             pct=round(100 * v / len(sub), 1)))
            print(f"  {e}  {k:18s} {v:2d}  ({100*v/len(sub):.0f}%)")
    pd.DataFrame(rows).to_csv(T / "failure_categories.csv", index=False)
    print("  저장: failure_analysis.csv, failure_categories.csv")


# ═══════════════════════════════════════════════════════════════════
# #7 데이터셋별 SHAP
# ═══════════════════════════════════════════════════════════════════
def shap_by_dataset(topk=10):
    print("\n═══ #7 데이터셋별 SHAP 랭킹 ═══")
    import xgboost as xgb
    rank = {}
    for ds in ALL:
        d = load(ds)
        cols = [c for c in COLS if c in d.columns]
        X, _ = clean(d, cols)
        yv = d["label"].astype(int).values
        if len(np.unique(yv)) < 2:
            continue
        m = fit_xgb(X, yv, seed=config.RANDOM_STATE)
        Xs = X.sample(min(2000, len(X)), random_state=config.RANDOM_STATE)
        # XGBoost 내장 TreeSHAP (shap 패키지 불필요, 결과 동일)
        contrib = m.get_booster().predict(
            xgb.DMatrix(Xs), pred_contribs=True)      # (n, F+1), 마지막은 bias
        imp = np.abs(contrib[:, :-1]).mean(0)
        s = pd.Series(imp, index=cols).sort_values(ascending=False)
        rank[ds] = s / s.sum()
        print(f"  {ds:6s}: " + " > ".join(s.head(4).index.tolist()))

    R = pd.DataFrame(rank).fillna(0)
    R["mean"] = R.mean(axis=1)
    R = R.sort_values("mean", ascending=False)
    R.round(4).to_csv(T / "shap_by_dataset.csv")

    top = R.head(topk).drop(columns="mean")
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(top.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(top.columns))); ax.set_xticklabels(top.columns, rotation=45)
    ax.set_yticks(range(len(top))); ax.set_yticklabels(top.index, fontsize=8)
    ax.set_title("Normalized SHAP importance by dataset")
    plt.colorbar(im, ax=ax)
    plt.tight_layout(); plt.savefig(F / "shap_by_dataset.png", dpi=150); plt.close()
    print("  저장: shap_by_dataset.csv, shap_by_dataset.png")


# ═══════════════════════════════════════════════════════════════════
# #8 OOD Calibration
# ═══════════════════════════════════════════════════════════════════
def calibration_ood():
    print("\n═══ #8 Calibration (in-domain + OOD) ═══")
    src = pd.concat([load(d) for d in SRC], ignore_index=True)
    cols = [c for c in COLS if c in src.columns]
    keys = gkey(src)
    X, med = clean(src, cols)
    y = src["label"].astype(int).values
    tr, te = gsplit(keys, config.RANDOM_STATE)
    m = fit_xgb(X[tr], y[tr], seed=config.RANDOM_STATE)

    rows = []
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], "--", c="gray", lw=1, label="Perfect")

    sets = {"in-domain": (X[te], y[te])}
    for t in TGT:
        d = load(t)
        Xt, _ = clean(d, cols, med=med)
        sets[t] = (Xt, d["label"].astype(int).values)

    for name, (Xq, yq) in sets.items():
        p = m.predict_proba(Xq)[:, 1]
        e, accs, confs = ece(yq, p)
        br = brier_score_loss(yq, p)
        au = roc(yq, p)
        rows.append(dict(eval_set=name, auc=round(au, 4),
                         ECE=round(e, 4), Brier=round(br, 4),
                         mean_pred=round(float(p.mean()), 4),
                         base_rate=round(float(yq.mean()), 4)))
        print(f"  {name:10s} AUC={au:.4f}  ECE={e:.4f}  Brier={br:.4f}")
        ax.plot(confs, accs, "o-", ms=4, label=f"{name} (ECE={e:.3f})")

    ax.set_xlabel("Confidence"); ax.set_ylabel("Empirical accuracy")
    ax.set_title("Reliability: in-domain vs OOD")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(F / "calibration_ood.png", dpi=150); plt.close()
    pd.DataFrame(rows).to_csv(T / "calibration_ood.csv", index=False)
    print("  저장: calibration_ood.csv, calibration_ood.png")


# ═══════════════════════════════════════════════════════════════════
# #9 Runtime scaling
# ═══════════════════════════════════════════════════════════════════
def runtime_scaling():
    print("\n═══ #9 Runtime Scaling ═══")
    src = pd.concat([load(d) for d in SRC], ignore_index=True)
    cols = [c for c in COLS if c in src.columns]
    X, _ = clean(src, cols)
    y = src["label"].astype(int).values

    rows = []
    # (a) feature 수
    for nf in [8, 16, 24, 34, len(cols)]:
        nf = min(nf, len(cols))
        c = cols[:nf]
        m = fit_xgb(X[c], y, seed=42)
        Xb = X[c].values[:2000]
        t0 = time.perf_counter()
        for _ in range(5):
            m.predict_proba(Xb)
        ms = (time.perf_counter() - t0) / 5 / len(Xb) * 1000
        rows.append(dict(axis="n_features", value=nf, ms_per_sample=round(ms, 5)))
        print(f"  features={nf:2d}  {ms:.5f} ms/sample")

    # (b) 학습 샘플 수 (모델 크기 영향)
    for frac in [0.1, 0.25, 0.5, 1.0]:
        n = int(len(X) * frac)
        m = fit_xgb(X.iloc[:n], y[:n], seed=42)
        Xb = X.values[:2000]
        t0 = time.perf_counter()
        for _ in range(5):
            m.predict_proba(Xb)
        ms = (time.perf_counter() - t0) / 5 / len(Xb) * 1000
        rows.append(dict(axis="n_train", value=n, ms_per_sample=round(ms, 5)))
        print(f"  n_train={n:6d}  {ms:.5f} ms/sample")

    # (c) batch 크기
    m = fit_xgb(X, y, seed=42)
    for b in [1, 10, 100, 1000, 5000]:
        Xb = X.values[:b]
        reps = 50 if b <= 100 else 5
        t0 = time.perf_counter()
        for _ in range(reps):
            m.predict_proba(Xb)
        ms = (time.perf_counter() - t0) / reps / b * 1000
        rows.append(dict(axis="batch_size", value=b, ms_per_sample=round(ms, 5)))
        print(f"  batch={b:5d}  {ms:.5f} ms/sample")

    df = pd.DataFrame(rows)
    df.to_csv(T / "runtime_scaling.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, a in zip(axes, ["n_features", "n_train", "batch_size"]):
        s = df[df.axis == a]
        ax.plot(s.value, s.ms_per_sample, "o-")
        ax.set_xlabel(a); ax.set_ylabel("ms / sample")
        ax.grid(alpha=0.3)
        if a in ("n_train", "batch_size"):
            ax.set_xscale("log")
    plt.tight_layout(); plt.savefig(F / "runtime_scaling.png", dpi=150); plt.close()
    print("  저장: runtime_scaling.csv, runtime_scaling.png")


def main():
    stat_tests()
    failure_analysis()
    shap_by_dataset()
    calibration_ood()
    runtime_scaling()
    print("\n완료.")


if __name__ == "__main__":
    main()
