#!/usr/bin/env python
"""[5단계 v2] Cross-domain adaptation — group-leakage 차단 + adaptation curve.

수정 이력
  (1) row-level permutation → (recording_id, vehicle_id) 그룹 분할
  (2) fine-tune을 "source+target 재학습"이 아니라 진짜 continued training으로 변경
      (source 부스터를 초기값으로 target subset에서 추가 학습, early stopping)
      → 리뷰 #3이 요구한 protocol(epoch/optimizer/lr/early stopping)을 명세 가능
  (3) adaptation curve 전체(0/10/20/40/60/80/100%)를 논문용 표·그림으로 저장
  (4) multi-seed 평균 ± std

출력:
  results/tables/domain_adapt_v2.csv          (raw, seed별)
  results/tables/adaptation_curve.csv         (논문 Table/Figure용 — 곡선 전체)
  results/tables/domain_adapt_v2_summary.csv  (zeroshot / 60% 요약)
  results/figures/adaptation_curve.png

실행: python scripts/05_domain_adapt_v2.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config
from src.features.kinematic import FEATURE_COLS as KIN
from src.features.game_theory import NASH_FEATURE_COLS as GT
from src.models.train import fit_xgb

COLS = KIN + GT
SRC_DATASETS = ["highD", "NGSIM", "MiTra"]
TGT_DATASETS = ["ETRI", "EMT", "uniD", "exiD"]

TEST_GROUP_FRAC = 0.30
FRACS = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
SEEDS = [42, 0, 1]

# ── Fine-tuning protocol (논문에 그대로 기재) ─────────────────
FT_ROUNDS = 200          # 추가 부스팅 라운드 상한
FT_LR = 0.02             # continued training learning rate
FT_EARLY_STOP = 20       # early stopping patience (rounds)
FT_VAL_FRAC = 0.20       # target adapt pool 중 validation 비율 (그룹 단위)


def load(ds, h):
    return pd.read_csv(config.PROCESSED_DIR / f"{ds}_gt_{h}s.csv")


def clean(df, cols, med=None):
    X = df[cols].replace([np.inf, -np.inf], np.nan)
    if med is None:
        med = X.median()
    return X.fillna(med), med


def roc(y, p):
    y = np.asarray(y)
    return roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")


def group_key(df):
    return (df["recording_id"].astype(str) + "|" + df["vehicle_id"].astype(str)).values


def split_groups(keys, frac, seed):
    u = np.unique(keys)
    rng = np.random.RandomState(seed); rng.shuffle(u)
    n = max(1, int(len(u) * frac))
    hold = set(u[:n])
    m = np.array([k in hold for k in keys])
    return ~m, m


def subsample_groups(keys, pool_mask, frac, seed):
    if frac <= 0:
        return np.zeros(len(keys), dtype=bool)
    g = np.unique(keys[pool_mask])
    if frac >= 1.0:
        sel = set(g)
    else:
        rng = np.random.RandomState(seed + 777); rng.shuffle(g)
        sel = set(g[:max(1, int(len(g) * frac))])
    return np.array([k in sel for k in keys]) & pool_mask


def finetune(base_model, Xft, yft, Xval, yval, seed):
    """source 모델을 초기 부스터로 하여 target subset에서 continued training.
    source 데이터는 replay하지 않는다. early stopping은 target validation 기준."""
    dft = xgb.DMatrix(Xft, label=yft)
    params = dict(objective="binary:logistic", eval_metric="auc",
                  eta=FT_LR, max_depth=6, subsample=0.8,
                  colsample_bytree=0.8, tree_method="hist", seed=seed)
    evals = []
    if Xval is not None and len(np.unique(yval)) > 1:
        evals = [(xgb.DMatrix(Xval, label=yval), "val")]
    bst = xgb.train(params, dft, num_boost_round=FT_ROUNDS,
                    xgb_model=base_model.get_booster(),
                    evals=evals,
                    early_stopping_rounds=FT_EARLY_STOP if evals else None,
                    verbose_eval=False)
    return bst


def run_one(src, tgt, cols, seed):
    Xs, med = clean(src, cols)
    ys = src["label"].astype(int).values
    Xt, _ = clean(tgt, cols, med=med)        # target도 source median으로 impute (배포 현실성)
    yt = tgt["label"].astype(int).values

    keys = group_key(tgt)
    pool_m, test_m = split_groups(keys, TEST_GROUP_FRAC, seed)
    if len(np.unique(yt[test_m])) < 2:
        return []

    Xte, yte = Xt[test_m], yt[test_m]
    dte = xgb.DMatrix(Xte)

    base = fit_xgb(Xs, ys, seed=seed)        # source-only 모델
    rows = []

    for frac in FRACS:
        ft_m = subsample_groups(keys, pool_m, frac, seed)
        if frac == 0.0:                       # zero-shot
            auc = roc(yte, base.predict_proba(Xte)[:, 1])
        else:
            # fine-tune subset 안에서 다시 그룹 단위 train/val 분할
            sub_keys = keys[ft_m]
            tr_sm, va_sm = split_groups(sub_keys, FT_VAL_FRAC, seed + 1)
            Xf, yf = Xt[ft_m], yt[ft_m]
            Xtr, ytr = Xf[tr_sm], yf[tr_sm]
            Xva, yva = Xf[va_sm], yf[va_sm]
            if len(np.unique(ytr)) < 2:
                auc = np.nan
            else:
                bst = finetune(base, Xtr, ytr,
                               Xva if len(Xva) > 0 else None,
                               yva if len(Xva) > 0 else None, seed)
                auc = roc(yte, bst.predict(dte))
        rows.append(dict(frac=frac, auc=auc,
                         n_ft_groups=int(len(np.unique(keys[ft_m]))),
                         n_ft_rows=int(ft_m.sum())))
    return rows


def main():
    all_rows = []
    for h in config.HORIZONS_SEC:
        src = pd.concat([load(d, h) for d in SRC_DATASETS], ignore_index=True)
        for tds in TGT_DATASETS:
            try:
                tgt = load(tds, h)
            except FileNotFoundError:
                print(f"  [skip] {tds} {h}s"); continue
            cols = [c for c in COLS if c in src.columns and c in tgt.columns]
            print(f"\n── {tds} | {h}s | rows={len(tgt)} "
                  f"groups={len(np.unique(group_key(tgt)))} feats={len(cols)}")
            for seed in SEEDS:
                for r in run_one(src, tgt, cols, seed):
                    r.update(horizon=h, target=tds, seed=seed)
                    all_rows.append(r)
            sub = pd.DataFrame([r for r in all_rows
                                if r["target"] == tds and r["horizon"] == h])
            s = sub.groupby("frac").auc.mean()
            print("   " + "  ".join(f"{int(f*100):>3d}%={v:.4f}" for f, v in s.items()))

    df = pd.DataFrame(all_rows)
    df.to_csv(config.TABLES_DIR / "domain_adapt_v2.csv", index=False)

    # ── adaptation curve (논문 Table/Figure)
    cur = (df.groupby(["target", "horizon", "frac"])
             .agg(auc_mean=("auc", "mean"), auc_std=("auc", "std"),
                  n_groups=("n_ft_groups", "mean"), n_rows=("n_ft_rows", "mean"))
             .reset_index().round(4))
    cur.to_csv(config.TABLES_DIR / "adaptation_curve.csv", index=False)

    # ── 요약 (zero-shot vs 60%)
    out = []
    for (t, h), g in cur.groupby(["target", "horizon"]):
        def get(f):
            r = g[np.isclose(g.frac, f)]
            return (float(r.auc_mean.iloc[0]), float(r.auc_std.iloc[0])) if len(r) else (np.nan, np.nan)
        zs, zss = get(0.0)
        ft, fts = get(0.6)
        fl, fls = get(1.0)
        out.append(dict(target=t, horizon=f"{h}s", zeroshot=zs, zeroshot_std=zss,
                        finetune60=ft, finetune60_std=fts,
                        finetune100=fl, finetune100_std=fls,
                        gain60=round(ft - zs, 4) if np.isfinite(ft) and np.isfinite(zs) else np.nan))
    summ = pd.DataFrame(out).sort_values(["target", "horizon"])
    summ.to_csv(config.TABLES_DIR / "domain_adapt_v2_summary.csv", index=False)

    # ── 그림 (3s)
    c3 = cur[cur.horizon == 3]
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for t in TGT_DATASETS:
        s = c3[c3.target == t].sort_values("frac")
        if len(s) == 0:
            continue
        ax.errorbar(s.frac * 100, s.auc_mean, yerr=s.auc_std,
                    marker="o", capsize=3, label=t)
    ax.axhline(0.5, ls="--", c="gray", lw=1)
    ax.set_xlabel("Target labels used for fine-tuning (%)")
    ax.set_ylabel("Target-domain ROC-AUC")
    ax.set_title("Adaptation curve (3 s horizon, 3 seeds)")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "adaptation_curve.png", dpi=150)
    plt.close()

    print("\n" + summ.to_string(index=False))
    print("\n저장: adaptation_curve.csv, domain_adapt_v2_summary.csv, adaptation_curve.png")


if __name__ == "__main__":
    main()
