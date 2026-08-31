#!/usr/bin/env python
"""[T-IV] Adaptation 통제 실험 — source pre-training이 실제로 기여하는가?

리뷰 예상 질문:
  "Fine-tuned 모델이 좋아진 것은 source에서 전이된 것인가,
   아니면 그냥 target 데이터를 봤기 때문인가?"

세 조건을 동일한 target subset / 동일한 group-disjoint test split에서 비교한다.
  (A) source-only   : source로만 학습, target 미사용            → zero-shot
  (B) target-only   : target subset으로만 scratch 학습 (source 미사용)
  (C) source→target : source 부스터에서 continued training      → fine-tuned

  transfer_gain = C - B   > 0 이면 source pre-training이 실제로 기여
                          ≈ 0 이면 target 데이터만으로 충분 (전이 이득 없음)

실행: python scripts/20_adapt_control.py
출력: results/tables/adapt_control.csv
      results/figures/adapt_control.png
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
SRC = ["highD", "NGSIM", "MiTra"]
TGT = ["ETRI", "EMT", "uniD", "exiD"]
TEST_FRAC = 0.30
FRACS = [0.1, 0.2, 0.4, 0.6, 1.0]
SEEDS = [42, 0, 1]
H = 3

FT_ROUNDS, FT_LR, FT_STOP, FT_VAL = 200, 0.02, 20, 0.20


def load(ds):
    return pd.read_csv(config.PROCESSED_DIR / f"{ds}_gt_{H}s.csv")


def clean(df, cols, med=None):
    X = df[cols].replace([np.inf, -np.inf], np.nan)
    if med is None:
        med = X.median()
    return X.fillna(med), med


def roc(y, p):
    y = np.asarray(y)
    return roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")


def gkey(df):
    return (df["recording_id"].astype(str) + "|" + df["vehicle_id"].astype(str)).values


def gsplit(keys, frac, seed):
    u = np.unique(keys)
    rng = np.random.RandomState(seed); rng.shuffle(u)
    hold = set(u[:max(1, int(len(u) * frac))])
    m = np.array([k in hold for k in keys])
    return ~m, m


def sub_groups(keys, pool, frac, seed):
    g = np.unique(keys[pool])
    if frac >= 1.0:
        sel = set(g)
    else:
        rng = np.random.RandomState(seed + 777); rng.shuffle(g)
        sel = set(g[:max(1, int(len(g) * frac))])
    return np.array([k in sel for k in keys]) & pool


PARAMS = dict(objective="binary:logistic", eval_metric="auc", eta=FT_LR,
              max_depth=6, subsample=0.8, colsample_bytree=0.8, tree_method="hist")


def train_boost(Xtr, ytr, Xva, yva, seed, base=None):
    """base=None → scratch 학습 (target-only). base=booster → continued training."""
    p = dict(PARAMS, seed=seed)
    d = xgb.DMatrix(Xtr, label=ytr)
    ev = []
    if Xva is not None and len(Xva) > 0 and len(np.unique(yva)) > 1:
        ev = [(xgb.DMatrix(Xva, label=yva), "val")]
    return xgb.train(p, d, num_boost_round=FT_ROUNDS, xgb_model=base,
                     evals=ev, early_stopping_rounds=FT_STOP if ev else None,
                     verbose_eval=False)


def main():
    src = pd.concat([load(d) for d in SRC], ignore_index=True)
    rows = []

    for t in TGT:
        tgt = load(t)
        cols = [c for c in COLS if c in src.columns and c in tgt.columns]
        Xs, med = clean(src, cols)
        ys = src["label"].astype(int).values
        Xt, _ = clean(tgt, cols, med=med)
        yt = tgt["label"].astype(int).values
        keys = gkey(tgt)
        print(f"\n── {t}  rows={len(tgt)}  groups={len(np.unique(keys))}")

        for seed in SEEDS:
            pool, test = gsplit(keys, TEST_FRAC, seed)
            if len(np.unique(yt[test])) < 2:
                continue
            Xte, yte = Xt[test], yt[test]
            dte = xgb.DMatrix(Xte)

            base = fit_xgb(Xs, ys, seed=seed)
            zs = roc(yte, base.predict_proba(Xte)[:, 1])
            rows.append(dict(target=t, seed=seed, frac=0.0, cond="source_only", auc=zs))

            for fr in FRACS:
                ft = sub_groups(keys, pool, fr, seed)
                sk = keys[ft]
                tr_m, va_m = gsplit(sk, FT_VAL, seed + 1)
                Xf, yf = Xt[ft], yt[ft]
                Xtr, ytr = Xf[tr_m], yf[tr_m]
                Xva, yva = Xf[va_m], yf[va_m]
                n_g = int(len(np.unique(sk)))
                if len(np.unique(ytr)) < 2:
                    continue

                # (B) target-only: scratch
                b_only = train_boost(Xtr, ytr, Xva, yva, seed, base=None)
                a_b = roc(yte, b_only.predict(dte))

                # (C) source → target: continued training
                b_ft = train_boost(Xtr, ytr, Xva, yva, seed, base=base.get_booster())
                a_c = roc(yte, b_ft.predict(dte))

                rows.append(dict(target=t, seed=seed, frac=fr, cond="target_only",
                                 auc=a_b, n_groups=n_g, n_rows=int(ft.sum())))
                rows.append(dict(target=t, seed=seed, frac=fr, cond="src_to_tgt",
                                 auc=a_c, n_groups=n_g, n_rows=int(ft.sum())))

        d = pd.DataFrame([r for r in rows if r["target"] == t])
        piv = d.pivot_table(index="frac", columns="cond", values="auc", aggfunc="mean")
        print(piv.round(4).to_string())

    df = pd.DataFrame(rows)
    summ = (df.groupby(["target", "frac", "cond"])
              .auc.agg(["mean", "std"]).reset_index().round(4))
    wide = summ.pivot_table(index=["target", "frac"], columns="cond", values="mean").reset_index()
    if "src_to_tgt" in wide and "target_only" in wide:
        wide["transfer_gain"] = (wide["src_to_tgt"] - wide["target_only"]).round(4)
    wide.to_csv(config.TABLES_DIR / "adapt_control.csv", index=False)

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), sharey=False)
    for ax, t in zip(axes, TGT):
        s = wide[wide.target == t].sort_values("frac")
        for c, lab in [("target_only", "target-only (scratch)"),
                       ("src_to_tgt", "source \u2192 target (fine-tuned)")]:
            if c in s:
                ax.plot(s.frac * 100, s[c], "o-", label=lab)
        zs = summ[(summ.target == t) & (summ.cond == "source_only")]["mean"]
        if len(zs):
            ax.axhline(float(zs.iloc[0]), ls=":", c="gray", label="zero-shot")
        ax.axhline(0.5, ls="--", c="lightgray", lw=1)
        ax.set_title(t); ax.set_xlabel("target labels (%)"); ax.grid(alpha=0.3)
    axes[0].set_ylabel("Target ROC-AUC")
    axes[0].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "adapt_control.png", dpi=150)
    plt.close()

    print("\n" + wide.to_string(index=False))
    print("\n저장: adapt_control.csv, adapt_control.png")
    print("transfer_gain > 0 → source pre-training이 기여 / ≈ 0 → target 데이터만으로 충분")


if __name__ == "__main__":
    main()
