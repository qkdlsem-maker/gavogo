#!/usr/bin/env python
"""[추가실험: SHAP] 설명가능성 분석. 위치: scripts/08_shap.py
joint 학습(highD+NGSIM+MiTra) full 모델에 SHAP → summary/bar 그림 + top20 표.
기대: TTC/speed/gap 상위, nash/urgency 중간 → "게임이론=해석" 스토리 시각 증명.
실행: python scripts/08_shap.py
필요: pip install shap"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config
from src.features.kinematic import FEATURE_COLS as KIN
from src.features.game_theory import NASH_FEATURE_COLS as GT
from src.models.train import fit_xgb, _xy

COLS = KIN + GT
H = 3   # 대표 horizon

def load(ds):
    df = pd.read_csv(config.PROCESSED_DIR / f"{ds}_gt_{H}s.csv"); df["dataset"] = ds
    return df

def main():
    train = pd.concat([load(ds) for ds in config.TRAIN_DATASETS], ignore_index=True)
    cols = [c for c in COLS if c in train.columns]
    X, y = _xy(train, cols)
    X = X.fillna(X.median())
    m = fit_xgb(X, y, seed=config.RANDOM_STATE)

    expl = shap.TreeExplainer(m)
    # 샘플링 (속도)
    Xs = X.sample(min(3000, len(X)), random_state=config.RANDOM_STATE)
    sv = expl.shap_values(Xs)

    fig = config.FIGURES_DIR
    # summary (beeswarm)
    plt.figure()
    shap.summary_plot(sv, Xs, show=False, max_display=20)
    plt.tight_layout(); plt.savefig(fig / "shap_summary.png", dpi=150); plt.close()
    # bar
    plt.figure()
    shap.summary_plot(sv, Xs, plot_type="bar", show=False, max_display=20)
    plt.tight_layout(); plt.savefig(fig / "shap_bar.png", dpi=150); plt.close()

    # top20 표 + 게임피처 표시
    imp = np.abs(sv).mean(0)
    rank = pd.DataFrame({"feature": cols, "mean_abs_shap": imp})
    rank["is_game"] = rank.feature.isin(GT)
    rank = rank.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    rank.head(20).to_csv(config.TABLES_DIR / "shap_top20.csv", index=False)

    print("── SHAP Top 20 ──")
    for i, r in rank.head(20).iterrows():
        tag = "[GAME]" if r.is_game else ""
        print(f"  {i+1:2d}. {r.feature:24s} {r.mean_abs_shap:.4f} {tag}")
    g = rank[rank.is_game]
    print(f"\n게임피처 최고 순위: {rank.index[rank.is_game][0]+1}위 ({g.iloc[0].feature})")
    print("저장: shap_summary.png, shap_bar.png, shap_top20.csv")

if __name__ == "__main__":
    main()
