#!/usr/bin/env python
"""[추가실험: Domain Shift] 38피처를 PCA/t-SNE로 2D 시각화. 위치: scripts/10_domain_shift.py
4개 데이터셋을 색으로 구분 → 클러스터가 분리되면 zero-shot 실패를 시각적으로 설명.
실행: python scripts/10_domain_shift.py"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config
from src.features.kinematic import FEATURE_COLS as KIN
from src.features.game_theory import NASH_FEATURE_COLS as GT

COLS = KIN + GT
ALL = ["highD", "NGSIM", "MiTra", "ETRI"]
COLORS = {"highD":"#2E6FB7","NGSIM":"#E08A1E","MiTra":"#3FA46A","ETRI":"#B7472E"}
PER = 800   # 데이터셋당 샘플 수
H = 3

def load(ds):
    return pd.read_csv(config.PROCESSED_DIR / f"{ds}_gt_{H}s.csv")

def main():
    parts, labels = [], []
    for ds in ALL:
        d = load(ds)
        cols = [c for c in COLS if c in d.columns]
        d = d.sample(min(PER, len(d)), random_state=config.RANDOM_STATE)
        parts.append(d[cols].replace([np.inf,-np.inf],np.nan).fillna(d[cols].median()))
        labels += [ds]*len(d)
    X = pd.concat(parts, ignore_index=True).fillna(0)
    labels = np.array(labels)
    Xs = StandardScaler().fit_transform(X)

    # PCA
    pca = PCA(n_components=2).fit_transform(Xs)
    # t-SNE
    tsne = TSNE(n_components=2, perplexity=30, init="pca",
                random_state=config.RANDOM_STATE).fit_transform(Xs)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for emb, ax, title in [(pca, axes[0], "PCA"), (tsne, axes[1], "t-SNE")]:
        for ds in ALL:
            mk = labels == ds
            ax.scatter(emb[mk,0], emb[mk,1], s=8, alpha=0.5, c=COLORS[ds], label=ds)
        ax.set_title(f"{title} of 38 features by dataset"); ax.legend(markerscale=2, fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout(); plt.savefig(config.FIGURES_DIR / "domain_shift.png", dpi=150); plt.close()
    print("저장: domain_shift.png")
    print("해석: 데이터셋별 클러스터가 분리될수록 도메인 갭이 크고 zero-shot이 어려움.")

if __name__ == "__main__":
    main()
