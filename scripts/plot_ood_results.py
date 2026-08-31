#!/usr/bin/env python
"""OOD 실험 결과 시각화 — 논문용 3개 그림.
위치: scripts/plot_ood_results.py
실행: python scripts/plot_ood_results.py
출력: results/figures/ood_comparison.png
              results/figures/ood_adaptation_curves.png
              results/figures/ood_similarity_scatter.png
"""
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config

FIGS = config.FIGURES_DIR
plt.rcParams.update({"font.size": 12, "figure.dpi": 150})

# ── 데이터 로드 ──────────────────────────────────────────
def load_adapt(name):
    return pd.read_csv(config.TABLES_DIR / f"domain_adapt_{name}.csv")

etri  = load_adapt("etri") if (config.TABLES_DIR/"domain_adapt_etri.csv").exists() \
        else pd.read_csv(config.TABLES_DIR / "domain_adapt.csv")
emt   = load_adapt("emt")
unid  = load_adapt("unid")
exid  = load_adapt("exid")

datasets = {
    "ETRI (KR)":  etri,
    "EMT (IT)":   emt,
    "uniD (DE)":  unid,
    "exiD (DE)":  exid,
}
colors = {
    "ETRI (KR)":  "#E07B54",
    "EMT (IT)":   "#5B8DB8",
    "uniD (DE)":  "#6AAF6A",
    "exiD (DE)":  "#A87DC8",
}
horizons = ["3s", "5s", "7s"]

# ── FIGURE 1: OOD 비교 바 차트 ─────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
fig.suptitle("OOD Cross-domain Performance", fontsize=14, fontweight="bold")

for ax, h in zip(axes, horizons):
    names = list(datasets.keys())
    x = np.arange(len(names))
    w = 0.35
    zs_vals = [datasets[n][datasets[n].horizon == h]["zeroshot"].values[0] for n in names]
    ft_vals  = [datasets[n][datasets[n].horizon == h]["std_finetune"].values[0] for n in names]

    b1 = ax.bar(x - w/2, zs_vals, w, label="Zero-shot",
                color=[colors[n] for n in names], alpha=0.45, edgecolor="white")
    b2 = ax.bar(x + w/2, ft_vals,  w, label="Std+FT",
                color=[colors[n] for n in names], alpha=0.95, edgecolor="white")

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_title(f"Horizon {h}", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([n.split()[0] for n in names], fontsize=10)
    ax.set_ylim(0.4, 1.02)
    ax.set_ylabel("ROC-AUC" if h == "3s" else "")
    for bar, v in zip(list(b1) + list(b2), zs_vals + ft_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=7)

patches = [mpatches.Patch(facecolor="gray", alpha=0.45, label="Zero-shot"),
           mpatches.Patch(facecolor="gray", alpha=0.95, label="Std+FT")]
axes[-1].legend(handles=patches, loc="lower right", fontsize=10)
plt.tight_layout()
plt.savefig(FIGS / "ood_comparison.png", bbox_inches="tight")
plt.close()
print("저장: ood_comparison.png")

# ── FIGURE 2: 적응 단계별 곡선 (3s 기준) ───────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
fig.suptitle("Cross-domain Adaptation Curves", fontsize=14, fontweight="bold")
steps = ["zeroshot", "std", "finetune", "std_finetune"]
step_labels = ["Zero-shot", "Standardize", "Fine-tune", "Std+FT"]

for ax, h in zip(axes, horizons):
    for name, df in datasets.items():
        row = df[df.horizon == h].iloc[0]
        vals = [row[s] for s in steps]
        ax.plot(step_labels, vals, marker="o", label=name,
                color=colors[name], linewidth=2, markersize=6)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_title(f"Horizon {h}", fontsize=12)
    ax.set_ylim(0.4, 1.02)
    ax.set_ylabel("ROC-AUC (OOD)" if h == "3s" else "")
    ax.tick_params(axis="x", labelsize=9)
    ax.grid(axis="y", alpha=0.3)

axes[-1].legend(loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig(FIGS / "ood_adaptation_curves.png", bbox_inches="tight")
plt.close()
print("저장: ood_adaptation_curves.png")

# ── FIGURE 3: Domain 유사도 scatter ─────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
fig.suptitle("Zero-shot vs Fine-tuned AUC (Domain Similarity)", fontsize=13, fontweight="bold")

domain_type = {
    "ETRI (KR)":  "Highway",
    "EMT (IT)":   "Urban",
    "uniD (DE)":  "Urban",
    "exiD (DE)":  "Highway",
}
marker_map = {"Highway": "o", "Urban": "s"}

for ax, h in zip(axes, horizons):
    for name, df in datasets.items():
        row = df[df.horizon == h].iloc[0]
        ax.scatter(row["zeroshot"], row["std_finetune"],
                   color=colors[name],
                   marker=marker_map[domain_type[name]],
                   s=120, zorder=5, label=name)
        ax.annotate(name.split()[0], (row["zeroshot"], row["std_finetune"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.plot([0.4, 1.0], [0.4, 1.0], "k--", alpha=0.3, linewidth=0.8)
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.4)
    ax.axvline(0.5, color="gray", linestyle=":", alpha=0.4)
    ax.set_xlabel("Zero-shot AUC", fontsize=10)
    ax.set_ylabel("Std+FT AUC" if h == "3s" else "", fontsize=10)
    ax.set_title(f"Horizon {h}", fontsize=11)
    ax.set_xlim(0.42, 0.82)
    ax.set_ylim(0.65, 1.02)
    ax.grid(alpha=0.25)

# 범례 (마커 타입 설명)
from matplotlib.lines import Line2D
legend_els = [
    Line2D([0], [0], marker="o", color="gray", linestyle="None", markersize=8, label="Highway"),
    Line2D([0], [0], marker="s", color="gray", linestyle="None", markersize=8, label="Urban"),
]
axes[-1].legend(handles=legend_els, loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig(FIGS / "ood_similarity_scatter.png", bbox_inches="tight")
plt.close()
print("저장: ood_similarity_scatter.png")

print("\n완료. results/figures/ 확인.")
