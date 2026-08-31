#!/usr/bin/env python
"""[IEEE Figure] 게임피처 domain-discriminability: 절대단위 vs 무차원.
scripts/fig_gamedi_scatter.py — gavogo 루트에서 실행.
game_di_domain_auc.csv (feature,GT_abs,GT_di,delta) 단독 사용."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config
BLUE,RED,GREY="#1F4E79","#C0392B","#8C8C8C"
plt.rcParams.update({"font.size":11,"font.family":"serif","figure.dpi":300,
    "axes.spines.top":False,"axes.spines.right":False})
d=pd.read_csv(config.TABLES_DIR/"game_di_domain_auc.csv")
fig,ax=plt.subplots(figsize=(5.2,4.8),constrained_layout=True)
lim=[0.45,max(0.95,d[["GT_abs","GT_di"]].max().max()+0.03)]
ax.plot(lim,lim,"--",c=GREY,lw=1)                       # y=x
ax.axhline(0.5,ls=":",c=GREY,lw=.8); ax.axvline(0.5,ls=":",c=GREY,lw=.8)
ax.scatter(d.GT_abs,d.GT_di,s=55,c=BLUE,edgecolor="k",lw=.5,zorder=3)
# 가장 leaky한 피처 라벨
for _,r in d.sort_values("GT_abs",ascending=False).head(3).iterrows():
    ax.annotate(r.feature,(r.GT_abs,r.GT_di),xytext=(5,-2),textcoords="offset points",fontsize=8,color=RED)
ax.set_xlabel("Domain-discrimination AUC — absolute-unit payoff")
ax.set_ylabel("Domain-discrimination AUC — dimensionless payoff")
ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
ax.text(.03,.80,"below diagonal:\nleakage removed",transform=ax.transAxes,fontsize=8,color=RED)
fig.savefig(config.FIGURES_DIR/"fig_gamedi_scatter.png",bbox_inches="tight",pad_inches=0.25); plt.close()
print("saved fig_gamedi_scatter.png")
