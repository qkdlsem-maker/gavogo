#!/usr/bin/env python
"""[IEEE Figure 리스타일] scripts/fig_ieee.py — gavogo 루트에서 실행.
zero-shot bar + reliability(12_extras 산출 곡선 없으면 skip) + domain-discriminability scatter."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config
T,F=config.TABLES_DIR,config.FIGURES_DIR
BLUE,RED,GREY="#1F4E79","#C0392B","#8C8C8C"
plt.rcParams.update({"font.size":11,"font.family":"serif","axes.linewidth":0.8,
    "figure.dpi":300,"savefig.bbox":"tight","axes.spines.top":False,"axes.spines.right":False})
def _csv(n): p=T/n; return pd.read_csv(p) if p.exists() else None

def zero_shot_bar():
    d=_csv("baselines_all.csv")
    if d is None: print("skip bar"); return
    m=d["model"]; x=np.arange(len(m)); w=0.38
    fig,ax=plt.subplots(figsize=(6.2,3.6))
    ax.bar(x-w/2,d["in_domain"],w,label="In-domain",color=BLUE,edgecolor="k",lw=.5)
    ax.bar(x+w/2,d["OOD_mean"],w,label="Zero-shot OOD",color=RED,edgecolor="k",lw=.5)
    ax.axhline(0.5,ls="--",c=GREY,lw=1); ax.text(len(m)-.5,0.51,"chance",color=GREY,fontsize=8,ha="right")
    ax.axhline(0.85,ls=":",c="#444",lw=1); ax.text(len(m)-.5,0.86,"deployment ~0.85",color="#444",fontsize=8,ha="right")
    ax.set_xticks(x); ax.set_xticklabels(m,rotation=20,ha="right"); ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0.4,1.0); ax.legend(frameon=False,ncol=2,loc="upper center",bbox_to_anchor=(.5,1.12))
    fig.savefig(F/"fig_zeroshot_bar.png"); plt.close(); print("saved fig_zeroshot_bar.png")

def domain_discriminability():
    dom=_csv("game_di_domain_auc.csv"); imp=_csv("shap_top20.csv")
    if dom is None or imp is None: print("skip scatter"); return
    # dom: feature,GT_abs,GT_di,delta  → GT_abs = 절대단위 domain-AUC
    d=dom[["feature","GT_abs"]].rename(columns={"GT_abs":"dauc"})
    d=d.merge(imp[["feature","mean_abs_shap"]].rename(columns={"mean_abs_shap":"imp"}),on="feature",how="inner").dropna()
    if len(d)<2: print("skip scatter: 매칭 부족"); return
    fig,ax=plt.subplots(figsize=(4.6,4.0))
    ax.scatter(d.dauc,d.imp,s=45,c=BLUE,edgecolor="k",lw=.5,zorder=3)
    z=np.polyfit(d.dauc,d.imp,1); xs=np.linspace(d.dauc.min(),d.dauc.max(),50)
    ax.plot(xs,np.polyval(z,xs),"--",c=RED,lw=1.5)
    r=np.corrcoef(d.dauc,d.imp)[0,1]
    ax.text(.05,.92,f"r = {r:.2f}",transform=ax.transAxes,fontsize=11,color=RED)
    ax.set_xlabel("Domain-discrimination AUC (abs-unit payoff)"); ax.set_ylabel("Predictive importance (|SHAP|)")
    fig.savefig(F/"fig_domain_discriminability.png"); plt.close(); print("saved fig_domain_discriminability.png")

if __name__=="__main__":
    zero_shot_bar(); domain_discriminability()
    print("reliability는 곡선 데이터 CSV가 없어 fig_reliability_standalone.py 로 별도 생성")
