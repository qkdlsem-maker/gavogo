#!/usr/bin/env python
"""[IEEE Figure - SHAP] scripts/fig_shap_ieee.py — gavogo 루트에서 실행.
xgboost 3.3.0 호환: shap.TreeExplainer 대신 booster.predict(pred_contribs=True) 사용.
- fig_shap_beeswarm.png : Top-5 볼드 강조
- fig_shap_by_dataset.png : dataset별 색 구분"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config
from src.features.kinematic import FEATURE_COLS as KIN
from src.features.game_theory import NASH_FEATURE_COLS as GT
from src.models.train import fit_xgb, _xy
COLS=KIN+GT; H=3; F=config.FIGURES_DIR
plt.rcParams.update({"font.size":11,"font.family":"serif","figure.dpi":300,"savefig.bbox":"tight"})
DS_COLORS=["#1F4E79","#C0392B","#2E8B57","#E08E0B","#6A3D9A","#17A2A2","#B5651D"]

def load(ds): d=pd.read_csv(config.PROCESSED_DIR/f"{ds}_gt_{H}s.csv"); d["dataset"]=ds; return d
def treeshap(m,X):
    contribs=m.get_booster().predict(xgb.DMatrix(X,feature_names=list(X.columns)),pred_contribs=True)
    return contribs[:,:-1]   # 마지막 열=bias 제거

def main():
    tr=pd.concat([load(d) for d in config.TRAIN_DATASETS],ignore_index=True)
    cols=[c for c in COLS if c in tr.columns]
    X,y=_xy(tr,cols); X=X.fillna(X.median())
    m=fit_xgb(X,y,seed=config.RANDOM_STATE)
    Xs=X.sample(min(3000,len(X)),random_state=config.RANDOM_STATE)
    sv=treeshap(m,Xs)

    # beeswarm 대신 안정적인 mean|SHAP| 수평막대 (Top-15, Top-5 강조)
    imp=np.abs(sv).mean(0); order=np.argsort(imp)[::-1][:15][::-1]
    names=[cols[i] for i in order]; vals=[imp[i] for i in order]
    top5=set(np.argsort(imp)[::-1][:5])
    barcol=["#1F4E79" if order[k] in top5 else "#B8C4D0" for k in range(len(order))]
    fig,ax=plt.subplots(figsize=(5.6,5.0))
    ax.barh(range(len(order)),vals,color=barcol,edgecolor="k",lw=.4)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(names,fontsize=9)
    for k,lbl in enumerate(ax.get_yticklabels()):
        if order[k] in top5: lbl.set_fontweight("bold")
    ax.set_xlabel("Mean |SHAP value|"); ax.spines[["top","right"]].set_visible(False)
    fig.savefig(F/"fig_shap_global.png"); plt.close(); print("saved fig_shap_global.png")

    # Per-dataset bar (dataset별 색)
    byds=config.TRAIN_DATASETS+[config.HOLDOUT_DATASET]+config.OOD_DATASETS
    med=X.median(); top=np.argsort(imp)[::-1][:8]; feats=[cols[i] for i in top]
    rows={}
    for ds in byds:
        d=load(ds); Xd=d[cols].fillna(med); Xd=Xd.sample(min(1500,len(Xd)),random_state=42)
        rows[ds]=np.abs(treeshap(m,Xd)).mean(0)
    fig,ax=plt.subplots(figsize=(7.2,4.2)); w=0.8/len(byds); xp=np.arange(len(feats))
    for i,ds in enumerate(byds):
        ax.bar(xp+i*w,[rows[ds][j] for j in top],w,label=ds,color=DS_COLORS[i%len(DS_COLORS)],edgecolor="k",lw=.3)
    ax.set_xticks(xp+w*len(byds)/2); ax.set_xticklabels(feats,rotation=30,ha="right",fontsize=8)
    ax.set_ylabel("Mean |SHAP|"); ax.legend(frameon=False,ncol=4,fontsize=8); ax.spines[["top","right"]].set_visible(False)
    fig.savefig(F/"fig_shap_by_dataset.png"); plt.close(); print("saved fig_shap_by_dataset.png")

if __name__=="__main__": main()
