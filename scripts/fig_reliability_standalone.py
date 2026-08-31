#!/usr/bin/env python
"""[IEEE Figure - Reliability] scripts/fig_reliability_standalone.py — gavogo 루트에서 실행.
곡선 데이터 CSV가 없어 모델을 다시 학습해 in-domain / OOD reliability를 직접 계산·작도.
in-domain 굵은파랑 / OOD 얇은빨강 / y=x 회색."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config
from src.features.kinematic import FEATURE_COLS as KIN
from src.features.game_theory import NASH_FEATURE_COLS as GT
from src.models.train import split_by_vehicle, fit_xgb, _xy
COLS=KIN+GT; H=3; F=config.FIGURES_DIR
BLUE,RED,GREY="#1F4E79","#C0392B","#8C8C8C"
plt.rcParams.update({"font.size":11,"font.family":"serif","figure.dpi":300,"savefig.bbox":"tight",
    "axes.spines.top":False,"axes.spines.right":False})
def load(ds): d=pd.read_csv(config.PROCESSED_DIR/f"{ds}_gt_{H}s.csv"); d["dataset"]=ds; return d
def clean(df,c): return df[c].replace([np.inf,-np.inf],np.nan)
def curve(y,p,bins=10):
    y=np.asarray(y); p=np.asarray(p); ed=np.linspace(0,1,bins+1); cf,ac=[],[]
    for i in range(bins):
        m=(p>=ed[i])&(p<ed[i+1])
        if m.sum()<5: continue
        cf.append(p[m].mean()); ac.append(y[m].mean())
    return cf,ac
def main():
    tr_all,tests=[],{}
    for ds in config.TRAIN_DATASETS:
        tr,te=split_by_vehicle(load(ds),seed=config.RANDOM_STATE); tr_all.append(tr); tests[ds]=te
    train=pd.concat(tr_all,ignore_index=True); cols=[c for c in COLS if c in train.columns]
    m=fit_xgb(*_xy(train,cols),seed=config.RANDOM_STATE)
    # in-domain
    inte=pd.concat(list(tests.values()),ignore_index=True)
    yi=inte["label"].astype(int).values; pi=m.predict_proba(clean(inte,cols))[:,1]
    # OOD (4 target 합)
    tgt=pd.concat([load(d) for d in [config.HOLDOUT_DATASET]+config.OOD_DATASETS],ignore_index=True)
    yo=tgt["label"].astype(int).values; po=m.predict_proba(clean(tgt,cols))[:,1]
    fig,ax=plt.subplots(figsize=(4.2,4.2))
    ax.plot([0,1],[0,1],"--",c=GREY,lw=1,label="perfect")
    cx,cy=curve(yi,pi); ax.plot(cx,cy,"-o",c=BLUE,lw=2.8,ms=5,label="In-domain")
    ox,oy=curve(yo,po); ax.plot(ox,oy,"-s",c=RED,lw=1.3,ms=4,label="Out-of-domain")
    ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy"); ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.set_aspect("equal"); ax.legend(frameon=False,loc="upper left")
    fig.savefig(F/"fig_reliability.png"); plt.close(); print("saved fig_reliability.png")
if __name__=="__main__": main()
