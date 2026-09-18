#!/usr/bin/env python
"""[#8 리뷰어 대응] Domain classifier confusion matrix. 위치: scripts/33_domain_confusion.py
48피처로 데이터셋 7-class 분류 → confusion matrix figure. domain separability 시각 증거.
실행: python scripts/33_domain_confusion.py  (xgboost, matplotlib 필요)"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config
from src.features.kinematic import FEATURE_COLS as KIN
from src.features.game_theory import NASH_FEATURE_COLS as GT
COLS=KIN+GT; H=3; PER=4000
ALL=config.TRAIN_DATASETS+[config.HOLDOUT_DATASET]+config.OOD_DATASETS
def load(ds):
    d=pd.read_csv(config.PROCESSED_DIR/f"{ds}_gt_{H}s.csv"); d["dataset"]=ds; return d
def gkey(df):
    return (df["dataset"].astype(str)+"|"+df["recording_id"].astype(str)+"|"+df["vehicle_id"].astype(str)).values
def main():
    import xgboost as xgb
    parts=[]
    for i,ds in enumerate(ALL):
        d=load(ds); d=d.sample(min(PER,len(d)),random_state=config.RANDOM_STATE).copy(); d["dom"]=i; parts.append(d)
    D=pd.concat(parts,ignore_index=True); cols=[c for c in COLS if c in D.columns]
    keys=gkey(D); u=np.unique(keys); rng=np.random.RandomState(42); rng.shuffle(u)
    te=set(u[:int(len(u)*0.3)]); mask=np.array([k in te for k in keys])
    X=D[cols].replace([np.inf,-np.inf],np.nan); X=X.fillna(X.median()); y=D["dom"].values
    clf=xgb.XGBClassifier(n_estimators=400,max_depth=6,learning_rate=0.05,tree_method="hist",
        num_class=len(ALL),objective="multi:softprob",n_jobs=-1,random_state=42,verbosity=0)
    clf.fit(X[~mask],y[~mask]); pred=clf.predict(X[mask])
    acc=accuracy_score(y[mask],pred); print(f"  accuracy={acc:.4f}")
    cm=confusion_matrix(y[mask],pred,normalize="true")
    pd.DataFrame(cm,index=ALL,columns=ALL).to_csv(config.TABLES_DIR/"domain_confusion.csv")
    fig,ax=plt.subplots(figsize=(5.2,4.6))
    im=ax.imshow(cm,cmap="Blues",vmin=0,vmax=1)
    ax.set_xticks(range(len(ALL))); ax.set_xticklabels(ALL,rotation=45,ha="right",fontsize=8)
    ax.set_yticks(range(len(ALL))); ax.set_yticklabels(ALL,fontsize=8)
    for i in range(len(ALL)):
        for j in range(len(ALL)):
            if cm[i,j]>=0.01: ax.text(j,i,f"{cm[i,j]:.2f}",ha="center",va="center",
                fontsize=7,color="white" if cm[i,j]>0.5 else "black")
    ax.set_xlabel("Predicted dataset"); ax.set_ylabel("True dataset")
    ax.set_title(f"Domain classifier confusion (acc={acc:.3f})")
    plt.colorbar(im,fraction=0.046); plt.tight_layout()
    plt.savefig(config.FIGURES_DIR/"domain_confusion.png",dpi=200); plt.close()
    print("저장: domain_confusion.csv, domain_confusion.png")
if __name__=="__main__": main()
