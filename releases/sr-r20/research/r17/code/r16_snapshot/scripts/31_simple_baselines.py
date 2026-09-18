#!/usr/bin/env python
"""[#9 리뷰어 대응] LogisticRegression + RandomForest baseline (overfitting 반박용).
위치: scripts/31_simple_baselines.py. in-domain + zero-shot OOD.
실행: python scripts/31_simple_baselines.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config
from src.features.kinematic import FEATURE_COLS as KIN
from src.features.game_theory import NASH_FEATURE_COLS as GT
from src.models.train import split_by_vehicle
COLS=KIN+GT; H=3
TGT=[config.HOLDOUT_DATASET]+config.OOD_DATASETS
def load(ds):
    d=pd.read_csv(config.PROCESSED_DIR/f"{ds}_gt_{H}s.csv"); d["dataset"]=ds; return d
def clean(df,c,med=None):
    X=df[c].replace([np.inf,-np.inf],np.nan)
    med=X.median() if med is None else med
    return X.fillna(med),med
def roc(y,p): 
    y=pd.Series(y); return roc_auc_score(y,p) if y.nunique()>1 else float("nan")
def main():
    trains,tests={},{}
    tr_all=[]
    for ds in config.TRAIN_DATASETS:
        tr,te=split_by_vehicle(load(ds),seed=config.RANDOM_STATE); tr_all.append(tr); tests[ds]=te
    train=pd.concat(tr_all,ignore_index=True)
    cols=[c for c in COLS if c in train.columns]
    Xtr,med=clean(train,cols); ytr=train["label"].astype(int)
    tgt={d:load(d) for d in TGT}
    sc=StandardScaler().fit(Xtr)
    models={"LogReg":LogisticRegression(max_iter=2000,C=1.0),
            "RandomForest":RandomForestClassifier(n_estimators=400,n_jobs=-1,random_state=config.RANDOM_STATE)}
    rows=[]
    for name,mdl in models.items():
        Xin = sc.transform(Xtr) if name=="LogReg" else Xtr
        mdl.fit(Xin,ytr)
        indom=np.mean([roc(tests[d]["label"].astype(int),
              mdl.predict_proba(sc.transform(clean(tests[d],cols,med)[0]) if name=="LogReg" else clean(tests[d],cols,med)[0])[:,1])
              for d in config.TRAIN_DATASETS])
        r=dict(model=name,in_domain=round(float(indom),4))
        oods=[]
        for d in TGT:
            Xo,_=clean(tgt[d],cols,med); Xo=sc.transform(Xo) if name=="LogReg" else Xo
            a=roc(tgt[d]["label"].astype(int),mdl.predict_proba(Xo)[:,1]); r[f"OOD_{d}"]=round(a,4); oods.append(a)
        r["OOD_mean"]=round(float(np.nanmean(oods)),4); rows.append(r)
        print(f"  {name:13s} in={r['in_domain']}  OOD_mean={r['OOD_mean']}")
    pd.DataFrame(rows).to_csv(config.TABLES_DIR/"simple_baselines.csv",index=False)
    print("저장: simple_baselines.csv")
if __name__=="__main__": main()
