#!/usr/bin/env python
"""[#5 리뷰어 대응] 7-way Leave-One-Dataset-Out. 위치: scripts/30_lodo7.py
각 데이터셋을 한 번씩 hold-out, 나머지 6개로 학습 → zero-shot AUC. 한눈에 보는 표.
실행: python scripts/30_lodo7.py"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config
from src.features.kinematic import FEATURE_COLS as KIN
from src.features.game_theory import NASH_FEATURE_COLS as GT
from src.models.train import fit_xgb
COLS=KIN+GT; H=3
ALL=config.TRAIN_DATASETS+[config.HOLDOUT_DATASET]+config.OOD_DATASETS
def load(ds):
    d=pd.read_csv(config.PROCESSED_DIR/f"{ds}_gt_{H}s.csv"); d["dataset"]=ds; return d
def clean(df,c): return df[c].replace([np.inf,-np.inf],np.nan)
def main():
    data={d:load(d) for d in ALL}
    cols=[c for c in COLS if all(c in data[d].columns for d in ALL)]
    rows=[]
    for test in ALL:
        tr=pd.concat([data[d] for d in ALL if d!=test],ignore_index=True)
        te=data[test]; y=te["label"].astype(int)
        m=fit_xgb(clean(tr,cols),tr["label"].astype(int),seed=config.RANDOM_STATE)
        auc=roc_auc_score(y,m.predict_proba(clean(te,cols))[:,1]) if y.nunique()>1 else float("nan")
        rows.append(dict(held_out=test,n=len(te),auc=round(auc,4)))
        print(f"  held-out={test:6s} (train=other 6)  AUC={auc:.4f}  n={len(te)}")
    df=pd.DataFrame(rows)
    print(f"\n  mean={df.auc.mean():.4f}  min={df.auc.min():.4f}  max={df.auc.max():.4f}")
    df.to_csv(config.TABLES_DIR/"lodo7.csv",index=False); print("저장: lodo7.csv")
if __name__=="__main__": main()
