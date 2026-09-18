#!/usr/bin/env python
"""[#7 리뷰어 대응] Purity vs AUC-inflation scatter. 위치: scripts/32_purity_auc.py
누수(관습 purity 높음)가 실제 AUC를 부풀렸음을 한눈에. 각 데이터셋에서:
  leaky in-domain AUC(관습 cross-vehicle) − clean in-domain AUC(within-vehicle) = inflation
  x=관습 purity, y=inflation. 관습 샘플은 메모리 빌드(파일 안 만듦).
실행: python scripts/32_purity_auc.py  (matplotlib 필요)"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config
from src.features.kinematic import build_samples, balance, build_features, FEATURE_COLS as KIN
from src.features.game_theory import NASH_FEATURE_COLS as GT
from src.models.train import split_by_vehicle, fit_xgb, eval_auc, _xy
ALL=config.TRAIN_DATASETS+[config.HOLDOUT_DATASET]+config.OOD_DATASETS
H=3
def purity(s):
    k=s["recording_id"].astype(str)+"|"+s["vehicle_id"].astype(str)
    per=s.assign(g=k).groupby("g").label.mean(); return float(((per==0)|(per==1)).mean())
def leaky_indomain_auc(ds):
    """관습 프로토콜 in-domain AUC (row-level split, 누수 노출)."""
    cdir=config.INTERIM_DIR/"canonical"/ds; events=pd.read_csv(config.INTERIM_DIR/f"events_{ds}.csv")
    fps=config.FPS[ds]; hf=H*fps; parts=[]
    for cf in sorted(cdir.glob("*.parquet")):
        canon=pd.read_parquet(cf)
        s=build_samples(canon,events,hf,random_state=config.RANDOM_STATE)
        if len(s): parts.append(build_features(balance(s,1.0,config.RANDOM_STATE),canon))
    if not parts: return np.nan, np.nan
    d=pd.concat(parts,ignore_index=True); pur=purity(d)
    cols=[c for c in KIN if c in d.columns]
    rng=np.random.RandomState(42); m=rng.rand(len(d))<0.7  # row-level split
    mdl=fit_xgb(d[cols][m].replace([np.inf,-np.inf],np.nan), d["label"].astype(int).values[m], seed=42)
    y=d["label"].astype(int).values[~m]
    auc=roc_auc_score(y, mdl.predict_proba(d[cols][~m].replace([np.inf,-np.inf],np.nan))[:,1]) if len(np.unique(y))>1 else np.nan
    return pur, auc
def clean_indomain_auc(ds):
    d=pd.read_csv(config.PROCESSED_DIR/f"{ds}_gt_{H}s.csv")
    cols=[c for c in (KIN+GT) if c in d.columns]
    tr,te=split_by_vehicle(d,seed=config.RANDOM_STATE)
    m=fit_xgb(*_xy(tr,cols),seed=config.RANDOM_STATE)
    return eval_auc(m,te,cols)
def main():
    rows=[]
    for ds in ALL:
        pur,leaky=leaky_indomain_auc(ds); clean=clean_indomain_auc(ds)
        infl=leaky-clean if (leaky==leaky and clean==clean) else np.nan
        rows.append(dict(dataset=ds,conv_purity=round(pur,4),leaky_auc=round(leaky,4),
                         clean_auc=round(clean,4),auc_inflation=round(infl,4)))
        print(f"  {ds:6s} purity={pur:.3f} leaky={leaky:.3f} clean={clean:.3f} inflation={infl:+.3f}")
    df=pd.DataFrame(rows); df.to_csv(config.TABLES_DIR/"purity_auc.csv",index=False)
    fig,ax=plt.subplots(figsize=(5,4))
    ax.scatter(df.conv_purity,df.auc_inflation,s=70,c="#B7472E",edgecolor="k",zorder=3)
    for _,r in df.iterrows():
        ax.annotate(r.dataset,(r.conv_purity,r.auc_inflation),xytext=(4,4),textcoords="offset points",fontsize=8)
    m=df.dropna()
    if len(m)>=2:
        z=np.polyfit(m.conv_purity,m.auc_inflation,1); xs=np.linspace(m.conv_purity.min(),m.conv_purity.max(),50)
        ax.plot(xs,np.polyval(z,xs),"--",c="#2E6FB7",zorder=2)
        r=np.corrcoef(m.conv_purity,m.auc_inflation)[0,1]; ax.set_title(f"Purity vs AUC inflation (r={r:.2f})")
    ax.set_xlabel("Conventional group–label purity"); ax.set_ylabel("In-domain AUC inflation (leaky − leak-free)")
    ax.grid(alpha=.3); plt.tight_layout(); plt.savefig(config.FIGURES_DIR/"purity_auc.png",dpi=200); plt.close()
    print("저장: purity_auc.csv, purity_auc.png")
if __name__=="__main__": main()
