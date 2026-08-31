#!/usr/bin/env python
"""[리뷰어 대응: 1s horizon control] 위치: scripts/28_horizon1.py

리뷰어 예상: "1초처럼 거의 즉각적인 horizon 에서도 zero-shot 이 실패하나?"
1s(거의 control)에서도 OOD≈chance 면 → 실패가 horizon 길이가 아니라 representation
문제라는 주장이 강해진다. (반대로 1s OOD 가 높게 나오면 신중히 해석 필요.)

within-vehicle(margin 2s) 로 horizon=1s 재빌드 + 게임피처 → in-domain / 4-target
zero-shot OOD + pooled bootstrap 95% CI. 기존 processed 파일은 건드리지 않음.
실행: python scripts/28_horizon1.py
"""
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config
from src.features.kinematic import build_features, FEATURE_COLS as KIN
from src.features.game_theory import compute_game_features, NASH_FEATURE_COLS as GT
from src.features.roadframe import add_road_frame, estimate_lateral_sign
from src.features.sampling import build_samples_within, balance_within, report_purity
from src.models.train import split_by_vehicle, fit_xgb, eval_auc, _xy

COLS = KIN + GT
HSEC = 1
MARGIN = 2.0
N_BOOT = 2000
ALL = config.TRAIN_DATASETS + [config.HOLDOUT_DATASET] + config.OOD_DATASETS
TARGETS = [config.HOLDOUT_DATASET] + config.OOD_DATASETS


def lat_sign(files, events, fps, mx=15):
    sc = 0.0
    for cf in files[:mx]:
        c = pd.read_parquet(cf, columns=["recording_id","vehicle_id","frame","x","y","lane_id"])
        s, n = estimate_lateral_sign(add_road_frame(c), events, fps=fps); sc += s*n
    return 1.0 if sc >= 0 else -1.0


def build(ds):
    cdir = config.INTERIM_DIR / "canonical" / ds
    events = pd.read_csv(config.INTERIM_DIR / f"events_{ds}.csv")
    files = sorted(cdir.glob("*.parquet"))
    fps = config.FPS[ds]; H = HSEC*fps; ls = lat_sign(files, events, fps)
    parts = []
    for cf in files:
        canon = pd.read_parquet(cf)
        s = build_samples_within(canon, events, H, fps=fps, margin_sec=MARGIN,
                                 neg_per_pos=3, require_both=True,
                                 random_state=config.RANDOM_STATE)
        if len(s)==0: continue
        s = balance_within(s, 1.0, config.RANDOM_STATE)
        if len(s)==0: continue
        feat = build_features(s, canon, fps=fps, lat_sign=ls, use_road_frame=True)
        parts.append(compute_game_features(feat, canon))
    if not parts: return None
    d = pd.concat(parts, ignore_index=True); d["dataset"]=ds; return d


def clean(df, cols): return df[cols].replace([np.inf,-np.inf], np.nan)
def roc(y,p):
    y=pd.Series(y); return roc_auc_score(y,p) if y.nunique()>1 else float("nan")
def boot(y,p,n=N_BOOT,seed=42):
    rng=np.random.RandomState(seed); y=np.asarray(y); p=np.asarray(p); a=[]
    for _ in range(n):
        i=rng.randint(0,len(y),len(y))
        if len(np.unique(y[i]))<2: continue
        a.append(roc_auc_score(y[i],p[i]))
    return np.mean(a), np.percentile(a,2.5), np.percentile(a,97.5)


def main():
    print(f"horizon={HSEC}s margin={MARGIN}s")
    data={}
    for ds in tqdm(ALL, desc="build 1s"):
        d=build(ds)
        if d is None: print(f"  {ds} 샘플없음"); continue
        data[ds]=d; rep=report_purity(d)
        print(f"  {ds:6s} rows={rep['n_rows']} purity={rep['group_label_purity']:.3f}")
    cols=[c for c in COLS if all(c in data[ds].columns for ds in data)]

    trains,tests=[],{}
    for ds in config.TRAIN_DATASETS:
        tr,te=split_by_vehicle(data[ds], seed=config.RANDOM_STATE)
        trains.append(tr); tests[ds]=te
    m=fit_xgb(*_xy(pd.concat(trains,ignore_index=True), cols), seed=config.RANDOM_STATE)
    indom=np.mean([eval_auc(m,tests[ds],cols) for ds in config.TRAIN_DATASETS])

    rows=[]; ys=[]; ps=[]
    for ds in TARGETS:
        te=data[ds]; y=te["label"].astype(int).values
        p=m.predict_proba(clean(te,cols))[:,1]; ys.append(y); ps.append(p)
        mean,lo,hi=boot(y,p)
        rows.append(dict(target=ds,n=len(y),auc=round(float(mean),4),
                         ci_low=round(float(lo),4),ci_high=round(float(hi),4)))
        print(f"  OOD {ds:6s} AUC={mean:.4f} [{lo:.4f},{hi:.4f}]")
    ya=np.concatenate(ys); pa=np.concatenate(ps); mean,lo,hi=boot(ya,pa)
    rows.append(dict(target="POOLED",n=len(ya),auc=round(float(mean),4),
                     ci_low=round(float(lo),4),ci_high=round(float(hi),4)))
    print(f"\n  in-domain(1s)={indom:.4f}  |  POOLED OOD={mean:.4f} [{lo:.4f},{hi:.4f}]")
    pd.DataFrame(rows).to_csv(config.TABLES_DIR/"horizon1.csv", index=False)
    pd.DataFrame([dict(horizon="1s",in_domain=round(float(indom),4),
                       ood_pooled=round(float(mean),4))]).to_csv(
                       config.TABLES_DIR/"horizon1_summary.csv", index=False)
    print("저장: horizon1.csv, horizon1_summary.csv")


if __name__ == "__main__":
    main()
