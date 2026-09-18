#!/usr/bin/env python3
"""Isolated R15 correction experiments. Never writes original data/results.

--out must be a new experiment directory under audits; all checkpoints,
sequences, predictions and split manifests are saved there. EMT25fps remains
legacy sensitivity-only input, not a certified physical trajectory benchmark.
"""
import os,sys,json,time,hashlib,argparse,importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,brier_score_loss
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.models.baselines import build_sequences,_ensure_road_frame,SEQ_COLS,fit_lstm
from src.models.metrics import calibration_bins
from src.models.train import fit_xgb
from src.features.kinematic import FEATURE_COLS as KIN,build_features
from src.features.game_theory import NASH_FEATURE_COLS as GT,compute_game_features
from src.data.base_adapter import compute_side_neighbors
SRC=['highD','NGSIM','MiTra'];TGT=['ETRI','EMT','uniD','exiD'];ALL=SRC+TGT;SEEDS=[42,0,1];COLS=KIN+GT
META=['dataset','recording_id','vehicle_id','frame','label'];SIDE=[s+'_'+k+'_id' for s in ['left','right'] for k in ['preceding','alongside','following']]
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(2**23),b''):h.update(b)
 return h.hexdigest()
def load(ds,h=3):
 d=pd.read_csv(ROOT/f'data/processed/{ds}_gt_{h}s.csv');d['dataset']=ds;return d

def split(d,seed):
 k=d.dataset.astype(str)+'|'+d.recording_id.astype(str)+'|'+d.vehicle_id.astype(str)
 u=np.unique(k);np.random.RandomState(seed).shuffle(u);te=k.isin(set(u[:max(1,int(len(u)*.3))])).to_numpy();return ~te,te

def savepred(o,name,d,p,seed,role,model_hash):
 q=d[META].copy();q['recording_id']=q['recording_id'].astype(str);q['score']=np.asarray(p);q['seed']=seed;q['role']=role;q['model_sha256']=model_hash
 q.to_parquet(o/'predictions'/f'{name}.parquet',index=False)
 y=d.label.to_numpy();e,acc,conf=calibration_bins(y,np.asarray(p))
 return dict(run=name,seed=seed,eval_set=role,n=len(q),auc=roc_auc_score(y,p),ece=e,brier=brier_score_loss(y,p),p_one=int((np.asarray(p)==1).sum()),mean_pred=float(np.mean(p)),base_rate=float(np.mean(y)),model_sha256=model_hash)

def init_out(o):
 for d in ['sequences','predictions','models','splits','tables','derived']:(o/d).mkdir(parents=True,exist_ok=True)
 if not (o/'run_config.json').exists():
  import platform,importlib.metadata as im,subprocess
  meta=dict(start=time.strftime('%Y-%m-%dT%H:%M:%S'),python=sys.version,platform=platform.platform(),packages={p:im.version(p) for p in ['numpy','pandas','scikit-learn','xgboost','torch']},git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),seeds=SEEDS,source=SRC,target=TGT,horizon='3s legacy configured units; EMT physical time unresolved',lstm=dict(K=10,epochs=30,hidden=64,batch=512,lr=.001),xgb=dict(trees=500,depth=6,lr=.05,threads=8),limitations='Offline roadframe and event-based lateral sign retained to isolate corrections. EMT coordinates/lanes uncertified. No causal benchmark claim.')
  (o/'run_config.json').write_text(json.dumps(meta,indent=2))

def sequences(o):
 stats=[]
 for ds in ALL:
  d=load(ds);parts=[];print('roadframe',ds,flush=True)
  for p in sorted((ROOT/f'data/interim/canonical/{ds}').glob('*.parquet')):
   c=pd.read_parquet(p);c=_ensure_road_frame(c);parts.append(c[['recording_id','vehicle_id','frame']+SEQ_COLS])
  c=pd.concat(parts,ignore_index=True);del parts
  print('build',ds,'canonical rows',len(c),flush=True)
  fixed,_=build_sequences(d,c,K=10)
  # Exact legacy selection order, but reuse grouped NumPy tracks for efficiency.
  idx=c.set_index(['vehicle_id','frame']).sort_index();tracks={}
  for vid,g in idx.groupby(level=0,sort=False):tracks[vid]=(g.index.get_level_values('frame').to_numpy(),g[SEQ_COLS].to_numpy(),g.recording_id.to_numpy())
  old=np.zeros_like(fixed);mixed=[];count=0
  for i,row in enumerate(d.itertuples()):
   fr,values,recs=tracks[row.vehicle_id];lo=np.searchsorted(fr,row.frame-10,side='right');hi=np.searchsorted(fr,row.frame,side='right');lo=max(lo,hi-10)
   h=values[lo:hi]
   if len(h):old[i,-len(h):]=np.nan_to_num(h)
   n=int((recs[lo:hi]!=row.recording_id).sum());count+=n;mixed.append(n)
  # Validate optimized legacy selection against the original function on real rows.
  ns={};code=(ROOT/'audits/r15_20260917/before/src/models/baselines.py').read_text();exec(compile(code,'legacy_baselines','exec'),ns)
  pick=np.arange(min(50,len(d)));ref,_=ns['build_sequences'](d.iloc[pick],c,K=10)
  np.testing.assert_array_equal(ref,old[pick])
  np.save(o/'sequences'/f'{ds}_before.npy',old);np.save(o/'sequences'/f'{ds}_after.npy',fixed)
  m=d[META].copy();m['foreign_recording_timesteps']=mixed;m['sequence_changed']=np.any(old!=fixed,axis=(1,2));m.to_parquet(o/'sequences'/f'{ds}_audit.parquet',index=False)
  stats.append(dict(dataset=ds,n=len(d),canonical_rows=len(c),mixed_samples=int((np.array(mixed)>0).sum()),foreign_steps=count,changed_samples=int(m.sequence_changed.sum()),optimized_legacy_checked=len(pick)))
  pd.DataFrame(stats).to_csv(o/'tables/sequence_impact.csv',index=False);print(stats[-1],flush=True)
  del c,idx,tracks,fixed,old

def emt(o):
 c=pd.concat([pd.read_parquet(p) for p in sorted((ROOT/'data/interim/canonical/EMT').glob('*.parquet'))],ignore_index=True)
 print('EMT neighbor correction',len(c),flush=True)
 fixed=compute_side_neighbors(c)
 stats=[]
 for col in SIDE:stats.append(dict(column=col,n=len(c),changed=int((c[col]!=fixed[col]).sum())))
 pd.DataFrame(stats).to_csv(o/'tables/emt_neighbor_impact.csv',index=False)
 fixed.to_parquet(o/'derived/EMT_canonical_recording_fixed.parquet',index=False)
 # Reconstruct old helper output once to establish source provenance.
 ns={};exec(compile((ROOT/'audits/r15_20260917/before/src/data/base_adapter.py').read_text(),'legacy_neighbor','exec'),ns)
 old=ns['compute_side_neighbors'](c.drop(columns=SIDE).copy())
 pd.DataFrame([dict(column=col,mismatch_to_saved=int((old[col]!=c[col]).sum())) for col in SIDE]).to_csv(o/'tables/emt_legacy_reproduction.csv',index=False)
 # Same sample rows, same fps/sign/RF policy; isolate six side-ID corrections.
 for h in [3,5,7]:
  d=load('EMT',h);parts=[];checks=[]
  for rec,g in d.groupby('recording_id',sort=False):
   sample=g[['recording_id','vehicle_id','frame','label']];can=c[c.recording_id==rec].reset_index(drop=True);new=fixed[fixed.recording_id==rec].reset_index(drop=True)
   oldf=build_features(sample,can,fps=25,lat_sign=1)
   oldf=compute_game_features(oldf[['recording_id','vehicle_id','frame','label']+KIN],can)
   newf=build_features(sample,new,fps=25,lat_sign=1)
   newf=compute_game_features(newf[['recording_id','vehicle_id','frame','label']+KIN],new)
   oldf.index=g.index;newf.index=g.index;parts.append(newf)
   for col in COLS:
    eq=np.isclose(oldf[col],g[col],atol=1e-7,rtol=1e-6,equal_nan=True)
    checks.append(dict(recording=rec,horizon=h,feature=col,legacy_saved_mismatch=int((~eq).sum())))
  z=pd.concat(parts).sort_index();z['dataset']='EMT';assert z[['recording_id','vehicle_id','frame','label']].equals(d[['recording_id','vehicle_id','frame','label']])
  z.to_parquet(o/'derived'/f'EMT_gt_{h}s_neighbors_fixed.parquet',index=False)
  pd.DataFrame(checks).to_csv(o/'tables'/f'emt_feature_reproduction_{h}s.csv',index=False)
  impact=[dict(horizon=h,feature=col,changed=int((~np.isclose(z[col],d[col],atol=1e-7,rtol=1e-6,equal_nan=True)).sum())) for col in COLS]
  pd.DataFrame(impact).to_csv(o/'tables'/f'emt_feature_impact_{h}s.csv',index=False);print('EMT features done',h,flush=True)

def lstm(o):
 import torch
 torch.set_num_threads(8);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
 data={d:load(d) for d in ALL};src=pd.concat([data[d] for d in SRC],ignore_index=True);metrics=[]
 for variant in ['before','after']:
  x=np.concatenate([np.load(o/'sequences'/f'{d}_{variant}.npy') for d in SRC]);targets={d:np.load(o/'sequences'/f'{d}_{variant}.npy') for d in TGT}
  for seed in SEEDS:
   tr,te=split(src,seed);manifest=src[META].copy();manifest['role']=np.where(te,'test','train');manifest.to_parquet(o/'splits'/f'source_seed{seed}.parquet',index=False)
   t=time.time();print('LSTM fit',variant,seed,flush=True)
   net,dev=fit_lstm(x[tr],src.label.to_numpy()[tr],epochs=30,seed=seed,device='cuda:0')
   mp=o/'models'/f'lstm_{variant}_{seed}.pt';torch.save(dict(state_dict=net.state_dict(),mu=net._mu,sd=net._sd,seed=seed,variant=variant),mp);mh=sha(mp)
   for name,dd,xx in [('in_domain',src.loc[te],x[te])]+[(d,data[d],targets[d]) for d in TGT]:
    net.eval();ps=[]
    with torch.no_grad():
     for st in range(0,len(xx),4096):
      xb=torch.tensor(((xx[st:st+4096]-net._mu)/net._sd).astype(np.float32),device=dev);ps.append(torch.sigmoid(net(xb)).cpu().numpy())
    p=np.concatenate(ps);row=savepred(o,f'lstm_{variant}_seed{seed}_{name}',dd,p,seed,name,mh);row.update(model='BiLSTM',variant=variant,fit_seconds=time.time()-t);metrics.append(row)
   pd.DataFrame(metrics).to_csv(o/'tables/lstm_metrics.csv',index=False);print('LSTM done',variant,seed,metrics[-5]['auc'],time.time()-t,flush=True)
   del net;torch.cuda.empty_cache()

def trees(o):
 data={d:load(d) for d in ALL};src=pd.concat([data[d] for d in SRC],ignore_index=True);raw=src[COLS].replace([np.inf,-np.inf],np.nan);metrics=[]
 for seed in SEEDS:
  tr,te=split(src,seed)
  for variant in ['source_median_before','train_median_after']:
   med=raw.median() if variant=='source_median_before' else raw.loc[tr].median();x=raw.fillna(med)
   med.to_csv(o/'models'/f'xgb_{variant}_{seed}_median.csv');print('XGB',variant,seed,flush=True)
   model=fit_xgb(x.loc[tr],src.label.loc[tr],seed=seed,params={'n_jobs':8});mp=o/'models'/f'xgb_{variant}_{seed}.ubj';model.save_model(mp);mh=sha(mp)
   sets=[('in_domain',src.loc[te],x.loc[te])]+[(ds,data[ds],data[ds][COLS].replace([np.inf,-np.inf],np.nan).fillna(med)) for ds in TGT]
   fixed=pd.read_parquet(o/'derived/EMT_gt_3s_neighbors_fixed.parquet');sets.append(('EMT_neighbors_fixed',fixed,fixed[COLS].replace([np.inf,-np.inf],np.nan).fillna(med)))
   for name,d,xx in sets:
    row=savepred(o,f'xgb_{variant}_seed{seed}_{name}',d,model.predict_proba(xx)[:,1],seed,name,mh);row.update(model='XGBoost',variant=variant);metrics.append(row)
   pd.DataFrame(metrics).to_csv(o/'tables/xgb_metrics.csv',index=False)

def loco(o):
 groups={'Germany':['highD','uniD','exiD'],'USA':['NGSIM'],'Italy':['MiTra'],'Korea':['ETRI'],'UAE':['EMT'],'legacy_Italy':['MiTra','EMT']};metrics=[]
 for h in [3,5,7]:
  data={d:load(d,h) for d in ALL}
  for name,members in groups.items():
   train=pd.concat([data[d] for d in sorted(ALL) if d not in members],ignore_index=True);test=pd.concat([data[d] for d in members],ignore_index=True)
   x=train[COLS].replace([np.inf,-np.inf],np.nan);med=x.median();x=x.fillna(med);xt=test[COLS].replace([np.inf,-np.inf],np.nan).fillna(med)
   for seed in [42,0,1,7,123]:
    print('LOCO',h,name,seed,flush=True);m=fit_xgb(x,train.label,seed=seed,params={'n_jobs':8});mp=o/'models'/f'loco_{h}_{name}_{seed}.ubj';m.save_model(mp)
    row=savepred(o,f'loco_{h}_{name}_{seed}',test,m.predict_proba(xt)[:,1],seed,name,sha(mp));row.update(horizon=h,held_out=name,members='+'.join(members),n_train=len(train),input='legacy saved features; EMT neighbor unchanged to isolate country mapping');metrics.append(row)
    pd.DataFrame(metrics).to_csv(o/'tables/loco_metrics.csv',index=False)

def adaptation(o):
 import xgboost as xgb
 from src.models.train import predict_best
 spec=importlib.util.spec_from_file_location('adapt_reference',ROOT/'scripts/20_adapt_control.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
 data={d:load(d) for d in ALL};src=pd.concat([data[d] for d in SRC],ignore_index=True);xs=src[COLS].replace([np.inf,-np.inf],np.nan);sm=xs.median();xs=xs.fillna(sm);rows=[]
 for seed in SEEDS:
  base=fit_xgb(xs,src.label,seed=seed,params={'n_jobs':8});bp=o/'models'/f'adapt_source_{seed}.ubj';base.save_model(bp);sm.to_csv(o/'models'/f'adapt_source_{seed}_median.csv')
  for ds in TGT:
   d=data[ds];raw=d[COLS].replace([np.inf,-np.inf],np.nan);xt=raw.fillna(sm);y=d.label.to_numpy();keys=a.gkey(d);pool,test=a.gsplit(keys,a.TEST_FRAC,seed)
   for frac in a.FRACS:
    ft=a.sub_groups(keys,pool,frac,seed);tr,va=a.gsplit(keys[ft],a.FT_VAL,seed+1);sub=raw.loc[ft];tm=sub.loc[tr].median();tm.to_csv(o/'models'/f'adapt_{ds}_{seed}_{frac}_target_median.csv')
    role=np.full(len(d),'unused_pool',dtype=object);role[test]='test';inds=np.flatnonzero(ft);role[inds[tr]]='train';role[inds[va]]='validation';manifest=d[META].copy();manifest['role']=role;manifest.to_parquet(o/'splits'/f'adapt_{ds}_{seed}_{frac}.parquet',index=False)
    print('adapt',ds,seed,frac,flush=True)
    for cond,med,initial in [('continuation',sm,base.get_booster()),('target_only_source_median',sm,None),('target_only_target_median',tm,None)]:
     xf=sub.fillna(med);xe=raw.loc[test].fillna(med);yt=y[ft]
     pars=dict(a.PARAMS,seed=seed,nthread=8);evals=[(xgb.DMatrix(xf.loc[va],label=yt[va]),'val')] if va.any() and len(np.unique(yt[va]))>1 else []
     b=xgb.train(pars,xgb.DMatrix(xf.loc[tr],label=yt[tr]),num_boost_round=a.FT_ROUNDS,xgb_model=initial,evals=evals,early_stopping_rounds=a.FT_STOP if evals else None,verbose_eval=False)
     mp=o/'models'/f'adapt_{ds}_{seed}_{frac}_{cond}.ubj';b.save_model(mp);mh=sha(mp);dm=xgb.DMatrix(xe)
     for stop,pred in [('last',b.predict(dm)),('best',predict_best(b,dm))]:
      row=savepred(o,f'adapt_{ds}_{seed}_{frac}_{cond}_{stop}',d.loc[test],pred,seed,ds,mh);row.update(target=ds,frac=frac,condition=cond,predict_iteration=stop,num_rounds=b.num_boosted_rounds(),best_iteration=getattr(b,'best_iteration',None),n_train=int(tr.sum()),n_val=int(va.sum()),n_groups=len(np.unique(keys[ft])));rows.append(row)
    pd.DataFrame(rows).to_csv(o/'tables/adaptation_metrics.csv',index=False)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['sequences','emt','lstm','trees','loco','adaptation']);ap.add_argument('--out',required=True,type=Path);a=ap.parse_args();o=a.out.resolve()
 if ROOT/'audits' not in o.parents:raise SystemExit('Output must be inside audits/')
 init_out(o);globals()[a.stage](o)
