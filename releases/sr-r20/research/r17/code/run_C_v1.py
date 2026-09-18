from common_v1 import *
import datetime
N=2000

def auc_engine(d,keys,levels=None,matrix=False):
 levels=np.unique(keys) if levels is None else np.asarray(levels);lookup={k:i for i,k in enumerate(levels)};g=np.array([lookup[k] for k in keys]);y=d.label.to_numpy().astype(float);p=d.score.to_numpy();G=len(levels)
 pos=np.bincount(g,weights=y,minlength=G);neg=np.bincount(g,weights=1-y,minlength=G)
 if matrix:
  M=np.zeros((G,G));pi=np.flatnonzero(y==1)
  for j in range(G):
   ns=np.sort(p[(g==j)&(y==0)]);v=(np.searchsorted(ns,p[pi],side='left')+np.searchsorted(ns,p[pi],side='right'))*.5
   M[:,j]=np.bincount(g[pi],weights=v,minlength=G)
  def calc(w):
   den=(w@pos)*(w@neg)
   return float(w@M@w/den) if den>0 else np.nan
 else:
  order=np.argsort(p,kind='stable');p=p[order];y=y[order];g=g[order];start=np.r_[0,np.flatnonzero(np.diff(p)!=0)+1]
  def calc(w):
   weights=w[g];pp=np.add.reduceat(weights*y,start);nn=np.add.reduceat(weights*(1-y),start);den=pp.sum()*nn.sum()
   return float(pp@(np.cumsum(nn)-nn*.5)/den) if den>0 else np.nan
 assert np.isclose(calc(np.ones(G)),roc_auc_score(d.label,d.score),atol=1e-12)
 return calc,levels

def reckeys(d,physical=False):return (d.dataset.astype(str)+'|'+d['physical_recording_id' if physical else 'recording_id'].astype(str)).to_numpy()
def cirow(v,**kw):
 v=np.asarray(v);ok=np.isfinite(v);lo,hi=np.quantile(v[ok],[.025,.975]) if ok.any() else [np.nan,np.nan]
 return dict(**kw,n_requested=N,n_valid=int(ok.sum()),n_invalid=int((~ok).sum()),ci_low=lo,ci_high=hi,confidence=.95,training_uncertainty_included=False,EMT_excluded=True)
def metrics(d):
 return dict(auc=roc_auc_score(d.label,d.score) if d.label.nunique()==2 else np.nan,ece=calibration_bins(d.label.to_numpy(),d.score.to_numpy())[0],brier=brier_score_loss(d.label,d.score),**counts(d))
def run():
 data={ds:load(ds).set_index(KEYS) for ds in DATASETS};validation=[];numeric=[];allmeta=[];cache={};splits={};modelhash={}
 for vp in sorted((O/'verification').glob('A*.json')):
  m=json.loads(vp.read_text());p=O/m['prediction'];d=pd.read_parquet(p);assert not d.duplicated(KEYS).any();assert set(d.dataset)<=set(DATASETS);assert d.label.isin([0,1]).all();assert np.isfinite(d.score).all() and d.score.between(0,1).all();assert (d.split_role=='test').all()
  for ds,q in d.groupby('dataset'):
   expected=data[ds].reindex(pd.MultiIndex.from_frame(q[KEYS]));assert expected.label.notna().all();assert np.array_equal(q.label,expected.label)
  for field,pathkey in [('prediction_sha256','prediction'),('config_sha256','config_path'),('model_sha256','model_path'),('split_sha256','split_path')]:
   path=O/m[pathkey]
   if str(path) not in modelhash:modelhash[str(path)]=sha(path)
   assert modelhash[str(path)]==m[field],str(path)
  assert d.model_sha256.eq(m['model_sha256']).all() and d.config_sha256.eq(m['config_sha256']).all()
  sp=m['split_path']
  if sp not in splits:splits[sp]=pd.read_parquet(O/sp)
  s=splits[sp];checks=validate_split(s,s.role,physical=m['protocol']=='A2_physical_recording')
  if m['protocol'].startswith('A2') or m['eval_set']=='source_test':
   a=d[META].sort_values(KEYS).reset_index(drop=True);b=s.loc[s.role.eq('test'),META].sort_values(KEYS).reset_index(drop=True);pd.testing.assert_frame_equal(a,b)
  else:assert len(d)==len(data[m['eval_set']])
  # Reproduce saved predictions, including early-stopping iteration range, without training.
  x=pd.concat([data[ds].reindex(pd.MultiIndex.from_frame(q[KEYS]))[COLS] for ds,q in d.groupby('dataset',sort=False)],ignore_index=True)
  # Preserve original prediction order independently of grouping.
  indexed=pd.concat(list(data.values()));x=indexed.reindex(pd.MultiIndex.from_frame(d[KEYS]))[COLS]
  med=pd.read_csv(O/'models'/f"{m['run']}_median.csv",index_col=0).iloc[:,0];x=x.replace([np.inf,-np.inf],np.nan).fillna(med)
  bst=xgb.Booster();bst.load_model(O/m['model_path']);bst.set_param({'nthread':8});pred=bst.predict(xgb.DMatrix(x),iteration_range=(0,int(m['iteration_range_end_exclusive'])))
  delta=float(np.max(np.abs(pred-d.score.to_numpy())));assert delta<1e-7
  mm=metrics(d)
  for key in ['auc','ece','brier']:assert np.isclose(mm[key],m[key],equal_nan=True,atol=1e-10)
  validation.append(dict(prediction=m['prediction'],n_rows=len(d),max_checkpoint_prediction_diff=delta,passed=True,**checks))
  info={k:m.get(k) for k in ['run','model','protocol','feature_set','seed','eval_set','fraction','condition']};info.update(EMT_excluded=True)
  for k in ['auc','ece','brier','positive_rate','positive_count']:numeric.append(dict(**info,aggregation='point_estimate',metric=k,value=mm[k],**{c:mm[c] for c in ['n_rows','n_vehicles','n_recordings','n_physical_recordings']}))
  allmeta.append(dict(**info,**mm));cache[m['run']+'__'+m['eval_set']]=d
  print('VERIFIED',m['run'],m['eval_set'],flush=True)
 pd.DataFrame(validation).to_csv(O/'verification/new_predictions_checks.csv',index=False)
 # Additional medians/nesting checks independent of fitting.
 medchecks=[]
 for cp in sorted((O/'configs').glob('A*.json')):
  cfg=json.loads(cp.read_text())
  if not isinstance(cfg,dict) or 'run' not in cfg:continue
  run=cfg['run'];mf=O/'models'/f'{run}_median.csv'
  if not mf.exists():continue
  if cfg['protocol']=='A1_recording':s=pd.read_parquet(O/'splits'/f'{run}.parquet');tr=s[s.role.eq('train')];base=pd.concat(list(data.values())).reindex(pd.MultiIndex.from_frame(tr[KEYS]))[COLS]
  elif cfg.get('condition')=='target_only':sp=run.rsplit('_target_only',1)[0];s=pd.read_parquet(O/'splits'/f'{sp}.parquet');tr=s[s.role.eq('train')];base=data[cfg['target']].reindex(pd.MultiIndex.from_frame(tr[KEYS]))[COLS]
  else:base=pd.concat([data[ds][COLS] for ds in SOURCES])
  expect=base.replace([np.inf,-np.inf],np.nan).median();actual=pd.read_csv(mf,index_col=0).iloc[:,0];assert np.allclose(actual,expect,equal_nan=True,atol=1e-10)
  medchecks.append(dict(run=run,training_median_matches=True))
 for cp in (O/'configs').glob('*nested_groups.json'):
  prior=set()
  for q in json.loads(cp.read_text()):assert prior<=set(q['groups']);prior=set(q['groups'])
 pd.DataFrame(medchecks).to_csv(O/'verification/training_median_checks.csv',index=False)
 # Pooled vs macro per seed; counts for macro describe union, not equal row weighting.
 for seed in SEEDS:
  dscores=[cache[f'A1_recording_full48_s{seed}__{ds}'] for ds in TARGETS];pool=pd.concat(dscores,ignore_index=True)
  for name,val in [('target_pooled',metrics(pool)['auc']),('target_macro',np.mean([metrics(q)['auc'] for q in dscores]))]:
   info=dict(run=f'A1_recording_full48_s{seed}',model='XGBoost',protocol='A1_recording',feature_set='full48',seed=seed,eval_set=name,fraction=None,condition=None,EMT_excluded=True,aggregation='pooled_point' if name.endswith('pooled') else 'unweighted_target_macro',metric='auc',value=val,**{k:v for k,v in counts(pool).items() if k.startswith('n_')});numeric.append(info)
 num=pd.DataFrame(numeric)
 groupcols=['model','protocol','feature_set','eval_set','fraction','condition','metric']
 extras=[]
 for key,q in num.groupby(groupcols,dropna=False):
  if len(q)!=3:continue
  for agg,val in [('seed_mean',q.value.mean()),('seed_sd_ddof1',q.value.std(ddof=1))]:
   extras.append(dict(zip(groupcols,key),run='three_fixed_seeds',seed='42,0,1',aggregation=agg,value=val,EMT_excluded=True,n_rows=np.nan,n_vehicles=np.nan,n_recordings=np.nan))
 pd.concat([num,pd.DataFrame(extras)],ignore_index=True).to_csv(O/'numeric_results.csv',index=False)
 meta=pd.DataFrame(allmeta);meta.to_csv(O/'tables/recomputed_metrics.csv',index=False)
 adapt=meta[meta.protocol.str.startswith('A2')].pivot(index=['protocol','eval_set','seed','fraction'],columns='condition',values='auc').reset_index();adapt['gain']=adapt.continuation-adapt.target_only;adapt.to_csv(O/'tables/A2_paired_seed_values.csv',index=False)
 adapt.groupby(['protocol','eval_set','fraction'])[['continuation','target_only','gain']].agg(['mean','std']).to_csv(O/'tables/A2_paired_summary.csv')
 cis=[];reps=[]
 for name in TARGETS+['pooled']:
  d=cache[f'A1_recording_full48_s42__{name}'] if name!='pooled' else pd.concat([cache[f'A1_recording_full48_s42__{ds}'] for ds in TARGETS],ignore_index=True)
  for unit in ['vehicle','recording']+(['physical_recording'] if name in ['exiD','pooled'] else []):
   keys=groupkeys(d).to_numpy() if unit=='vehicle' else reckeys(d,unit=='physical_recording');calc,levels=auc_engine(d,keys,matrix=unit!='vehicle');rng=np.random.RandomState(170042);values=[]
   strata=[np.flatnonzero(np.array([k.split('|')[0] for k in levels])==ds) for ds in TARGETS] if name=='pooled' and unit!='vehicle' else [np.arange(len(levels))]
   for b in range(N):
    w=np.zeros(len(levels))
    for inds in strata:
     if len(inds):w+=np.bincount(rng.choice(inds,len(inds),replace=True),minlength=len(levels))
    values.append(calc(w))
   cis.append(cirow(values,analysis='A1',eval_set=name,unit=unit,seed=42,point_auc=roc_auc_score(d.label,d.score),n_groups=len(levels),n_recordings=counts(d)['n_recordings'],n_physical_recordings=counts(d)['n_physical_recordings'],pooled_stratified_by_dataset=name=='pooled' and unit!='vehicle'))
   reps.extend(dict(analysis='A1',eval_set=name,unit=unit,replicate=i,value=v) for i,v in enumerate(values));print('BOOTSTRAP A1',name,unit,flush=True)
 # All seed test unions use the same resampled recording multiplicity, both methods.
 for (protocol,ds,fr),z in adapt.groupby(['protocol','eval_set','fraction']):
  pairs=[];union=set();physical=ds=='exiD'
  for seed in SEEDS:
   a=cache[f'{protocol}_{ds}_s{seed}_f{fr}_continuation__{ds}'];b=cache[f'{protocol}_{ds}_s{seed}_f{fr}_target_only__{ds}'];pd.testing.assert_frame_equal(a[META].reset_index(drop=True),b[META].reset_index(drop=True));union.update(reckeys(a,physical));pairs.append((a,b))
  levels=np.array(sorted(union));eng=[(auc_engine(a,reckeys(a,physical),levels,True)[0],auc_engine(b,reckeys(b,physical),levels,True)[0]) for a,b in pairs];rng=np.random.RandomState(170043);values=[]
  for i in range(N):
   w=np.bincount(rng.randint(0,len(levels),len(levels)),minlength=len(levels));diff=np.array([a(w)-b(w) for a,b in eng]);values.append(float(diff.mean()) if np.isfinite(diff).all() else np.nan)
  cis.append(cirow(values,analysis=protocol,eval_set=ds,fraction=fr,unit='physical_recording' if physical else 'recording',seed='42,0,1',point_gain=z.gain.mean(),n_groups=len(levels),conditioning='3 fixed models per method; union recording weights shared across methods and all seeds; all 3 AUC required',budget_adjustment='unadjusted'))
  reps.extend(dict(analysis=protocol,eval_set=ds,fraction=fr,unit='physical_recording' if physical else 'recording',replicate=i,value=v) for i,v in enumerate(values));print('BOOTSTRAP A2',protocol,ds,fr,flush=True)
 pd.DataFrame(cis).to_csv(O/'tables/bootstrap_ci.csv',index=False);pd.DataFrame(reps).to_parquet(O/'tables/bootstrap_replicates.parquet',index=False)
 jwrite(O/'logs/C_v1_completed.json',dict(completed=datetime.datetime.now().isoformat(),predictions_verified=len(validation),medians_verified=len(medchecks),CI_rows=len(cis)))
if __name__=='__main__':
 if '--freeze' in sys.argv:jwrite(O/'configs/code_freeze_C_v1.json',dict(frozen_before_execution=datetime.datetime.now().isoformat(),runner_sha256=sha(__file__),bootstrap_replicates=N,auc_ties='exact half-credit; score grouping',ECE='repository calibration_bins default 10 uniform bins',A2_exid_bootstrap_unit='original physical recording for both literal-ID and physical-ID holdout',A1_additional='physical recording bootstrap for exiD and pooled; requested ID bootstrap separately preserved',checks='all new predictions metrics, keys, labels, hashes, roles, saved checkpoint prediction, train-only medians, nesting'))
 else:run()
