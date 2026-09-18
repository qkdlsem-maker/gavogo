"""Portable bootstrap reproduction from archived predictions; no training/raw input."""
from run_C_v1 import *
import argparse
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);args=p.parse_args();O=args.root.resolve();out=args.output_dir;out.mkdir(parents=True,exist_ok=False)
cache={p.stem:pd.read_parquet(p) for p in (O/'predictions').glob('*.parquet')}
adapt=pd.read_csv(O/'tables/A2_paired_seed_values.csv')
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
pd.DataFrame(cis).to_csv(out/'bootstrap_ci.csv',index=False);pd.DataFrame(reps).to_parquet(out/'bootstrap_replicates.parquet',index=False)
print("Completed bootstrap reproduction",out)
