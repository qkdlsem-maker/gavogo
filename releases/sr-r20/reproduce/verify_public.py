"""Verify public manifests and R19 table arithmetic; never claims prediction/raw replay."""
from pathlib import Path
import argparse,csv,hashlib,json,re,sys
import numpy as np,pandas as pd

def verify(root):
 root=root.resolve();results=[]
 manifest=root/'MANIFEST.csv'
 if not manifest.exists():raise FileNotFoundError('Public MANIFEST.csv required; obtain the versioned release.')
 for q in csv.DictReader(manifest.open()):
  p=(root/q['path']).resolve();assert p.is_relative_to(root) and p.is_file(),q['path'];b=p.read_bytes();assert len(b)==int(q['bytes']) and hashlib.sha256(b).hexdigest()==q['sha256'],q['path']
 def read(path):return pd.read_csv(root/path)
 def table(prefix,i):return read(f'tables/{prefix}_{i:02d}.csv')
 def f(x):return float(str(x).replace(',','').replace('−','-').replace('%',''))
 def close(x,y,dec=4):assert abs(f(x)-float(y))<=.50001*10**(-dec),(x,y)
 def interval(x,lo,hi):
  q=re.findall(r'[+-]?\d+\.\d+',str(x).replace('−','-'));assert len(q)==2,x;close(q[0],lo);close(q[1],hi)
 r16=root/'aggregates/R16';r17=root/'aggregates/R17'
 pop={q['dataset']:q for q in json.loads((r16/'populations.json').read_text())}
 for _,q in table('main',1).iterrows():
  z=pop[q.Dataset];assert f(q.Rows)==z['n'] and f(q.Vehicles)==z['vehicles'] and f(q['Positive rows'])==z['positives']
 results.append({'table':'Main1','check':'aggregate population counts'})
 seq=read('aggregates/R16/sequence_impact.csv').set_index('dataset')
 for _,q in table('main',2).iterrows():
  z=seq.loc[q.Dataset];assert f(q['Sample rows'])==z.n and f(q['Affected rows'])==z.mixed_samples;close(q['Affected fraction'],100*z.mixed_samples/z.n,1)
 results.append({'table':'Main2','check':'archived aggregate counts; no sequence reconstruction'})
 a=read('aggregates/R17/A1_seed_metrics.csv');summ=read('aggregates/R17/A1_summary.csv').set_index('eval_set');names={'Source test':'source_test','Target macro':'target_macro','Pooled targets':'target_pooled'}
 for _,q in table('main',3).iterrows():
  ds=names.get(q['Evaluation set'],q['Evaluation set']);z=a[a.eval_set.eq(ds)].auc.to_numpy();assert len(z)==3;close(q['Mean AUC'],z.mean());close(q['Seed SD (denominator 3)'],z.std(ddof=0));assert np.isclose(summ.loc[ds,'mean'],z.mean())
 results.append({'table':'Main3','check':'recomputed three-seed means and population SD from public seed aggregates'})
 a2=read('aggregates/R17/A2_seed_gains.csv');ci=read('aggregates/R17/bootstrap_ci.csv')
 def check_adapt(tab,literal=False):
  for _,q in tab.iterrows():
   ds='exiD' if literal else q.Target;fr=f(q['Pool fraction'])/100;pr='A2_physical_recording' if ds=='exiD' and not literal else 'A2_recording';z=a2[(a2.protocol==pr)&(a2.eval_set==ds)&np.isclose(a2.fraction,fr)];assert len(z)==3;close(q['Target only'],z.target_only.mean());close(q.Continuation,z.continuation.mean());close(q.Gain,z.gain.mean());c=ci[(ci.analysis==pr)&(ci.eval_set==ds)&np.isclose(ci.fraction,fr)].iloc[0]
   if ds=='ETRI':assert str(q.iloc[-1]).startswith('Not estimable');assert c.n_valid==1003 and c.n_invalid==997 and c.n_groups==2
   else:interval(q.iloc[-1],c.ci_low,c.ci_high)
 check_adapt(table('main',4));check_adapt(table('supp',3),True)
 primary=a2[(a2.eval_set!='exiD')|(a2.protocol=='A2_physical_recording')].groupby(['eval_set','fraction']).gain.mean();assert len(primary)==15 and (primary<0).all()
 results.extend([{'table':'Main4','check':'paired seed-aggregate means and archived CI; bootstrap not rerun'},{'table':'S3','check':'literal-ID sensitivity; positive full-budget exception retained'}])
 for _,q in table('main',5).iterrows():close(q['Absolute change'],abs(f(q['Full input'])-f(q['Censored input'])),3)
 b=json.loads((r17/'B2_counts.json').read_text());assert b['total_pairs']==234 and b['both_missing']==28 and b['finite_pairs']==206 and b['finite_equal']==203 and b['finite_changed']==3
 results.append({'table':'Main5','check':'published counterexample arithmetic and audit counts only; raw timing experiment not rerun'})
 m=read('aggregates/R16/prediction_metrics.csv');configs={'Trees before':('xgb','source_median_before'),'Trees after':('xgb','train_median_after'),'Recurrent before':('lstm','before'),'Recurrent after':('lstm','after')}
 for _,q in table('supp',1).iterrows():
  fam,cond=configs[q.Configuration];z=m[(m.family==fam)&(m.condition==cond)]
  for col,ds in [('Source test','in_domain'),('ETRI','ETRI'),('uniD','uniD'),('exiD','exiD'),('Target macro','macro')]:
   ar=z[z.eval_set.isin(['ETRI','uniD','exiD'])].groupby('seed').auc.mean().to_numpy() if ds=='macro' else z[z.eval_set.eq(ds)].auc.to_numpy();assert len(ar)==3;v=str(q[col]).split('±');close(v[0],ar.mean());close(v[1],ar.std(ddof=0))
 results.append({'table':'S1','check':'recomputed means/population SD; EMT excluded'})
 for _,q in table('supp',2).iterrows():
  ds='pooled' if q['Target set']=='Pooled targets' else q['Target set'];z=ci[(ci.analysis=='A1')&(ci.eval_set==ds)];v=z[z.unit=='vehicle'].iloc[0];r=z[z.unit==('physical_recording' if ds in ['pooled','exiD'] else 'recording')].iloc[0];close(q.AUC,v.point);interval(q['Vehicle 95% CI'],v.ci_low,v.ci_high);interval(q['Recording 95% CI'],r.ci_low,r.ci_high);assert f(q.Recordings)==r.n_groups
 results.append({'table':'S2','check':'fixed seed42 physical-recording vs vehicle CI labels/endpoints'})
 sf=read('aggregates/R16/same_fit_metrics.csv').set_index('eval_set');oldci=read('aggregates/R16/same_fit_ci.csv')
 for _,q in table('supp',4).iterrows():
  ds={'Source test':'in_domain','Pooled targets':'POOLED_NO_EMT'}.get(q['Evaluation set'],q['Evaluation set']);z=sf.loc[ds];assert f(q.Rows)==z.n
  for col,k in [('AUC','auc'),('ECE','ece'),('Brier','brier')]:close(q[col],z[k])
  if ds!='in_domain':c=oldci[(oldci.eval_set==ds)&(oldci.unit=='vehicle')].iloc[0];interval(q['Vehicle 95% CI'],c.lo,c.hi)
 results.append({'table':'S4','check':'earlier fixed-fit aggregates; not A1'})
 ad=read('aggregates/R16/adaptation_no_emt.csv');pc=read('aggregates/R16/paired_ci.csv')
 for _,q in table('supp',5).iterrows():
  ds=q.Target;fr=f(q['Pool fraction'])/100;z=ad[(ad.eval_set==ds)&np.isclose(ad.fraction,fr)];t=z[z.condition=='target_only_target_median'].iloc[0]['mean'];c=z[z.condition=='continuation'].iloc[0]['mean'];close(q['Target only'],t);close(q.Continuation,c);close(q.Gain,c-t);r=pc[(pc.target==ds)&np.isclose(pc.frac,fr)].iloc[0];interval(q.iloc[-1],r.lo,r.hi)
 results.append({'table':'S5','check':'vehicle-split aggregate comparison separate from recording splits'})
 ar=read('aggregates/R16/archived_sampling.csv').set_index('dataset')
 for _,q in table('supp',6).iterrows():
  ds=q.Dataset.split()[0];z=ar.loc[ds];close(q['Purity by group'],z.purity,3);close(q['AUC (all vehicles)'],z.auc_all);close(q['AUC (mixed-label vehicles only)'],z.auc_mixed_only);close(q['Δ'],z.auc_mixed_only-z.auc_all)
 results.append({'table':'S6','check':'archived aggregate transcription only; no population replay'})
 sys.path.insert(0,str(root/'research/r17/code/r16_snapshot'))
 from src.features.kinematic import FEATURE_COLS,BASE_COLS,GAP_COLS,VOS_COLS,ROLL_COLS,LAT_COLS,INT_COLS
 from src.features.game_theory import NASH_FEATURE_COLS
 counts=[len(c) for c in [BASE_COLS,GAP_COLS,VOS_COLS,ROLL_COLS,LAT_COLS,INT_COLS,NASH_FEATURE_COLS]];assert table('supp',7)['Count'].astype(int).tolist()==counts+[len(FEATURE_COLS)+len(NASH_FEATURE_COLS)]
 results.append({'table':'S7','check':'counts from published corrected source'})
 for _,q in table('supp',8).iterrows():
  ds='POOLED_NO_EMT' if q['Evaluation set']=='Pooled targets' else q['Evaluation set'];z=oldci[oldci.eval_set.eq(ds)]
  for unit,col in [('row','Row bootstrap'),('vehicle','Vehicle bootstrap')]:
   c=z[z.unit.eq(unit)].iloc[0];interval(q[col],c.lo,c.hi)
  assert f(q['Vehicle groups'])==z[z.unit.eq('vehicle')].iloc[0].n_units
 results.append({'table':'S8','check':'archived fixed-fit row/vehicle CI'})
 assert len(results)==13
 return dict(passed=True,tables=results,full_training_replayed=False,predictions_recomputed=False,bootstrap_recomputed=False,SHAP_recomputed=False)
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args();print(json.dumps(verify(a.root),indent=2))
