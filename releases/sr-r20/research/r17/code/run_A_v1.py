import time,json,sys
from common_v1 import *

def a1(data):
 src=pd.concat([data[d] for d in SOURCES],ignore_index=True);rows=[]
 for seed in SEEDS:
  run=f'A1_recording_full48_s{seed}'
  try:
   test=np.zeros(len(src),bool);details={}
   for ds in SOURCES:
    inds=np.flatnonzero(src.dataset.eq(ds));_,te,info=recording_holdout(src.iloc[inds],seed);test[inds]=te;details[ds]=info
   roles=np.where(test,'test','train');sp,checks=save_split(run,src,roles)
   cp,cfg=model_config(run,'A1_recording',seed,COLS,split_rule=details,split_checks=checks)
   tr=~test;x,med=clean(src.loc[tr],COLS);med.to_csv(O/'models'/f'{run}_median.csv');missing=[]
   for name,d in [('source_train',src.loc[tr]),('source_test',src.loc[test])]+[(ds,data[ds]) for ds in TARGETS]:
    for col in COLS:missing.append(dict(eval_set=name,feature=col,missing_rate=float(d[col].replace([np.inf,-np.inf],np.nan).isna().mean()),train_median=None if pd.isna(med[col]) else med[col]))
   pd.DataFrame(missing).to_csv(O/'tables'/f'{run}_missing.csv',index=False)
   start=time.monotonic();m=fit_source(x,src.label.loc[tr],seed);mp,mh=save_model(run,m)
   for ds,d in [('source_test',src.loc[test])]+[(ds,data[ds]) for ds in TARGETS]:
    xx,_=clean(d,COLS,med);rows.append(predict_save(run,'A1_recording',seed,ds,d,m.predict_proba(xx)[:,1],mp,mh,cp,sp))
   jwrite(O/'logs'/f'{run}_completed.json',dict(seconds=time.monotonic()-start,split=checks));print('DONE',run,rows[-4]['auc'],flush=True)
  except Exception as e:failure('A1',run,e)
 pd.DataFrame(rows).to_csv(O/'tables/A1_metrics.csv',index=False)

def a2(data):
 src=pd.concat([data[d] for d in SOURCES],ignore_index=True);xs,sm=clean(src,COLS);rows=[];checks=[]
 for seed in SEEDS:
  baserun=f'A2_source_all_s{seed}'
  try:
   cp,cfg=model_config(baserun,'A2_source_all',seed,COLS,imputation='all source rows; all are training')
   bsp,_=save_split(baserun,src,np.repeat('train',len(src)),require_recording=False)
   sm.to_csv(O/'models'/f'{baserun}_median.csv');base=fit_source(xs,src.label,seed);bp,bh=save_model(baserun,base)
  except Exception as e:failure('A2',baserun,e);continue
  for ds,physical in [(ds,False) for ds in TARGETS]+[('exiD',True)]:
   protocol='A2_physical_recording' if physical else 'A2_recording';d=data[ds];raw=d[COLS].replace([np.inf,-np.inf],np.nan);y=d.label.to_numpy();keys=groupkeys(d).to_numpy()
   try:pool,test,info=recording_holdout(d,seed,physical)
   except Exception as e:failure('A2',f'{protocol}_{ds}_s{seed}_split',e);continue
   groups=np.unique(keys[pool]);np.random.RandomState(seed+777).shuffle(groups);prior=set();selection=[]
   for fr in [.1,.2,.4,.6,1.]:
    n=max(1,int(np.floor(len(groups)*fr)));selected=set(groups[:n]);assert prior<=selected;prior=selected;ft=np.isin(keys,list(selected))&pool
    selected_ids=np.unique(keys[ft]);np.random.RandomState(seed+1).shuffle(selected_ids)
    nv=max(1,int(np.floor(.2*len(selected_ids)))) if len(selected_ids)>1 else 0;vg=set(selected_ids[:nv]);va=ft & np.isin(keys,list(vg));tr=ft & ~va
    roles=np.full(len(d),'unused_pool',dtype=object);roles[test]='test';roles[va]='validation';roles[tr]='train'
    tag=f'{protocol}_{ds}_s{seed}_f{fr}'
    try:sp,ch=save_split(tag,d,roles,physical=physical)
    except Exception as e:failure('A2',tag+'_split',e);continue
    selection.append(dict(fraction=fr,n_pool_groups=len(groups),n_selected=n,groups=sorted(selected),nested=True,rounding='max(1,floor(fraction*n_pool_groups)); first n of one seed+777 permutation'))
    checks.append(dict(protocol=protocol,dataset=ds,seed=seed,fraction=fr,**ch,n_train=int(tr.sum()),n_validation=int(va.sum()),n_test=int(test.sum())))
    early=bool(va.any() and len(np.unique(y[va]))>1)
    for condition in ['continuation','target_only']:
     run=tag+'_'+condition
     try:
      if not tr.any() or len(np.unique(y[tr]))<2:raise ValueError('Empty or single-class training subset; no redraw')
      med=sm if condition=='continuation' else raw.loc[tr].median();med.to_csv(O/'models'/f'{run}_median.csv');x=raw.fillna(med)
      pd.DataFrame([dict(feature=c,train_missing_rate=float(raw.loc[tr,c].isna().mean()),validation_missing_rate=float(raw.loc[va,c].isna().mean()) if va.any() else None,test_missing_rate=float(raw.loc[test,c].isna().mean()),median=None if pd.isna(med[c]) else med[c]) for c in COLS]).to_csv(O/'tables'/f'{run}_missing.csv',index=False)
      cp,cfg=model_config(run,protocol,seed,COLS,target=ds,fraction=fr,condition=condition,recording_rule=info,group_budget=selection[-1],validation_rule='20% selected vehicle groups; recordings may overlap train/validation; no test recording overlap',split_checks=ch,source_checkpoint=str(bp.relative_to(O)) if condition=='continuation' else None,source_checkpoint_sha256=bh if condition=='continuation' else None,median_fit='all source training rows' if condition=='continuation' else 'target selected training rows only',adaptation=dict(rounds=200,eta=.02,max_depth=6,subsample=.8,colsample_bytree=.8,patience=20,eval_metric='auc',nthread=8),early_stopping_enabled=early,early_stopping_fallback=None if early else 'No/tiny or single-class validation: use all200 rounds; no validation AUC selection')
      pars=dict(objective='binary:logistic',eval_metric='auc',eta=.02,max_depth=6,subsample=.8,colsample_bytree=.8,tree_method='hist',seed=seed,nthread=8)
      ev=[(xgb.DMatrix(x.loc[va],label=y[va]),'validation')] if early else []
      start=time.monotonic();m=xgb.train(pars,xgb.DMatrix(x.loc[tr],label=y[tr]),num_boost_round=200,xgb_model=base.get_booster() if condition=='continuation' else None,evals=ev,early_stopping_rounds=20 if early else None,verbose_eval=False)
      assert base.get_booster().num_boosted_rounds()==500
      best=int(m.best_iteration) if early else None;end=best+1 if best is not None else m.num_boosted_rounds();mp,mh=save_model(run,m);p=m.predict(xgb.DMatrix(x.loc[test]),iteration_range=(0,end))
      row=predict_save(run,protocol,seed,ds,d.loc[test],p,mp,mh,cp,sp,fraction=fr,condition=condition,iteration_end=end);row.update(best_iteration=best,num_rounds=m.num_boosted_rounds(),early_stopping_enabled=early,n_train=int(tr.sum()),n_validation=int(va.sum()),selected_groups=n,pool_groups=len(groups));rows.append(row)
      jwrite(O/'logs'/f'{run}_completed.json',dict(seconds=time.monotonic()-start,best_iteration=best,num_rounds=m.num_boosted_rounds(),iteration_end=end));print('DONE',run,row['auc'],flush=True)
     except Exception as e:failure('A2',run,e)
   jwrite(O/'configs'/f'{protocol}_{ds}_s{seed}_nested_groups.json',selection)
 pd.DataFrame(rows).to_csv(O/'tables/A2_metrics.csv',index=False);pd.DataFrame(checks).to_csv(O/'tables/A2_split_checks.csv',index=False)

if __name__=='__main__':
 frozen=json.loads((O/'configs/code_freeze_A_v1.json').read_text())
 for name,h in frozen['code_hashes'].items():assert sha(O/name)==h,name
 started=time.time();data={ds:load(ds) for ds in DATASETS};print('INPUT_ROWS',{d:len(x) for d,x in data.items()},flush=True)
 a1(data);a2(data);jwrite(O/'logs/A_v1_completed.json',dict(elapsed_seconds=time.time()-started,completed_at=time.strftime('%Y-%m-%dT%H:%M:%S'),failures=[p.name for p in (O/'logs').glob('failure_*.json')]))
