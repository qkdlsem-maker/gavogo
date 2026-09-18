from pathlib import Path
import json,hashlib,sys,time,traceback
import numpy as np,pandas as pd,xgboost as xgb
from sklearn.metrics import roc_auc_score,brier_score_loss
O=Path(__file__).resolve().parents[1];R=O.parents[1];sys.path.insert(0,str(O/'code/r16_snapshot'))
from src.features.kinematic import FEATURE_COLS as KIN
from src.features.game_theory import NASH_FEATURE_COLS as GT
from src.models.metrics import calibration_bins
SOURCES=['highD','NGSIM','MiTra'];TARGETS=['ETRI','uniD','exiD'];DATASETS=SOURCES+TARGETS;SEEDS=[42,0,1];COLS=KIN+GT;META=['dataset','recording_id','vehicle_id','frame','label'];KEYS=META[:-1]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**23),b''):h.update(b)
 return h.hexdigest()
def jwrite(p,d):
 Path(p).parent.mkdir(parents=True,exist_ok=True)
 with Path(p).open('x') as f:json.dump(d,f,ensure_ascii=False,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))
def load(ds):
 assert ds in DATASETS
 d=pd.read_csv(R/f'data/processed/{ds}_gt_3s.csv');d['dataset']=ds
 for c in ['recording_id','vehicle_id','frame','label']:d[c]=d[c].astype(int)
 if d.duplicated(KEYS).any():raise ValueError(f'duplicate keys in {ds}')
 return d

def groupkeys(d):return d.dataset.astype(str)+'|'+d.recording_id.astype(str)+'|'+d.vehicle_id.astype(str)
def physical_recording(d):return np.where(d.dataset.eq('exiD') & d.recording_id.ge(1000),d.recording_id-1000,d.recording_id)
def recording_holdout(d,seed,physical=False):
 rec=physical_recording(d) if physical else d.recording_id.to_numpy();u=np.sort(np.unique(rec))
 if len(u)<2:raise ValueError('Fewer than two recordings; no substitute grouping permitted')
 u=u.copy();np.random.RandomState(seed).shuffle(u);n=max(1,int(np.floor(.3*len(u))))
 assert n<len(u);held=u[:n];test=np.isin(rec,held)
 return ~test,test,dict(unit='physical_original_recording' if physical else 'saved_recording_id',n_recordings=len(u),test_recordings=sorted(map(int,held)),rounding='max(1,floor(0.30*n_recordings))',seed=seed)
def validate_split(d,roles,require_recording=True,physical=False):
 a=d[np.asarray(roles)=='train'];b=d[np.asarray(roles)=='test'];va=d[np.asarray(roles)=='validation']
 def recset(q,ph=False):return set(zip(q.dataset,physical_recording(q) if ph else q.recording_id))
 g=lambda q:set(groupkeys(q));out={'train_test_vehicle_overlap':len(g(a)&g(b)),'train_validation_vehicle_overlap':len(g(a)&g(va)),'validation_test_vehicle_overlap':len(g(va)&g(b)),'train_test_recording_overlap':len(recset(a)&recset(b)),'validation_test_recording_overlap':len(recset(va)&recset(b)),'train_validation_recording_overlap':len(recset(a)&recset(va)),'train_test_physical_recording_overlap':len(recset(a,True)&recset(b,True)),'validation_test_physical_recording_overlap':len(recset(va,True)&recset(b,True))}
 assert not any(out[k] for k in ['train_test_vehicle_overlap','train_validation_vehicle_overlap','validation_test_vehicle_overlap'])
 if require_recording:assert out['train_test_recording_overlap']==out['validation_test_recording_overlap']==0
 if physical:assert out['train_test_physical_recording_overlap']==out['validation_test_physical_recording_overlap']==0
 return out

def save_split(name,d,roles,require_recording=True,physical=False):
 out=validate_split(d,roles,require_recording,physical);q=d[META].copy();q['physical_recording_id']=physical_recording(d);q['role']=roles
 p=O/'splits'/f'{name}.parquet';assert not p.exists();q.to_parquet(p,index=False);return p,out

def counts(d):return dict(n_rows=len(d),n_vehicles=int(groupkeys(d).nunique()),n_recordings=int(d[['dataset','recording_id']].drop_duplicates().shape[0]),n_physical_recordings=len(set(zip(d.dataset,physical_recording(d)))),positive_count=int(d.label.sum()),positive_rate=float(d.label.mean()) if len(d) else None)
def clean(d,cols,median=None):
 x=d[cols].replace([np.inf,-np.inf],np.nan);median=x.median() if median is None else median
 return x.fillna(median),median

def model_config(run,protocol,seed,cols,**extra):
 cfg={'run':run,'protocol':protocol,'seed':seed,'source_datasets':SOURCES,'target_datasets':TARGETS,'EMT_excluded':True,'feature_set':'full48' if cols==COLS else 'within_pipeline_temporal_subset','features':cols,'horizon_sec':3,'threads':8,'xgb_source':dict(n_estimators=500,max_depth=6,learning_rate=.05,tree_method='hist',n_jobs=8),'offline_roadframe_event_sign_retained':True,'runner_code_hashes':json.loads((O/'configs/code_freeze_A_v1.json').read_text())['code_hashes'],**extra}
 p=O/'configs'/f'{run}.json';jwrite(p,cfg);return p,cfg

def fit_source(x,y,seed):
 if len(np.unique(y))<2:raise ValueError('Source training labels have fewer than two classes')
 m=xgb.XGBClassifier(n_estimators=500,max_depth=6,learning_rate=.05,tree_method='hist',n_jobs=8,random_state=seed,eval_metric='auc');m.fit(x,y,verbose=False);return m

def save_model(run,m):
 p=O/'models'/f'{run}.ubj';assert not p.exists();m.save_model(p);return p,sha(p)
def predict_save(run,protocol,seed,ds,d,p,model_path,model_hash,config_path,split_path,feature_set='full48',fraction=None,condition=None,iteration_end=500):
 if len(p)!=len(d) or not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError('Invalid predictions')
 q=d[META].copy();q['score']=p;q['seed']=seed;q['split_role']='test';q['eval_set']=ds;q['physical_recording_id']=physical_recording(d);q['model_sha256']=model_hash;q['config_sha256']=sha(config_path);q['protocol']=protocol;q['iteration_range_end_exclusive']=iteration_end
 if q.duplicated(KEYS).any():raise ValueError('duplicate prediction keys')
 path=O/'predictions'/f'{run}__{ds}.parquet';assert not path.exists();q.to_parquet(path,index=False)
 y=d.label.to_numpy();auc=roc_auc_score(y,p) if len(np.unique(y))>1 else None
 row=dict(run=run,model='XGBoost',protocol=protocol,feature_set=feature_set,seed=seed,eval_set=ds,fraction=fraction,condition=condition,EMT_excluded=True,auc=auc,auc_defined=auc is not None,ece=calibration_bins(y,np.asarray(p))[0],brier=brier_score_loss(y,p),**counts(d),prediction=str(path.relative_to(O)),prediction_sha256=sha(path),model_path=str(model_path.relative_to(O)),model_sha256=model_hash,config_path=str(config_path.relative_to(O)),config_sha256=sha(config_path),split_path=str(split_path.relative_to(O)),split_sha256=sha(split_path),iteration_range_start=0,iteration_range_end_exclusive=iteration_end)
 jwrite(O/'verification'/f'{run}__{ds}.json',row);return row

def failure(stage,run,error):
 print('FAILED',stage,run,repr(error),flush=True);jwrite(O/'logs'/f'failure_{run}.json',{'stage':stage,'run':run,'error':repr(error),'traceback':traceback.format_exc()})
