from common_v1 import *
import importlib,datetime
CAND=['ego_speed','ego_acc','preceding_vx','dhw','thw','ttc']
mods={'highD':('highd_adapter','HighdAdapter'),'NGSIM':('ngsim_adapter','NgsimAdapter'),'MiTra':('mitra_adapter','MitraAdapter'),'ETRI':('etri_adapter','EtriAdapter'),'uniD':('unid_adapter','UnidAdapter'),'exiD':('exid_adapter','ExidAdapter')}
# Side IDs have no data-flow into these six base columns. Do not compute their expensive unrelated branch.
def skip_side(d,**kw):
 d=d.copy()
 for side in ['left','right']:
  for p in ['preceding','alongside','following']:d[f'{side}_{p}_id']=0
 return d

def frame_raw(ds,raw):
 return (raw['Time [s]']*30).round().astype(int) if ds=='MiTra' else raw[{'NGSIM':'Frame_ID','ETRI':'프레임 인덱스'}.get(ds,'frame')].astype(int)
def project(c):return c.rename(columns={'vx':'ego_speed','ax':'ego_acc'}).set_index(['vehicle_id','frame'])[CAND]

def select():
 rows=[]
 for ds in DATASETS:
  p=O/'predictions'/f'A1_recording_full48_s42__{ds if ds in TARGETS else "source_test"}.parquet'
  d=pd.read_parquet(p);d=d[d.dataset.eq(ds)]
  rr=np.sort(d.recording_id.unique());np.random.RandomState(42).shuffle(rr)
  for rid in (rr if ds=='ETRI' else rr[:2]):
   z=d[d.recording_id.eq(rid)].sort_values(['frame','vehicle_id']);tt=np.sort(z.frame.unique())
   for ix in np.unique([0,len(tt)//2,len(tt)-1]):
    q=z[z.frame.eq(tt[ix])].iloc[0];rows.append({k:q[k] for k in KEYS})
 return pd.DataFrame(rows)
def execute():
 ids=pd.read_csv(O/'configs/B_sample_ids.csv');out=[];fail=[];rawinfo=[];mapping=[]
 for (ds,rid),idsr in ids.groupby(['dataset','recording_id'],sort=False):
  try:
   mn,cn=mods[ds];mod=importlib.import_module('src.data.'+mn)
   if hasattr(mod,'compute_side_neighbors'):mod.compute_side_neighbors=skip_side
   a=getattr(mod,cn)(R/'data/raw'/ds);raw=a.load_raw(int(rid));fr=frame_raw(ds,raw)
   full=project(a.to_canonical(raw.copy(),int(rid)));saved=load(ds).query('recording_id == @rid').set_index(['vehicle_id','frame'])
   rawinfo.append(dict(dataset=ds,recording_id=rid,raw_rows=len(raw),frame_min=int(fr.min()),frame_max=int(fr.max()),columns='|'.join(raw.columns),upstream_processed=ds in ['uniD','exiD']))
   if ds in ['uniD','exiD']:
    for dst,src in [('vx','lonVelocity'),('vx','xVelocity'),('ax','lonAcceleration'),('ax','xAcceleration')]:
     if src in raw:
      u=raw[dst].to_numpy();v=raw[src].to_numpy();v=np.abs(v) if dst=='vx' else v
      mapping.append(dict(dataset=ds,recording_id=rid,derived=dst,native=src,max_abs_diff=float(np.nanmax(np.abs(u-v))),n_rows=len(raw)))
   for q in idsr.itertuples(index=False):
    cut=raw.loc[fr.le(q.frame)].copy();part=project(a.to_canonical(cut,int(rid)))
    key=(q.vehicle_id,q.frame)
    for c in CAND:
     x=float(full.loc[key,c]);y=float(part.loc[key,c]);s=float(saved.loc[key,c]);same=bool(np.isclose(x,y,atol=1e-8,rtol=1e-7,equal_nan=True));link=bool(np.isclose(x,s,atol=1e-7,rtol=1e-6,equal_nan=True))
     out.append(dict(dataset=ds,recording_id=rid,vehicle_id=q.vehicle_id,frame=q.frame,feature=c,full=x,censored=y,stored=s,abs_diff=abs(x-y),invariant=same,stored_value_matches=link,raw_full_rows=len(raw),raw_censored_rows=len(cut)))
   print('B2 DONE',ds,rid,flush=True)
  except Exception as e:fail.append(dict(dataset=ds,recording_id=rid,error=repr(e),traceback=traceback.format_exc()));print('B2 FAILED',ds,rid,repr(e),flush=True)
 pd.DataFrame(out).to_csv(O/'tables/B2_censoring.csv',index=False)
 pd.DataFrame(rawinfo).to_csv(O/'tables/B2_raw_inputs.csv',index=False)
 pd.DataFrame(mapping).to_csv(O/'tables/B2_upstream_velocity_mapping.csv',index=False)
 jwrite(O/'logs/B_v1_completed.json',dict(completed=datetime.datetime.now().isoformat(),failures=fail,n_comparisons=len(out),B3_status='not executable under available provenance'))
if __name__=='__main__':
 if '--freeze' in sys.argv:
  select().to_csv(O/'configs/B_sample_ids.csv',index=False)
  jwrite(O/'configs/code_freeze_B_v1.json',dict(frozen_before_execution=datetime.datetime.now().isoformat(),runner_sha256=sha(__file__),sample_ids_sha256=sha(O/'configs/B_sample_ids.csv'),candidate_features=CAND,selected_features=[],selection_reason='ETRI all six depend on full-record range orientation; uniD relative features are constants, not measured values; exiD preceding_vx is abs(leadDV), not established leader absolute speed. No common established temporal subset. B3 blocked before performance fitting.',tolerance=dict(atol=1e-8,rtol=1e-7),selection='seed42 A1 actual evaluation rows; shuffled recordings seed42 first two, all five ETRI; earliest/middle/latest unique evaluation frame, smallest vehicle at each time',scope='Fresh actual adapter candidate branch on raw and all-vehicle raw<=t, no canonical/stat cache. Unused side-ID branch bypassed and never used for safe-feature declaration. Vendor preprocessing and uniD/exiD upstream final CSV not rerun; no sensor causal claim.'))
 else:execute()
