"""Portable, no-training metric verification using the extracted R17 evidence only."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score,brier_score_loss
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
rows=[]
for path in sorted((a.root/'predictions').glob('*.parquet')):
 d=pd.read_parquet(path);y=d.label.to_numpy();s=d.score.to_numpy();assert not d.duplicated(['dataset','recording_id','vehicle_id','frame']).any();assert np.isfinite(s).all() and ((s>=0)&(s<=1)).all()
 ece=0.;edges=np.linspace(0,1,11)
 for i in range(10):
  m=(s>=edges[i])&((s<=edges[i+1]) if i==9 else (s<edges[i+1]))
  if m.any():ece+=m.mean()*abs(y[m].mean()-s[m].mean())
 row=dict(prediction='predictions/'+path.name,auc=roc_auc_score(y,s) if len(np.unique(y))==2 else np.nan,ece=ece,brier=brier_score_loss(y,s),n_rows=len(d),n_vehicles=len(d[['dataset','recording_id','vehicle_id']].drop_duplicates()),n_recordings=len(d[['dataset','recording_id']].drop_duplicates()))
 ref=a.root/'verification'/path.with_suffix('.json').name
 if ref.exists():
  saved=json.loads(ref.read_text())
  for k in ['auc','ece','brier']:assert np.isclose(row[k],saved[k],atol=1e-10,equal_nan=True)
 rows.append(row)
pd.DataFrame(rows).to_csv(a.output,index=False);print('Verified',len(rows),'prediction files; output',a.output)
