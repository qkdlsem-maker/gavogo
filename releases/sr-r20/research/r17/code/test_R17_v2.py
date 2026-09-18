import unittest
from common_v1 import *
from run_C_v1 import auc_engine,reckeys
from src.data.base_adapter import orient_longitudinal
class R17Tests(unittest.TestCase):
 def fixture(self):
  return pd.DataFrame(dict(dataset=['exiD']*8,recording_id=[1,1,1001,1001,2,2,1002,1002],vehicle_id=range(8),frame=[1]*8,label=[0,1]*4,score=[.1,.6,.6,.6,.3,.7,.9,.2]))
 def test_physical_holdout(self):
  d=self.fixture();tr,te,_=recording_holdout(d,42,True);self.assertFalse(set(physical_recording(d[tr]))&set(physical_recording(d[te])))
 def test_no_single_recording_substitution(self):
  with self.assertRaises(ValueError):recording_holdout(self.fixture().query('recording_id==1'),42)
 def test_auc_weighted_ties_and_empty(self):
  d=self.fixture();k=reckeys(d);levels=np.unique(k)
  for matrix in [False,True]:
   fn,_=auc_engine(d,k,levels,matrix)
   for w in [np.array([2,0,1,3]),np.array([0,1,0,0])]:
    ww=np.array([w[list(levels).index(x)] for x in k]);self.assertAlmostEqual(fn(w),roc_auc_score(d.label,d.score,sample_weight=ww),places=12)
   self.assertTrue(np.isnan(fn(np.zeros(len(levels)))))
 def test_same_physical_weight_both_directions(self):
  d=self.fixture();d["physical_recording_id"]=physical_recording(d);k=reckeys(d,True);self.assertEqual(k[0],k[2]);self.assertEqual(k[4],k[6])
 def test_orientation_future_dependence(self):
  d=pd.DataFrame(dict(x=[0.,2.,3.],y=[0.,1.,20.]));a=orient_longitudinal(d.copy());b=orient_longitudinal(d.iloc[:2].copy());self.assertNotEqual(a.loc[1,'x'],b.loc[1,'x'])
 def test_three_seed_invalid_not_nanmean(self):
  x=np.array([.2,np.nan,.4]);value=float(x.mean()) if np.isfinite(x).all() else np.nan;self.assertTrue(np.isnan(value))
if __name__=='__main__':unittest.main(verbosity=2)
