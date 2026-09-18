import unittest
import numpy as np
import pandas as pd
from src.models.baselines import build_sequences, SEQ_COLS
from src.data.base_adapter import compute_side_neighbors
from src.models.metrics import calibration_bins

class RecordingBoundaries(unittest.TestCase):
    def track(self, rec, speed, dataset='a'):
        d=pd.DataFrame({c:[0.]*4 for c in SEQ_COLS})
        d['vx']=speed;d['recording_id']=rec;d['dataset']=dataset
        d['vehicle_id']=1;d['frame']=[1,2,3,4]
        return d
    def test_sequence_recording_invariance(self):
        a=self.track('A',1);b=self.track('B',9)
        samples=pd.DataFrame([dict(dataset='a',recording_id='A',vehicle_id=1,frame=4,label=1)])
        x,_=build_sequences(samples,a,K=4);mixed,_=build_sequences(samples,pd.concat([a,b]),K=4)
        np.testing.assert_array_equal(x,mixed);np.testing.assert_array_equal(x[0,:,0],[1]*4)
    def test_dataset_boundary_and_future(self):
        a=self.track('A',1);b=self.track('A',9,'b')
        samples=pd.DataFrame([dict(dataset='a',recording_id='A',vehicle_id=1,frame=2,label=1)])
        x,_=build_sequences(samples,pd.concat([a,b]),K=4)
        np.testing.assert_array_equal(x[0,:,0],[0,0,1,1])
    def test_duplicate_identity_rejected(self):
        a=self.track('A',1);samples=a.iloc[:1].assign(label=1)
        with self.assertRaises(ValueError):build_sequences(samples,pd.concat([a,a]))
    def test_neighbors_recording_invariance(self):
        a=pd.DataFrame(dict(recording_id=['A','A'],frame=[1,1],vehicle_id=[1,2],x=[0.,10.],lane_id=[2,1]))
        b=pd.DataFrame(dict(recording_id=['B'],frame=[1],vehicle_id=[9],x=[1.],lane_id=[1]))
        isolated=compute_side_neighbors(a);mixed=compute_side_neighbors(pd.concat([a,b],ignore_index=True))
        self.assertEqual(mixed.loc[0,'left_preceding_id'],2)
        np.testing.assert_array_equal(isolated.filter(like='_id').values,mixed.iloc[:2].filter(like='_id').values)
        self.assertNotIn('left_preceding_id',a.columns)
    def test_probability_one_and_endpoints(self):
        self.assertEqual(calibration_bins([0],[1])[0],1.)
        self.assertEqual(calibration_bins([0,1],[0,1])[0],0.)
        self.assertAlmostEqual(calibration_bins([1,0],[0,1])[0],1.)
    def test_invalid_probability_rejected(self):
        with self.assertRaises(ValueError):calibration_bins([0],[1.01])
    def test_pixel_annotation_columns(self):
        import tempfile
        from pathlib import Path
        from src.data.emt_annotations import read_kitti_pixel_tracks
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'video_test.txt'
            p.write_text('1 1 Car 0 0 0 100 200 300 600 0 0 0 0 0 0 0\n')
            d=read_kitti_pixel_tracks(p)
            self.assertEqual(d.bbox_center_x_px.iloc[0],200)
            self.assertEqual(d.bbox_center_y_px.iloc[0],400)
            self.assertNotIn('lane_id',d.columns)
    def test_best_iteration_on_fitted_booster(self):
        import xgboost as xgb
        from src.models.train import predict_best
        rng=np.random.RandomState(7);x=rng.normal(size=(160,4));y=rng.randint(0,2,160)
        train=xgb.DMatrix(x[:100],label=y[:100]);valid=xgb.DMatrix(x[100:],label=y[100:])
        b=xgb.train(dict(objective='binary:logistic',eval_metric='logloss',max_depth=2,seed=7,nthread=1),train,num_boost_round=50,evals=[(valid,'val')],early_stopping_rounds=5,verbose_eval=False)
        self.assertLess(b.best_iteration+1,b.num_boosted_rounds())
        np.testing.assert_array_equal(predict_best(b,valid),b[:b.best_iteration+1].predict(valid))
        self.assertTrue(np.any(predict_best(b,valid)!=b.predict(valid)))
if __name__=='__main__':unittest.main()
