"""ngsim_adapter.py — NGSIM → canonical. 10fps, ft→m, Local_Y=종/Local_X=횡.
좌우 이웃 미제공 → compute_side_neighbors 로 계산."""
import numpy as np
import pandas as pd
from src.data.base_adapter import BaseAdapter, compute_side_neighbors
from src.data.schema import CANONICAL_COLUMNS

FT2M = 0.3048


class NgsimAdapter(BaseAdapter):
    FILES = {1: "trajectories-0400-0415.csv",
             2: "trajectories-0500-0515.csv",
             3: "trajectories-0515-0530.csv"}

    def load_raw(self, recording_id):
        df = pd.read_csv(self.raw_dir / self.FILES[recording_id])
        df.columns = df.columns.str.strip()
        return df

    def to_canonical(self, raw, recording_id):
        df = raw
        out = pd.DataFrame({
            "dataset": "NGSIM", "recording_id": recording_id,
            "vehicle_id": df["Vehicle_ID"].astype(int), "frame": df["Frame_ID"].astype(int),
            "x": df["Local_Y"] * FT2M, "y": df["Local_X"] * FT2M,
            "vx": df["v_Vel"] * FT2M, "vy": 0.0,
            "ax": df.get("v_Acc", 0.0) * FT2M, "ay": 0.0,
            "lane_id": df["Lane_ID"].astype(int),
            "dhw": df.get("Space_Headway", np.nan) * FT2M,
            "thw": df.get("Time_Headway", np.nan), "ttc": np.nan,
            "preceding_id": df.get("Preceding", 0).fillna(0).astype(int),
            "following_id": df.get("Following", 0).fillna(0).astype(int),
        })
        # 전방차 속도 + TTC 근사
        sm = out.set_index(["frame", "vehicle_id"])["vx"]
        pv = sm.reindex(list(zip(out["frame"], out["preceding_id"]))).values
        out["preceding_vx"] = np.where(out["preceding_id"].values == 0, out["vx"].values, pv)
        out["preceding_vx"] = out["preceding_vx"].astype(float)
        clo = out["vx"] - out["preceding_vx"]
        out["ttc"] = np.clip(np.where(clo > 0.1, out["dhw"] / clo, 999.0), 0, 999)
        # 좌우 이웃 계산 (NGSIM lane 작을수록 왼쪽)
        out = compute_side_neighbors(out, lane_smaller_is_left=True)
        return out[CANONICAL_COLUMNS]
