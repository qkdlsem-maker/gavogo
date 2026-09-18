"""mitra_adapter.py — MiTra → canonical. 30fps, m단위, Speed km/h.
좌우 이웃 직접 제공(Left/Right_Leader/Follower_ID). 종방향 자동 판정.
파일: Data_T{n}/T{n}_DAll.csv (recording_id = n)."""
import numpy as np
import pandas as pd
from src.data.base_adapter import BaseAdapter, orient_longitudinal
from src.data.schema import CANONICAL_COLUMNS, NO_NEIGHBOR_ID


class MitraAdapter(BaseAdapter):
    def load_raw(self, recording_id):
        n = recording_id
        path = self.raw_dir / f"Data_T{n}" / f"T{n}_DAll.csv"
        return pd.read_csv(path)

    def to_canonical(self, raw, recording_id):
        df = raw
        fps = 30
        out = pd.DataFrame({
            "dataset": "MiTra", "recording_id": recording_id,
            "vehicle_id": df["Vehicle_ID"].astype(int),
            "frame": (df["Time [s]"] * fps).round().astype(int),
            "x": df["x [m]"], "y": df["y [m]"],
            "vx": df["Speed [km/h]"] / 3.6, "vy": 0.0,
            "ax": df["Lon. Acc. [ms-2]"], "ay": df["Lat. Acc. [ms-2]"],
            "lane_id": df["Lane"].astype(int),
            "dhw": np.nan, "thw": np.nan, "ttc": np.nan,
            "preceding_id": df["Leader_ID"].replace(-1, 0).fillna(0).astype(int),
            "following_id": df["Follower_ID"].replace(-1, 0).fillna(0).astype(int),
            "left_preceding_id":  df["Left_Leader_ID"].replace(-1, 0).fillna(0).astype(int),
            "left_following_id":  df["Left_Follower_ID"].replace(-1, 0).fillna(0).astype(int),
            "right_preceding_id": df["Right_Leader_ID"].replace(-1, 0).fillna(0).astype(int),
            "right_following_id": df["Right_Follower_ID"].replace(-1, 0).fillna(0).astype(int),
            "left_alongside_id": NO_NEIGHBOR_ID,
            "right_alongside_id": NO_NEIGHBOR_ID,
        })
        out = orient_longitudinal(out)   # UTM → 종방향 정렬

        # dhw/ttc 를 leader 위치로 근사 계산
        sm_x = out.set_index(["frame", "vehicle_id"])["x"]
        sm_v = out.set_index(["frame", "vehicle_id"])["vx"]
        key = list(zip(out["frame"], out["preceding_id"]))
        lead_x = sm_x.reindex(key).values
        lead_v = sm_v.reindex(key).values
        has = out["preceding_id"].values != 0
        dhw = np.where(has, np.abs(lead_x - out["x"].values), np.nan)
        out["dhw"] = dhw
        out["preceding_vx"] = np.where(has, lead_v, out["vx"].values)
        clo = out["vx"] - out["preceding_vx"]
        out["ttc"] = np.clip(np.where(clo > 0.1, out["dhw"] / clo, 999.0), 0, 999)
        out["thw"] = np.where(out["vx"] > 0.1, out["dhw"] / out["vx"], np.nan)
        return out[CANONICAL_COLUMNS]
