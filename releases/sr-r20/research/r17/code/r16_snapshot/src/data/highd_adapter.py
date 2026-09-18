"""highd_adapter.py — highD → canonical. 25fps, m단위, 좌우이웃 직접 제공.
양방향 도로: drivingDirection 2는 좌우 반대 → 보정."""
import pandas as pd
from src.data.base_adapter import BaseAdapter
from src.data.schema import CANONICAL_COLUMNS, NO_NEIGHBOR_ID


class HighdAdapter(BaseAdapter):
    def load_raw(self, recording_id):
        rid = f"{recording_id:02d}"
        t = pd.read_csv(self.raw_dir / f"{rid}_tracks.csv")
        meta = pd.read_csv(self.raw_dir / f"{rid}_tracksMeta.csv")
        t["dir"] = t["id"].map(dict(zip(meta["id"], meta["drivingDirection"])))
        return t

    def to_canonical(self, raw, recording_id):
        df = raw
        d2 = df["dir"] == 2
        out = pd.DataFrame({
            "dataset": "highD", "recording_id": recording_id,
            "vehicle_id": df["id"].astype(int), "frame": df["frame"].astype(int),
            "x": df["x"], "y": df["y"], "vx": df["xVelocity"].abs(),
            "vy": df["yVelocity"], "ax": df["xAcceleration"], "ay": df["yAcceleration"],
            "lane_id": df["laneId"].astype(int),
            "dhw": df["dhw"], "thw": df["thw"], "ttc": df["ttc"],
            "preceding_vx": df["precedingXVelocity"].abs(),
            "preceding_id": df["precedingId"].fillna(0).astype(int),
            "following_id": df["followingId"].fillna(0).astype(int),
        })

        def pick(cl, cr):
            v = df[cl].copy(); v[d2] = df.loc[d2, cr]
            return v.fillna(NO_NEIGHBOR_ID).astype(int)
        out["left_preceding_id"]  = pick("leftPrecedingId", "rightPrecedingId")
        out["left_alongside_id"]  = pick("leftAlongsideId", "rightAlongsideId")
        out["left_following_id"]  = pick("leftFollowingId", "rightFollowingId")
        out["right_preceding_id"] = pick("rightPrecedingId", "leftPrecedingId")
        out["right_alongside_id"] = pick("rightAlongsideId", "leftAlongsideId")
        out["right_following_id"] = pick("rightFollowingId", "leftFollowingId")
        return out[CANONICAL_COLUMNS]
