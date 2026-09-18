"""etri_adapter.py — 한국 ETRI → canonical. cp949, 10fps 가정.
TM좌표(East/North) 사용. 속도·이웃 미제공 → 위치차분으로 속도, 계산으로 이웃.
파일: 180906_s2d_l3_{NN}_crop_pp_lane_crt_lccrt.csv (recording_id = NN)."""
import numpy as np
import pandas as pd
from src.data.base_adapter import BaseAdapter, compute_side_neighbors, orient_longitudinal
from src.data.schema import CANONICAL_COLUMNS

FPS = 10


class EtriAdapter(BaseAdapter):
    def load_raw(self, recording_id):
        path = self.raw_dir / f"180906_s2d_l3_{recording_id:02d}_crop_pp_lane_crt_lccrt.csv"
        return pd.read_csv(path, encoding="cp949")

    def to_canonical(self, raw, recording_id):
        df = raw.rename(columns={
            "프레임 인덱스": "frame", "차량 ID": "vid",
            "East (TM 좌표)": "east", "North (TM 좌표)": "north", "차선": "lane",
        })
        out = pd.DataFrame({
            "dataset": "ETRI", "recording_id": recording_id,
            "vehicle_id": df["vid"].astype(int), "frame": df["frame"].astype(int),
            "x": df["east"], "y": df["north"],
            "vx": 0.0, "vy": 0.0, "ax": 0.0, "ay": 0.0,
            "lane_id": df["lane"].astype(int),
            "dhw": np.nan, "thw": np.nan, "ttc": np.nan, "preceding_vx": np.nan,
            "preceding_id": 0, "following_id": 0,
        })
        out = orient_longitudinal(out)   # TM → 종방향 정렬

        # 속도/가속도 = 위치 차분 (차량별, 시간순)
        out = out.sort_values(["vehicle_id", "frame"]).reset_index(drop=True)
        dt = 1.0 / FPS
        g = out.groupby("vehicle_id")
        out["vx"] = (g["x"].diff() / dt).fillna(0.0)
        out["vy"] = (g["y"].diff() / dt).fillna(0.0)
        out["ax"] = (g["vx"].diff() / dt).fillna(0.0)
        out["ay"] = (g["vy"].diff() / dt).fillna(0.0)
        out["vx"] = out["vx"].abs()

        # 좌우 이웃 + 전방차/dhw/ttc 계산
        out = compute_side_neighbors(out, lane_smaller_is_left=True)
        out = self._front_relations(out)
        return out[CANONICAL_COLUMNS]

    def _front_relations(self, out):
        """같은 lane 앞차를 preceding 으로, dhw/ttc 근사."""
        pid = np.zeros(len(out), dtype=int)
        dhw = np.full(len(out), np.nan)
        pvx = np.full(len(out), np.nan)
        for _, g in out.groupby("frame", sort=False):
            for ln, sub in g.groupby("lane_id"):
                s = sub.sort_values("x")
                xs, ids, vs = s["x"].values, s["vehicle_id"].values, s["vx"].values
                idxs = s.index.values
                for i in range(len(s) - 1):
                    pid[idxs[i]] = ids[i + 1]
                    dhw[idxs[i]] = xs[i + 1] - xs[i]
                    pvx[idxs[i]] = vs[i + 1]
        out["preceding_id"] = pid
        out["dhw"] = dhw
        out["preceding_vx"] = np.where(np.isnan(pvx), out["vx"].values, pvx)
        clo = out["vx"] - out["preceding_vx"]
        out["ttc"] = np.clip(np.where(clo > 0.1, out["dhw"] / clo, 999.0), 0, 999)
        out["thw"] = np.where(out["vx"] > 0.1, out["dhw"] / out["vx"], np.nan)
        return out
