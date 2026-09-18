"""unid_adapter.py — uniD(00_final_unid.csv) → canonical.
uniD는 이웃 ID 없음 → compute_side_neighbors() 사용.
dhw/thw/ttc 미제공 → 0/999 처리.
"""
import pandas as pd
from pathlib import Path
from src.data.schema import CANONICAL_COLUMNS, NO_NEIGHBOR_ID, validate_canonical
from src.data.base_adapter import compute_side_neighbors


class UnidAdapter:
    def __init__(self, raw_dir: Path):
        self.raw_dir = Path(raw_dir)

    def load_raw(self, recording_id: int) -> pd.DataFrame:
        rid = f"{recording_id:02d}"
        return pd.read_csv(self.raw_dir / f"{rid}_final_unid.csv", low_memory=False)

    def to_canonical(self, raw: pd.DataFrame, recording_id: int) -> pd.DataFrame:
        df = raw.rename(columns={"trackId": "vehicle_id"}).copy()
        df["recording_id"] = recording_id
        df = compute_side_neighbors(df, lane_smaller_is_left=True)

        out = pd.DataFrame({
            "dataset":            "uniD",
            "recording_id":       recording_id,
            "vehicle_id":         df["vehicle_id"].astype(int),
            "frame":              df["frame"].astype(int),
            "x":                  df["x"],
            "y":                  df["y"],
            "vx":                 df["vx"].abs(),
            "vy":                 df["vy"],
            "ax":                 df["ax"],
            "ay":                 df["ay"],
            "lane_id":            df["lane_id"].astype(int),
            "dhw":                0.0,
            "thw":                0.0,
            "ttc":                999.0,
            "preceding_vx":       0.0,
            "preceding_id":       NO_NEIGHBOR_ID,
            "following_id":       NO_NEIGHBOR_ID,
            "left_preceding_id":  df["left_preceding_id"].astype(int),
            "left_alongside_id":  df["left_alongside_id"].astype(int),
            "left_following_id":  df["left_following_id"].astype(int),
            "right_preceding_id": df["right_preceding_id"].astype(int),
            "right_alongside_id": df["right_alongside_id"].astype(int),
            "right_following_id": df["right_following_id"].astype(int),
        })
        return out[CANONICAL_COLUMNS]

    def get_canonical(self, recording_id: int) -> pd.DataFrame:
        raw = self.load_raw(recording_id)
        canon = self.to_canonical(raw, recording_id)
        validate_canonical(canon, strict=True)
        return canon
