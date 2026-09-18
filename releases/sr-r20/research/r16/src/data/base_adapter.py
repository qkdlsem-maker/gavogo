"""
base_adapter.py — 어댑터 추상 베이스 + 공용 헬퍼.

새 데이터셋: BaseAdapter 상속 → load_raw(), to_canonical() 두 개만 구현.
좌우 이웃이 원본에 없으면 compute_side_neighbors() 헬퍼를 쓰면 된다.
"""
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

from src.data.schema import validate_canonical, NO_NEIGHBOR_ID


class BaseAdapter(ABC):
    def __init__(self, raw_dir):
        self.raw_dir = raw_dir

    @abstractmethod
    def load_raw(self, recording_id): ...

    @abstractmethod
    def to_canonical(self, raw, recording_id): ...

    def get_canonical(self, recording_id):
        raw = self.load_raw(recording_id)
        canon = self.to_canonical(raw, recording_id)
        validate_canonical(canon, strict=True)
        return canon


def compute_side_neighbors(df, lane_smaller_is_left=True):
    """
    좌우 이웃 ID 미제공 데이터셋용 (NGSIM/ETRI).
    각 (recording, frame, vehicle)에서 lane±1 차선의 전방/후방/병렬 가장 가까운 차 ID.
    df는 columns: frame, vehicle_id, x, lane_id 필요. (x=종방향)
    여러 recording을 전달할 때 recording_id가 필수이며 dataset도 있으면 구분한다.
    recording_id가 없는 입력은 호출자가 단일 recording임을 보장해야 한다.
    """
    df = df.copy()
    if not df.index.is_unique:
        raise ValueError("Neighbor assignment requires unique row indices")
    group_cols = [c for c in ("dataset", "recording_id") if c in df] + ["frame"]
    if df[group_cols + ["vehicle_id"]].isna().any().any():
        raise ValueError("Neighbor identity keys must not be missing")
    if df.duplicated(group_cols + ["vehicle_id"]).any():
        raise ValueError("Duplicate neighbor identity keys; supply recording IDs")
    for c in ["left_preceding_id", "left_alongside_id", "left_following_id",
              "right_preceding_id", "right_alongside_id", "right_following_id"]:
        df[c] = NO_NEIGHBOR_ID

    left_delta = -1 if lane_smaller_is_left else +1
    right_delta = -left_delta

    for _, g in df.groupby(group_cols, sort=False):
        by_lane = {ln: sub for ln, sub in g.groupby("lane_id")}
        for idx, row in g.iterrows():
            ex, lane = row["x"], row["lane_id"]
            for side, delta in [("left", left_delta), ("right", right_delta)]:
                sub = by_lane.get(lane + delta)
                if sub is None or len(sub) == 0:
                    continue
                dx = sub["x"].values - ex
                ids = sub["vehicle_id"].values
                f, b = dx > 0, dx < 0
                if f.any():
                    df.at[idx, f"{side}_preceding_id"] = int(ids[f][np.argmin(dx[f])])
                if b.any():
                    df.at[idx, f"{side}_following_id"] = int(ids[b][np.argmax(dx[b])])
                df.at[idx, f"{side}_alongside_id"] = int(ids[np.argmin(np.abs(dx))])
    return df


def orient_longitudinal(df):
    """UTM/TM 좌표에서 종방향 자동 판정: 범위 큰 축을 x(종)로, 작은 축을 y(횡)로.
    highway 직선 가정. df는 임시 x,y 컬럼을 가짐."""
    xr = df["x"].max() - df["x"].min()
    yr = df["y"].max() - df["y"].min()
    if yr > xr:                       # y축이 주행방향이면 swap
        df["x"], df["y"] = df["y"].copy(), df["x"].copy()
    return df
