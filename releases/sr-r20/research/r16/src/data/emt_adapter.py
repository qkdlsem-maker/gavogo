"""emt_adapter.py — EMT(dataset_emt_raw.csv) → canonical.

EMT 원본 컬럼: frame, vehicle_id, label, localX, localY, lane_id, recording_id
- 속도/가속도: frame 간 차분으로 계산 (legacy fps=25; 실제 시간 검증 아님)
- R15 감사: upstream finalize_csv.py의 KITTI bbox 열 오프셋 오류 확인.
  localX/localY는 검증된 미터 좌표가 아니며 lane_id는 KMeans 생성값.
  recording 수정 실험은 이 입력 한계를 해결하지 않는다.
- dhw/thw/ttc: 같은 차선 앞차 기준 계산
- 이웃 ID: compute_side_neighbors() 사용
- width/height 컬럼 제외 (카메라 투영값, 신뢰 불가)
"""
import numpy as np
import pandas as pd
from pathlib import Path
from src.data.schema import CANONICAL_COLUMNS, NO_NEIGHBOR_ID
from src.data.base_adapter import compute_side_neighbors

FPS = 25.0
TTC_CAP = 999.0


def _compute_kinematics(df: pd.DataFrame) -> pd.DataFrame:
    """frame 간 차분으로 vx, vy, ax, ay 계산."""
    df = df.sort_values(["recording_id", "vehicle_id", "frame"]).copy()
    dt = 1.0 / FPS

    df = df.sort_values(["recording_id", "vehicle_id", "frame"]).copy()
    grp = df.groupby(["recording_id", "vehicle_id"])
    frame_diff = grp["frame"].diff().fillna(1)
    actual_dt = frame_diff / FPS  # 실제 시간 간격

    dx = grp["x"].diff()
    dy = grp["y"].diff()

    df["vx"] = (dx / actual_dt).fillna(0.0)
    df["vy"] = (dy / actual_dt).fillna(0.0)

    # 비연속 구간(frame_diff > 2) → 속도 신뢰 불가, 0 처리
    bad = frame_diff > 2
    df.loc[bad, "vx"] = 0.0
    df.loc[bad, "vy"] = 0.0

    df["vx"] = df["vx"].abs().clip(0, 60)  # 60 m/s ≈ 216 km/h cap
    df["ax"] = grp["vx"].diff().fillna(0.0) / actual_dt
    df["ay"] = grp["vy"].diff().fillna(0.0) / actual_dt
    df.loc[bad, "ax"] = 0.0
    df.loc[bad, "ay"] = 0.0

    # 첫 프레임 결측 → 0 채움
    for c in ["vx", "vy", "ax", "ay"]:
        df[c] = df[c].fillna(0.0)

    df["vx"] = df["vx"].abs()
    return df


def _compute_lead_features(df: pd.DataFrame) -> pd.DataFrame:
    """같은 (frame, recording_id, lane_id) 내 앞차(x 큰 쪽) 기준 dhw/thw/ttc."""
    df = df.copy()
    df["dhw"] = np.nan
    df["thw"] = np.nan
    df["ttc"] = np.nan
    df["preceding_id"] = NO_NEIGHBOR_ID
    df["preceding_vx"] = 0.0

    for (rec, frame, lane), g in df.groupby(
        ["recording_id", "frame", "lane_id"], sort=False
    ):
        if len(g) < 2:
            continue
        g_sorted = g.sort_values("x")
        xs = g_sorted["x"].values
        vxs = g_sorted["vx"].values
        ids = g_sorted["vehicle_id"].values
        idxs = g_sorted.index

        for i, idx in enumerate(idxs):
            # 앞차 = 더 큰 x 값 중 가장 가까운 것
            ahead = xs > xs[i]
            if not ahead.any():
                continue
            j = np.where(ahead)[0][0]  # g_sorted 기준 바로 앞
            dx = xs[j] - xs[i]
            rel_v = vxs[i] - vxs[j]   # 접근 속도

            df.at[idx, "dhw"] = dx
            df.at[idx, "thw"] = (dx / vxs[i]) if vxs[i] > 0.5 else TTC_CAP
            df.at[idx, "ttc"] = (dx / rel_v) if rel_v > 0.1 else TTC_CAP
            df.at[idx, "preceding_id"] = int(ids[j])
            df.at[idx, "preceding_vx"] = vxs[j]

    df["dhw"] = df["dhw"].fillna(0.0).clip(0, 999)
    df["thw"] = df["thw"].fillna(TTC_CAP).clip(0, TTC_CAP)
    df["ttc"] = df["ttc"].fillna(TTC_CAP).clip(0, TTC_CAP)
    return df


class EmtAdapter:
    """BaseAdapter 미상속 — EMT는 단일 CSV 파일 구조."""

    def __init__(self, raw_dir: Path):
        self.raw_dir = Path(raw_dir)

    def load_raw(self) -> pd.DataFrame:
        path = self.raw_dir / "dataset_emt_raw.csv"
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        return df

    def to_canonical(self, raw: pd.DataFrame) -> pd.DataFrame:
        df = raw.rename(columns={"localY": "x", "localX": "y"}).copy()

        # 운동학 계산
        df = _compute_kinematics(df)

        # 앞차 기반 dhw/thw/ttc
        df = _compute_lead_features(df)

        # 뒷차 following_id: 앞차의 반대 방향
        df["following_id"] = NO_NEIGHBOR_ID

        # 좌우 이웃
        df = compute_side_neighbors(df, lane_smaller_is_left=True)

        out = pd.DataFrame({
            "dataset":      "EMT",
            "recording_id": df["recording_id"],
            "vehicle_id":   df["vehicle_id"].astype(int),
            "frame":        df["frame"].astype(int),
            "x":            df["x"],
            "y":            df["y"],
            "vx":           df["vx"],
            "vy":           df["vy"],
            "ax":           df["ax"],
            "ay":           df["ay"],
            "lane_id":      df["lane_id"].astype(int),
            "dhw":          df["dhw"],
            "thw":          df["thw"],
            "ttc":          df["ttc"],
            "preceding_vx": df["preceding_vx"],
            "preceding_id": df["preceding_id"].fillna(0).astype(int),
            "following_id": df["following_id"].astype(int),
            "left_preceding_id":  df["left_preceding_id"].astype(int),
            "left_alongside_id":  df["left_alongside_id"].astype(int),
            "left_following_id":  df["left_following_id"].astype(int),
            "right_preceding_id": df["right_preceding_id"].astype(int),
            "right_alongside_id": df["right_alongside_id"].astype(int),
            "right_following_id": df["right_following_id"].astype(int),
        })

        return out[CANONICAL_COLUMNS]

    def get_canonical(self) -> pd.DataFrame:
        raw = self.load_raw()
        canon = self.to_canonical(raw)
        from src.data.schema import validate_canonical
        validate_canonical(canon, strict=True)
        return canon
