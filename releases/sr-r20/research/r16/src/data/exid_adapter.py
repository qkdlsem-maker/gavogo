"""exid_adapter.py — exiD(00_final_exid.csv) → canonical.
방향 분리: dir=0(원본 rec_id), dir=1(1000+rec_id)로 분리됨.
get_canonical(recording_id)에서 rec_id < 1000이면 dir=0, >= 1000이면 dir=1.
"""
import pandas as pd
from pathlib import Path
from src.data.schema import CANONICAL_COLUMNS, NO_NEIGHBOR_ID, validate_canonical


class ExidAdapter:
    def __init__(self, raw_dir: Path):
        self.raw_dir = Path(raw_dir)

    def load_raw(self, recording_id: int) -> pd.DataFrame:
        # dir=1이면 실제 파일 rec_id = recording_id - 1000
        if recording_id >= 1000:
            actual_rid = recording_id - 1000
            dir_val = 1
        else:
            actual_rid = recording_id
            dir_val = 0
        rid = f"{actual_rid:02d}"
        df = pd.read_csv(self.raw_dir / f"{rid}_final_exid.csv", low_memory=False)
        # 해당 방향만 필터링
        df = df[df['dir'] == dir_val].copy()
        return df

    def to_canonical(self, raw: pd.DataFrame, recording_id: int) -> pd.DataFrame:
        df = raw.copy()

        def clean_id(col):
            if col not in df.columns:
                return pd.Series([NO_NEIGHBOR_ID] * len(df), index=df.index)
            s = df[col].astype(str).str.split(";").str[0]
            return pd.to_numeric(s, errors="coerce").fillna(-1).replace(-1, NO_NEIGHBOR_ID).astype(int)

        def clean_dhw(col):
            if col not in df.columns:
                return pd.Series([0.0] * len(df), index=df.index)
            return df[col].replace(-1, 0.0).fillna(0.0).clip(lower=0)

        def clean_ttc(col):
            if col not in df.columns:
                return pd.Series([999.0] * len(df), index=df.index)
            return df[col].replace(-1, 999.0).fillna(999.0).clip(0, 999)

        out = pd.DataFrame({
            "dataset":            "exiD",
            "recording_id":       recording_id,
            "vehicle_id":         df["trackId"].astype(int),
            "frame":              df["frame"].astype(int),
            "x":                  df["x"],  # 전처리에서 dir=0은 -xCenter, dir=1은 xCenter
            "y":                  df["y"],
            "vx":                 df["vx"].abs(),
            "vy":                 df["vy"],
            "ax":                 df["ax"],
            "ay":                 df["ay"],
            "lane_id":            df["lane_id"].astype(int),
            "dhw":                clean_dhw("leadDHW"),
            "thw":                clean_dhw("leadTHW"),
            "ttc":                clean_ttc("leadTTC"),
            "preceding_vx":       df["leadDV"].fillna(0.0).abs() if "leadDV" in df.columns else 0.0,
            "preceding_id":       clean_id("leadId"),
            "following_id":       clean_id("rearId"),
            "left_preceding_id":  clean_id("leftLeadId"),
            "left_alongside_id":  clean_id("leftAlongsideId"),
            "left_following_id":  clean_id("leftRearId"),
            "right_preceding_id": clean_id("rightLeadId"),
            "right_alongside_id": clean_id("rightAlongsideId"),
            "right_following_id": clean_id("rightRearId"),
        })
        return out[CANONICAL_COLUMNS]

    def get_canonical(self, recording_id: int) -> pd.DataFrame:
        raw = self.load_raw(recording_id)
        if len(raw) == 0:
            raise FileNotFoundError(f"exiD rec={recording_id} 데이터 없음")
        canon = self.to_canonical(raw, recording_id)
        validate_canonical(canon, strict=True)
        return canon