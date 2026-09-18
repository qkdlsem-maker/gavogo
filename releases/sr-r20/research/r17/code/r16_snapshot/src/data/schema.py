"""
schema.py — GAVOGO 공통 스키마.
모든 데이터셋(highD/NGSIM/MiTra/ETRI)은 어댑터로 이 스키마로 변환된다.
피처·게임이론·모델 코드는 오직 이 스키마만 본다.

좌표 규약: x=종방향(주행) [m], y=횡방향 [m], vx=종속도(양수) [m/s]
단위는 전부 SI. 이웃 없음 = 0.
"""
CANONICAL_COLUMNS = [
    "dataset", "recording_id", "vehicle_id", "frame",
    "x", "y", "vx", "vy", "ax", "ay", "lane_id",
    "dhw", "thw", "ttc", "preceding_vx",
    "preceding_id", "following_id",
    "left_preceding_id", "left_alongside_id", "left_following_id",
    "right_preceding_id", "right_alongside_id", "right_following_id",
]
NO_NEIGHBOR_ID = 0


def validate_canonical(df, strict=True):
    missing = [c for c in CANONICAL_COLUMNS if c not in df.columns]
    if missing:
        msg = f"[schema] 누락 컬럼: {missing}"
        if strict:
            raise ValueError(msg)
        print("WARN:", msg)
        return False
    return True
