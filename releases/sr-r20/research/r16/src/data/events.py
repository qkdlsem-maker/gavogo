"""events.py — canonical 기반 차선변경 이벤트 감지 (데이터셋 무관)."""
import pandas as pd


def detect_events(canon, persistence_k=5):
    df = canon.sort_values(["vehicle_id", "frame"], ignore_index=True)
    rows = []
    for vid, g in df.groupby("vehicle_id", sort=False):
        g = g.reset_index(drop=True)
        lanes, frames = g["lane_id"].values, g["frame"].values
        n = len(g)
        rec_id = g["recording_id"].iloc[0]  # str or int 그대로 유지
        for i in range(1, n):
            if lanes[i] == lanes[i-1]:
                continue
            end = min(i + persistence_k, n)
            if end - i < persistence_k or not (lanes[i:end] == lanes[i]).all():
                continue
            old, new = int(lanes[i-1]), int(lanes[i])
            rows.append(dict(
                recording_id=rec_id,
                vehicle_id=int(vid),
                event_frame=int(frames[i]),
                old_lane=old,
                new_lane=new,
                event_type="left" if new < old else "right",
            ))
    return pd.DataFrame(rows)
