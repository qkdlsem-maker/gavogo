"""features/kinematic.py — 공통 스키마 기반 피처. 데이터셋 무관.

v4: 횡방향 피처를 '도로 국소좌표(road frame)' 기준으로 계산.
    전역 y축을 쓰면 곡선(exiD)·도심(uniD)에서 곡률 주행이 횡속도로 오인되어
    지름길 피처가 된다. roadframe.add_road_frame()이 차선 중심선을 복원해
    lat_offset을 만들고, 여기서는 그 위에서 시간 미분/집계만 한다.
"""
import numpy as np
import pandas as pd

from src.features.roadframe import (add_road_frame, add_global_frame,
                                    estimate_lateral_sign)  # noqa: F401

ROLLING_K = 5

BASE_COLS = ["ego_speed", "ego_acc", "lane_id", "ttc", "thw", "dhw", "preceding_vx"]
GAP_COLS  = ["gap_left_front", "gap_left_back", "gap_right_front", "gap_right_back"]
VOS_COLS  = ["vos_left", "vos_right"]
ROLL_COLS = ["ego_speed_mean5", "ego_speed_std5", "ego_speed_slope5",
             "ttc_mean5", "ttc_min5", "ttc_slope5"]
LAT_COLS  = ["lat_offset", "lat_offset_norm", "lat_speed", "lat_acc", "lat_drift",
             "lat_offset_slope5", "lat_speed_std5", "abs_lat_speed_max5"]
INT_COLS  = ["ttc_x_speed", "gap_diff", "speed_ratio", "vos_diff", "risk_x_gapdiff",
             "lat_speed_x_gapdiff", "lat_drift_x_vosdiff"]

FEATURE_COLS = BASE_COLS + GAP_COLS + VOS_COLS + ROLL_COLS + LAT_COLS + INT_COLS


# ── 구 프로토콜(cross-vehicle) 샘플러. 비교/재현용 ────────────────────
def build_samples(canon, events, horizon_frames, lk_interval=25, random_state=42):
    rid = canon["recording_id"].iloc[0]
    H = horizon_frames
    ev = events[events["recording_id"] == rid]
    pos = ev[["vehicle_id", "event_frame"]].copy()
    pos["frame"] = pos["event_frame"] - H
    pos = pos[["vehicle_id", "frame"]]; pos["label"] = 1
    neg = []
    for vid, g in canon.sort_values("frame").groupby("vehicle_id", sort=False):
        lanes, frames = g["lane_id"].values, g["frame"].values
        for i in range(0, len(g) - H, lk_interval):
            if (lanes[i:i+H] == lanes[i]).all():
                neg.append((vid, frames[i]))
    neg = pd.DataFrame(neg, columns=["vehicle_id", "frame"]); neg["label"] = 0
    s = pd.concat([pos, neg], ignore_index=True); s["recording_id"] = rid
    key = canon[["recording_id", "vehicle_id", "frame"]]
    return s.merge(key, on=["recording_id", "vehicle_id", "frame"], how="inner")


def balance(s, ratio=1.0, random_state=42):
    pos, neg = s[s.label == 1], s[s.label == 0]
    n = int(len(pos) * ratio)
    if len(neg) > n:
        neg = neg.sample(n=n, random_state=random_state)
    return pd.concat([pos, neg], ignore_index=True)


def _slope(a):
    a = np.asarray(a, dtype=float)
    a = a[~np.isnan(a)]
    return float(np.polyfit(np.arange(len(a)), a, 1)[0]) if len(a) >= 2 else 0.0


# ── 메인 피처 빌더 ───────────────────────────────────────────────────
def build_features(samples, canon, fps=25, lat_sign=1.0, use_road_frame=True):
    """canon: 원본 canonical.
    use_road_frame: True면 차선 중심선 복원(정상), False면 전역 y 사용(ablation).
    lat_sign: 데이터셋 단위로 정한 ±1 (lat_offset 부호 정규화). 1.0이면 정규화 없음."""
    canon = add_road_frame(canon) if use_road_frame else add_global_frame(canon)
    canon["lat_offset"] = canon["lat_offset"] * float(lat_sign)

    cols = ["recording_id", "vehicle_id", "frame", "x", "vx", "ax", "lane_id",
            "dhw", "thw", "ttc", "preceding_vx",
            "lat_offset", "lane_halfwidth",
            "left_preceding_id", "left_following_id",
            "right_preceding_id", "right_following_id"]
    df = samples.merge(canon[cols], on=["recording_id", "vehicle_id", "frame"], how="left")
    df = df.rename(columns={"vx": "ego_speed", "ax": "ego_acc"})

    # ── gaps
    pos = canon[["recording_id", "frame", "vehicle_id", "x"]].rename(
        columns={"vehicle_id": "nid", "x": "nx"})

    def gap(idc, kind):
        m = df[["recording_id", "frame", idc, "x"]].rename(columns={idc: "nid"})
        m = m.merge(pos, on=["recording_id", "frame", "nid"], how="left")
        g = (m["nx"] - df["x"].values) if kind == "front" else (df["x"].values - m["nx"])
        g[df[idc].values == 0] = np.nan
        return g.values

    df["gap_left_front"]  = gap("left_preceding_id", "front")
    df["gap_left_back"]   = gap("left_following_id", "back")
    df["gap_right_front"] = gap("right_preceding_id", "front")
    df["gap_right_back"]  = gap("right_following_id", "back")
    lf, lb = df["gap_left_front"].clip(lower=0), df["gap_left_back"].clip(lower=0)
    rf, rb = df["gap_right_front"].clip(lower=0), df["gap_right_back"].clip(lower=0)
    df["vos_left"]  = lf.fillna(0) + lb.fillna(0)
    df["vos_right"] = rf.fillna(0) + rb.fillna(0)

    # ── rolling (종방향 + 횡방향, 직전 K프레임)
    idx = canon.set_index(["vehicle_id", "frame"]).sort_index()
    sm, ss, ssl, tm, tmn, tsl = ([] for _ in range(6))
    lat_spd, lat_acc, lat_off_sl, lat_spd_std, abs_lat_max = [], [], [], [], []

    for _, r in df.iterrows():
        vid, f = int(r["vehicle_id"]), int(r["frame"])
        try:
            g = idx.loc[vid]
            hist = g[(g.index >= f - ROLLING_K) & (g.index <= f)]
        except KeyError:
            hist = None

        if hist is None or len(hist) == 0:
            sm.append(r["ego_speed"]); ss.append(0.0); ssl.append(0.0)
            tm.append(r["ttc"]); tmn.append(r["ttc"]); tsl.append(0.0)
        else:
            sv = hist["vx"].values
            tv = np.clip(hist["ttc"].values, 0, 999)
            sm.append(np.nanmean(sv)); ss.append(np.nanstd(sv)); ssl.append(_slope(sv))
            tm.append(np.nanmean(tv)); tmn.append(np.nanmin(tv)); tsl.append(_slope(tv))

        if hist is None or len(hist) < 2:
            lat_spd.append(0.0); lat_acc.append(0.0)
            lat_off_sl.append(0.0); lat_spd_std.append(0.0); abs_lat_max.append(0.0)
        else:
            ov = hist["lat_offset"].values.astype(float)
            ov = np.nan_to_num(ov, nan=0.0)
            v_lat = _slope(ov) * fps                    # m/s (도로 수직 성분)
            lat_spd.append(v_lat)
            dv = np.diff(ov) * fps
            lat_acc.append(float(np.mean(np.diff(dv)) * fps) if len(dv) >= 2 else 0.0)
            lat_spd_std.append(float(np.std(dv)) if len(dv) >= 1 else 0.0)
            abs_lat_max.append(float(np.max(np.abs(dv))) if len(dv) >= 1 else 0.0)
            lat_off_sl.append(v_lat)

    df["ego_speed_mean5"], df["ego_speed_std5"], df["ego_speed_slope5"] = sm, ss, ssl
    df["ttc_mean5"], df["ttc_min5"], df["ttc_slope5"] = tm, tmn, tsl

    hw = df["lane_halfwidth"].replace(0, np.nan).fillna(1.75)
    df["lat_offset_norm"]    = df["lat_offset"] / hw
    df["lat_speed"]          = lat_spd
    df["lat_acc"]            = lat_acc
    df["lat_offset_slope5"]  = lat_off_sl
    df["lat_speed_std5"]     = lat_spd_std
    df["abs_lat_speed_max5"] = abs_lat_max
    # 차선 중심에서 '멀어지는' 방향 성분 (>0 이면 이탈 중)
    df["lat_drift"] = np.sign(df["lat_offset"].fillna(0)) * df["lat_speed"].fillna(0)

    # ── interaction
    spd = df["ego_speed"].fillna(30.0)
    ttc = df["ttc"].clip(0, 999).fillna(10.0)
    df["ttc_x_speed"] = ttc * spd
    df["gap_diff"] = df["vos_left"].fillna(0) - df["vos_right"].fillna(0)
    df["speed_ratio"] = spd / (df["preceding_vx"].abs().fillna(30.0) + 1e-3)
    df["vos_diff"] = (df["vos_left"].fillna(0) - df["vos_right"].fillna(0)).abs()
    df["risk_x_gapdiff"] = (1.0 / (ttc + 1e-3)) * df["vos_diff"]
    df["lat_speed_x_gapdiff"] = df["lat_speed"].fillna(0) * df["gap_diff"]
    df["lat_drift_x_vosdiff"] = df["lat_drift"].fillna(0) * df["vos_diff"]

    for c in FEATURE_COLS:
        if c in df.columns:
            df[c] = df[c].replace([np.inf, -np.inf], np.nan)
    return df
