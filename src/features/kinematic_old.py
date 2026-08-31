"""features/kinematic.py — 공통 스키마 기반 피처. 데이터셋 무관.

v2 변경점: 횡방향(lateral) 피처 추가.
  기존 피처는 종방향(속도/가속/TTC/gap)만 사용했다. cross-vehicle negative
  샘플링에서는 '누가 차선을 바꾸는 차인가'가 풀려서 문제가 없었지만,
  within-vehicle 샘플링(=언제 바꾸는가)에서는 차선변경 의도의 최강 신호인
  '차선 내 횡방향 드리프트'가 반드시 필요하다.

  추가:
    lat_offset       차선 중심 대비 횡위치 (m)
    lat_offset_norm  차선 반폭으로 정규화 (무차원 → 도메인 간 전이에 유리)
    lat_speed        횡속도 (m/s), y의 K프레임 기울기 × fps (모든 데이터셋 동일 방식)
    lat_acc          횡가속도 근사
    lat_drift        중심에서 멀어지는 방향 성분 (sign(lat_offset) × lat_speed)
    lat_offset_slope5 / lat_speed_std5 / abs_lat_speed_max5
  ※ 원본 vy는 데이터셋마다 0으로 채워진 경우가 있어(NGSIM/MiTra) 신뢰 불가 →
     y의 차분으로 통일해서 계산한다.
"""
import numpy as np
import pandas as pd

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


# ── 구 프로토콜(cross-vehicle) 샘플러. 비교/재현용으로 남겨둠 ──────────
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


# ── 헬퍼 ─────────────────────────────────────────────────────────────
def _slope(a):
    a = np.asarray(a, dtype=float)
    a = a[~np.isnan(a)]
    return float(np.polyfit(np.arange(len(a)), a, 1)[0]) if len(a) >= 2 else 0.0


def _lane_geometry(canon):
    """차선별 중심/반폭 추정. 원본에 차선 중심선이 없으므로 관측 분포로 근사.
    center = 해당 차선 y의 중앙값, half_width = (p95 - p05)/2."""
    g = canon.groupby("lane_id")["y"]
    center = g.median()
    hw = (g.quantile(0.95) - g.quantile(0.05)) / 2.0
    hw = hw.replace(0, np.nan)
    # 반폭이 비정상(0/NaN)이면 전체 차선 평균으로 대체, 그것도 없으면 1.75m(표준 차선 반폭)
    fallback = float(hw.median()) if np.isfinite(hw.median()) else 1.75
    hw = hw.fillna(fallback)
    return center.to_dict(), hw.to_dict(), fallback


# ── 메인 피처 빌더 ───────────────────────────────────────────────────
def build_features(samples, canon, fps=25):
    cols = ["recording_id", "vehicle_id", "frame", "x", "y", "vx", "ax", "lane_id",
            "dhw", "thw", "ttc", "preceding_vx",
            "left_preceding_id", "left_following_id",
            "right_preceding_id", "right_following_id"]
    df = samples.merge(canon[cols], on=["recording_id", "vehicle_id", "frame"], how="left")
    df = df.rename(columns={"vx": "ego_speed", "ax": "ego_acc"})

    # ── gaps (이웃 x 위치 merge)
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

    # ── 차선 기하 (횡방향 오프셋 계산용)
    centers, halfws, hw_fb = _lane_geometry(canon)

    # ── rolling (직전 K프레임): 종방향 + 횡방향
    idx = canon.set_index(["vehicle_id", "frame"]).sort_index()
    sm, ss, ssl, tm, tmn, tsl = ([] for _ in range(6))
    lat_off, lat_offn, lat_spd, lat_acc = [], [], [], []
    lat_off_sl, lat_spd_std, abs_lat_max = [], [], []

    for _, r in df.iterrows():
        vid, f = int(r["vehicle_id"]), int(r["frame"])
        lane = r["lane_id"]
        try:
            g = idx.loc[vid]
            hist = g[(g.index >= f - ROLLING_K) & (g.index <= f)]
        except KeyError:
            hist = None

        # 종방향
        if hist is None or len(hist) == 0:
            sm.append(r["ego_speed"]); ss.append(0.0); ssl.append(0.0)
            tm.append(r["ttc"]); tmn.append(r["ttc"]); tsl.append(0.0)
        else:
            sv = hist["vx"].values
            tv = np.clip(hist["ttc"].values, 0, 999)
            sm.append(np.nanmean(sv)); ss.append(np.nanstd(sv)); ssl.append(_slope(sv))
            tm.append(np.nanmean(tv)); tmn.append(np.nanmin(tv)); tsl.append(_slope(tv))

        # 횡방향
        c = centers.get(lane, np.nan)
        hw = halfws.get(lane, hw_fb)
        hw = hw if (hw and np.isfinite(hw) and hw > 1e-3) else hw_fb
        off = (r["y"] - c) if np.isfinite(c) else np.nan
        lat_off.append(off)
        lat_offn.append(off / hw if np.isfinite(off) else np.nan)

        if hist is None or len(hist) < 2:
            lat_spd.append(0.0); lat_acc.append(0.0)
            lat_off_sl.append(0.0); lat_spd_std.append(0.0); abs_lat_max.append(0.0)
        else:
            yv = hist["y"].values.astype(float)
            # 횡속도: y의 프레임당 기울기 × fps  (원본 vy는 데이터셋마다 0이라 신뢰 불가)
            v_lat = _slope(yv) * fps
            lat_spd.append(v_lat)
            # 프레임별 횡속도 시퀀스 → 가속/변동/최대
            dy = np.diff(yv) * fps
            lat_acc.append(float(np.mean(np.diff(dy)) * fps) if len(dy) >= 2 else 0.0)
            lat_spd_std.append(float(np.std(dy)) if len(dy) >= 1 else 0.0)
            abs_lat_max.append(float(np.max(np.abs(dy))) if len(dy) >= 1 else 0.0)
            # 차선 중심 대비 오프셋의 추세
            if np.isfinite(c):
                lat_off_sl.append(_slope(yv - c) * fps)
            else:
                lat_off_sl.append(0.0)

    df["ego_speed_mean5"], df["ego_speed_std5"], df["ego_speed_slope5"] = sm, ss, ssl
    df["ttc_mean5"], df["ttc_min5"], df["ttc_slope5"] = tm, tmn, tsl

    df["lat_offset"]        = lat_off
    df["lat_offset_norm"]   = lat_offn
    df["lat_speed"]         = lat_spd
    df["lat_acc"]           = lat_acc
    df["lat_offset_slope5"] = lat_off_sl
    df["lat_speed_std5"]    = lat_spd_std
    df["abs_lat_speed_max5"] = abs_lat_max
    # 중심에서 '멀어지는' 방향 성분 (>0 이면 이탈 중)
    df["lat_drift"] = np.sign(df["lat_offset"].fillna(0)) * df["lat_speed"].fillna(0)

    # ── interaction
    spd = df["ego_speed"].fillna(30.0)
    ttc = df["ttc"].clip(0, 999).fillna(10.0)
    df["ttc_x_speed"] = ttc * spd
    df["gap_diff"] = df["vos_left"].fillna(0) - df["vos_right"].fillna(0)
    df["speed_ratio"] = spd / (df["preceding_vx"].abs().fillna(30.0) + 1e-3)
    df["vos_diff"] = (df["vos_left"].fillna(0) - df["vos_right"].fillna(0)).abs()
    df["risk_x_gapdiff"] = (1.0 / (ttc + 1e-3)) * df["vos_diff"]
    # 횡방향 × 공간: '어느 쪽으로 드리프트하는데 그쪽 gap이 있는가'
    df["lat_speed_x_gapdiff"] = df["lat_speed"].fillna(0) * df["gap_diff"]
    df["lat_drift_x_vosdiff"] = df["lat_drift"].fillna(0) * df["vos_diff"]

    for c in FEATURE_COLS:
        if c in df.columns:
            df[c] = df[c].replace([np.inf, -np.inf], np.nan)
    return df
