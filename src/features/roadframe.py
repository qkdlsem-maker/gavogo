"""features/roadframe.py — 도로 국소좌표(road frame) 기반 횡방향 정규화.

배경
----
전역 y축을 '횡방향'으로 쓰면 직선 고속도로(highD/NGSIM/MiTra)에서는 맞지만
곡선·램프(exiD)나 도심(uniD)에서는 틀린다. 곡선을 따라 정상 주행하는 것만으로도
전역 y가 계속 변하기 때문에, 그 변화량이 '횡속도'로 오인된다.
→ lat_speed가 의도가 아니라 '도로 위 어느 곡률 구간에 있는가'를 인코딩하고,
   합류 지점이 곧 라벨이 되어 exiD 단독 AUC 0.93 같은 지름길이 생긴다.

해결: 차선 중심선(centerline)을 복원해 국소 도로좌표계를 만든다.
  1) recording 단위 PCA → 도로 주축 u(종), 부축 v(횡)
  2) s = proj_u, d = proj_v
  3) 차량 주행방향 sx로 정렬 (양방향 도로 대응): s_drv = sx*s, d_drv = sx*d
  4) (lane_id, sx) 그룹별로 s 구간을 나눠 d_drv의 중앙값 → 차선 중심선 c(s)
     (곡선이면 c(s)가 s에 따라 휘어지므로 곡률이 흡수된다)
  5) lat_offset = d_drv - c(s)          ← 차선 중심 대비 횡방향 이탈
     lat_speed  = d(lat_offset)/dt      ← 곡선 정상주행 시 ≈ 0, 차선변경 시에만 커짐

이 단계가 Adapter의 Topology Reconstruction / Normalization 에 해당한다.
"""
import numpy as np
import pandas as pd

N_BINS = 60          # 종방향 s 구간 수 (중심선 해상도)
MIN_PTS_PER_BIN = 5


def _pca_axes(x, y):
    """recording 전체 점군의 주축(u)·부축(v) 반환."""
    P = np.column_stack([x, y]).astype(float)
    mu = P.mean(0)
    Q = P - mu
    # 공분산 고유분해 (2x2)
    C = np.cov(Q.T)
    w, V = np.linalg.eigh(C)
    order = np.argsort(w)[::-1]
    u = V[:, order[0]]
    v = V[:, order[1]]
    return mu, u, v


def _centerline_offset(s, d, lane, sx, n_bins=N_BINS):
    """(lane, sx) 그룹별 중심선 c(s)를 s-구간 중앙값으로 추정하고 offset 반환."""
    off = np.full(len(s), np.nan)
    key = pd.Series([f"{int(l)}|{int(k)}" for l, k in zip(lane, sx)])
    for g, idx in key.groupby(key).groups.items():
        idx = np.asarray(idx)
        sg, dg = s[idx], d[idx]
        if len(idx) < 10:
            off[idx] = dg - np.median(dg)
            continue
        lo, hi = np.percentile(sg, [0.5, 99.5])
        if hi - lo < 1e-6:
            off[idx] = dg - np.median(dg)
            continue
        edges = np.linspace(lo, hi, n_bins + 1)
        bi = np.clip(np.digitize(sg, edges) - 1, 0, n_bins - 1)
        centers = np.full(n_bins, np.nan)
        for b in range(n_bins):
            m = bi == b
            if m.sum() >= MIN_PTS_PER_BIN:
                centers[b] = np.median(dg[m])
        # 빈 구간 보간 + 평활
        valid = ~np.isnan(centers)
        if valid.sum() < 2:
            off[idx] = dg - np.median(dg)
            continue
        xb = np.arange(n_bins)
        centers = np.interp(xb, xb[valid], centers[valid])
        k = 3
        pad = np.pad(centers, (k // 2, k // 2), mode="edge")
        centers = np.convolve(pad, np.ones(k) / k, mode="valid")
        off[idx] = dg - centers[bi]
    return off


def add_global_frame(canon):
    """[Ablation용] 도로좌표 복원 없이 전역 y를 그대로 횡방향으로 사용.
    직선 도로에서는 근사적으로 맞지만 곡선·도심에서는 곡률이 횡속도로 오인된다."""
    c = canon.copy()
    c["lat_offset"] = c["y"] - c.groupby("lane_id")["y"].transform("median")
    hw = c.groupby("lane_id")["lat_offset"].apply(
        lambda z: (np.nanpercentile(z, 95) - np.nanpercentile(z, 5)) / 2.0)
    hw = hw.replace(0, np.nan)
    med = hw.median()
    fb = float(med) if (np.isfinite(med) and med > 1e-3) else 1.75
    c["lane_halfwidth"] = c["lane_id"].map(hw.fillna(fb).to_dict()).fillna(fb).values
    return c


def add_road_frame(canon):
    """canon에 s, lat_offset, lane_halfwidth 컬럼 추가. (recording 단위 호출)"""
    c = canon.copy()
    mu, u, v = _pca_axes(c["x"].values, c["y"].values)
    P = np.column_stack([c["x"].values, c["y"].values]).astype(float) - mu
    s = P @ u
    d = P @ v
    c["_s"] = s
    c["_d"] = d

    # 차량별 주행방향 (s가 증가하는가)
    cs = c.sort_values(["vehicle_id", "frame"])
    g = cs.groupby("vehicle_id")["_s"]
    ds = g.last() - g.first()
    sx_map = np.sign(ds).replace(0, 1.0)
    sx = c["vehicle_id"].map(sx_map).fillna(1.0).values

    s_drv = s * sx
    d_drv = d * sx

    c["lat_offset"] = _centerline_offset(s_drv, d_drv,
                                         c["lane_id"].values, sx)
    c["_sx"] = sx
    c["s_drv"] = s_drv

    # 차선 반폭 (정규화용)
    hw = c.groupby("lane_id")["lat_offset"].apply(
        lambda z: (np.nanpercentile(z, 95) - np.nanpercentile(z, 5)) / 2.0)
    hw = hw.replace(0, np.nan)
    med = hw.median()
    fb = float(med) if (np.isfinite(med) and med > 1e-3) else 1.75
    hw = hw.fillna(fb).to_dict()
    c["lane_halfwidth"] = c["lane_id"].map(hw).fillna(fb).values
    return c


def estimate_lateral_sign(canon_rf, events, fps=25, k=None):
    """lat_offset의 +방향을 'lane_id 증가 방향'으로 맞추는 부호 s(±1) 추정."""
    if k is None:
        k = max(2, int(round(0.6 * fps)))
    rid = canon_rf["recording_id"].iloc[0]
    ev = events[events["recording_id"] == rid]
    if len(ev) == 0:
        return 1.0, 0
    idx = canon_rf.set_index(["vehicle_id", "frame"])["lat_offset"].sort_index()
    votes = []
    for _, r in ev.iterrows():
        vid, ef = int(r["vehicle_id"]), int(r["event_frame"])
        inc = int(r["new_lane"]) > int(r["old_lane"])
        try:
            y0 = float(idx.loc[(vid, ef - k)])
            y1 = float(idx.loc[(vid, ef + k)])
        except (KeyError, TypeError):
            continue
        dy = y1 - y0
        if not np.isfinite(dy) or abs(dy) < 1e-6:
            continue
        votes.append(1.0 if ((dy > 0) == inc) else -1.0)
    if not votes:
        return 1.0, 0
    return (1.0 if float(np.mean(votes)) >= 0 else -1.0), len(votes)
