"""features/sampling.py — within-vehicle negative sampling (GAVOGO v2 protocol).

기존 build_samples()의 문제
---------------------------
negative를 "H프레임 동안 차선을 유지하는 아무 차량"에서 뽑았다.
→ positive 차량과 negative 차량이 서로 다른 차량 집합.
→ 모델이 '언제 바꾸는가(when)'가 아니라 '누가 바꾸는 차인가(who)'를 학습.
→ vehicle-identity shortcut. AUC가 부풀려짐 (highD: 0.88 → within-vehicle 0.74).

본 모듈의 프로토콜
------------------
positive : 차선변경 이벤트 e의 event_frame - H
negative : *같은 차량*의 궤적에서, 모든 이벤트로부터 충분히 떨어져 있고
           [f, f+H] 구간 동안 차선을 유지하는 시점
require_both=True : positive와 negative를 모두 가진 차량만 채택
                    → group_label_purity = 0 (차량 정체성으로는 라벨을 못 맞춤)

결과적으로 과제가 "이 차량이 지금 차선을 바꾸려 하는가"로 정직해진다.
"""
import numpy as np
import pandas as pd


def build_samples_within(canon, events, horizon_frames, fps,
                         margin_sec=2.0,
                         neg_per_pos=3,
                         neg_min_spacing_sec=1.0,
                         require_both=True,
                         random_state=42):
    """
    Parameters
    ----------
    canon           : canonical DataFrame (단일 recording)
    events          : detect_events() 결과 (전체)
    horizon_frames  : H (= horizon_sec * fps)
    fps             : 프레임레이트
    margin_sec      : 이벤트 주변 배제 구간(초). negative는 event_frame - H - margin
                      이전이거나 event_frame + margin 이후여야 함.
    neg_per_pos     : positive 1개당 뽑을 negative 최대 개수
    neg_min_spacing_sec : negative끼리 최소 간격(초). 시간적 중복 방지.
    require_both    : True면 pos/neg 둘 다 가진 차량만 사용 (purity=0 보장)

    Returns
    -------
    DataFrame[recording_id, vehicle_id, frame, label]
    """
    rng = np.random.RandomState(random_state)
    rid = canon["recording_id"].iloc[0]
    H = int(horizon_frames)
    margin = int(round(margin_sec * fps))
    spacing = max(1, int(round(neg_min_spacing_sec * fps)))

    ev = events[events["recording_id"] == rid]
    ev_by_veh = {v: np.sort(g["event_frame"].values)
                 for v, g in ev.groupby("vehicle_id", sort=False)}

    rows = []
    canon_s = canon.sort_values(["vehicle_id", "frame"])

    for vid, g in canon_s.groupby("vehicle_id", sort=False):
        efs = ev_by_veh.get(vid)
        if efs is None or len(efs) == 0:
            continue                       # 이벤트 없는 차량은 아예 사용 안 함

        frames = g["frame"].values
        lanes = g["lane_id"].values
        fmin, fmax = frames[0], frames[-1]
        f2i = {int(f): i for i, f in enumerate(frames)}

        # ── positive: event_frame - H (해당 프레임이 궤적 안에 있어야 함)
        pos_frames = []
        for ef in efs:
            pf = int(ef) - H
            if pf in f2i:
                pos_frames.append(pf)
        if not pos_frames:
            if require_both:
                continue
            # positive 없으면 negative만 뽑는 건 의미 없음
            continue

        # ── negative 후보: 모든 이벤트로부터 안전거리 확보 + [f, f+H] 차선 유지
        cand = []
        for i, f in enumerate(frames):
            f = int(f)
            if f + H > fmax:
                continue
            # 이벤트 근처 배제
            ok = True
            for ef in efs:
                ef = int(ef)
                # negative는 (예측시점 f) 기준으로 미래 H 안에 이벤트가 없어야 하고,
                # 경계 모호성 회피를 위해 margin만큼 더 떨어뜨린다.
                if (f > ef - H - margin) and (f < ef + margin):
                    ok = False
                    break
            if not ok:
                continue
            # [f, f+H] 차선 유지 확인
            j = f2i.get(f + H)
            if j is None:
                # 프레임 불연속 → 인덱스 기반 fallback
                seg = lanes[i:i + H + 1]
            else:
                seg = lanes[i:j + 1]
            if len(seg) < 2 or not (seg == seg[0]).all():
                continue
            cand.append(f)

        if not cand and require_both:
            continue

        # negative 다운샘플 (시간 간격 확보 + 개수 제한)
        neg_frames = []
        if cand:
            cand = np.array(cand)
            rng.shuffle(cand)
            n_want = neg_per_pos * len(pos_frames)
            for f in cand:
                if len(neg_frames) >= n_want:
                    break
                if all(abs(int(f) - int(nf)) >= spacing for nf in neg_frames):
                    neg_frames.append(int(f))

        if require_both and (not pos_frames or not neg_frames):
            continue

        for f in pos_frames:
            rows.append((rid, int(vid), int(f), 1))
        for f in neg_frames:
            rows.append((rid, int(vid), int(f), 0))

    s = pd.DataFrame(rows, columns=["recording_id", "vehicle_id", "frame", "label"])
    if len(s) == 0:
        return s
    # canon에 실제 존재하는 (rec,veh,frame)만 남김
    key = canon[["recording_id", "vehicle_id", "frame"]]
    return s.merge(key, on=["recording_id", "vehicle_id", "frame"], how="inner")


def balance_within(s, ratio=1.0, random_state=42):
    """차량 단위로 pos:neg 비율 맞추기. 전역 balance()와 달리 차량 구조를 보존."""
    if len(s) == 0:
        return s
    out = []
    rng = np.random.RandomState(random_state)
    for vid, g in s.groupby(["recording_id", "vehicle_id"], sort=False):
        p = g[g.label == 1]
        n = g[g.label == 0]
        k = int(len(p) * ratio)
        if len(n) > k and k > 0:
            n = n.sample(n=k, random_state=rng.randint(0, 10 ** 6))
        out.append(pd.concat([p, n]))
    return pd.concat(out, ignore_index=True)


def report_purity(s):
    """검증용: group_label_purity가 0인지 확인."""
    if len(s) == 0:
        return dict(n=0)
    k = s["recording_id"].astype(str) + "|" + s["vehicle_id"].astype(str)
    per = s.assign(g=k).groupby("g").label.mean()
    purity = float(((per == 0) | (per == 1)).mean())
    return dict(n_rows=len(s), n_groups=len(per),
                pos_rate=round(float(s.label.mean()), 4),
                group_label_purity=round(purity, 4))
