"""features/game_theory_di.py — 무차원 게임이론 피처 빌더."""
import numpy as np
import pandas as pd
from src.features import game_core_di as gc

NASH_FEATURE_COLS = gc.NASH_FEATURE_COLS


def _inputs(row, speed_map):
    ego = float(row.get("ego_speed") or 30.0)
    if not np.isfinite(ego):
        ego = 30.0

    def nb(c):
        nid = int(row.get(c, 0) or 0)
        return speed_map.get(nid, ego) if nid else ego

    def g(c, d):
        v = row.get(c)
        try:
            v = float(v)
        except (TypeError, ValueError):
            return d
        return d if not np.isfinite(v) else v

    return gc.to_dimensionless(
        ego_speed=ego,
        prec_speed=g("preceding_vx", ego),
        ttc=g("ttc", 999.0),
        gap_left=g("gap_left_front", 50.0),
        gap_right=g("gap_right_front", 50.0),
        surr_speed_left=nb("left_preceding_id"),
        surr_speed_right=nb("right_preceding_id"),
        rel_speed_back=0.0,
    )


def _nash_with_kind(A, B):
    p, q = gc.find_mixed_nash(A, B)
    if gc._is_nash(A, B, p, q, tol=1e-4):
        pure = (np.max(p) > 1 - 1e-6) and (np.max(q) > 1 - 1e-6)
        kind = "pure" if pure else "mixed"
    else:
        kind = "fallback"
    return p, q, kind


def _one(z, stats=None):
    Al, Bl = gc.build_matrix(z, "left")
    Ar, Br = gc.build_matrix(z, "right")
    pl, ql, kl = _nash_with_kind(Al, Bl)
    pr, qr, kr = _nash_with_kind(Ar, Br)
    if stats is not None:
        stats[kl] += 1
        stats[kr] += 1

    p_left, p_right = float(pl[1]), float(pr[1])
    lc_prob = max(p_left, p_right)

    lk_pay = (float(Al[0] @ ql) + float(Ar[0] @ qr)) / 2.0
    lc_l, lc_r = float(Al[1] @ ql), float(Ar[1] @ qr)
    lc_pay = max(lc_l, lc_r)
    gap = lc_pay - lk_pay

    _, sj_l, _ = gc.stackelberg(Al, Bl)
    _, sj_r, _ = gc.stackelberg(Ar, Br)
    stk_lk = max(float(Al[0, int(np.argmax(Bl[0, :]))]),
                 float(Ar[0, int(np.argmax(Br[0, :]))]))
    stk_lc = max(float(Al[1, int(np.argmax(Bl[1, :]))]),
                 float(Ar[1, int(np.argmax(Br[1, :]))]))
    commit = 1.0 if ((lc_pay > lk_pay) == (stk_lc > stk_lk)) else 0.0

    return {
        "nash_lc_prob": lc_prob,
        "nash_lk_prob": 1.0 - lc_prob,
        "nash_lc_prob_left": p_left,
        "nash_lc_prob_right": p_right,
        "ego_best_action": 1 if lc_l >= lc_r else 2,
        "payoff_lc_max": lc_pay,
        "payoff_lk": lk_pay,
        "payoff_gap": gap,
        "stackelberg_lc_payoff": stk_lc,
        "stackelberg_lk_payoff": stk_lk,
        "stackelberg_advantage": stk_lc - stk_lk,
        "follower_best_response": float((sj_l + sj_r) / 2.0),
        "stackelberg_nash_gap": stk_lc - lc_pay,
        "nash_urgency": gc.strategic_urgency(gap, z),
        "threat_asymmetry": gc.threat_asymmetry(z),
        "strategic_commitment": commit,
    }


def compute_game_features(feat_df, canon, stats=None):
    df = feat_df.copy()
    lut = {f: dict(zip(g["vehicle_id"], g["vx"])) for f, g in canon.groupby("frame")}
    out = {c: [] for c in NASH_FEATURE_COLS}
    for _, r in df.iterrows():
        try:
            v = _one(_inputs(r, lut.get(int(r["frame"]), {})), stats=stats)
        except Exception:
            v = {c: 0.0 for c in NASH_FEATURE_COLS}
        for c in NASH_FEATURE_COLS:
            out[c].append(v[c])
    for c in NASH_FEATURE_COLS:
        df[c] = out[c]
    return df
