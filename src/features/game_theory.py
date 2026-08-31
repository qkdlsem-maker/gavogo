"""features/game_theory.py — Nash-Stackelberg 해석 피처 (canonical 기반).
위치: src/features/game_theory.py
포지션: 성능이 아니라 해석(interpretability)용. 현재프레임 14개 피처."""
import numpy as np
import pandas as pd
from src.features import game_core as gc

NASH_FEATURE_COLS = [
    "nash_lc_prob", "nash_lk_prob", "ego_best_action",
    "payoff_lc_max", "payoff_lk", "payoff_gap",
    "stackelberg_lc_payoff", "stackelberg_lk_payoff", "stackelberg_advantage",
    "follower_best_response", "stackelberg_nash_gap",
    "nash_urgency", "threat_asymmetry", "strategic_commitment",
]


def _safe(v, d):
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return d
        return float(v)
    except Exception:
        return d


def _inputs(row, sm):
    ego = _safe(row.get("ego_speed"), 30.0)
    prec = _safe(row.get("preceding_vx"), ego)
    def nb(c):
        nid = int(row.get(c, 0) or 0)
        return sm.get(nid, ego) if nid else ego
    return dict(ego_speed=ego, prec_speed=prec,
        gap_left=_safe(row.get("gap_left_front"), 50.0),
        gap_right=_safe(row.get("gap_right_front"), 50.0),
        gap_back=_safe(row.get("dhw"), 30.0), rel_speed_back=0.0,
        ttc=_safe(row.get("ttc"), 10.0), gap_to_ego=_safe(row.get("dhw"), 40.0),
        delta_gap_left=0.0, delta_gap_right=0.0,
        surr_speed_left=nb("left_preceding_id"), surr_speed_right=nb("right_preceding_id"))


def compute_game_features(feat_df, canon):
    df = feat_df.copy()
    lut = {f: dict(zip(g["vehicle_id"], g["vx"])) for f, g in canon.groupby("frame")}
    out = {c: [] for c in NASH_FEATURE_COLS}
    for _, r in df.iterrows():
        v = _one(_inputs(r, lut.get(int(r["frame"]), {})))
        for c in NASH_FEATURE_COLS:
            out[c].append(v[c])
    for c in NASH_FEATURE_COLS:
        df[c] = out[c]
    return df


def _one(f):
    try:
        Al, Bl = gc.build_directional_payoff_matrix("left", **f)
        pl, ql = gc.find_mixed_nash(Al, Bl)
        Ar, Br = gc.build_directional_payoff_matrix("right", **f)
        pr, qr = gc.find_mixed_nash(Ar, Br)
        lc = (float(pl[1]) + float(pr[1])) / 2
        plk = (float(Al[0] @ ql) + float(Ar[0] @ qr)) / 2
        plcl, plcr = float(Al[1] @ ql), float(Ar[1] @ qr)
        plc = max(plcl, plcr); pgap = plc - plk
        LK, LLC, RLC = gc.LK, gc.LLC, gc.RLC
        A3, B3 = np.zeros((3, 3)), np.zeros((3, 3))
        for j in range(3):
            A3[LK, j]  = gc.compute_ego_payoff_directional(LK, j, f["ego_speed"], f["prec_speed"], max(f["gap_left"], f["gap_right"]), f["gap_back"], f["rel_speed_back"], f["ttc"], 0.0, f["prec_speed"])
            A3[LLC, j] = gc.compute_ego_payoff_directional(LLC, j, f["ego_speed"], f["prec_speed"], f["gap_left"], f["gap_back"], f["rel_speed_back"], f["ttc"], f["delta_gap_left"], f["surr_speed_left"])
            A3[RLC, j] = gc.compute_ego_payoff_directional(RLC, j, f["ego_speed"], f["prec_speed"], f["gap_right"], f["gap_back"], f["rel_speed_back"], f["ttc"], f["delta_gap_right"], f["surr_speed_right"])
            B3[LK, j]  = gc.compute_surr_payoff(LK, j, f["prec_speed"], f["ego_speed"], f["gap_to_ego"])
            B3[LLC, j] = gc.compute_surr_payoff(LLC, j, f["surr_speed_left"], f["ego_speed"], f["gap_left"])
            B3[RLC, j] = gc.compute_surr_payoff(RLC, j, f["surr_speed_right"], f["ego_speed"], f["gap_right"])
        stk = gc.compute_stackelberg_features(A3, B3, plc, plk)
        asym = gc.compute_threat_asymmetry(f["gap_left"], f["gap_right"], f["surr_speed_left"], f["surr_speed_right"], f["ego_speed"])
        urg = gc.compute_nash_urgency(plcl - float(Al[0] @ ql), plcr - float(Ar[0] @ qr), f["ttc"], f["rel_speed_back"])
        return {"nash_lc_prob": lc, "nash_lk_prob": 1 - lc,
            "ego_best_action": 1 if plcl >= plcr else 2,
            "payoff_lc_max": plc, "payoff_lk": plk, "payoff_gap": pgap,
            "stackelberg_lc_payoff": stk.get("stackelberg_lc_payoff", 0.0),
            "stackelberg_lk_payoff": stk.get("stackelberg_lk_payoff", 0.0),
            "stackelberg_advantage": stk.get("stackelberg_advantage", 0.0),
            "follower_best_response": stk.get("follower_best_response", 0.0),
            "stackelberg_nash_gap": stk.get("stackelberg_nash_gap", 0.0),
            "nash_urgency": urg, "threat_asymmetry": asym,
            "strategic_commitment": stk.get("strategic_commitment", abs(pgap))}
    except Exception:
        return {c: 0.0 for c in NASH_FEATURE_COLS}
