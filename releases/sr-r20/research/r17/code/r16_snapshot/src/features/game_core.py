"""game_core.py — 게임이론 순수 수학 코어 (원본 game_theory.py에서 추출).
위치: src/features/game_core.py"""
import numpy as np
import logging
from itertools import combinations
from typing import Tuple, Optional, Dict
logger=logging.getLogger(__name__)
TTC_CAP_VALUE=999.0
# ── 행동 인덱스 ──────────────────────────────────
LK  = 0
LLC = 1
RLC = 2
ACC  = 0
HOLD = 1
DEC  = 2

EGO_ACTIONS       = [LK, LLC, RLC]
SURR_ACTIONS      = [ACC, HOLD, DEC]
EGO_ACTION_NAMES  = ["LK", "LLC", "RLC"]
SURR_ACTION_NAMES = ["ACC", "HOLD", "DEC"]

# ── 피처 컬럼 (Nash + Stackelberg 혼합) ──────────
NASH_FEATURE_COLS = [
    # Nash 피처 (동시 의사결정)
    "nash_lc_prob",           # 전체 LC 확률
    "nash_lk_prob",           # LK 확률
    "ego_best_action",        # 우세 전략
    "payoff_lc_max",          # LC 최대 payoff
    "payoff_lk",              # LK payoff
    "payoff_gap",             # Nash payoff gap
    # Stackelberg 피처 (순차 의사결정, ego=leader)
    "stackelberg_lc_payoff",  # ego LC 시 follower 최적반응 후 payoff
    "stackelberg_lk_payoff",  # ego LK 시 follower 최적반응 후 payoff
    "stackelberg_advantage",  # stackelberg_lc - stackelberg_lk (핵심!)
    "follower_best_response", # follower 최적반응 전략
    "stackelberg_nash_gap",   # Stackelberg vs Nash payoff 차이
    # 혼합 파생 피처
    "nash_urgency",           # 종합 변경 긴박도
    "threat_asymmetry",       # 좌우 위협 비대칭성
    "strategic_commitment",   # ego 전략 확신도 (Nash+Stackelberg 일치 여부)
]


# ──────────────────────────────────────────────
# Payoff 함수 (개선)
# ──────────────────────────────────────────────

def compute_ego_payoff_directional(
    ego_action: int,
    surr_action: int,
    ego_speed: float,
    prec_speed: float,
    gap_target: float,
    gap_back: float,
    rel_speed_back: float,
    ttc: float,
    delta_gap: float,        # 신규: 최근 1s gap 변화율
    surr_speed: float,       # 신규: 타겟 차선 차량 속도
) -> float:
    """
    방향별 ego Payoff 계산 (개선판).
    delta_gap: 양수=gap 증가(안전), 음수=gap 감소(위험)
    """
    ego_speed      = _safe(ego_speed,      30.0)
    prec_speed     = _safe(prec_speed,     ego_speed)
    gap_target     = _safe(gap_target,     50.0)
    gap_back       = _safe(gap_back,       50.0)
    rel_speed_back = _safe(rel_speed_back, 0.0)
    ttc            = _safe(ttc,            10.0, cap=TTC_CAP_VALUE)
    delta_gap      = _safe(delta_gap,      0.0)
    surr_speed     = _safe(surr_speed,     ego_speed)

    payoff = 0.0
    speed_pressure = ego_speed - prec_speed

    if ego_action == LK:
        payoff -= 0.8 * max(speed_pressure, 0)
        payoff -= _ttc_penalty(ttc)
        if rel_speed_back > 2.0:
            payoff -= 0.5 * min(rel_speed_back, 10.0)
        # gap 감소 추세면 추가 페널티
        if delta_gap < -2.0:
            payoff -= 0.3 * min(abs(delta_gap), 5.0)

    elif ego_action in [LLC, RLC]:
        payoff += _gap_reward(gap_target)

        # 타겟 차선 속도 이득
        speed_gain = surr_speed - ego_speed
        if speed_gain > 0:
            payoff += 0.3 * min(speed_gain, 10.0)

        # 주변 차량 반응
        if surr_action == DEC:
            payoff += 2.5
        elif surr_action == ACC:
            payoff -= 3.5
            payoff -= _gap_penalty(gap_target)
        else:  # HOLD
            payoff += 0.5 if gap_target > 20.0 else -0.5

        # 후방 위협
        if rel_speed_back > 2.0:
            payoff += 0.5 * min(rel_speed_back, 10.0)

        # 전방 압박 해소
        if speed_pressure > 0:
            payoff += 0.5 * min(speed_pressure, 15.0)

        # gap 변화 추세 반영
        if delta_gap > 2.0:
            payoff += 0.4 * min(delta_gap, 5.0)  # gap 증가 추세 → 유리
        elif delta_gap < -2.0:
            payoff -= 0.4 * min(abs(delta_gap), 5.0)  # gap 감소 추세 → 위험

    return float(np.clip(payoff, -20.0, 20.0))


def compute_surr_payoff(
    ego_action: int,
    surr_action: int,
    surr_speed: float,
    ego_speed: float,
    gap_to_ego: float,
) -> float:
    surr_speed = _safe(surr_speed, 30.0)
    ego_speed  = _safe(ego_speed,  30.0)
    gap_to_ego = _safe(gap_to_ego, 50.0)

    payoff = 0.0
    if ego_action in [LLC, RLC]:
        if surr_action == DEC:
            payoff -= 1.5
            payoff += 1.0
        elif surr_action == ACC:
            payoff -= 5.0 if gap_to_ego < 20.0 else -0.5
        else:
            payoff -= 0.5 if gap_to_ego < 15.0 else 0.2
    else:
        if surr_action == ACC:
            payoff += 1.0
        elif surr_action == HOLD:
            payoff += 0.5
        else:
            payoff -= 0.5
    return float(np.clip(payoff, -10.0, 10.0))


# ──────────────────────────────────────────────
# 방향별 Payoff Matrix (신규: 좌/우 분리)
# ──────────────────────────────────────────────

def build_directional_payoff_matrix(
    direction: str,   # "left" or "right"
    ego_speed: float,
    prec_speed: float,
    gap_left: float,
    gap_right: float,
    gap_back: float,
    rel_speed_back: float,
    ttc: float,
    delta_gap_left: float,
    delta_gap_right: float,
    surr_speed_left: float,
    surr_speed_right: float,
    gap_to_ego: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    방향별 2×3 게임 (LK vs LLC or LK vs RLC).
    ego: {LK, LC_dir}, surr: {ACC, HOLD, DEC}
    """
    if direction == "left":
        lc_action  = LLC
        gap_target = gap_left
        delta_gap  = delta_gap_left
        surr_speed = surr_speed_left
    else:
        lc_action  = RLC
        gap_target = gap_right
        delta_gap  = delta_gap_right
        surr_speed = surr_speed_right

    A = np.zeros((2, 3))
    B = np.zeros((2, 3))

    for j, sa in enumerate(SURR_ACTIONS):
        # LK
        A[0, j] = compute_ego_payoff_directional(
            LK, sa, ego_speed, prec_speed,
            max(gap_left, gap_right), gap_back,
            rel_speed_back, ttc, 0.0, prec_speed
        )
        B[0, j] = compute_surr_payoff(LK, sa, surr_speed, ego_speed, gap_to_ego)

        # LC (방향별)
        A[1, j] = compute_ego_payoff_directional(
            lc_action, sa, ego_speed, prec_speed,
            gap_target, gap_back, rel_speed_back, ttc,
            delta_gap, surr_speed
        )
        B[1, j] = compute_surr_payoff(lc_action, sa, surr_speed, ego_speed, gap_to_ego)

    return A, B


# ──────────────────────────────────────────────
# Nash Equilibrium 계산
# ──────────────────────────────────────────────

def find_mixed_nash(A: np.ndarray, B: np.ndarray, tol: float = 1e-6):
    n_ego  = A.shape[0]
    n_surr = A.shape[1]
    best_p = np.ones(n_ego)  / n_ego
    best_q = np.ones(n_surr) / n_surr
    best_score = -np.inf

    for i in range(n_ego):
        for j in range(n_surr):
            p = np.zeros(n_ego);  p[i] = 1.0
            q = np.zeros(n_surr); q[j] = 1.0
            if _is_nash(A, B, p, q, tol):
                score = A[i, j]
                if score > best_score:
                    best_score = score
                    best_p, best_q = p.copy(), q.copy()

    if best_score == -np.inf:
        best_p, best_q = _solve_mixed(A, B)

    return best_p, best_q


def _is_nash(A, B, p, q, tol=1e-6):
    ego_payoff  = p @ A @ q
    surr_payoff = p @ B @ q
    for i in range(A.shape[0]):
        ei = np.zeros(A.shape[0]); ei[i] = 1.0
        if ei @ A @ q > ego_payoff + tol:
            return False
    for j in range(B.shape[1]):
        ej = np.zeros(B.shape[1]); ej[j] = 1.0
        if p @ B @ ej > surr_payoff + tol:
            return False
    return True


def _solve_mixed(A, B):
    n, m = A.shape
    best_p = np.ones(n) / n
    best_q = np.ones(m) / m
    best_val = -np.inf

    for supp_p in combinations(range(n), min(2, n)):
        for supp_q in combinations(range(m), min(2, m)):
            p, q = _solve_support(A, B, list(supp_p), list(supp_q))
            if p is not None and _is_nash(A, B, p, q, tol=1e-4):
                val = p @ A @ q
                if val > best_val:
                    best_val = val
                    best_p, best_q = p, q
    return best_p, best_q


def _solve_support(A, B, supp_p, supp_q):
    try:
        n_p, n_q = len(supp_p), len(supp_q)
        if n_p == 2 and n_q >= 2:
            i0, i1 = supp_p
            j0, j1 = supp_q[0], supp_q[1]
            diff = A[i0, [j0, j1]] - A[i1, [j0, j1]]
            if abs(diff[0] - diff[1]) < 1e-9:
                return None, None
            q_sub = np.array([diff[1], -diff[0]])
            if q_sub.sum() == 0:
                return None, None
            q_sub = q_sub / q_sub.sum()
            if np.any(q_sub < -1e-6):
                return None, None
            q = np.zeros(A.shape[1])
            q[[j0, j1]] = np.clip(q_sub, 0, 1)

            diff2 = B[[i0, i1], j0] - B[[i0, i1], j1]
            if abs(diff2[0] - diff2[1]) < 1e-9:
                return None, None
            p_sub = np.array([diff2[1], -diff2[0]])
            if p_sub.sum() == 0:
                return None, None
            p_sub = p_sub / p_sub.sum()
            if np.any(p_sub < -1e-6):
                return None, None
            p = np.zeros(A.shape[0])
            p[[i0, i1]] = np.clip(p_sub, 0, 1)
            return p, q
        return None, None
    except Exception:
        return None, None


# ──────────────────────────────────────────────
# Stackelberg 게임 (ego=leader, surr=follower)
# ──────────────────────────────────────────────

def compute_stackelberg(
    A: np.ndarray,
    B: np.ndarray,
) -> tuple:
    """
    Stackelberg Equilibrium 계산.
    ego(leader)가 먼저 전략 선택 → surr(follower)가 최적 반응.

    Returns
    -------
    ego_action   : int    leader 최적 전략
    surr_action  : int    follower 최적 반응
    ego_payoff   : float  leader payoff
    surr_payoff  : float  follower payoff
    """
    best_ego_payoff = -np.inf
    best_ego_action = 0
    best_surr_action = 0

    for i in range(A.shape[0]):
        # follower 최적 반응: ego 전략 i 고정 시 B[i,:] 최대화
        surr_best = int(np.argmax(B[i, :]))
        ego_payoff = A[i, surr_best]

        if ego_payoff > best_ego_payoff:
            best_ego_payoff = ego_payoff
            best_ego_action = i
            best_surr_action = surr_best

    ego_p  = float(best_ego_payoff)
    surr_p = float(B[best_ego_action, best_surr_action])
    return best_ego_action, best_surr_action, ego_p, surr_p


def compute_stackelberg_features(
    A: np.ndarray,
    B: np.ndarray,
    nash_lc_payoff: float,
    nash_lk_payoff: float,
) -> dict:
    """
    Stackelberg 기반 피처 계산.

    stackelberg_advantage  : LC vs LK Stackelberg payoff 차이
    stackelberg_nash_gap   : Stackelberg LC payoff - Nash LC payoff
                             양수 → leader 우위 (순차 게임이 유리)
    strategic_commitment   : Nash와 Stackelberg 전략 일치 여부
                             1=일치(확신), 0=불일치(불확실)
    """
    ego_action, surr_action, ego_payoff, surr_payoff = compute_stackelberg(A, B)

    # Stackelberg에서 LK 선택 시 payoff
    surr_best_lk = int(np.argmax(B[LK, :]))
    stk_lk_payoff = float(A[LK, surr_best_lk])

    # LC 최적 (LLC or RLC 중 더 나은 것)
    surr_best_llc = int(np.argmax(B[LLC, :]))
    surr_best_rlc = int(np.argmax(B[RLC, :]))
    stk_llc = float(A[LLC, surr_best_llc])
    stk_rlc = float(A[RLC, surr_best_rlc])
    stk_lc_payoff = max(stk_llc, stk_rlc)

    stk_advantage = stk_lc_payoff - stk_lk_payoff
    stk_nash_gap  = stk_lc_payoff - nash_lc_payoff

    # strategic_commitment: Nash와 Stackelberg 모두 LC 선호 or 둘 다 LK 선호
    nash_prefers_lc = 1 if nash_lc_payoff > nash_lk_payoff else 0
    stk_prefers_lc  = 1 if stk_lc_payoff  > stk_lk_payoff  else 0
    commitment = 1.0 if nash_prefers_lc == stk_prefers_lc else 0.0

    return dict(
        stackelberg_lc_payoff  = float(np.clip(stk_lc_payoff, -20, 20)),
        stackelberg_lk_payoff  = float(np.clip(stk_lk_payoff, -20, 20)),
        stackelberg_advantage  = float(np.clip(stk_advantage, -20, 20)),
        follower_best_response = float(surr_action),
        stackelberg_nash_gap   = float(np.clip(stk_nash_gap, -20, 20)),
        strategic_commitment   = commitment,
    )


# ──────────────────────────────────────────────
# 신규 파생 피처 계산
# ──────────────────────────────────────────────

def compute_cooperation_index(A_left, A_right, B_left, B_right) -> float:
    """
    주변 차량 협조 가능성 지수.
    DEC 전략의 평균 surr payoff가 높을수록 협조 가능성 높음.
    """
    dec_payoff_left  = np.mean(B_left[:, DEC])
    dec_payoff_right = np.mean(B_right[:, DEC])
    raw = (dec_payoff_left + dec_payoff_right) / 2.0
    return float(np.clip(raw / 10.0, -1.0, 1.0))  # 정규화


def compute_threat_asymmetry(
    gap_left: float, gap_right: float,
    surr_speed_left: float, surr_speed_right: float,
    ego_speed: float,
) -> float:
    """
    좌우 위협 비대칭성.
    양수: 우측이 더 위협적 (좌측변경 선호)
    음수: 좌측이 더 위협적 (우측변경 선호)
    """
    left_threat  = _gap_penalty(gap_left)  + max(surr_speed_left  - ego_speed, 0) * 0.1
    right_threat = _gap_penalty(gap_right) + max(surr_speed_right - ego_speed, 0) * 0.1
    return float(np.clip(right_threat - left_threat, -5.0, 5.0))


def compute_nash_urgency(
    payoff_gap_left: float,
    payoff_gap_right: float,
    ttc: float,
    rel_speed_back: float,
) -> float:
    """
    종합 변경 긴박도.
    payoff_gap이 크고, TTC가 짧고, 후방 위협이 클수록 높음.
    """
    pg_score  = max(payoff_gap_left, payoff_gap_right) / 20.0
    ttc_score = max(1.0 - ttc / 10.0, 0.0)
    back_score = min(max(rel_speed_back, 0.0), 10.0) / 10.0
    urgency = 0.4 * pg_score + 0.4 * ttc_score + 0.2 * back_score
    return float(np.clip(urgency, 0.0, 1.0))


def compute_nash_stability(p_left: np.ndarray, p_right: np.ndarray) -> float:
    """
    Nash 안정성: LK 전략 확률이 높을수록 안정적.
    0(불안정, 변경 임박) ~ 1(안정, 유지)
    """
    lk_prob_left  = float(p_left[0])
    lk_prob_right = float(p_right[0])
    return float((lk_prob_left + lk_prob_right) / 2.0)


# ──────────────────────────────────────────────
# 메인 피처 계산
# ──────────────────────────────────────────────


def _safe(val, default=0.0, cap=None):
    if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
        return default
    try:
        v = float(val)
        if np.isnan(v) or np.isinf(v):
            return default
        if cap is not None:
            v = min(v, cap)
        return v
    except (TypeError, ValueError):
        return default


def _ttc_penalty(ttc: float) -> float:
    if ttc < 2.0:  return 5.0
    elif ttc < 5.0: return 2.0
    elif ttc < 10.0: return 0.5
    return 0.0


def _gap_reward(gap: float) -> float:
    if gap > 40.0:   return 3.0
    elif gap > 20.0: return 1.5
    elif gap > 10.0: return 0.5
    return -2.0


def _gap_penalty(gap: float) -> float:
    if gap < 5.0:    return 5.0
    elif gap < 10.0: return 2.0
    elif gap < 20.0: return 0.5
    return 0.0

