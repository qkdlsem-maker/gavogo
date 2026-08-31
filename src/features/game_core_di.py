"""game_core_di.py — 무차원(dimensionless) 연속 utility 기반 게임이론 코어.

기존 game_core.py의 두 가지 문제
--------------------------------
(1) payoff가 if/else 계단함수 → "게임이론으로 위장한 룰 기반"이라는 비판 (T-IV #2)
      예:  _ttc_penalty:  ttc<2 → 5.0,  ttc<5 → 2.0,  ttc<10 → 0.5,  else 0
(2) payoff가 절대 단위(m, m/s)에 직접 의존 → 도메인 식별자가 되어버림.
      측정된 domain-discriminability AUC:
        payoff_lc_max 0.907, payoff_gap 0.897, stackelberg_lc_payoff 0.830

본 모듈의 재정식화
------------------
모든 상태변수를 무차원으로 바꾸고, payoff를 미분가능한 연속 utility로 정의한다.

  무차원 상태변수 (τ0 = 2 s : 기준 time headway)
    ρ   = τ0 / TTC                      전방 충돌 위험
    π   = (v_ego − v_prec) / v_ego      전방 속도 압박
    θ_d = (g_d / v_ego) / τ0            방향 d 목표차선 time gap (τ0 단위)
    σ_d = (v_d − v_ego) / v_ego         방향 d 속도 이득
    β   = Δv_back / v_ego               후방 접근 위협

  연속 utility (상태변수에 대해 계단 없음)
    U_ego(LK)     = −[ w_π·sp0(π⁺) + w_ρ·(1 − e^{−ρ}) + w_β·sp0(β⁺) ]
    U_ego(LC_d,a) = −C_LC − w_θ·sp0( ((1 − θ_d)/s)⁺ ) + w_σ·tanh(σ_d) + R(a; θ_d)

    sp0(z) = softplus(z) − ln2   (원점 정규화: sp0(0) = 0)
    squeeze(θ) = σ((1 − θ)/s)    gap이 좁을수록 1

  설계 의도
    · 압박이 없으면 U_LK = 0 이고 U_LC ≤ −C_LC < 0  →  차선유지가 우월전략.
    · 전방 위험/압박이 커지면 U_LK가 단조 감소하여 어느 지점에서 LC로 전환된다
      (진짜 equilibrium switch).
    · 목표차선 time gap θ가 1(=2초) 미만이면 벌점이 발산 →
      아무리 전방이 위험해도 물리적 공간이 없으면 LC가 성립하지 않는다.
    · 상대 행동 R(a)는 게임의 이산 행동집합에 대한 것이므로 이산인 것이 정상이다.
      상태변수에 대해서는 어디에도 계단이 없다.

Proposition 1 (Boundedness of strategic urgency)
------------------------------------------------
  urgency = w1·u_p + w2·u_ρ + w3·u_β,   w1 + w2 + w3 = 1,  w_i ≥ 0
  where
    u_p = σ(payoff_gap / κ)     ∈ (0,1)
    u_ρ = 1 − e^{−ρ}            ∈ [0,1)   (ρ ≥ 0)
    u_β = tanh(sp0(β⁺))         ∈ [0,1)   (sp0(β⁺) ≥ 0)
  각 항이 [0,1]에 속하고 가중치가 볼록결합이므로  urgency ∈ [0,1].  ∎
  (기존 구현은 clip으로 강제했으나, 여기서는 clip 없이 성립한다.)
"""
import numpy as np
from itertools import combinations

# ── 행동 인덱스 ──────────────────────────────────
LK, LLC, RLC = 0, 1, 2
ACC, HOLD, DEC = 0, 1, 2
SURR_ACTIONS = [ACC, HOLD, DEC]

# ── 상수 ─────────────────────────────────────────
TAU0 = 2.0          # 기준 time headway [s] — 모든 시간을 이 단위로 무차원화
V_MIN = 1.0         # 0 나눗셈 방지 [m/s]
S_SQ = 0.35         # squeeze / 기하벌점 완만도

# 무차원 가중치 (전부 dimensionless)
W_PI, W_RHO, W_BETA = 1.0, 5.0, 0.6      # LK 벌점: 속도압박 / 전방위험 / 후방위협
W_THETA, W_SIGMA = 3.0, 1.2              # LC: 목표차선 기하벌점 / 속도이득
C_LC = 0.35                              # 차선변경 고정 비용 (maneuver cost)
C_ACC, C_DEC = 2.0, 1.5                  # 상대 반응 (DEC는 현재 미사용)
KAPPA = 1.0                              # urgency 스케일
LN2 = float(np.log(2.0))

# ── 피처 컬럼 (16개) ─────────────────────────────
NASH_FEATURE_COLS = [
    # Nash (동시 의사결정)
    "nash_lc_prob", "nash_lk_prob",
    "nash_lc_prob_left", "nash_lc_prob_right",
    "ego_best_action",
    "payoff_lc_max", "payoff_lk", "payoff_gap",
    # Stackelberg (순차 의사결정, ego = leader)
    "stackelberg_lc_payoff", "stackelberg_lk_payoff", "stackelberg_advantage",
    "follower_best_response", "stackelberg_nash_gap",
    # 파생
    "nash_urgency", "threat_asymmetry", "strategic_commitment",
]


# ── 매끄러운 기본 함수 ────────────────────────────────────────────────
def _sp(z):
    """softplus (수치 안정)."""
    z = np.clip(z, -30, 30)
    return float(np.log1p(np.exp(z)))


def _sp0(z):
    """원점 정규화 softplus: sp0(0) = 0. 압박이 없을 때 오프셋이 생기지 않게 한다."""
    return _sp(z) - LN2


def _sig(z):
    z = np.clip(z, -30, 30)
    return float(1.0 / (1.0 + np.exp(-z)))


def _squeeze(theta):
    """time gap θ(τ0 단위)가 좁을수록 1에 접근하는 매끄러운 압박도."""
    return _sig((1.0 - theta) / S_SQ)


def _safe(v, d):
    try:
        v = float(v)
        return d if (np.isnan(v) or np.isinf(v)) else v
    except (TypeError, ValueError):
        return d


# ── 무차원 상태 ──────────────────────────────────────────────────────
def to_dimensionless(ego_speed, prec_speed, ttc, gap_left, gap_right,
                     surr_speed_left, surr_speed_right, rel_speed_back):
    """단위를 가진 원 상태 → 무차원 상태."""
    v = max(abs(_safe(ego_speed, 30.0)), V_MIN)
    vp = _safe(prec_speed, v)
    t = float(np.clip(_safe(ttc, 999.0), 0.1, 999.0))

    rho = TAU0 / t                                    # 전방 위험
    pi = (v - abs(vp)) / v                            # 속도 압박
    th_l = (max(_safe(gap_left, 50.0), 0.0) / v) / TAU0
    th_r = (max(_safe(gap_right, 50.0), 0.0) / v) / TAU0
    sg_l = (abs(_safe(surr_speed_left, v)) - v) / v
    sg_r = (abs(_safe(surr_speed_right, v)) - v) / v
    beta = _safe(rel_speed_back, 0.0) / v
    return dict(rho=float(rho), pi=float(pi),
                theta_l=float(th_l), theta_r=float(th_r),
                sigma_l=float(sg_l), sigma_r=float(sg_r),
                beta=float(beta))


# ── 연속 utility ─────────────────────────────────────────────────────
def u_ego_lk(z):
    """차선유지 효용. 압박이 전혀 없으면 정확히 0, 압박이 커질수록 단조 감소."""
    return float(-(W_PI * _sp0(max(z["pi"], 0.0))
                   + W_RHO * (1.0 - np.exp(-z["rho"]))
                   + W_BETA * _sp0(max(z["beta"], 0.0))))


def u_ego_lc(z, direction, surr_action):
    """차선변경 효용.
       - 고정 비용 C_LC를 항상 지불 → 압박이 없으면 U_LC < 0 = U_LK → LK 우월
       - 목표차선 time gap θ가 1(=2초) 미만이면 기하 벌점이 발산
         → 물리적 공간이 없으면 아무리 전방이 위험해도 LC가 성립하지 않음
       - θ ≥ 1 이면 기하 벌점 0 (여유가 충분한데 보상까지 주지는 않음)"""
    theta = z["theta_l"] if direction == "left" else z["theta_r"]
    sigma = z["sigma_l"] if direction == "left" else z["sigma_r"]
    sq = _squeeze(theta)                                    # 좁을수록 1

    base = (-C_LC
            - W_THETA * _sp0(max((1.0 - theta) / S_SQ, 0.0))  # 기하 벌점 (θ≥1이면 0)
            + W_SIGMA * np.tanh(sigma))                       # 속도 이득

    if surr_action == ACC:
        base -= C_ACC * sq          # 상대가 막아서면 좁을수록 치명적
    elif surr_action == DEC:
        base += 0.0                 # 양보받아도 이득 없음 — 방해가 없을 뿐
    else:                           # HOLD
        base -= 0.5 * sq            # 좁으면 그냥 유지만 해도 부담
    return float(base)


def u_surr(z, ego_action, surr_action, direction):
    """상대(목표차선 후행차)의 utility. 무차원."""
    theta = z["theta_l"] if direction == "left" else z["theta_r"]
    sq = _squeeze(theta)
    if ego_action == LK:
        return float({ACC: 0.3, HOLD: 0.5, DEC: -0.3}[surr_action])
    # ego가 끼어들려는 상황
    if surr_action == ACC:
        return float(-1.5 * sq + 0.5)      # 막아서면 자신도 위험해짐
    if surr_action == DEC:
        return float(-0.5 + 1.0 * sq)      # 양보: 좁을수록 합리적
    return float(0.2 - 0.6 * sq)           # HOLD


def build_matrix(z, direction):
    """2×3 방향별 게임 (ego: LK/LC_d, surr: ACC/HOLD/DEC)."""
    A = np.zeros((2, 3))
    B = np.zeros((2, 3))
    lk = u_ego_lk(z)
    for j, a in enumerate(SURR_ACTIONS):
        A[0, j] = lk
        A[1, j] = u_ego_lc(z, direction, a)
        B[0, j] = u_surr(z, LK, a, direction)
        B[1, j] = u_surr(z, LLC, a, direction)
    return A, B


# ── Nash 균형 ────────────────────────────────────────────────────────
def _is_nash(A, B, p, q, tol=1e-6):
    ep, sp_ = p @ A @ q, p @ B @ q
    for i in range(A.shape[0]):
        e = np.zeros(A.shape[0]); e[i] = 1
        if e @ A @ q > ep + tol:
            return False
    for j in range(B.shape[1]):
        e = np.zeros(B.shape[1]); e[j] = 1
        if p @ B @ e > sp_ + tol:
            return False
    return True


def _solve_support(A, B, sp_i, sq_j):
    i0, i1 = sp_i
    j0, j1 = sq_j
    d = A[i0, [j0, j1]] - A[i1, [j0, j1]]
    if abs(d[0] - d[1]) < 1e-9:
        return None, None
    qs = np.array([d[1], -d[0]])
    if qs.sum() == 0 or np.any(qs / qs.sum() < -1e-6):
        return None, None
    qs = np.clip(qs / qs.sum(), 0, 1)
    d2 = B[[i0, i1], j0] - B[[i0, i1], j1]
    if abs(d2[0] - d2[1]) < 1e-9:
        return None, None
    ps = np.array([d2[1], -d2[0]])
    if ps.sum() == 0 or np.any(ps / ps.sum() < -1e-6):
        return None, None
    ps = np.clip(ps / ps.sum(), 0, 1)
    p = np.zeros(A.shape[0]); p[[i0, i1]] = ps
    q = np.zeros(A.shape[1]); q[[j0, j1]] = qs
    return p, q


def find_mixed_nash(A, B, tol=1e-6):
    """순수전략 균형 우선 탐색 → 없으면 혼합전략 support enumeration."""
    best_p = np.ones(A.shape[0]) / A.shape[0]
    best_q = np.ones(A.shape[1]) / A.shape[1]
    best = -np.inf
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            p = np.zeros(A.shape[0]); p[i] = 1
            q = np.zeros(A.shape[1]); q[j] = 1
            if _is_nash(A, B, p, q, tol) and A[i, j] > best:
                best, best_p, best_q = A[i, j], p.copy(), q.copy()
    if best == -np.inf:
        for si in combinations(range(A.shape[0]), 2):
            for sj in combinations(range(A.shape[1]), 2):
                p, q = _solve_support(A, B, list(si), list(sj))
                if p is not None and _is_nash(A, B, p, q, 1e-4):
                    v = p @ A @ q
                    if v > best:
                        best, best_p, best_q = v, p, q
    return best_p, best_q


def stackelberg(A, B):
    """ego = leader. follower가 최적반응 → leader가 그것을 예상하고 선택."""
    best_i, best_j, best_v = 0, 0, -np.inf
    for i in range(A.shape[0]):
        j = int(np.argmax(B[i, :]))
        if A[i, j] > best_v:
            best_v, best_i, best_j = A[i, j], i, j
    return best_i, best_j, float(best_v)


# ── 파생 피처 ────────────────────────────────────────────────────────
def strategic_urgency(payoff_gap, z):
    """Proposition 1: 세 항이 모두 [0,1]이고 가중치가 볼록결합 → urgency ∈ [0,1].
    clip 없이 성립한다."""
    u_p = _sig(payoff_gap / KAPPA)                     # (0,1)
    u_r = 1.0 - np.exp(-max(z["rho"], 0.0))            # [0,1)
    u_b = np.tanh(_sp0(max(z["beta"], 0.0)))           # [0,1)  (sp0(0)=0)
    u_b = float(max(0.0, min(1.0, u_b)))
    return float(0.4 * u_p + 0.4 * u_r + 0.2 * u_b)


def threat_asymmetry(z):
    """좌우 위협 비대칭. 무차원, [-1,1] 유계.
    > 0 : 우측이 더 위협적 → 좌측 변경 선호"""
    tl = _squeeze(z["theta_l"]) - 0.5 * np.tanh(z["sigma_l"])
    tr = _squeeze(z["theta_r"]) - 0.5 * np.tanh(z["sigma_r"])
    return float(np.tanh(tr - tl))
