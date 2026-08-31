#!/usr/bin/env python
"""[sanity] 무차원 게임 payoff가 실제로 상황에 따라 전략을 바꾸는지 확인.
죽은 피처(상수) 방지. 실행: python scripts/sanity_game_di.py"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.features import game_core_di as gc
from src.features.game_theory_di import _one

CASES = {
    "여유(압박없음, 앞차 동속, 옆 널널)":
        dict(ego_speed=30, prec_speed=30, ttc=999, gap_left=80, gap_right=80,
             surr_speed_left=30, surr_speed_right=30, rel_speed_back=0),
    "전방 느린차 + 좌측 여유":
        dict(ego_speed=30, prec_speed=20, ttc=6, gap_left=80, gap_right=15,
             surr_speed_left=32, surr_speed_right=25, rel_speed_back=0),
    "전방 매우 위험 + 좌측 여유":
        dict(ego_speed=30, prec_speed=15, ttc=2, gap_left=90, gap_right=20,
             surr_speed_left=33, surr_speed_right=25, rel_speed_back=0),
    "전방 위험이나 양옆 막힘":
        dict(ego_speed=30, prec_speed=15, ttc=2, gap_left=6, gap_right=6,
             surr_speed_left=28, surr_speed_right=28, rel_speed_back=0),
    "도심 저속 (8 m/s), 압박없음":
        dict(ego_speed=8, prec_speed=8, ttc=999, gap_left=25, gap_right=25,
             surr_speed_left=8, surr_speed_right=8, rel_speed_back=0),
    "도심 저속, 전방 정체":
        dict(ego_speed=8, prec_speed=3, ttc=3, gap_left=25, gap_right=6,
             surr_speed_left=9, surr_speed_right=7, rel_speed_back=0),
}

print(f"{'상황':34s} {'lc_prob':>8s} {'p_lk':>7s} {'p_lc':>7s} {'gap':>7s} {'urg':>6s} {'asym':>6s}")
print("-" * 82)
for name, kw in CASES.items():
    z = gc.to_dimensionless(**kw)
    v = _one(z)
    print(f"{name:34s} {v['nash_lc_prob']:8.3f} {v['payoff_lk']:7.2f} "
          f"{v['payoff_lc_max']:7.2f} {v['payoff_gap']:7.2f} "
          f"{v['nash_urgency']:6.3f} {v['threat_asymmetry']:6.2f}")

print("\n기대: 압박없음 → lc_prob≈0 / 전방위험+옆여유 → lc_prob≈1 / 양옆막힘 → lc_prob≈0")
print("     고속도로(30m/s)와 도심(8m/s)의 동일 시나리오가 비슷한 값 → 무차원성 확인")

# Proposition 1 검증
print("\n── Proposition 1: urgency ∈ [0,1] (clip 없이) ──")
rng = np.random.RandomState(0)
us = []
for _ in range(20000):
    z = gc.to_dimensionless(
        ego_speed=rng.uniform(1, 45), prec_speed=rng.uniform(0, 45),
        ttc=rng.choice([rng.uniform(0.1, 20), 999]),
        gap_left=rng.uniform(0, 150), gap_right=rng.uniform(0, 150),
        surr_speed_left=rng.uniform(0, 45), surr_speed_right=rng.uniform(0, 45),
        rel_speed_back=rng.uniform(-10, 15))
    v = _one(z)
    us.append(v["nash_urgency"])
us = np.array(us)
print(f"  min={us.min():.4f}  max={us.max():.4f}  "
      f"{'OK (유계 성립)' if (us.min() >= 0 and us.max() <= 1) else '<<< 위반!'}")
print(f"  std={us.std():.4f} {'<<< 상수! 죽은 피처' if us.std() < 1e-6 else ''}")
