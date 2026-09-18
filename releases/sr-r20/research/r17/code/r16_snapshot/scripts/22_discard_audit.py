#!/usr/bin/env python
"""[리뷰어 대응: Negative Sampling Generality] 위치: scripts/22_discard_audit.py

리뷰어 예상 질문:
  "Your protocol removes all vehicles without both positive and negative
   samples. Could this introduce a different sampling bias?"

require_both=True 로 인해 버려지는 차량의 비율과 사유를 데이터셋별로 계측한다.
논문의 주장("기존 benchmark는 biased")에 대한 자기방어:
   "그럼 당신 benchmark는 새 bias를 만들지 않았는가?" 를 Appendix 표로 봉쇄.

사유 분해 (build_samples_within 을 두 번 호출해 순수 비교로 도출):
  E  = 이벤트를 가진 차량 수 (canon에 존재)
  A  = 유효 positive 를 하나라도 가진 차량 (require_both=False 결과의 차량)
  B  = positive+negative 둘 다 가진 차량 (require_both=True, 최종 채택)
  → discard(no positive window) = E - A
  → discard(no negative window) = A - B
  → kept = B,  discard_rate = 1 - B/E

기본 margin=2s, 3s horizon (논문 기본 설정).
실행: python scripts/22_discard_audit.py
"""
import argparse, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config
from src.features.sampling import build_samples_within

ALL = config.TRAIN_DATASETS + [config.HOLDOUT_DATASET] + config.OOD_DATASETS


def veh_set(samples):
    if samples is None or len(samples) == 0:
        return set()
    k = samples["recording_id"].astype(str) + "|" + samples["vehicle_id"].astype(str)
    return set(k.unique())


def audit_ds(ds, margin_sec, hsec):
    cdir = config.INTERIM_DIR / "canonical" / ds
    events = pd.read_csv(config.INTERIM_DIR / f"events_{ds}.csv")
    files = sorted(cdir.glob("*.parquet"))
    assert files, f"canonical 없음: {cdir}"
    fps = config.FPS[ds]
    H = hsec * fps

    E, A, B = set(), set(), set()
    for cf in files:
        canon = pd.read_parquet(cf)
        rid = str(canon["recording_id"].iloc[0])
        # E: 이벤트 있는 차량 ∩ canon 차량
        ev = events[events["recording_id"].astype(str) == rid]
        ev_veh = set(ev["vehicle_id"].unique())
        canon_veh = set(canon["vehicle_id"].unique())
        for v in (ev_veh & canon_veh):
            E.add(f"{rid}|{v}")

        # A: positive 하나라도 (require_both=False)
        sa = build_samples_within(canon, events, H, fps=fps,
                                  margin_sec=margin_sec, neg_per_pos=3,
                                  require_both=False,
                                  random_state=config.RANDOM_STATE)
        A |= veh_set(sa)
        # B: pos+neg 둘 다 (require_both=True)
        sb = build_samples_within(canon, events, H, fps=fps,
                                  margin_sec=margin_sec, neg_per_pos=3,
                                  require_both=True,
                                  random_state=config.RANDOM_STATE)
        B |= veh_set(sb)

    nE, nA, nB = len(E), len(A & E), len(B & E)
    no_pos = nE - nA
    no_neg = nA - nB
    return dict(dataset=ds,
                event_vehicles=nE,
                kept=nB,
                discard_no_positive=no_pos,
                discard_no_negative=no_neg,
                discard_rate=round(1 - nB / nE, 4) if nE else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--margin_sec", type=float, default=2.0)
    ap.add_argument("--horizon", type=int, default=3)
    a = ap.parse_args()

    rows = []
    for ds in tqdm(ALL, desc="audit"):
        try:
            r = audit_ds(ds, a.margin_sec, a.horizon)
        except Exception as e:
            print(f"  [{ds}] 실패: {e}"); continue
        rows.append(r)
        print(f"  [{ds:6s}] event_veh={r['event_vehicles']:5d} kept={r['kept']:5d} "
              f"no_pos={r['discard_no_positive']:4d} no_neg={r['discard_no_negative']:4d} "
              f"discard_rate={r['discard_rate']}")

    res = pd.DataFrame(rows)
    out = config.TABLES_DIR / "discard_audit.csv"
    res.to_csv(out, index=False)
    print(f"\n저장: {out}")
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
