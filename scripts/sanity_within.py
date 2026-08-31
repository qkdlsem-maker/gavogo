#!/usr/bin/env python
"""[sanity] within-vehicle negative sampling 검증.
highD 몇 개 recording으로 purity=0 나오는지, 샘플 수 얼마나 남는지 확인.

실행: cd ~/gavogo && python scripts/sanity_within.py
"""
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config
from src.data.highd_adapter import HighdAdapter
from src.data.events import detect_events
from src.features.kinematic import build_samples, balance
from src.features.sampling import build_samples_within, balance_within, report_purity

DS = "highD"
RECS = [1, 2, 3]          # 샘플 확인용 3개만
HORIZONS = [3, 5, 7]


def main():
    fps = config.FPS[DS]
    ad = HighdAdapter(config.RAW_DIRS[DS])

    for rec in RECS:
        canon = ad.get_canonical(rec)
        ev = detect_events(canon, persistence_k=config.EVENT_PERSISTENCE_K)
        print(f"\n{'='*66}\nhighD rec={rec}  frames={len(canon)}  "
              f"vehicles={canon.vehicle_id.nunique()}  events={len(ev)}")

        for h in HORIZONS:
            H = h * fps

            # 기존 프로토콜 (cross-vehicle)
            s_old = balance(build_samples(canon, ev, H))
            r_old = report_purity(s_old)

            # 신규 프로토콜 (within-vehicle)
            s_new = build_samples_within(canon, ev, H, fps=fps)
            s_new = balance_within(s_new)
            r_new = report_purity(s_new)

            print(f"  [{h}s] old: rows={r_old.get('n_rows',0):5d} "
                  f"veh={r_old.get('n_groups',0):4d} "
                  f"pos={r_old.get('pos_rate',0):.2f} "
                  f"purity={r_old.get('group_label_purity',float('nan')):.3f}")
            print(f"        new: rows={r_new.get('n_rows',0):5d} "
                  f"veh={r_new.get('n_groups',0):4d} "
                  f"pos={r_new.get('pos_rate',0):.2f} "
                  f"purity={r_new.get('group_label_purity',float('nan')):.3f}"
                  f"   {'OK' if r_new.get('group_label_purity',1)==0.0 else '<<< purity!=0'}")


if __name__ == "__main__":
    main()
