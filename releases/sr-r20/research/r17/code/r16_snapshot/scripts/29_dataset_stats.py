#!/usr/bin/env python
"""[Figure 1 수치 검증] 데이터셋별 trajectories / frames / hours 집계.
위치: scripts/29_dataset_stats.py
canonical parquet(형식 통일)에서 계산. "frames" 정의가 애매해 후보 3종 모두 출력.
실행: python scripts/29_dataset_stats.py
"""
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config

ALL = ["highD", "NGSIM", "MiTra", "ETRI", "EMT", "uniD", "exiD"]


def main():
    rows = []
    tot = dict(traj=0, uframes=0, vehframes=0, sec=0.0)
    for ds in ALL:
        cdir = config.INTERIM_DIR / "canonical" / ds
        files = sorted(cdir.glob("*.parquet"))
        fps = config.FPS[ds]
        traj = uframes = vehframes = 0; sec = 0.0
        for cf in files:
            d = pd.read_parquet(cf, columns=["recording_id", "vehicle_id", "frame"])
            traj += d["vehicle_id"].nunique()           # 파일(=recording)당 고유 차량
            uf = d["frame"].nunique()
            uframes += uf                                # 고유 프레임(=recording 길이)
            vehframes += len(d)                          # 차량-프레임 관측 수(행 수)
            sec += (d["frame"].max() - d["frame"].min() + 1) / fps
        rows.append(dict(dataset=ds, recordings=len(files),
                         trajectories=traj, unique_frames=uframes,
                         vehicle_frames=vehframes, hours=round(sec/3600, 2)))
        tot["traj"] += traj; tot["uframes"] += uframes
        tot["vehframes"] += vehframes; tot["sec"] += sec
        print(f"  {ds:6s} traj={traj:7d} uframes={uframes:8d} vehframes={vehframes:10d} hours={sec/3600:6.2f}")

    print(f"\n  TOTAL  trajectories={tot['traj']:,}"
          f"  unique_frames={tot['uframes']:,}"
          f"  vehicle_frames={tot['vehframes']:,}"
          f"  hours={tot['sec']/3600:.2f}")
    df = pd.DataFrame(rows)
    df.to_csv(config.TABLES_DIR / "dataset_stats.csv", index=False)
    print("\n저장: dataset_stats.csv")
    print("매칭: 216,598↔trajectories, 4.75M↔unique_frames 또는 vehicle_frames, 11.6↔hours 중 확인")


if __name__ == "__main__":
    main()
