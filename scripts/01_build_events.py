#!/usr/bin/env python
"""[1단계] 데이터셋 → canonical → 차선변경 이벤트.
실행:
  python scripts/01_build_events.py --dataset highD|NGSIM|MiTra|ETRI|uniD|exiD
  python scripts/01_build_events.py --dataset EMT
"""
import argparse, sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config
from src.data.events import detect_events
from src.data.highd_adapter import HighdAdapter
from src.data.ngsim_adapter import NgsimAdapter
from src.data.mitra_adapter import MitraAdapter
from src.data.etri_adapter import EtriAdapter
from src.data.emt_adapter import EmtAdapter
from src.data.unid_adapter import UnidAdapter
from src.data.exid_adapter import ExidAdapter

ADAPTERS = {
    "highD": HighdAdapter,
    "NGSIM": NgsimAdapter,
    "MiTra": MitraAdapter,
    "ETRI":  EtriAdapter,
    "uniD":  UnidAdapter,
    "exiD":  ExidAdapter,
    "EMT":   EmtAdapter,
}


def parquet_path(cdir, rid):
    try:
        return cdir / f"{int(rid):02d}.parquet"
    except (ValueError, TypeError):
        return cdir / f"{rid}.parquet"


def run_emt(cdir):
    adapter = EmtAdapter(config.RAW_DIRS["EMT"])
    canon = adapter.get_canonical()
    allev = []
    for vid, grp in canon.groupby("recording_id"):
        grp = grp.reset_index(drop=True)
        grp.to_parquet(cdir / f"{vid}.parquet")
        ev = detect_events(grp, config.EVENT_PERSISTENCE_K)
        allev.append(ev)
        print(f"  {vid}: events={len(ev)}")
    return pd.concat(allev, ignore_index=True)


def run_standard(dataset, cdir):
    adapter = ADAPTERS[dataset](config.RAW_DIRS[dataset])
    recs = config.DEFAULT_RECS[dataset]
    allev = []
    for rid in recs:
        try:
            canon = adapter.get_canonical(rid)
        except FileNotFoundError:
            print(f"  [skip] {rid}: 파일없음"); continue
        canon.to_parquet(parquet_path(cdir, rid))
        ev = detect_events(canon, config.EVENT_PERSISTENCE_K)
        allev.append(ev)
        print(f"  rec {rid:02d}: events={len(ev)}")
    if not allev:
        return pd.DataFrame()
    return pd.concat(allev, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(ADAPTERS))
    a = ap.parse_args()

    cdir = config.INTERIM_DIR / "canonical" / a.dataset
    cdir.mkdir(parents=True, exist_ok=True)

    if a.dataset == "EMT":
        ev = run_emt(cdir)
    else:
        ev = run_standard(a.dataset, cdir)

    if len(ev) == 0:
        print("이벤트 없음"); return

    out = config.INTERIM_DIR / f"events_{a.dataset}.csv"
    ev.to_csv(out, index=False)
    print(f"총 {len(ev)} 이벤트 "
          f"(L={int((ev.event_type=='left').sum())}, "
          f"R={int((ev.event_type=='right').sum())}) → {out.name}")


if __name__ == "__main__":
    main()
