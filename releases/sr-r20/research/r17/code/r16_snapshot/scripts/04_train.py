#!/usr/bin/env python
"""[4단계] joint 학습 + ablation + ETRI hold-out + Scaling Law.
실행: python scripts/04_train.py
Scaling Law: 도메인 누적 추가하면서 ETRI OOD AUC 변화 측정"""
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config
from src.features.kinematic import FEATURE_COLS as KIN
from src.features.game_theory import NASH_FEATURE_COLS as GT
from src.models.train import split_by_vehicle, fit_xgb, eval_auc, _xy

GROUPS = {
    "full":    KIN + GT,
    "no_game": KIN,
    "no_vos":  [c for c in KIN if not c.startswith("vos")] + GT,
}

# Scaling Law: 도메인 누적 순서
SCALING_ORDER = ["highD", "NGSIM", "MiTra", "EMT"]


def load(ds, h):
    df = pd.read_csv(config.PROCESSED_DIR / f"{ds}_gt_{h}s.csv")
    df["dataset"] = ds
    return df


def main():
    # ── 1. Joint 학습 + Ablation ──────────────────────────────────────
    rows = []
    for h in config.HORIZONS_SEC:
        trains, tests = [], {}
        for ds in config.TRAIN_DATASETS:
            tr, te = split_by_vehicle(load(ds, h), seed=config.RANDOM_STATE)
            trains.append(tr); tests[ds] = te
        train = pd.concat(trains, ignore_index=True)
        holdout = load(config.HOLDOUT_DATASET, h)

        for gname, gcols in GROUPS.items():
            cols = [c for c in gcols if c in train.columns]
            m = fit_xgb(*_xy(train, cols), seed=config.RANDOM_STATE)
            r = {"horizon": f"{h}s", "group": gname}
            for ds in config.TRAIN_DATASETS:
                r[ds] = round(eval_auc(m, tests[ds], cols), 4)
            r["ETRI_OOD"] = round(eval_auc(m, holdout, cols), 4)
            rows.append(r)
            print(f"  [{h}s/{gname:8s}] "
                  + " ".join(f"{ds}={r[ds]}" for ds in config.TRAIN_DATASETS)
                  + f" | ETRI(OOD)={r['ETRI_OOD']}")

    res = pd.DataFrame(rows)
    res.to_csv(config.TABLES_DIR / "joint_results.csv", index=False)
    print("\n저장: joint_results.csv")

    print("\n── 게임이론 기여 (full−no_game, in-domain 평균) ──")
    for h in res.horizon.unique():
        f  = res[(res.horizon==h)&(res.group=="full")][config.TRAIN_DATASETS].mean(axis=1).values[0]
        ng = res[(res.horizon==h)&(res.group=="no_game")][config.TRAIN_DATASETS].mean(axis=1).values[0]
        print(f"  {h}: {f-ng:+.4f}")

    # ── 2. Scaling Law ────────────────────────────────────────────────
    print("\n── Scaling Law (누적 도메인 → ETRI OOD AUC) ──")
    scaling_rows = []
    cols = [c for c in KIN + GT if True]  # full feature set
    for h in config.HORIZONS_SEC:
        holdout = load(config.HOLDOUT_DATASET, h)
        accumulated = []
        for ds in SCALING_ORDER:
            tr, _ = split_by_vehicle(load(ds, h), seed=config.RANDOM_STATE)
            accumulated.append(tr)
            train = pd.concat(accumulated, ignore_index=True)
            use_cols = [c for c in cols if c in train.columns]
            m = fit_xgb(*_xy(train, use_cols), seed=config.RANDOM_STATE)
            auc = round(eval_auc(m, holdout, use_cols), 4)
            scaling_rows.append({
                "horizon": f"{h}s",
                "n_domains": len(accumulated),
                "domains": "+".join(SCALING_ORDER[:len(accumulated)]),
                "ETRI_OOD_AUC": auc,
            })
            print(f"  [{h}s] {len(accumulated)}개 도메인 "
                  f"({'+'.join(SCALING_ORDER[:len(accumulated)])}): ETRI OOD={auc}")

    sl = pd.DataFrame(scaling_rows)
    sl.to_csv(config.TABLES_DIR / "scaling_law.csv", index=False)
    print("\n저장: scaling_law.csv")


if __name__ == "__main__":
    main()
