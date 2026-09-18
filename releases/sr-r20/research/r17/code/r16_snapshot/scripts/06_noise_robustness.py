#!/usr/bin/env python
"""[추가실험1] 노이즈 강건성. 위치: scripts/06_noise_robustness.py
가설: 게임이론 결합 모델(full)이 노이즈에 더 강건 (운동학만보다 성능 하락폭 작음).
방법: test 피처에 가우시안 노이즈 주입 → full vs no_game AUC 하락폭 비교.
실행: python scripts/06_noise_robustness.py"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config
from src.features.kinematic import FEATURE_COLS as KIN
from src.features.game_theory import NASH_FEATURE_COLS as GT
from src.models.train import split_by_vehicle, fit_xgb, _xy

# 노이즈는 '운동학 입력'에만 주입 (게임피처는 그로부터 계산된 것이므로 입력 오염을 모사)
NOISE_TARGETS = ["ego_speed", "ego_acc", "ttc", "dhw",
                 "gap_left_front", "gap_right_front"]
LEVELS = [0.0, 0.1, 0.2, 0.3, 0.5]   # 표준편차 대비 노이즈 비율

GROUPS = {"full": KIN + GT, "no_game": KIN}

def load(ds, h):
    df = pd.read_csv(config.PROCESSED_DIR / f"{ds}_gt_{h}s.csv"); df["dataset"] = ds
    return df

def add_noise(df, level, seed=42):
    out = df.copy()
    rng = np.random.RandomState(seed)
    for c in NOISE_TARGETS:
        if c in out.columns:
            s = out[c].std()
            out[c] = out[c] + rng.normal(0, level * s, len(out))
    return out

def roc(y, p):
    return roc_auc_score(y, p) if y.nunique() > 1 else float("nan")

def main():
    h = 3   # 대표 horizon
    trains, tests = [], []
    for ds in config.TRAIN_DATASETS:
        tr, te = split_by_vehicle(load(ds, h), seed=config.RANDOM_STATE)
        trains.append(tr); tests.append(te)
    train = pd.concat(trains, ignore_index=True)
    test = pd.concat(tests, ignore_index=True)

    rows = []
    models = {}
    for g, cols in GROUPS.items():
        c = [x for x in cols if x in train.columns]
        models[g] = (fit_xgb(*_xy(train, c), seed=config.RANDOM_STATE), c)

    for lv in LEVELS:
        noisy = add_noise(test, lv)
        r = {"noise": lv}
        for g, (m, c) in models.items():
            X = noisy[c].replace([np.inf, -np.inf], np.nan)
            r[g] = round(roc(noisy["label"].astype(int), m.predict_proba(X)[:, 1]), 4)
        r["full_advantage"] = round(r["full"] - r["no_game"], 4)
        rows.append(r)
        print(f"  noise={lv:.1f}: full={r['full']:.4f} no_game={r['no_game']:.4f} (full우세 {r['full_advantage']:+.4f})")

    df = pd.DataFrame(rows)
    df.to_csv(config.TABLES_DIR / "noise_robustness.csv", index=False)
    # 하락폭 비교
    print("\n── 노이즈 0→0.5 AUC 하락폭 (작을수록 강건) ──")
    for g in GROUPS:
        drop = df[df.noise == 0.0][g].values[0] - df[df.noise == 0.5][g].values[0]
        print(f"  {g}: {drop:+.4f}")
    print("\n저장: noise_robustness.csv")

if __name__ == "__main__":
    main()
