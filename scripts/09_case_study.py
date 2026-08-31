#!/usr/bin/env python
"""[추가실험: Case Study] 개별 사례 설명. 위치: scripts/09_case_study.py
joint 모델로 실제 lane-change 직전 차량 사례를 골라
예측확률 + nash_lc_prob + nash_urgency + threat_asymmetry + 실제결과를 제시.
"해석 가능하다"는 주장을 구체 사례로 뒷받침.
실행: python scripts/09_case_study.py"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config
from src.features.kinematic import FEATURE_COLS as KIN
from src.features.game_theory import NASH_FEATURE_COLS as GT
from src.models.train import split_by_vehicle, fit_xgb, _xy

COLS = KIN + GT
H = 3
SHOW = ["nash_lc_prob", "nash_urgency", "threat_asymmetry", "payoff_gap"]

def load(ds):
    df = pd.read_csv(config.PROCESSED_DIR / f"{ds}_gt_{H}s.csv"); df["dataset"] = ds
    return df

def main():
    trains, tests = [], []
    for ds in config.TRAIN_DATASETS:
        tr, te = split_by_vehicle(load(ds), seed=config.RANDOM_STATE)
        trains.append(tr); tests.append(te)
    train = pd.concat(trains, ignore_index=True)
    test = pd.concat(tests, ignore_index=True)
    cols = [c for c in COLS if c in train.columns]
    m = fit_xgb(*_xy(train, cols), seed=config.RANDOM_STATE)

    X = test[cols].replace([np.inf, -np.inf], np.nan)
    test = test.copy(); test["pred"] = m.predict_proba(X)[:, 1]

    # 실제 lane-change(label=1) 중 고확신 사례 3개 + lane-keep(label=0) 1개
    lc = test[test.label == 1].nlargest(3, "pred")
    lk = test[test.label == 0].nsmallest(1, "pred")
    cases = pd.concat([lc, lk])

    print("── Case Study ──")
    rows = []
    for i, (_, r) in enumerate(cases.iterrows(), 1):
        d = {"case": i, "dataset": r["dataset"], "vehicle": int(r["vehicle_id"]),
             "pred_prob": round(r["pred"], 3), "actual": "LANE-CHANGE" if r.label == 1 else "keep"}
        for c in SHOW:
            d[c] = round(r[c], 3) if c in r else np.nan
        rows.append(d)
        print(f"  Case {i}: {r['dataset']} veh#{int(r['vehicle_id'])} | pred={r['pred']:.3f} "
              f"nash_lc={r.get('nash_lc_prob',0):.2f} urgency={r.get('nash_urgency',0):.2f} "
              f"| actual={d['actual']}")
    cdf = pd.DataFrame(rows)
    cdf.to_csv(config.TABLES_DIR / "case_study.csv", index=False)

    # 그림: 사례별 예측확률 + 게임피처 (0~1 범위만; threat_asymmetry는 표에만)
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(cdf)); w = 0.25
    ax.bar(x - w, cdf["pred_prob"], w, label="Pred. prob", color="#2E6FB7")
    ax.bar(x, cdf["nash_lc_prob"], w, label="Nash LC prob", color="#E08A1E")
    ax.bar(x + w, cdf["nash_urgency"], w, label="Strategic urgency", color="#3FA46A")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels([f"#{r.vehicle}\n({r.actual})" for _, r in cdf.iterrows()], fontsize=8)
    ax.set_ylabel("Value"); ax.set_title("Case Study: Prediction vs Game-theoretic Signals")
    ax.legend(fontsize=8, loc="upper center", ncol=3, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3); ax.set_ylim(0, 1.15)
    plt.tight_layout(); plt.savefig(config.FIGURES_DIR / "case_study.png", dpi=150); plt.close()
    print("\n저장: case_study.csv, case_study.png")

if __name__ == "__main__":
    main()
