#!/usr/bin/env python
"""[Table 5 재현] 관습(cross-vehicle) 프로토콜의 group-label purity와 AUC.

배경: 논문 Table 5(및 Introduction/Highlights가 인용하는 exiD purity=0.91, AUC=0.98 등)를
     실제로 만들어내는 스크립트가 저장소에 없었다.
       - scripts/27_conventional_leakage.py 는 vehicle-id 5-fold target-encoding으로 만든
         대리 AUC(conv_id_auc)만 계산한다 — 실제 피처로 모델을 학습하지 않는다.
       - scripts/32_purity_auc.py 는 실제 XGBoost를 학습하지만 row-level(무작위 행 단위)
         분할을 의도적으로 쓴다 — Table 5가 쓰는 그룹(차량) 단위 분할이 아니다.
     둘 다 Table 5의 "Purity / AUC(all vehicles) / AUC(mixed-label vehicles only)" 세
     컬럼을 만들지 않는다. 이 스크립트가 그 계산을 담당한다: (recording, vehicle) 그룹
     단위 70/30 분할 + fit_xgb(논문 Section 5의 하이퍼파라미터와 동일: 500 trees, depth 6,
     learning_rate 0.05, hist)로 실제 특징 기반 모델을 학습·평가한다.

두 가지 데이터 소스를 지원한다 (--source):
  backup    (기본값) data/processed_crossveh_backup/{dataset}_{h}s.csv를 그대로 읽는다.
            파이프라인이 within-vehicle 샘플링으로 개편되기 전에 남은 백업이며, 24개의
            구형 피처(횡방향/road-frame 피처 없음)만 담고 있다. 논문 Table 5의 원 수치와
            정확히 대응한다(±0.01 이내) — 이 스크립트가 그 사실을 검증했다.
  canonical data/interim/canonical/{ds}/*.parquet + events_{ds}.csv로부터 매번 새로
            관습적(cross-vehicle) 샘플을 만들고, *현재* build_features() — 즉 논문의
            나머지 모든 표(Table 6 이하)가 쓰는 것과 동일한, road-frame 횡방향 피처가
            포함된 34개 피처 — 로 학습·평가한다. data/processed_crossveh_backup/ 없이도
            동작하므로 재현성 공백을 메운다. 다만 backup과는 피처 집합이 달라(구버전은
            횡방향 피처가 없었다 — 본문 4.1절 "It was absent from our own earlier
            iterations" 참고) 숫자가 backup 기반 결과 및 Table 5의 원 수치와 달라질 수
            있다는 점을 스크립트가 그대로 보여준다 — 이 역시 사실확인 결과의 일부다.

실행: python scripts/35_table5_repro.py [--source backup|canonical]
출력: results/tables/table5_repro.csv          (source=backup)
      results/tables/table5_repro_canonical.csv (source=canonical)
컬럼: dataset,horizon,purity,auc_all,auc_mixed_only,n_test_vehicles,n_mixed_test_vehicles
"""
import argparse
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config
from src.models.train import fit_xgb
from src.features.kinematic import build_samples, balance, build_features, FEATURE_COLS
from src.features.roadframe import add_road_frame, estimate_lateral_sign

BACKUP_DIR = config.DATA_DIR / "processed_crossveh_backup"
DATASETS = ["highD", "NGSIM", "MiTra", "ETRI", "EMT", "uniD", "exiD"]
HORIZONS = config.HORIZONS_SEC  # [3, 5, 7]
TEST_FRAC = 0.30
SEED = config.RANDOM_STATE  # 42

# 논문 Table 5가 실제로 보고하는 네 개 (dataset, horizon) 행.
TABLE5_ROWS = {("highD", 3), ("NGSIM", 3), ("MiTra", 3), ("exiD", 7)}


# ── 소스 1: 고정 백업 CSV (구형 24피처, Table 5 원 수치와 대응) ─────────────
def load_backup(ds, h):
    path = BACKUP_DIR / f"{ds}_{h}s.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    feat_cols = [c for c in df.columns if c not in ("recording_id", "vehicle_id", "frame", "label")]
    return df, feat_cols


# ── 소스 2: canonical에서 매번 재구성 (현재 34피처, road-frame 포함) ────────
def dataset_lateral_sign(files, events, fps, max_files=15):
    """scripts/02_build_features.py의 dataset_lateral_sign()과 동일한 로직
    (다수결로 데이터셋 전체의 횡방향 부호를 정한다, 본문 3.2절)."""
    score = 0.0
    for cf in files[:max_files]:
        canon = pd.read_parquet(cf, columns=["recording_id", "vehicle_id",
                                             "frame", "x", "y", "lane_id"])
        s, n = estimate_lateral_sign(add_road_frame(canon), events, fps=fps)
        score += s * n
    return 1.0 if score >= 0 else -1.0


def build_conventional_canonical(ds, h):
    """canonical + events로부터 관습적(cross-vehicle) 샘플을 새로 만들고,
    *현재* build_features() (road-frame 횡방향 피처 포함, 34개)로 특징을 계산한다."""
    cdir = config.INTERIM_DIR / "canonical" / ds
    events_path = config.INTERIM_DIR / f"events_{ds}.csv"
    if not cdir.exists() or not events_path.exists():
        return None
    files = sorted(cdir.glob("*.parquet"))
    if not files:
        return None
    events = pd.read_csv(events_path)
    fps = config.FPS[ds]
    lat_sign = dataset_lateral_sign(files, events, fps)
    H = h * fps

    parts = []
    for cf in files:
        canon = pd.read_parquet(cf)
        s = build_samples(canon, events, H, random_state=SEED)  # 구 프로토콜(cross-vehicle)
        if len(s) == 0:
            continue
        s = balance(s, 1.0, SEED)
        if len(s) == 0:
            continue
        parts.append(build_features(s, canon, fps=fps, lat_sign=lat_sign, use_road_frame=True))
    if not parts:
        return None
    df = pd.concat(parts, ignore_index=True)
    feat_cols = [c for c in FEATURE_COLS if c in df.columns]
    return df, feat_cols


# ── 공통 평가: (recording,vehicle) 그룹 70/30 분할 + fit_xgb ────────────────
def evaluate(df, feat_cols, ds, h, seed=SEED, test_frac=TEST_FRAC):
    key = (df["recording_id"].astype(str) + "|" + df["vehicle_id"].astype(str)).values
    per = pd.DataFrame({"g": key, "y": df["label"].astype(int).values}).groupby("g").y.mean()
    pure = float(((per == 0) | (per == 1)).mean())

    rng = np.random.RandomState(seed)
    groups = per.index.values.copy()
    rng.shuffle(groups)
    n_test = max(1, int(len(groups) * test_frac))
    test_g = set(groups[:n_test])
    is_test = pd.Series(key).isin(test_g).values
    is_train = ~is_test

    X = df[feat_cols].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    y = df["label"].astype(int).values

    if len(np.unique(y[is_train])) < 2 or len(np.unique(y[is_test])) < 2:
        return dict(dataset=ds, horizon=h, purity=round(pure, 4), auc_all=float("nan"),
                    auc_mixed_only=float("nan"), n_test_vehicles=len(test_g),
                    n_mixed_test_vehicles=0)

    m = fit_xgb(X[is_train], y[is_train], seed=seed)
    auc_all = roc_auc_score(y[is_test], m.predict_proba(X[is_test])[:, 1])

    mixed_groups = set(per[(per > 0) & (per < 1)].index)
    is_test_mixed = pd.Series(key).isin(mixed_groups & test_g).values
    if is_test_mixed.sum() > 0 and len(np.unique(y[is_test_mixed])) > 1:
        auc_mixed = float(roc_auc_score(y[is_test_mixed], m.predict_proba(X[is_test_mixed])[:, 1]))
    else:
        auc_mixed = float("nan")

    return dict(dataset=ds, horizon=h, purity=round(pure, 4),
                auc_all=round(float(auc_all), 4),
                auc_mixed_only=round(auc_mixed, 4) if auc_mixed == auc_mixed else float("nan"),
                n_test_vehicles=len(test_g), n_mixed_test_vehicles=int(is_test_mixed.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["backup", "canonical"], default="backup")
    a = ap.parse_args()

    loader = load_backup if a.source == "backup" else build_conventional_canonical
    out_path = config.TABLES_DIR / (
        "table5_repro.csv" if a.source == "backup" else "table5_repro_canonical.csv")

    rows = []
    for ds in DATASETS:
        for h in HORIZONS:
            t0 = time.time()
            built = loader(ds, h)
            if built is None:
                print(f"  {ds:6s} h={h}s  데이터 없음, 건너뜀")
                continue
            df, feat_cols = built
            r = evaluate(df, feat_cols, ds, h)
            tag = "  <- Table 5" if (ds, h) in TABLE5_ROWS else ""
            print(f"  {ds:6s} h={h}s  purity={r['purity']:.4f}  "
                  f"AUC(all)={r['auc_all']:.4f}  AUC(mixed)={r['auc_mixed_only']:.4f}"
                  f"  ({time.time()-t0:.1f}s){tag}", flush=True)
            rows.append(r)
            # 도중에 중단되어도 지금까지의 결과를 잃지 않도록 매 조합마다 즉시 저장
            pd.DataFrame(rows).to_csv(out_path, index=False)

    print(f"\n저장: {out_path.name}")
    if a.source == "backup":
        print("주의: data/processed_crossveh_backup/ 은 파이프라인 개편 이전 관습적")
        print("(cross-vehicle) 샘플링 데이터의 백업이며, 24개의 구형 피처(횡방향 피처 없음)만")
        print("담고 있다. 이 폴더가 없는 환경에서는 --source canonical 을 사용하라.")
    else:
        print("주의: 이 실행은 canonical 레코드로부터 관습적(cross-vehicle) 샘플링을 매번")
        print("새로 재구성하고, *현재* 34개 피처(road-frame 횡방향 피처 포함)로 평가한다.")
        print("backup(구형 24피처, 횡방향 없음) 기반 결과 및 논문 Table 5의 원 수치와")
        print("숫자가 다를 수 있다 — 두 피처 집합이 다르기 때문이며, 이는 재현 실패가")
        print("아니라 파이프라인이 반복 개발되며 피처 집합이 바뀌었다는 사실 자체를 보여준다.")


if __name__ == "__main__":
    main()
