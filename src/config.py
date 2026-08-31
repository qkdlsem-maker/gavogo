"""config.py — 경로/상수 중앙 관리."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"

RAW_DIRS = {
    "highD": DATA_DIR / "raw" / "highD",
    "NGSIM": DATA_DIR / "raw" / "NGSIM",
    "MiTra": DATA_DIR / "raw" / "MiTra",
    "ETRI":  DATA_DIR / "raw" / "ETRI",
    "EMT":   DATA_DIR / "raw" / "EMT",
    "uniD":  DATA_DIR / "raw" / "uniD",
    "exiD":  DATA_DIR / "raw" / "exiD",
}

INTERIM_DIR   = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR   = PROJECT_ROOT / "results"
MODELS_DIR  = RESULTS_DIR / "models"
TABLES_DIR  = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"

# 데이터셋별 프레임레이트
FPS = {"highD": 25, "NGSIM": 10, "MiTra": 30, "ETRI": 10,
       "EMT": 25, "uniD": 25, "exiD": 25}

HORIZONS_SEC = [3, 5, 7]
EVENT_PERSISTENCE_K = 5
RANDOM_STATE = 42
SEEDS = [42, 0, 1, 7, 123]

# 학습/hold-out 역할
TRAIN_DATASETS  = ["highD", "NGSIM", "MiTra"]
HOLDOUT_DATASET = "ETRI"
OOD_DATASETS    = ["EMT", "uniD", "exiD"]  # 추가 OOD 검증용

# recording_id 범위
DEFAULT_RECS = {
    "highD": list(range(1, 61)),
    "NGSIM": [1, 2, 3],
    "MiTra": [1, 3, 4, 7, 8, 9],
    "ETRI":  [0, 1, 2, 3, 4],
    "uniD":  list(range(0, 13)),
    "exiD":  list(range(0, 93)) + list(range(1000, 1093)),
}

# EMT: 단일 CSV, recording_id = video 이름(문자열)
EMT_SINGLE_FILE = True

for d in [INTERIM_DIR, PROCESSED_DIR, MODELS_DIR, TABLES_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)
