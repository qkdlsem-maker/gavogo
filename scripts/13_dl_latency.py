#!/usr/bin/env python
"""[추가실험: 딥러닝 추론 지연] scripts/13_dl_latency.py
TabTransformer/BiLSTM (GPU) vs 트리 (CPU) 추론 지연 측정.
batch=1 (실시간) + batch=5000 (throughput) 둘 다.
가중치 불필요 — 아키텍처+입력shape만으로 측정.
실행: CUDA_VISIBLE_DEVICES=1 python scripts/13_dl_latency.py
"""
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config
from src.features.kinematic import FEATURE_COLS as KIN
from src.features.game_theory import NASH_FEATURE_COLS as GT

COLS = KIN + GT
H = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REPS = 100          # batch=1 반복
BATCH_REPS = 30     # batch=5000 반복


# ---- 아키텍처 (기존 phase7과 동일) ----
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=1, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model), nn.LayerNorm(d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128,
            dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1))

    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer(x)
        return self.classifier(x).squeeze(-1)


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, bidirectional=False, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, bidirectional=bidirectional,
                            dropout=dropout if num_layers > 1 else 0.0)
        fc_in = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Sequential(nn.Linear(fc_in, 32), nn.ReLU(),
                                nn.Dropout(dropout), nn.Linear(32, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


def time_dl(model, n_feat, batch):
    model.eval().to(DEVICE)
    x = torch.randn(batch, 1, n_feat, device=DEVICE)  # seq_len=1
    reps = REPS if batch == 1 else BATCH_REPS
    with torch.no_grad():
        for _ in range(10): model(x)          # warmup
        if DEVICE == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps): model(x)
        if DEVICE == "cuda": torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    return dt / reps / batch * 1000  # ms/sample


def main():
    # 피처 개수만 데이터에서 추출
    ds = config.TRAIN_DATASETS[0]
    df = pd.read_csv(config.PROCESSED_DIR / f"{ds}_gt_{H}s.csv")
    cols = [c for c in COLS if c in df.columns]
    n_feat = len(cols)
    print(f"device={DEVICE} | n_features={n_feat}")

    models = {
        "TabTransformer-1L": TransformerClassifier(n_feat, num_layers=1),
        "TabTransformer-2L": TransformerClassifier(n_feat, num_layers=2),
        "BiLSTM":            LSTMClassifier(n_feat, bidirectional=True),
    }

    rows = []
    for name, m in models.items():
        l1 = time_dl(m, n_feat, 1)
        l2 = time_dl(m, n_feat, 5000)
        rows.append(dict(model=name, device=DEVICE,
                         ms_per_sample_b1=round(l1, 4),
                         ms_per_sample_b5000=round(l2, 6)))
        print(f"  {name:18s} b1={l1:.4f}  b5000={l2:.6f} ms/sample")

    out = config.TABLES_DIR / "latency_dl.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\n저장: {out}")
    print("(트리 latency.csv와 합쳐서 비교 표 작성)")


if __name__ == "__main__":
    main()
