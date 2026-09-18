"""models/baselines.py — baseline 모델. 위치: src/models/baselines.py

공정 비교(리뷰 Q4) 원칙
  - 모든 모델이 동일한 정보에 접근해야 한다.
  - 기존 SEQ_COLS에는 횡방향 정보가 없었다(원본 vy는 NGSIM/MiTra에서 0으로 채워져 무용).
    그 결과 BiLSTM만 lateral cue 없이 학습되어 트리 대비 불공정하게 낮은 성능(0.59)을 보였다.
  - → build_sequences()가 road frame을 복원해 lat_offset을 시퀀스 피처에 포함시킨다.
  - GPU 추론은 배치 단위로 수행 (exiD 82k행 한 번에 올리면 CUDA 커널 오류).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


# ── LightGBM ──────────────────────────────────
def fit_lgbm(Xtr, ytr, seed=42):
    import lightgbm as lgb
    m = lgb.LGBMClassifier(n_estimators=500, max_depth=6, learning_rate=0.05,
                           num_leaves=31, random_state=seed, n_jobs=-1, verbose=-1)
    m.fit(Xtr, ytr)
    return m


# ── CatBoost ──────────────────────────────────
def fit_catboost(Xtr, ytr, seed=42):
    from catboost import CatBoostClassifier
    m = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.05,
                           random_seed=seed, verbose=False, allow_writing_files=False)
    m.fit(Xtr, ytr)
    return m


# ── LSTM 시퀀스 빌더 ──────────────────────────
# lat_offset 포함: 도로 국소좌표(차선 중심선 기준) 횡방향 이탈.
# 원본 vy는 데이터셋마다 0으로 채워져 있어 제외한다.
SEQ_COLS = ["vx", "ax", "lane_id", "ttc", "dhw", "thw", "preceding_vx", "lat_offset"]


def _ensure_road_frame(canon):
    """canon에 lat_offset이 없으면 recording 단위로 road frame을 복원해 추가."""
    if "lat_offset" in canon.columns:
        return canon
    from src.features.roadframe import add_road_frame
    parts = []
    for _, g in canon.groupby((["dataset"] if "dataset" in canon else []) + ["recording_id"], sort=False):
        try:
            parts.append(add_road_frame(g.reset_index(drop=True)))
        except Exception:
            g = g.reset_index(drop=True).copy()
            g["lat_offset"] = 0.0
            parts.append(g)
    return pd.concat(parts, ignore_index=True)


def build_sequences(samples, canon, K=10):
    """Past K-frame window, isolated by recording/vehicle (and dataset if present).

    Missing history is left-zero-padded; frame gaps are not interpolated.
    This fixes identity mixing, not the offline road-frame estimation policy.
    """
    if not isinstance(K, (int, np.integer)) or K <= 0:
        raise ValueError("K must be a positive integer")
    keys = ["recording_id", "vehicle_id"]
    if "dataset" in samples and "dataset" in canon:
        keys = ["dataset"] + keys
    elif any("dataset" in d and d["dataset"].nunique() > 1 for d in (samples, canon)):
        raise ValueError("Multiple datasets require dataset IDs in both inputs")
    for d in (samples, canon):
        if not set(keys + ["frame"]).issubset(d.columns):
            raise ValueError("Sequences require recording_id, vehicle_id and frame")
        if d[keys + ["frame"]].isna().any().any():
            raise ValueError("Sequence identity keys must not be missing")
    if canon.duplicated(keys + ["frame"]).any():
        raise ValueError("Duplicate canonical recording/vehicle/frame keys")
    canon = _ensure_road_frame(canon)
    tracks = {}
    for key, g in canon.groupby(keys, sort=False):
        g = g.sort_values("frame", kind="stable")
        tracks[key] = (g.frame.to_numpy(), g[SEQ_COLS].to_numpy())
    X = np.zeros((len(samples), K, len(SEQ_COLS)), dtype=np.float32)
    for i, row in enumerate(samples[keys + ["frame"]].itertuples(index=False, name=None)):
        track = tracks.get(tuple(row[:-1]))
        if track is None:
            continue
        frames, values = track
        lo = np.searchsorted(frames, row[-1] - K, side="right")
        hi = np.searchsorted(frames, row[-1], side="right")
        h = values[lo:hi][-K:]
        if len(h):
            X[i, -len(h):] = np.nan_to_num(h)
    return X, samples["label"].astype(int).to_numpy()


# ── BiLSTM + Attention (PyTorch) ──────────────
def fit_lstm(Xtr, ytr, Xval=None, epochs=30, seed=42, device=None):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    F = Xtr.shape[2]

    # 시퀀스 표준화 (피처축 기준) — 트리와 달리 스케일에 민감
    mu = Xtr.reshape(-1, F).mean(0)
    sd = Xtr.reshape(-1, F).std(0) + 1e-6

    class Net(nn.Module):
        def __init__(s):
            super().__init__()
            s.lstm = nn.LSTM(F, 64, batch_first=True, bidirectional=True)
            s.att = nn.Linear(128, 1)
            s.fc = nn.Sequential(nn.Linear(128, 64), nn.ReLU(),
                                 nn.Dropout(0.2), nn.Linear(64, 1))

        def forward(s, x):
            o, _ = s.lstm(x)
            w = torch.softmax(s.att(o), dim=1)
            ctx = (w * o).sum(1)
            return s.fc(ctx).squeeze(-1)

    net = Net().to(device)
    net._mu, net._sd = mu, sd          # 추론 시 동일 표준화 적용
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossf = nn.BCEWithLogitsLoss()

    Xn = ((Xtr - mu) / sd).astype(np.float32)
    Xt = torch.tensor(Xn).to(device)
    yt = torch.tensor(ytr, dtype=torch.float32).to(device)

    net.train()
    for _ in range(epochs):
        perm = torch.randperm(len(Xt))
        for b in range(0, len(Xt), 512):
            i = perm[b:b + 512]
            opt.zero_grad()
            loss = lossf(net(Xt[i]), yt[i])
            loss.backward()
            opt.step()
    return net, device


def lstm_auc(net, device, X, y, bs=4096):
    import torch
    net.eval()
    mu = getattr(net, "_mu", 0.0)
    sd = getattr(net, "_sd", 1.0)
    Xn = ((X - mu) / sd).astype(np.float32)
    ps = []
    with torch.no_grad():
        for i in range(0, len(Xn), bs):
            xb = torch.tensor(Xn[i:i + bs]).to(device)
            ps.append(torch.sigmoid(net(xb)).cpu().numpy())
    p = np.concatenate(ps) if ps else np.zeros(0)
    return roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")


# ── TabTransformer ────────────────────────────
def fit_tabtransformer(Xtr, ytr, epochs=30, seed=42, device=None):
    import torch, torch.nn as nn
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    F = Xtr.shape[1]
    d = 32

    class TabT(nn.Module):
        def __init__(s):
            super().__init__()
            s.emb = nn.Linear(1, d)
            s.pos = nn.Parameter(torch.randn(1, F, d) * 0.02)
            enc = nn.TransformerEncoderLayer(d, nhead=4, dim_feedforward=64,
                                             batch_first=True, dropout=0.1)
            s.tr = nn.TransformerEncoder(enc, num_layers=2)
            s.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 32), nn.ReLU(),
                                   nn.Dropout(0.2), nn.Linear(32, 1))

        def forward(s, x):
            t = s.emb(x.unsqueeze(-1)) + s.pos
            t = s.tr(t).mean(1)
            return s.head(t).squeeze(-1)

    net = TabT().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    lossf = nn.BCEWithLogitsLoss()
    Xv = np.nan_to_num(Xtr.values if hasattr(Xtr, "values") else Xtr)
    Xt = torch.tensor(Xv, dtype=torch.float32).to(device)
    yt = torch.tensor(np.asarray(ytr), dtype=torch.float32).to(device)

    net.train()
    for _ in range(epochs):
        perm = torch.randperm(len(Xt))
        for b in range(0, len(Xt), 512):
            i = perm[b:b + 512]
            opt.zero_grad()
            loss = lossf(net(Xt[i]), yt[i])
            loss.backward()
            opt.step()
    return net, device


def tabt_auc(net, device, X, y, bs=4096):
    import torch
    net.eval()
    Xv = np.nan_to_num(X.values if hasattr(X, "values") else X)
    ps = []
    with torch.no_grad():
        for i in range(0, len(Xv), bs):
            xb = torch.tensor(Xv[i:i + bs], dtype=torch.float32).to(device)
            ps.append(torch.sigmoid(net(xb)).cpu().numpy())
    p = np.concatenate(ps) if ps else np.zeros(0)
    return roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")
