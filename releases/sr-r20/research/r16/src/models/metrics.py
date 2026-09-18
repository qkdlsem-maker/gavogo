"""Binary calibration metrics with explicit probability-one handling."""
import numpy as np


def calibration_bins(y, p, bins=10):
    """ECE and per-bin positive fraction/mean probability, including p == 1."""
    y, p = np.asarray(y), np.asarray(p)
    if y.ndim != 1 or p.shape != y.shape or not len(p):
        raise ValueError("Expected nonempty, equally sized 1D labels and probabilities")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Expected binary labels and finite probabilities in [0, 1]")
    if not isinstance(bins, (int, np.integer)) or bins <= 0:
        raise ValueError("bins must be a positive integer")
    edges = np.linspace(0, 1, bins + 1)
    e, positive_fractions, confidences = 0.0, [], []
    for i in range(bins):
        m = (p >= edges[i]) & ((p <= edges[i+1]) if i == bins-1 else (p < edges[i+1]))
        if not m.any():
            positive_fractions.append(np.nan)
            confidences.append((edges[i] + edges[i+1])/2)
            continue
        a, c = float(y[m].mean()), float(p[m].mean())
        e += float(m.mean()) * abs(a-c)
        positive_fractions.append(a); confidences.append(c)
    return float(e), np.array(positive_fractions), np.array(confidences)
