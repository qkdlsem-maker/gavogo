"""Read EMT KITTI boxes in pixels; do not invent meters, lanes or timestamps.

Unlike the legacy external finalize_csv.py, KITTI bbox columns are 6:10.
These pixel tracks are not canonical road trajectories and cannot be passed
as metre-valued inputs to the lane-change benchmark without calibration.
"""
from pathlib import Path
import numpy as np
import pandas as pd


def read_kitti_pixel_tracks(path, recording_id=None):
    path = Path(path)
    d = pd.read_csv(path, sep=r"\s+", header=None)
    if d.shape[1] != 17:
        raise ValueError("Expected the 17-column EMT KITTI annotation format")
    boxes = d.iloc[:, 6:10].to_numpy(dtype=float)
    if not np.isfinite(boxes).all() or (boxes[:, 2:] < boxes[:, :2]).any():
        raise ValueError("Invalid KITTI bounding boxes")
    out = pd.DataFrame({
        "recording_id": recording_id or path.stem,
        "frame": d[0].astype(int), "vehicle_id": d[1].astype(int), "label": d[2],
        "bbox_center_x_px": (boxes[:, 0] + boxes[:, 2]) / 2,
        "bbox_center_y_px": (boxes[:, 1] + boxes[:, 3]) / 2,
    })
    if out.duplicated(["recording_id", "vehicle_id", "frame"]).any():
        raise ValueError("Duplicate EMT track/frame keys")
    return out
