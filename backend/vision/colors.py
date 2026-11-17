"""Vehicle colour recognition - pure OpenCV, no extra models needed."""
from __future__ import annotations

import cv2
import numpy as np


def classify_vehicle_color(bgr: np.ndarray) -> str:
    """Classify the dominant body colour of a vehicle crop."""
    if bgr is None or bgr.size == 0:
        return "Unknown"
    h, w = bgr.shape[:2]
    if h < 8 or w < 8:
        return "Unknown"
    roi = bgr[int(h * 0.18):int(h * 0.75), int(w * 0.12):int(w * 0.88)]
    if roi.size == 0:
        roi = bgr
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hh, ss, vv = hsv[:, :, 0].astype(np.float32), hsv[:, :, 1].astype(np.float32), \
        hsv[:, :, 2].astype(np.float32)
    mask = vv > 30
    if not mask.any():
        return "Black"
    sat = float(np.median(ss[mask]))
    val = float(np.median(vv[mask]))

    if sat < 48:  # achromatic body
        if val >= 190:
            return "White"
        if val >= 115:
            return "Silver"
        if val >= 55:
            return "Gray"
        return "Black"

    hue = float(np.median(hh[mask]))
    if hue < 10 or hue >= 168:
        return "Red"
    if hue < 22:
        return "Brown" if val < 150 else "Orange"
    if hue < 36:
        return "Yellow"
    if hue < 86:
        return "Green"
    if hue < 100:
        return "Cyan"
    if hue < 132:
        return "Blue"
    if hue < 168:
        return "Purple"
    return "Unknown"
