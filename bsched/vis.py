"""Visualisation helpers for eyeballing correctness."""
from __future__ import annotations

import cv2
import numpy as np

from bsched.models import CITYSCAPES_NAMES, CITYSCAPES_PALETTE, COCO_NAMES


def draw_detections(bgr: np.ndarray, dets: np.ndarray) -> np.ndarray:
    img = bgr.copy()
    for x1, y1, x2, y2, score, cls in dets:
        c = int(cls)
        color = tuple(int(v) for v in np.random.default_rng(c).integers(64, 256, 3))
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(img, p1, p2, color, 2)
        label = f"{COCO_NAMES[c]} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (p1[0], p1[1] - th - 4), (p1[0] + tw + 2, p1[1]), color, -1)
        cv2.putText(img, label, (p1[0] + 1, p1[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 1, cv2.LINE_AA)
    return img


def draw_segmentation(bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    color = CITYSCAPES_PALETTE[mask][:, :, ::-1]  # RGB palette -> BGR
    img = cv2.addWeighted(bgr, 1 - alpha, color, alpha, 0)
    # Legend of classes present, largest first.
    ids, counts = np.unique(mask, return_counts=True)
    y = 20
    for i in ids[np.argsort(-counts)][:10]:
        cv2.rectangle(img, (8, y - 12), (22, y + 2), tuple(int(v) for v in CITYSCAPES_PALETTE[i][::-1]), -1)
        cv2.putText(img, CITYSCAPES_NAMES[i], (28, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        y += 20
    return img


def side_by_side(*imgs: np.ndarray, width: int = 960) -> np.ndarray:
    out = []
    for im in imgs:
        h, w = im.shape[:2]
        out.append(cv2.resize(im, (width, round(h * width / w))))
    return np.concatenate(out, axis=0)
