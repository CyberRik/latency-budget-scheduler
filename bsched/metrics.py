"""Accuracy/equivalence metrics shared by the verification and evaluation scripts."""
from __future__ import annotations

import numpy as np

IGNORE = 255


def match_by_iou(a: np.ndarray, b: np.ndarray, thr: float = 0.5):
    """Greedy one-to-one matching of [N,6] (x1,y1,x2,y2,score,cls) boxes by IoU,
    within the same class. Returns (pairs [(i, j, iou)], n_unmatched)."""
    if len(a) == 0 or len(b) == 0:
        return [], len(a) + len(b)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    area = lambda d: (d[:, 2] - d[:, 0]) * (d[:, 3] - d[:, 1])  # noqa: E731
    iou = inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)
    # YOLOv10 can emit the same box twice with different classes; match within class.
    iou = np.where(a[:, None, 5] == b[None, :, 5], iou, 0.0)
    pairs, used_a, used_b = [], set(), set()
    for i, j in zip(*np.unravel_index(np.argsort(-iou, axis=None), iou.shape)):
        if iou[i, j] < thr:
            break
        if i not in used_a and j not in used_b:
            pairs.append((int(i), int(j), float(iou[i, j])))
            used_a.add(i); used_b.add(j)
    return pairs, (len(a) - len(pairs)) + (len(b) - len(pairs))


def det_agreement(a: np.ndarray, b: np.ndarray) -> dict:
    pairs, unmatched = match_by_iou(a, b)
    out = {"n_a": len(a), "n_b": len(b), "matched": len(pairs), "unmatched": unmatched}
    if pairs:
        A, B = a[[p[0] for p in pairs]], b[[p[1] for p in pairs]]
        out.update(min_iou=min(p[2] for p in pairs),
                   max_box_abs_diff_px=float(np.abs(A[:, :4] - B[:, :4]).max()),
                   max_score_abs_diff=float(np.abs(A[:, 4] - B[:, 4]).max()))
    return out


def confusion(pred: np.ndarray, gt: np.ndarray, n_cls: int) -> np.ndarray:
    """n_cls x n_cls confusion (rows = GT, cols = pred), ignoring GT == 255."""
    valid = gt != IGNORE
    idx = gt[valid].astype(np.int64) * n_cls + pred[valid].astype(np.int64)
    return np.bincount(idx, minlength=n_cls * n_cls).reshape(n_cls, n_cls)


def miou(conf: np.ndarray) -> dict:
    """Standard dataset-level mIoU from an accumulated confusion matrix.
    Classes absent from both GT and prediction are excluded (NaN)."""
    tp = np.diag(conf).astype(np.float64)
    denom = conf.sum(0) + conf.sum(1) - tp
    iou = np.where(denom > 0, tp / np.maximum(denom, 1), np.nan)
    return {"miou": float(np.nanmean(iou)), "per_class_iou": [None if np.isnan(v) else float(v) for v in iou],
            "pixel_acc": float(tp.sum() / max(conf.sum(), 1)),
            "classes_present_in_gt": int((conf.sum(1) > 0).sum())}
