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


# ------------------------------------------------- COCO-style detection AP
# Re-implements pycocotools' COCOeval for bbox, area "all", no crowd boxes:
# per (image, class) top-maxDets by score; each detection, in score order,
# takes the unmatched GT with the highest IoU >= t; precision made monotone;
# sampled at 101 recall points. scripts/eval_det.py cross-checks this
# against pycocotools.

IOU_THRS = np.linspace(0.5, 0.95, 10)
REC_THRS = np.linspace(0.0, 1.0, 101)


def _iou_xyxy(d: np.ndarray, g: np.ndarray) -> np.ndarray:
    x1 = np.maximum(d[:, None, 0], g[None, :, 0]); y1 = np.maximum(d[:, None, 1], g[None, :, 1])
    x2 = np.minimum(d[:, None, 2], g[None, :, 2]); y2 = np.minimum(d[:, None, 3], g[None, :, 3])
    inter = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    area = lambda b: (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])  # noqa: E731
    union = area(d)[:, None] + area(g)[None, :] - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1), 0.0)


def coco_ap(dets: list[np.ndarray], gts: list[np.ndarray], n_cls: int, max_dets: int = 100) -> dict:
    """dets[i]: [N,6] x1,y1,x2,y2,score,cls; gts[i]: [M,5] x1,y1,x2,y2,cls (same image order).

    Returns AP per class (averaged over IoU 0.50:0.95), AP50, AP75, and their
    means over classes that have at least one GT box.
    """
    T = len(IOU_THRS)
    per_cls = []
    for c in range(n_cls):
        scores, tps, n_gt = [], [], 0
        for d, g in zip(dets, gts):
            gc = g[g[:, 4] == c]
            dc = d[d[:, 5] == c]
            dc = dc[np.argsort(-dc[:, 4], kind="mergesort")][:max_dets]
            n_gt += len(gc)
            if len(dc) == 0:
                continue
            tp = np.zeros((T, len(dc)), bool)
            if len(gc):
                ious = _iou_xyxy(dc, gc)
                for ti, t in enumerate(IOU_THRS):
                    used = np.zeros(len(gc), bool)
                    thr = min(t, 1 - 1e-10)
                    for di in range(len(dc)):
                        cand = np.where(used, -1.0, ious[di])
                        m = len(cand) - 1 - int(np.argmax(cand[::-1]))   # last max, as COCOeval's loop
                        if cand[m] >= thr:
                            used[m] = True
                            tp[ti, di] = True
            scores.append(dc[:, 4])
            tps.append(tp)
        if n_gt == 0:
            per_cls.append(None)
            continue
        if not scores:
            per_cls.append({"n_gt": n_gt, "n_det": 0, "ap_per_iou": [0.0] * T})
            continue
        s = np.concatenate(scores)
        order = np.argsort(-s, kind="mergesort")
        tp = np.concatenate(tps, axis=1)[:, order]
        ap_t = []
        for ti in range(T):
            tpc = np.cumsum(tp[ti]).astype(np.float64)
            fpc = np.cumsum(~tp[ti]).astype(np.float64)
            rc = tpc / n_gt
            pr = tpc / np.maximum(tpc + fpc, np.spacing(1))
            pr = np.maximum.accumulate(pr[::-1])[::-1]
            idx = np.searchsorted(rc, REC_THRS, side="left")
            q = np.where(idx < len(pr), pr[np.minimum(idx, len(pr) - 1)], 0.0)
            ap_t.append(float(q.mean()))
        per_cls.append({"n_gt": n_gt, "n_det": int(len(s)), "ap_per_iou": ap_t})
    valid = [p for p in per_cls if p is not None]
    mean = lambda f: float(np.mean([f(p) for p in valid])) if valid else None  # noqa: E731
    return {
        "per_class": [None if p is None else {"n_gt": p["n_gt"], "n_det": p["n_det"],
                                              "AP": float(np.mean(p["ap_per_iou"])),
                                              "AP50": p["ap_per_iou"][0], "AP75": p["ap_per_iou"][5]}
                      for p in per_cls],
        "mAP": mean(lambda p: np.mean(p["ap_per_iou"])), "mAP50": mean(lambda p: p["ap_per_iou"][0]),
        "mAP75": mean(lambda p: p["ap_per_iou"][5]), "n_classes_with_gt": len(valid),
    }
