"""Optional check: bsched.metrics.coco_ap vs pycocotools on saved predictions.

pycocotools is NOT a project dependency; install it anywhere and pass its
directory. Usage:
  python scripts/crosscheck_coco.py <project_root> <preds.npz> <n_frames> <out.json> <pycocotools_dir> [start]

Checks frames [start, start + n_frames) as one evaluation, so a subset or
disjoint chunks can be checked too. All 10 000 frames (~2.3M detections)
take ~6 min on the dev laptop.
"""
import sys, json
from pathlib import Path
root = Path(sys.argv[1]); npz = sys.argv[2]; n = int(sys.argv[3]); out = sys.argv[4]
start = int(sys.argv[6]) if len(sys.argv) > 6 else 0
sys.path.insert(0, str(root)); sys.path.insert(0, str(root / "scripts")); sys.path.append(sys.argv[5])
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from bsched.config import load_config
from bsched.metrics import coco_ap
import eval_det
cfg = load_config(None)
names, _ = eval_det.eval_classes(cfg)
paths, gts, _, _ = eval_det.load_frames_and_gt(cfg, start + n)
paths, gts = paths[start:], gts[start:]
z = np.load(npz)
all_dets, img_idx = z["dets"], z["image_index"]   # each z[...] access re-reads the whole array
bounds = np.searchsorted(img_idx, np.arange(start, start + len(paths) + 1))   # image_index is sorted
dets = [all_dets[bounds[i]:bounds[i + 1]] for i in range(len(paths))]
assert list(z["frames"][start: start + len(paths)]) == [Path(p).name for p in paths]
images = [{"id": i, "width": 1280, "height": 720} for i in range(len(paths))]
anns, k = [], 1
for i, g in enumerate(gts):
    for x1, y1, x2, y2, c in g:
        anns.append({"id": k, "image_id": i, "category_id": int(c) + 1, "bbox": [x1, y1, x2 - x1, y2 - y1],
                     "area": (x2 - x1) * (y2 - y1), "iscrowd": 0}); k += 1
gt = COCO(); gt.dataset = {"images": images, "annotations": anns,
                           "categories": [{"id": j + 1, "name": nm} for j, nm in enumerate(names)]}; gt.createIndex()
dt_list = [{"image_id": i, "category_id": int(d[5]) + 1, "bbox": [d[0], d[1], d[2] - d[0], d[3] - d[1]], "score": float(d[4])}
           for i, ds in enumerate(dets) for d in ds]
dt = gt.loadRes(dt_list)
E = COCOeval(gt, dt, "bbox"); E.params.maxDets = [1, 10, 100]; E.evaluate(); E.accumulate(); E.summarize()
prec = E.eval["precision"]  # [T, R, K, A, M]
ref = {}
for j, nm in enumerate(names):
    p = prec[:, :, j, 0, 2]
    ref[nm] = None if (p == -1).all() else {"AP": float(p[p > -1].mean()), "AP50": float(p[0][p[0] > -1].mean())}
ours = coco_ap(dets, gts, len(names), 100)
rows = {}
for j, nm in enumerate(names):
    o = ours["per_class"][j]; r = ref[nm]
    rows[nm] = {"ours_AP": o and o["AP"], "pycocotools_AP": r and r["AP"],
                "ours_AP50": o and o["AP50"], "pycocotools_AP50": r and r["AP50"]}
res = {"frames": [start, start + len(paths)], "n_frames": len(paths), "ours_mAP": ours["mAP"], "pycocotools_mAP": float(E.stats[0]),
       "ours_mAP50": ours["mAP50"], "pycocotools_mAP50": float(E.stats[1]),
       "max_abs_diff_per_class_AP": max(abs(v["ours_AP"] - v["pycocotools_AP"]) for v in rows.values() if v["ours_AP"] is not None),
       "per_class": rows}
Path(out).write_text(json.dumps(res, indent=1)); print(json.dumps({k: v for k, v in res.items() if k != "per_class"}, indent=1))
