"""Organise downloaded BDD100K archives and extract the val splits.

Archives are identified by their *contents*, not their filenames (some mirrors
ship misnamed zips). Finished zips in data/ are moved to data/archives/, and
val splits are extracted into a fixed layout:

  data/bdd100k/images/10k/val/*.jpg           (seg images, 1000)
  data/bdd100k/images/100k/val/*.jpg          (det images, 10000)
  data/bdd100k/images/frame100/val/*.jpg      (1 frame per video, no labels)
  data/bdd100k/labels/det_v1/val/*.json       (per-image v1 box2d labels)
  data/bdd100k/labels/sem_seg/val/*_train_id.png

Writes data/manifest.json (archive sha256, what was found, ID cross-checks).
Safe to re-run; it skips what is already extracted.

Usage:  python scripts/prepare_data.py [--data-dir data]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent.parent


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def classify(names: list[str]) -> tuple[str, str] | None:
    """Return (kind, member-prefix-regex for the val split) or None."""
    files = [n for n in names if not n.endswith("/")]
    if any(n.endswith("_train_id.png") for n in files):
        return "sem_seg_masks", r"(^|/)labels/val/[^/]+_train_id\.png$"
    if any(re.search(r"100k/val/[^/]+\.json$", n) for n in files):
        return "det_v1_labels", r"100k/val/[^/]+\.json$"
    jpgs = [n for n in files if n.lower().endswith(".jpg")]
    if jpgs:
        if any(re.search(r"-\d{7}\.jpg$", n) for n in jpgs):
            return "frame100_images", r"(^|/)val/[^/]+\.jpg$"
        if any("/100k/" in "/" + n for n in jpgs):
            return "images_100k", r"(^|/)100k/val/[^/]+\.jpg$"
        if any("/10k/" in "/" + n for n in jpgs):
            return "images_10k", r"(^|/)10k/val/[^/]+\.jpg$"
        # Unprefixed image zip: decide by val count (10k set has 1000 val).
        n_val = sum(bool(re.search(r"(^|/)val/[^/]+\.jpg$", n)) for n in jpgs)
        if n_val == 1000:
            return "images_10k", r"(^|/)val/[^/]+\.jpg$"
        if n_val == 10000:
            return "images_100k", r"(^|/)val/[^/]+\.jpg$"
    return None


DEST = {
    "sem_seg_masks": "bdd100k/labels/sem_seg/val",
    "det_v1_labels": "bdd100k/labels/det_v1/val",
    "frame100_images": "bdd100k/images/frame100/val",
    "images_100k": "bdd100k/images/100k/val",
    "images_10k": "bdd100k/images/10k/val",
}


def extract_val(zpath: Path, pattern: str, dest: Path) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    rx = re.compile(pattern)
    n = 0
    with zipfile.ZipFile(zpath) as zf:
        for info in zf.infolist():
            if info.is_dir() or not rx.search(info.filename):
                continue
            # Flatten to basename: guards against path traversal in the archive.
            name = PurePosixPath(info.filename).name
            if not name or name in (".", ".."):
                continue
            out = dest / name
            if not out.exists() or out.stat().st_size != info.file_size:
                with zf.open(info) as src, open(out, "wb") as dst:
                    shutil.copyfileobj(src, dst)
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    args = ap.parse_args()
    data = Path(args.data_dir)
    archives = data / "archives"
    archives.mkdir(parents=True, exist_ok=True)

    # Move finished zips (never .crdownload/.part) into archives/.
    for z in data.glob("*.zip"):
        shutil.move(str(z), archives / z.name)

    manifest_path = data / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"archives": {}}
    for z in sorted(archives.glob("*.zip")):
        entry = manifest["archives"].get(z.name, {})
        if entry.get("size") == z.stat().st_size and "kind" in entry:
            print(f"skip (already processed): {z.name} -> {entry['kind']}")
            continue
        with zipfile.ZipFile(z) as zf:
            names = zf.namelist()
        cls = classify(names)
        if cls is None:
            print(f"UNRECOGNISED archive, left untouched: {z.name}")
            manifest["archives"][z.name] = {"kind": None, "size": z.stat().st_size}
            continue
        kind, pattern = cls
        n = extract_val(z, pattern, data / DEST[kind])
        print(f"{z.name}: {kind}, extracted {n} val files -> {DEST[kind]}")
        manifest["archives"][z.name] = {"kind": kind, "size": z.stat().st_size,
                                        "sha256": sha256(z), "val_files": n,
                                        "dest": DEST[kind]}

    # Cross-check IDs between images and labels.
    def stems(rel, suffix):
        d = data / rel
        return {p.name[: -len(suffix)] for p in d.glob("*" + suffix)} if d.exists() else set()

    img10k = stems(DEST["images_10k"], ".jpg")
    img100k = stems(DEST["images_100k"], ".jpg")
    seg = stems(DEST["sem_seg_masks"], "_train_id.png")
    det = stems(DEST["det_v1_labels"], ".json")
    manifest["counts"] = {"images_10k_val": len(img10k), "images_100k_val": len(img100k),
                          "frame100_val": len(stems(DEST["frame100_images"], ".jpg")),
                          "sem_seg_masks_val": len(seg), "det_v1_labels_val": len(det)}
    manifest["id_checks"] = {
        "seg_masks_with_image": len(seg & img10k) if img10k else None,
        "seg_masks_without_image": len(seg - img10k) if img10k else None,
        "det_labels_with_image": len(det & img100k) if img100k else None,
        "det_labels_without_image": len(det - img100k) if img100k else None,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"counts": manifest["counts"], "id_checks": manifest["id_checks"]}, indent=2))


if __name__ == "__main__":
    main()
