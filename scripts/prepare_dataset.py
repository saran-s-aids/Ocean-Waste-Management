import json
import os
import re
from pathlib import Path
from urllib.parse import urlparse

import requests
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
ANN_PATH = ROOT / "data" / "taco_annotations.json"
OUT_DIR = ROOT / "data"
CLASS_DIRS = {
    "plastic": OUT_DIR / "plastic",
    "organic": OUT_DIR / "organic",
    "other": OUT_DIR / "other",
}

# Heuristics to map TACO category names to our 3 classes
PLASTIC_PAT = re.compile(r"plastic|polystyrene|styrofoam|foam|pet\b|poly", re.I)
ORGANIC_PAT = re.compile(r"vegetation|organic|leaf|seaweed|wood|banana|apple|food|fruit|vegetable|bread|egg|fish|branch|plant", re.I)

MAX_PER_CLASS = 30  # adjust as needed (smaller for quicker setup)
TIMEOUT = 20


def safe_filename(url: str) -> str:
    path = urlparse(url).path
    name = os.path.basename(path)
    if not name:
        name = re.sub(r"[^a-zA-Z0-9]", "", url)[:30] + ".jpg"
    return name


def main():
    for d in CLASS_DIRS.values():
        d.mkdir(parents=True, exist_ok=True)

    with open(ANN_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Build id -> image record
    images = {img["id"]: img for img in data["images"]}

    # Build image_id -> list of category names
    cat_id_to_name = {c["id"]: c["name"] for c in data["categories"]}
    img_to_cats = {}
    for ann in data["annotations"]:
        img_id = ann["image_id"]
        cat_name = cat_id_to_name.get(ann["category_id"], "")
        img_to_cats.setdefault(img_id, set()).add(cat_name)

    selected = {"plastic": [], "organic": [], "other": []}

    def classify(cat_names: set[str]) -> str | None:
        # Prioritize plastic if any plastic-like category appears
        joined = " ".join(cat_names)
        if PLASTIC_PAT.search(joined):
            return "plastic"
        if ORGANIC_PAT.search(joined):
            return "organic"
        # fallback to other if common non-plastic/organic trash
        return "other"

    # Iterate images and assign
    for img_id, cats in img_to_cats.items():
        if all(len(v) >= MAX_PER_CLASS for v in selected.values()):
            break
        label = classify(cats)
        if len(selected[label]) < MAX_PER_CLASS:
            selected[label].append(img_id)

    # Download helper
    def download(url: str, dest: Path) -> bool:
        try:
            with requests.get(url, stream=True, timeout=TIMEOUT) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name, leave=False) as pbar:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            return True
        except Exception:
            return False

    # Download images for each class
    for label, ids in selected.items():
        print(f"Downloading {len(ids)} images for {label}...")
        for img_id in ids:
            rec = images[img_id]
            url = rec.get("flickr_640_url") or rec.get("flickr_url")
            if not url:
                # some are OLM S3 or others
                url = rec.get("coco_url") or rec.get("url") or rec.get("flickr_url")
            if not url:
                continue
            fname = safe_filename(url)
            dest = CLASS_DIRS[label] / fname
            if dest.exists():
                continue
            ok = download(url, dest)
            if not ok:
                # try fallback to flickr_url if 640 failed
                if url != rec.get("flickr_url") and rec.get("flickr_url"):
                    url2 = rec.get("flickr_url")
                    fname2 = safe_filename(url2)
                    dest2 = CLASS_DIRS[label] / fname2
                    if not dest2.exists():
                        download(url2, dest2)

    print("Done. Class distribution:")
    for k, v in selected.items():
        print(k, len(v))


if __name__ == "__main__":
    main()
