from pathlib import Path
from PIL import Image

import shutil

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
BAD_DIR = DATA_DIR / "_bad"


def is_image_ok(p: Path) -> bool:
    try:
        with Image.open(p) as im:
            im.verify()
        # reopen to load (verify doesn't load all data)
        with Image.open(p) as im:
            im.transpose(Image.FLIP_LEFT_RIGHT)
        return True
    except Exception:
        return False


def main():
    BAD_DIR.mkdir(exist_ok=True)
    count_bad = 0
    for cls_dir in [p for p in DATA_DIR.iterdir() if p.is_dir() and p.name != "_bad"]:
        for img_path in cls_dir.glob("**/*"):
            if img_path.is_file():
                ok = is_image_ok(img_path)
                if not ok:
                    count_bad += 1
                    dest = BAD_DIR / f"{cls_dir.name}__{img_path.name}"
                    try:
                        shutil.move(str(img_path), str(dest))
                    except Exception:
                        img_path.unlink(missing_ok=True)
    print(f"Moved/removed {count_bad} bad images to {BAD_DIR}")


if __name__ == "__main__":
    main()
