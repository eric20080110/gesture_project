"""檢查手勢資料集:資料夾結構、類別一致、張數、圖片是否損壞、模糊、重複。

用法:
    python check_dataset.py /path/to/dataset
"""
import hashlib
import sys
from pathlib import Path

from PIL import Image

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None

SPLITS = ["train", "val", "test_roommate"]
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
BLUR_THRESHOLD = 60.0  # Laplacian 變異數低於此值視為可能模糊(可自行調整)
MIN_SIDE = 224


def list_images(folder: Path):
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in EXTS)


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def blur_score(path: Path):
    if cv2 is None:
        return None
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def main(root: Path):
    problems = 0
    class_sets = {}
    hashes = {}  # md5 -> [paths]

    for split in SPLITS:
        d = root / split
        if not d.is_dir():
            print(f"[缺少] 找不到資料夾: {d}")
            problems += 1
            continue
        class_sets[split] = sorted(p.name for p in d.iterdir() if p.is_dir())

    # 類別名稱一致性
    if class_sets:
        ref_split = next(iter(class_sets))
        ref = class_sets[ref_split]
        print(f"類別({ref_split}): {ref}")
        for split, classes in class_sets.items():
            if classes != ref:
                print(f"[錯誤] {split} 的類別 {classes} 與 {ref_split} 的 {ref} 不一致")
                problems += 1

    # 逐 split / 類別檢查
    print("\n=== 張數統計 ===")
    for split, classes in class_sets.items():
        for c in classes:
            imgs = list_images(root / split / c)
            print(f"{split:15s} {c:10s} {len(imgs):4d} 張")
            for p in imgs:
                try:
                    with Image.open(p) as im:
                        im.verify()
                    with Image.open(p) as im:
                        w, h = im.size
                    if min(w, h) < MIN_SIDE:
                        print(f"  [解析度偏低] {p} ({w}x{h})")
                except Exception as e:
                    print(f"  [損壞] {p}: {e}")
                    problems += 1
                    continue
                s = blur_score(p)
                if s is not None and s < BLUR_THRESHOLD:
                    print(f"  [可能模糊] {p} (score={s:.1f})")
                hashes.setdefault(md5(p), []).append(p)

    # 各 split 類別平衡
    print("\n=== 類別平衡 ===")
    for split, classes in class_sets.items():
        counts = [len(list_images(root / split / c)) for c in classes]
        if counts and max(counts) != min(counts):
            print(f"[注意] {split} 各類張數不一致: {dict(zip(classes, counts))}")

    # 完全重複的檔案(特別是跨 split,會造成資料洩漏)
    print("\n=== 重複檢查 ===")
    dup_found = False
    for h, paths in hashes.items():
        if len(paths) > 1:
            dup_found = True
            splits_involved = {p.relative_to(root).parts[0] for p in paths}
            tag = "跨 split 洩漏" if len(splits_involved) > 1 else "同 split 重複"
            print(f"[{tag}]")
            for p in paths:
                print(f"   {p}")
            if len(splits_involved) > 1:
                problems += 1
    if not dup_found:
        print("沒有發現完全相同的圖片。")

    print(f"\n檢查完成,嚴重問題數: {problems}")
    if cv2 is None:
        print("(未安裝 opencv,已略過模糊檢查)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    main(Path(sys.argv[1]))