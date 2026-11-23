"""把判錯 / 沒偵測到手的圖拼成一張縮圖拼貼,用來看失敗原因,也可以直接放進 README。

用法:
    # ok 被判錯的圖(附上 MediaPipe 偵測到的關鍵點,方便看關鍵點準不準)
    python failure_montage.py --csv results_keypoints/pred_mlp_val.csv \
        --kind wrong --true ok --kp keypoints/val.npz --out montage_ok_wrong.png

    # 沒偵測到手的圖
    python failure_montage.py --csv results_keypoints/pred_mlp_val.csv \
        --kind nohand --out montage_nohand.png

    # YOLO 的預測明細也能用(沒有 detected 欄位,只能用 --kind wrong)
    python failure_montage.py --csv results_val/predictions_val.csv --kind wrong --true ok
"""
import argparse
import csv
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

MAX_SIDE = 1280  # 要和 extract_landmarks.py 一致,關鍵點座標才對得上
CHAINS = [[0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12],
          [0, 13, 14, 15, 16], [0, 17, 18, 19, 20]]


def load_img(path):
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    if max(img.size) > MAX_SIDE:
        s = MAX_SIDE / max(img.size)
        img = img.resize((int(img.width * s), int(img.height * s)))
    return img


def draw_landmarks(img, lm):
    d = ImageDraw.Draw(img)
    r = max(3, max(img.size) // 150)
    w = max(2, max(img.size) // 300)
    for chain in CHAINS:
        d.line([(lm[i][0], lm[i][1]) for i in chain], fill=(0, 255, 0), width=w)
    for x, y, _ in lm:
        d.ellipse([x - r, y - r, x + r, y + r], fill=(255, 0, 0))


def blur_background(img, lm):
    import cv2

    pts = np.array([[x, y] for x, y, _ in lm], dtype=np.int32)
    hull = cv2.convexHull(pts).reshape(-1, 2)
    cx, cy = hull[:, 0].mean(), hull[:, 1].mean()
    hull = ((hull - [cx, cy]) * 1.08 + [cx, cy]).astype(np.int32)
    size = max(pts[:, 0].max() - pts[:, 0].min(), pts[:, 1].max() - pts[:, 1].min())
    mask = Image.new("L", img.size, 0)
    d = ImageDraw.Draw(mask)
    d.polygon([tuple(p) for p in hull], fill=255)
    for x, y, _ in lm:
        r = int(size * 0.09)
        d.ellipse([x - r, y - r, x + r, y + r], fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(5))
    blurred = img.filter(ImageFilter.GaussianBlur(35))
    return Image.composite(img, blurred, mask)


def crop_around_hand(img, lm, margin):
    xs = [p[0] for p in lm]
    ys = [p[1] for p in lm]
    side = max(max(xs) - min(xs), max(ys) - min(ys)) * (1 + 2 * margin)
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    left = max(0, int(cx - side / 2))
    top = max(0, int(cy - side / 2))
    right = min(img.width, int(cx + side / 2))
    bottom = min(img.height, int(cy + side / 2))
    return img.crop((left, top, right, bottom))


def make_tile(path, caption, thumb, lm=None, crop_hand=False, crop_margin=0.4,
              draw_lm=True, blur_bg=False):
    img = load_img(path)
    if lm is not None:
        if blur_bg:
            img = blur_background(img, lm)
        if draw_lm:
            draw_landmarks(img, lm)
        if crop_hand:
            img = crop_around_hand(img, lm, crop_margin)
    img.thumbnail((thumb, thumb))
    tile = Image.new("RGB", (thumb, thumb + 22), "white")
    tile.paste(img, ((thumb - img.width) // 2, 0))
    ImageDraw.Draw(tile).text((4, thumb + 5), caption, fill="black")
    return tile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--kind", choices=["wrong", "nohand"], default="wrong")
    ap.add_argument("--true", default=None, help="只看某個真實類別,例如 ok")
    ap.add_argument("--pred", default=None, help="只看被判成某類別的,例如 paper")
    ap.add_argument("--kp", default=None, help="keypoints/val.npz,提供就會畫出關鍵點")
    ap.add_argument("--crop-hand", action="store_true", help="依關鍵點裁切手部,避免露出臉")
    ap.add_argument("--crop-margin", type=float, default=0.4)
    ap.add_argument("--no-landmarks", action="store_true", help="只用關鍵點裁切,不畫出骨架")
    ap.add_argument("--blur-background", action="store_true", help="模糊手部以外的區域,避免露出臉")
    ap.add_argument("--path-replace", nargs=2, metavar=("OLD", "NEW"), default=None,
                    help="顯示時替換圖片路徑,例如把 data/ 換成 data_crop/")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--cols", type=int, default=5)
    ap.add_argument("--thumb", type=int, default=256)
    ap.add_argument("--out", default="montage.png")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with open(args.csv) as f:
        rows = list(csv.DictReader(f))

    if args.kind == "nohand":
        if "detected" not in rows[0]:
            raise SystemExit("這個 csv 沒有 detected 欄位,請用方法 B 的 pred_*.csv")
        rows = [r for r in rows if r["detected"] == "0"]
    else:
        rows = [r for r in rows if r["correct"] == "0"]
        if "detected" in rows[0] if rows else False:
            rows = [r for r in rows if r["detected"] == "1"]  # 沒偵測到的另外看
    if args.true:
        rows = [r for r in rows if r["true"] == args.true]
    if args.pred:
        rows = [r for r in rows if r["pred"] == args.pred]

    total = len(rows)
    if total == 0:
        raise SystemExit("沒有符合條件的圖")
    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.n]
    print(f"符合條件共 {total} 張,隨機取 {len(rows)} 張")

    kp = {}
    if args.kp:
        d = np.load(args.kp, allow_pickle=True)
        kp = {p: (lm, ok) for p, lm, ok in zip(d["paths"], d["lms"], d["detected"])}

    tiles = []
    for r in rows:
        lm = None
        if r["path"] in kp and kp[r["path"]][1]:
            lm = kp[r["path"]][0]
        img_path = r["path"]
        if args.path_replace:
            img_path = img_path.replace(args.path_replace[0], args.path_replace[1])
        tiles.append(make_tile(img_path, f'{r["true"]} -> {r["pred"]}', args.thumb,
                               lm, args.crop_hand, args.crop_margin,
                               not args.no_landmarks, args.blur_background))

    cols = min(args.cols, len(tiles))
    nrows = (len(tiles) + cols - 1) // cols
    tw, th = tiles[0].size
    sheet = Image.new("RGB", (cols * tw, nrows * th), "white")
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * tw, (i // cols) * th))
    sheet.save(args.out)
    print(f"已存到 {args.out}")


if __name__ == "__main__":
    main()