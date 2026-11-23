"""依「場景」與「亮度」拆開統計各方法在 test_roommate 的準確率。

不需要手動標場景:
  1) label  : 用圖片邊緣的背景顏色自動分群成場景,並量測每張圖的平均亮度(所以不用人工標暗光)。
              會輸出 scene_labels.csv 與每個場景的縮圖拼貼,讓你確認分群是否合理。
  2) report : 讀入各方法的預測明細 csv,按場景與亮度分組算準確率。

用法:
    python scene_analysis.py label --data data/test_roommate --k 3 --out scene_out
    # 看 scene_out/scene_montage_*.png,如果一個場景混了兩種背景就把 --k 調大再跑一次,
    # 或直接手動編輯 scene_out/scene_labels.csv 的 scene 欄位。

    python scene_analysis.py report --labels scene_out/scene_labels.csv \
        --csv yolo_exp1=results_test_exp1/predictions_test_roommate.csv \
        --csv yolo_exp2=results_test_exp2/predictions_test_roommate.csv \
        --csv kp_rf=results_keypoints/pred_rf_test_roommate.csv \
        --csv kp_mlp=results_keypoints/pred_mlp_test_roommate.csv
"""
import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def key_of(path):
    """用「類別/檔名」當 key,避免不同腳本寫入的路徑格式不同(絕對/相對)而對不上。"""
    return "/".join(Path(path).parts[-2:])


def load_small(path, size=64):
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    return np.asarray(img.resize((size, size)), dtype=np.float32)


def bg_feature(arr):
    """取圖片最外圈的 8x8 格平均顏色(手通常在中間,外圈主要是背景)。"""
    g = arr.reshape(8, 8, 8, 8, 3).mean(axis=(1, 3))  # 8x8 格,每格 RGB 平均
    mask = np.ones((8, 8), dtype=bool)
    mask[2:6, 2:6] = False
    return g[mask].flatten() / 255.0


def cmd_label(args):
    from sklearn.cluster import KMeans

    root = Path(args.data)
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in EXTS)
    if not files:
        raise SystemExit(f"{root} 底下沒有圖片")
    print(f"共 {len(files)} 張,計算背景特徵與亮度...")

    feats, bright = [], []
    for p in files:
        arr = load_small(p)
        feats.append(bg_feature(arr))
        bright.append(float(arr.mean() / 255.0))
    feats = np.array(feats)
    bright = np.array(bright)

    km = KMeans(n_clusters=args.k, n_init=10, random_state=0).fit(feats)
    scenes = km.labels_

    # 亮度分三組(相對的:這份測試集裡偏暗/中等/偏亮)
    q1, q2 = np.quantile(bright, [1 / 3, 2 / 3])
    bins = np.where(bright <= q1, "dim", np.where(bright <= q2, "mid", "bright"))
    print(f"亮度範圍 {bright.min():.2f} ~ {bright.max():.2f}(0=全黑,1=全白),"
          f"分界 {q1:.2f} / {q2:.2f}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "scene_labels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "path", "true", "scene", "brightness", "brightness_bin"])
        for p, s, b, bn in zip(files, scenes, bright, bins):
            w.writerow([key_of(p), str(p), p.parent.name, f"scene{s}", f"{b:.3f}", bn])

    # 每個場景的縮圖拼貼,方便人工確認
    rng = random.Random(0)
    for s in range(args.k):
        members = [p for p, sc in zip(files, scenes) if sc == s]
        pick = rng.sample(members, min(12, len(members)))
        thumb = 200
        cols = 4
        rows = (len(pick) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * thumb, rows * (thumb + 18)), "white")
        for i, p in enumerate(pick):
            img = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
            img.thumbnail((thumb, thumb))
            x, y = (i % cols) * thumb, (i // cols) * (thumb + 18)
            sheet.paste(img, (x, y))
            ImageDraw.Draw(sheet).text((x + 3, y + thumb + 3), p.parent.name, fill="black")
        sheet.save(out / f"scene_montage_scene{s}.png")
        print(f"scene{s}: {len(members)} 張 -> {out}/scene_montage_scene{s}.png")

    print(f"\n標籤已存到 {out}/scene_labels.csv")
    print("請先看拼貼確認分群合理,再跑 report。")


def cmd_report(args):
    labels = {}
    with open(args.labels) as f:
        for r in csv.DictReader(f):
            labels[r["key"]] = r

    methods = {}
    for item in args.csv:
        name, path = item.split("=", 1)
        with open(path) as f:
            methods[name] = {key_of(r["path"]): r for r in csv.DictReader(f)}

    def report(group_field, title):
        groups = defaultdict(list)
        for k, r in labels.items():
            groups[r[group_field]].append(k)
        print(f"\n=== 依 {title} ===")
        header = f"{'組別':12s}{'張數':>6s}" + "".join(f"{m:>11s}" for m in methods)
        print(header)
        rows_out = []
        for g in sorted(groups):
            keys = groups[g]
            accs = []
            for m, data in methods.items():
                ok = [int(data[k]["correct"]) for k in keys if k in data]
                accs.append(np.mean(ok) if ok else float("nan"))
            print(f"{g:12s}{len(keys):6d}" + "".join(f"{a:11.3f}" for a in accs))
            rows_out.append([g, len(keys)] + [f"{a:.4f}" for a in accs])
        return rows_out

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for field, title in [("scene", "場景"), ("brightness_bin", "亮度(dim=最暗的三分之一)")]:
        rows = report(field, title)
        with open(out / f"accuracy_by_{field}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["group", "n"] + list(methods))
            w.writerows(rows)

    # 場景 x 真實類別(找出哪個場景的哪個手勢最糟)
    print("\n=== 場景 x 類別 的準確率(只列出張數 >= 10 的格子) ===")
    cells = defaultdict(list)
    for k, r in labels.items():
        cells[(r["scene"], r["true"])].append(k)
    print(f"{'場景':10s}{'類別':10s}{'張數':>6s}" + "".join(f"{m:>11s}" for m in methods))
    for (s, c), keys in sorted(cells.items()):
        if len(keys) < 10:
            continue
        accs = []
        for m, data in methods.items():
            ok = [int(data[k]["correct"]) for k in keys if k in data]
            accs.append(np.mean(ok) if ok else float("nan"))
        print(f"{s:10s}{c:10s}{len(keys):6d}" + "".join(f"{a:11.3f}" for a in accs))

    print(f"\n已存到 {out}/")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("label")
    a.add_argument("--data", required=True, help="data/test_roommate")
    a.add_argument("--k", type=int, default=3, help="場景數(先猜,看拼貼再調)")
    a.add_argument("--out", default="scene_out")
    a.set_defaults(func=cmd_label)

    b = sub.add_parser("report")
    b.add_argument("--labels", required=True)
    b.add_argument("--csv", action="append", required=True, help="名稱=預測明細csv,可重複")
    b.add_argument("--out", default="scene_out")
    b.set_defaults(func=cmd_report)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()