"""事後探索實驗:先用 MediaPipe 把手裁出來(去掉大部分背景),再丟給 YOLO。
目的:檢驗「YOLO 在別人的手上變差,主因是背景/環境線索」這個推測。

步驟:
  1) crop : 偵測手 -> 以手的外框(加邊界)裁成正方形 -> 存到新資料夾,並記錄每類總張數與偵測到的張數
  2) 用 train_and_eval.py 在裁切後的資料夾上重新訓練 YOLO(只用 train 與 val)
  3) eval : 在裁切後的各測試集上評估。沒偵測到手(沒辦法裁)的圖一律算判錯,
            才能和方法 B、以及先前的 YOLO 在同一個標準下比較。

用法:
    python crop_experiment.py crop --data data --out data_crop --margin 0.3 \
        --splits train val test_roommate test_dark test_cluttered
    python crop_experiment.py eval --weights <best.pt> --crop_data data_crop \
        --splits val test_roommate test_dark test_cluttered
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

from extract_landmarks import EXTS, build_detector, load_rgb

MAX_SAVE = 448


def cmd_crop(args):
    root, out = Path(args.data), Path(args.out)
    classes = sorted(p.name for p in (root / "train").iterdir() if p.is_dir())
    print("類別:", classes, " margin =", args.margin)

    detect, close = build_detector(args.min_conf)
    manifest = []
    try:
        for split in args.splits:
            for c in classes:
                folder = root / split / c
                if not folder.is_dir():
                    print(f"[警告] 找不到 {folder}")
                    continue
                files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in EXTS)
                (out / split / c).mkdir(parents=True, exist_ok=True)
                n_det = 0
                for p in files:
                    rgb = load_rgb(p)
                    h, w = rgb.shape[:2]
                    pts = detect(rgb)
                    if pts is None:
                        continue
                    xs = [x * w for x, _, _ in pts]
                    ys = [y * h for _, y, _ in pts]
                    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
                    side = max(max(xs) - min(xs), max(ys) - min(ys)) * (1 + 2 * args.margin)
                    side = max(side, 32)
                    half = side / 2
                    # 正方形裁切;超出畫面的部分 PIL 會補黑色(所有 split 一致)
                    crop = Image.fromarray(rgb).crop(
                        (int(cx - half), int(cy - half), int(cx + half), int(cy + half)))
                    if crop.width > MAX_SAVE:
                        crop = crop.resize((MAX_SAVE, MAX_SAVE))
                    crop.save(out / split / c / f"{p.stem}.jpg", quality=95)
                    n_det += 1
                manifest.append((split, c, len(files), n_det))
                print(f"{split:15s} {c:10s} 裁切成功 {n_det}/{len(files)}")
    finally:
        close()

    out.mkdir(parents=True, exist_ok=True)
    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "class", "total", "detected"])
        w.writerows(manifest)
    print(f"\n已存到 {out}/(含 manifest.csv)。接著用 train_and_eval.py 在這個資料夾上訓練。")


def cmd_eval(args):
    from ultralytics import YOLO

    model = YOLO(args.weights)
    names = [model.names[i] for i in sorted(model.names)]
    idx = {n: i for i, n in enumerate(names)}
    crop_root = Path(args.crop_data)

    totals = {}
    with open(crop_root / "manifest.csv") as f:
        for r in csv.DictReader(f):
            totals[(r["split"], r["class"])] = int(r["total"])

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    C = len(names)

    for split in args.splits:
        paths, y_true = [], []
        for c in names:
            d = crop_root / split / c
            if d.is_dir():
                for p in sorted(d.iterdir()):
                    if p.suffix.lower() in EXTS:
                        paths.append(p)
                        y_true.append(idx[c])
        if not paths:
            print(f"[跳過] {split} 沒有裁切後的圖片")
            continue
        y_true = np.array(y_true)
        y_pred = np.array([
            int(r.probs.top1) for r in model.predict(
                [str(p) for p in paths], imgsz=args.imgsz, device=args.device,
                stream=True, verbose=False)
        ])
        correct = y_pred == y_true
        n_total = sum(totals[(split, c)] for c in names)
        acc_det = correct.mean()
        acc_all = correct.sum() / n_total  # 沒偵測到手(沒辦法裁)的圖算判錯
        print(f"\n[{split}] 原始 {n_total} 張,裁切成功 {len(paths)} 張 ({len(paths) / n_total:.1%})")
        print(f"  整體準確率(沒偵測到算錯): {acc_all:.4f}   只看裁切成功的: {acc_det:.4f}")

        cm = np.zeros((C, C + 1), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1
        for c in names:
            cm[idx[c], C] = totals[(split, c)] - int((y_true == idx[c]).sum())
        cols = names + ["no_hand"]
        print("  混淆矩陣(列=真實,欄=預測):")
        print("            " + " ".join(f"{c:>9s}" for c in cols))
        for c, row in zip(names, cm):
            print(f"  {c:>9s} " + " ".join(f"{v:9d}" for v in row))

        with open(out / f"crop_pred_{split}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["path", "true", "pred", "correct"])
            for p, t, pr, ok in zip(paths, y_true, y_pred, correct):
                w.writerow([p, names[t], names[pr], int(ok)])
    print(f"\n預測明細已存到 {out}/")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("crop")
    a.add_argument("--data", required=True)
    a.add_argument("--out", default="data_crop")
    a.add_argument("--splits", nargs="+",
                   default=["train", "val", "test_roommate", "test_dark", "test_cluttered"])
    a.add_argument("--margin", type=float, default=0.3,
                   help="手的外框每邊外擴的比例(0.3 = 每邊多 30%%)。越大保留越多背景")
    a.add_argument("--min_conf", type=float, default=0.5)
    a.set_defaults(func=cmd_crop)

    b = sub.add_parser("eval")
    b.add_argument("--weights", required=True)
    b.add_argument("--crop_data", default="data_crop")
    b.add_argument("--splits", nargs="+",
                   default=["val", "test_roommate", "test_dark", "test_cluttered"])
    b.add_argument("--imgsz", type=int, default=224)
    b.add_argument("--device", default="mps")
    b.add_argument("--out", default="results_crop")
    b.set_defaults(func=cmd_eval)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()