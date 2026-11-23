"""YOLOv8n-cls 訓練,以及在 test_roommate 上評估(準確率、混淆矩陣、錯誤清單)。

用法:
    # 1) 訓練(device 預設 mps,失敗可改 cpu)
    python train_and_eval.py train --data /path/to/dataset --epochs 30

    # 2) 在室友測試集上評估(只在最後跑一次,不要拿來調參數)
    python train_and_eval.py eval --data /path/to/dataset \
        --weights runs/classify/gesture_baseline/weights/best.pt
"""
import argparse
import csv
from pathlib import Path

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def cmd_train(args):
    from ultralytics import YOLO

    # 強化 augmentation:針對「背景/光線依賴」與「換人、換手」的泛化問題
    # 不加 --aug 時使用 ultralytics 預設值(用來保留可重現的 baseline)
    aug = {}
    if args.aug:
        aug = dict(
            hsv_h=0.03,   # 色相擾動,降低對牆面/物體顏色的依賴
            hsv_s=0.8,    # 飽和度
            hsv_v=0.6,    # 亮度,模擬暗光與逆光
            degrees=20,   # 隨機旋轉,模擬手的角度變化
            scale=0.7,    # 縮放,模擬距離變化
            fliplr=0.5,   # 左右翻轉,順便讓模型見過另一隻手
            erasing=0.5,  # 隨機遮擋,模擬手被部分擋住,也削弱背景線索
        )
        print("使用強化 augmentation:", aug)

    model = YOLO("yolov8n-cls.pt")
    model.train(
        data=args.data,  # 資料夾根目錄,內含 train/ val/
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        seed=args.seed,
        patience=args.patience,
        **aug,
    )
    print(f"\n訓練完成。權重: {args.project}/{args.name}/weights/best.pt")
    print("訓練曲線: results.png,val 混淆矩陣: confusion_matrix.png(同資料夾)")


def cmd_eval(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import classification_report, confusion_matrix
    from ultralytics import YOLO

    model = YOLO(args.weights)
    names = [model.names[i] for i in sorted(model.names)]
    name_to_idx = {n: i for i, n in enumerate(names)}

    test_dir = Path(args.data) / args.split
    if not test_dir.is_dir():
        raise SystemExit(f"找不到測試資料夾: {test_dir}")

    paths, y_true = [], []
    for cls_dir in sorted(p for p in test_dir.iterdir() if p.is_dir()):
        if cls_dir.name not in name_to_idx:
            print(f"[警告] 類別 {cls_dir.name} 不在模型類別 {names} 中,略過")
            continue
        for p in sorted(cls_dir.rglob("*")):
            if p.suffix.lower() in EXTS:
                paths.append(p)
                y_true.append(name_to_idx[cls_dir.name])

    if not paths:
        raise SystemExit("測試集沒有圖片")

    y_pred, confs = [], []
    for r in model.predict(
        [str(p) for p in paths], imgsz=args.imgsz, device=args.device,
        stream=True, verbose=False,
    ):
        y_pred.append(int(r.probs.top1))
        confs.append(float(r.probs.top1conf))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    acc = float((y_true == y_pred).mean())
    print(f"\n[{args.split}] 樣本數: {len(paths)}  整體準確率: {acc:.4f}\n")

    labels = list(range(len(names)))
    print(classification_report(y_true, y_pred, labels=labels,
                                target_names=names, digits=3, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("混淆矩陣(列=真實,欄=預測):")
    print("          " + " ".join(f"{n:>9s}" for n in names))
    for n, row in zip(names, cm):
        print(f"{n:>9s} " + " ".join(f"{v:9d}" for v in row))

    # 最容易混淆的組合
    pairs = [(cm[i, j], names[i], names[j])
             for i in labels for j in labels if i != j and cm[i, j] > 0]
    pairs.sort(reverse=True)
    if pairs:
        print("\n最常見的錯誤(真實 -> 預測):")
        for cnt, t, p in pairs[:5]:
            print(f"  {t} -> {p}: {cnt} 張")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # 混淆矩陣圖
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(labels)
    ax.set_yticks(labels)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"YOLOv8n-cls on {args.split} (acc={acc:.3f})")
    thresh = cm.max() / 2 if cm.max() else 0
    for i in labels:
        for j in labels:
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(out / f"confusion_matrix_{args.split}.png", dpi=150)

    # 預測明細與錯誤清單(失敗分析用)
    with open(out / f"predictions_{args.split}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "true", "pred", "confidence", "correct"])
        for p, t, pr, c in zip(paths, y_true, y_pred, confs):
            w.writerow([p, names[t], names[pr], f"{c:.4f}", int(t == pr)])

    wrong = [(p, names[t], names[pr], c)
             for p, t, pr, c in zip(paths, y_true, y_pred, confs) if t != pr]
    print(f"\n判錯 {len(wrong)} 張,明細已存到 {out}/predictions_{args.split}.csv")
    print(f"混淆矩陣圖已存到 {out}/confusion_matrix_{args.split}.png")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--data", required=True)
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--patience", type=int, default=10)
    t.add_argument("--seed", type=int, default=0,
                   help="隨機種子。先前所有訓練都是 0;重複實驗請用 1、2")
    t.add_argument("--aug", action="store_true",
                   help="開啟較強的 augmentation(色彩、旋轉、縮放、翻轉、遮擋)")
    t.add_argument("--imgsz", type=int, default=224)
    t.add_argument("--batch", type=int, default=32)
    t.add_argument("--device", default="mps")
    t.add_argument("--project", default="runs/classify")
    t.add_argument("--name", default="gesture_baseline")
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("eval")
    e.add_argument("--data", required=True)
    e.add_argument("--weights", required=True)
    e.add_argument("--split", default="test_roommate")
    e.add_argument("--imgsz", type=int, default=224)
    e.add_argument("--device", default="mps")
    e.add_argument("--out", default="results")
    e.set_defaults(func=cmd_eval)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()