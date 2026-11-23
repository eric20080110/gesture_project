"""方法 B:用 MediaPipe 關鍵點訓練小分類器(隨機森林與 MLP),並評估。
沒偵測到手的樣本一律算判錯,混淆矩陣多一欄 no_hand。

用法:
    # 訓練並在 val 上評估
    python train_keypoints.py

    # 模型凍結後,不重新訓練,直接載入已存的模型去評估其他資料集
    python train_keypoints.py --no_train --eval_split test_dark test_cluttered
    python train_keypoints.py --no_train --test        # 等同評估 test_roommate

評估前要先用 extract_landmarks.py --splits <名稱> 產生 keypoints/<名稱>.npz。
這些額外的資料集只拿來評估,不要拿來調參數。
"""
import argparse
import csv
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from keypoints_common import features


def load(path):
    d = np.load(path, allow_pickle=True)
    return d["lms"], d["detected"], d["y"], d["paths"], [str(c) for c in d["classes"]]


def build_xy(lms, detected, y, mirror_aug=False):
    X, Y = [], []
    for lm, ok, label in zip(lms, detected, y):
        if not ok:
            continue
        X.append(features(lm))
        Y.append(label)
        if mirror_aug:  # 左右翻轉擴增,讓模型同時學到左手與右手
            X.append(features(lm, mirror=True))
            Y.append(label)
    return np.array(X), np.array(Y)


def evaluate(model, name, split, data, classes, out_dir):
    lms, detected, y, paths, _ = data
    pred = np.full(len(y), -1)  # -1 = 沒偵測到手
    idx = np.where(detected)[0]
    if len(idx):
        Xd = np.array([features(lms[i]) for i in idx])
        pred[idx] = model.predict(Xd)

    correct = pred == y
    acc_all = correct.mean()
    acc_det = correct[detected].mean() if detected.any() else float("nan")
    print(f"\n[{name} / {split}] 樣本數 {len(y)}  "
          f"偵測成功率 {detected.mean():.1%}  "
          f"整體準確率(沒偵測到算錯) {acc_all:.4f}  "
          f"只看有偵測到的 {acc_det:.4f}")

    C = len(classes)
    cm = np.zeros((C, C + 1), dtype=int)  # 最後一欄 = no_hand
    for t, p in zip(y, pred):
        cm[t, C if p < 0 else p] += 1
    cols = classes + ["no_hand"]
    print("混淆矩陣(列=真實,欄=預測):")
    print("          " + " ".join(f"{c:>9s}" for c in cols))
    for c, row in zip(classes, cm):
        print(f"{c:>9s} " + " ".join(f"{v:9d}" for v in row))

    per_class = {c: cm[i, i] / max(cm[i].sum(), 1) for i, c in enumerate(classes)}
    print("各類召回率:", {k: round(float(v), 3) for k, v in per_class.items()})

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(C + 1))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticks(range(C))
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{name} on {split} (acc={acc_all:.3f})")
    th = cm.max() / 2 if cm.max() else 0
    for i in range(C):
        for j in range(C + 1):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > th else "black")
    fig.tight_layout()
    fig.savefig(out_dir / f"cm_{name}_{split}.png", dpi=150)
    plt.close(fig)

    with open(out_dir / f"pred_{name}_{split}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "true", "pred", "detected", "correct"])
        for pth, t, p, d, ok in zip(paths, y, pred, detected, correct):
            w.writerow([pth, classes[t], "no_hand" if p < 0 else classes[p], int(d), int(ok)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kp", default="keypoints")
    ap.add_argument("--out", default="results_keypoints")
    ap.add_argument("--test", action="store_true", help="在 test_roommate 上評估")
    ap.add_argument("--eval_split", nargs="*", default=[],
                    help="額外要評估的資料集名稱,例如 test_dark test_cluttered")
    ap.add_argument("--no_train", action="store_true",
                    help="不重新訓練,載入 --out 裡已存的 rf.joblib / mlp.joblib")
    args = ap.parse_args()

    kp, out = Path(args.kp), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    val = load(kp / "val.npz")
    holdouts = list(args.eval_split)
    if args.test and "test_roommate" not in holdouts:
        holdouts.insert(0, "test_roommate")

    holdout_data = {}
    for s in holdouts:
        p = kp / f"{s}.npz"
        if not p.exists():
            raise SystemExit(
                f"找不到 {p}。請先執行:\n"
                f"  python extract_landmarks.py --data <data路徑> --splits {s}"
            )
        holdout_data[s] = load(p)

    if args.no_train:
        models, classes = {}, None
        for name in ["rf", "mlp"]:
            pth = out / f"{name}.joblib"
            if not pth.exists():
                raise SystemExit(f"找不到 {pth},請先不加 --no_train 訓練一次")
            blob = joblib.load(pth)
            models[name], classes = blob["model"], blob["classes"]
        print("已載入先前訓練好的模型(未重新訓練),類別:", classes)
    else:
        train = load(kp / "train.npz")
        classes = train[4]
        print("類別:", classes)
        print(f"train 偵測成功率 {train[1].mean():.1%}(沒偵測到的不會拿來訓練)")
        Xtr, ytr = build_xy(train[0], train[1], train[2], mirror_aug=True)
        print(f"訓練樣本數(含左右翻轉擴增): {len(ytr)}")
        models = {
            "rf": RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1),
            "mlp": make_pipeline(
                StandardScaler(),
                MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000,
                              early_stopping=True, random_state=0),
            ),
        }
        for name, model in models.items():
            model.fit(Xtr, ytr)
            joblib.dump({"model": model, "classes": classes}, out / f"{name}.joblib")

    for name, model in models.items():
        evaluate(model, name, "val", val, classes, out)
        for s, data in holdout_data.items():
            evaluate(model, name, s, data, classes, out)

    print(f"\n模型、混淆矩陣圖與預測明細已存到 {out}/")


if __name__ == "__main__":
    main()