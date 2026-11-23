"""test_roommate 一次性最終評估:YOLO exp2 + 方法 B RF/MLP。

預設會拒絕重跑,避免看了 test 結果又回頭調參。確定要重跑才加 --force。

用法:
    python final_eval.py
    python final_eval.py --check-dataset
    python final_eval.py --force
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

from summarize_predictions import comparison_table, details, load_rows

BASE = Path(__file__).resolve().parent
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
BASE_CONDS = ["normal", "dark"]
EXTRA_CONDS = ["cluttered", "backlight"]


def images_in(folder: Path):
    return [p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS]


def test_counts(root: Path):
    counts = {}
    classes = sorted(p.name for p in root.iterdir() if p.is_dir())
    for c in classes:
        cls_dir = root / c
        direct = [p for p in cls_dir.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        if direct:
            counts[(c, "all")] = len(direct)
        for cond_dir in sorted(p for p in cls_dir.iterdir() if p.is_dir()):
            counts[(c, cond_dir.name)] = len(images_in(cond_dir))
    return classes, counts


def preflight(root: Path, allow_incomplete: bool):
    if not root.is_dir():
        raise SystemExit(f"找不到 {root}")
    classes, counts = test_counts(root)
    total = sum(counts.values())
    if total == 0:
        raise SystemExit(f"{root} 裡沒有圖片,請先拍攝 test_roommate")

    print("\n=== test_roommate 張數 ===")
    for c in classes:
        for cond in [k[1] for k in counts if k[0] == c]:
            print(f"{c:10s} {cond:12s} {counts[(c, cond)]:4d}")

    missing = [(c, cond) for c in classes for cond in BASE_CONDS
               if counts.get((c, cond), 0) == 0]
    no_extra = [c for c in classes
                if not any(counts.get((c, cond), 0) > 0 for cond in EXTRA_CONDS)]
    if missing or no_extra:
        parts = []
        if missing:
            parts.append("缺少基本情境:" + ", ".join(f"{c}/{cond}" for c, cond in missing))
        if no_extra:
            parts.append("缺少第三種情境(cluttered 或 backlight):" + ", ".join(no_extra))
        msg = ";".join(parts)
        if allow_incomplete:
            print(f"[警告] {msg}(--allow-incomplete)")
        else:
            raise SystemExit(
                f"{msg}\n請補拍,或確定要繼續才加 --allow-incomplete。"
            )

    low = [(c, cond, n) for (c, cond), n in counts.items() if n < 5]
    if low:
        print("[警告] 以下情境少於 5 張,分情境結果會很不穩定:")
        for c, cond, n in low:
            print(f"  {c}/{cond}: {n}")

    print(f"總張數:{total}")
    return classes, counts


def run(cmd):
    print("\n$ " + " ".join(str(x) for x in cmd), flush=True)
    subprocess.run([str(x) for x in cmd], cwd=BASE, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo-weights",
                    default="runs/classify/runs/classify/exp2_more_bg_aug/weights/best.pt")
    ap.add_argument("--yolo-out", default="results_test_yolo_exp2")
    ap.add_argument("--kp-out", default="results_keypoints_final")
    ap.add_argument("--skip-yolo", action="store_true")
    ap.add_argument("--skip-kp", action="store_true")
    ap.add_argument("--allow-incomplete", action="store_true")
    ap.add_argument("--check-dataset", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    marker = BASE / ".test_evaluated"
    if marker.exists() and not args.force:
        print(f"{marker} 已存在,代表 test_roommate 已經評估過。")
        print(marker.read_text())
        print("為了避免偷看 test 調參,預設不再重跑;確定要重跑請加 --force。")
        return

    test_root = BASE / "data" / "test_roommate"
    classes, counts = preflight(test_root, args.allow_incomplete)
    has_conditions = any(
        (test_root / c / cond).is_dir()
        for c in classes for cond in BASE_CONDS + EXTRA_CONDS
    )

    if args.check_dataset:
        run([sys.executable, "check_dataset.py", "data"])

    if not args.skip_yolo:
        run([sys.executable, "train_and_eval.py", "eval",
             "--data", "data", "--weights", args.yolo_weights,
             "--split", "test_roommate", "--out", args.yolo_out])

    if not args.skip_kp:
        run([sys.executable, "extract_landmarks.py",
             "--data", "data", "--splits", "test_roommate"])
        run([sys.executable, "train_keypoints.py",
             "--test", "--out", args.kp_out])

    items = [
        ("YOLO exp2", BASE / args.yolo_out / "predictions_test_roommate.csv"),
        ("B RF", BASE / args.kp_out / "pred_rf_test_roommate.csv"),
        ("B MLP", BASE / args.kp_out / "pred_mlp_test_roommate.csv"),
    ]
    loaded = []
    for name, path in items:
        if not path.exists():
            raise SystemExit(f"找不到 {path},請確認前面的步驟有沒有成功")
        loaded.append((name, load_rows(path)))

    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# test_roommate 最終評估",
        "",
        f"- 產生時間:{stamp}",
        f"- YOLO 權重:`{args.yolo_weights}`(用 val 選出的 best.pt,偏樂觀)",
        "- 方法 B:RF / MLP,沒有用 val 調參;沒偵測到手算判錯。",
        "- test_roommate 只評估這一次,結果不得拿來回頭調參。",
        "",
        "## 比較表",
        "",
        comparison_table(loaded),
        "",
    ]
    for name, rows in loaded:
        lines += [details(rows, name, group_condition=has_conditions), ""]

    out = BASE / "results_test_summary.md"
    out.write_text("\n".join(lines))
    marker.write_text(
        f"evaluated_at={stamp}\n"
        f"yolo_weights={args.yolo_weights}\n"
        f"yolo_out={args.yolo_out}\n"
        f"kp_out={args.kp_out}\n"
    )
    print(f"\n已寫入 {out}")
    print(f"已建立 {marker}(下次要重跑需加 --force)")
    print("\n" + comparison_table(loaded))


if __name__ == "__main__":
    main()
