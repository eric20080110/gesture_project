"""把 predictions CSV 整理成 markdown 表格,給 README 與最終評估使用。

用法:
    python summarize_predictions.py --csv results_keypoints/pred_mlp_val.csv
    python summarize_predictions.py --csv results_test_yolo_exp2/predictions_test_roommate.csv \
        --csv results_keypoints_final/pred_mlp_test_roommate.csv --group-condition
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

CLASS_ORDER = ["ok", "paper", "rock", "scissors"]


def load_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _ok(row):
    return row.get("correct") == "1"


def _detected(row):
    return row.get("detected") == "1"


def summarize(rows):
    n = len(rows)
    classes = [c for c in CLASS_ORDER if any(r["true"] == c for r in rows)]
    classes += sorted({r["true"] for r in rows} - set(classes))
    has_det = bool(rows) and "detected" in rows[0]

    per_class = {}
    for c in classes:
        sub = [r for r in rows if r["true"] == c]
        per_class[c] = {
            "n": len(sub),
            "recall": sum(_ok(r) for r in sub) / len(sub) if sub else 0.0,
            "det": (sum(_detected(r) for r in sub) / len(sub)
                    if has_det and sub else None),
        }

    detected = [r for r in rows if _detected(r)] if has_det else rows
    return {
        "n": n,
        "acc": sum(_ok(r) for r in rows) / n if n else 0.0,
        "has_det": has_det,
        "nohand": n - len(detected) if has_det else None,
        "acc_det": (sum(_ok(r) for r in detected) / len(detected)
                    if has_det and detected else None),
        "classes": classes,
        "per_class": per_class,
    }


def _fmt(v, digits=3):
    return "-" if v is None else f"{v:.{digits}f}"


def comparison_table(items):
    """items: [(名稱, rows), ...]"""
    summaries = [(name, summarize(rows)) for name, rows in items]
    classes = []
    for _, s in summaries:
        for c in s["classes"]:
            if c not in classes:
                classes.append(c)
    header = ["方法", "n", "整體 acc", "no_hand", "只看偵測到"] + classes
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for name, s in summaries:
        row = [
            name,
            str(s["n"]),
            _fmt(s["acc"]),
            "-" if s["nohand"] is None else str(s["nohand"]),
            _fmt(s["acc_det"]),
        ]
        row += [_fmt(s["per_class"].get(c, {}).get("recall")) for c in classes]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def condition_table(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[Path(r["path"]).parent.name].append(r)
    header = ["情境", "n", "整體 acc", "no_hand"]
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for cond in sorted(groups):
        s = summarize(groups[cond])
        lines.append("| " + " | ".join([
            cond,
            str(s["n"]),
            _fmt(s["acc"]),
            "-" if s["nohand"] is None else str(s["nohand"]),
        ]) + " |")
    return "\n".join(lines)


def details(rows, name, group_condition=False):
    s = summarize(rows)
    lines = [f"### {name}", "",
             f"- 樣本數:{s['n']}",
             f"- 整體準確率:{_fmt(s['acc'])}"]
    if s["has_det"]:
        lines.append(f"- 沒偵測到手:{s['nohand']} 張(算判錯)")
        lines.append(f"- 只看有偵測到的準確率:{_fmt(s['acc_det'])}")
    lines += ["", "| 類別 | n | 召回率 | 偵測率 |", "|---|---:|---:|---:|"]
    for c in s["classes"]:
        d = s["per_class"][c]
        lines.append(f"| {c} | {d['n']} | {_fmt(d['recall'])} | {_fmt(d['det'])} |")
    if group_condition:
        lines += ["", "#### 分情境", "", condition_table(rows)]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", required=True)
    ap.add_argument("--name", nargs="+", default=None)
    ap.add_argument("--group-condition", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    names = args.name or [Path(p).stem for p in args.csv]
    if len(names) != len(args.csv):
        raise SystemExit("--name 的數量要和 --csv 一樣")
    items = [(name, load_rows(path)) for name, path in zip(names, args.csv)]
    parts = ["## 比較表", "", comparison_table(items), ""]
    for name, rows in items:
        parts += [details(rows, name, args.group_condition), ""]
    text = "\n".join(parts)
    print(text)
    if args.out:
        Path(args.out).write_text(text)
        print(f"\n已寫入 {args.out}")


if __name__ == "__main__":
    main()
