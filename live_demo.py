"""即時 demo:筆電鏡頭 + MediaPipe 關鍵點 + 方法 B 分類器,可選擇同時顯示 YOLO 對照。

用法:
    # 只跑方法 B(預設載入 results_keypoints/mlp.joblib)
    python live_demo.py

    # 同時顯示 YOLO 的預測做對照(錄影時很有說服力)
    python live_demo.py --yolo runs/classify/runs/classify/exp1_more_bg/weights/best.pt

    # 自動錄成影片(也可以直接用 macOS 的 Cmd+Shift+5 錄螢幕)
    python live_demo.py --yolo <best.pt> --record demo.mp4

按鍵:q 離開。
第一次執行時 macOS 會要求相機權限,請在「系統設定 > 隱私權與安全性 > 相機」允許你執行的程式(終端機或 PyCharm)。
"""
import argparse
import time
from collections import Counter, deque

import cv2
import joblib
import numpy as np

from extract_landmarks import build_detector
from keypoints_common import features

CHAINS = [[0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12],
          [0, 13, 14, 15, 16], [0, 17, 18, 19, 20]]
GREEN, RED, WHITE, BLACK = (0, 255, 0), (0, 0, 255), (255, 255, 255), (0, 0, 0)


def draw_skeleton(frame, pts):
    for chain in CHAINS:
        for a, b in zip(chain[:-1], chain[1:]):
            cv2.line(frame, tuple(map(int, pts[a][:2])), tuple(map(int, pts[b][:2])), GREEN, 2)
    for x, y, _ in pts:
        cv2.circle(frame, (int(x), int(y)), 4, RED, -1)


def put_label(frame, text, line, color=WHITE):
    y = 40 + line * 42
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    cv2.rectangle(frame, (10, y - th - 8), (20 + tw, y + 10), BLACK, -1)
    cv2.putText(frame, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="results_keypoints/mlp.joblib")
    ap.add_argument("--yolo", default=None, help="YOLO best.pt 路徑,提供就會顯示對照")
    ap.add_argument("--device", default="mps", help="YOLO 推論裝置(mps 有問題就改 cpu)")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--min_conf", type=float, default=0.5)
    ap.add_argument("--threshold", type=float, default=0.6,
                    help="方法 B 的信心低於此值就顯示 ?,避免亂猜")
    ap.add_argument("--smooth", type=int, default=5, help="取最近幾幀投票,讓顯示比較穩定")
    ap.add_argument("--record", default=None, help="輸出影片檔名,例如 demo.mp4")
    args = ap.parse_args()

    blob = joblib.load(args.model)
    model, classes = blob["model"], blob["classes"]

    yolo = None
    if args.yolo:
        from ultralytics import YOLO
        yolo = YOLO(args.yolo)

    detect, close = build_detector(args.min_conf)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("打不開相機。請確認相機權限,或用 --camera 1 試試別的相機編號。")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    writer = None
    history = deque(maxlen=args.smooth)
    last_t, fps = time.time(), 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # 鏡像顯示比較自然;訓練時有左右翻轉擴增,不影響預測
            h, w = frame.shape[:2]

            rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            raw = detect(rgb)

            if raw is None:
                history.append("no hand")
                conf_text = ""
            else:
                pts = np.array([[x * w, y * h, z * w] for x, y, z in raw], dtype=np.float32)
                proba = model.predict_proba(features(pts)[None])[0]
                i = int(np.argmax(proba))
                history.append(classes[i] if proba[i] >= args.threshold else "?")
                conf_text = f" ({proba[i]:.2f})"
                draw_skeleton(frame, pts)

            shown = Counter(history).most_common(1)[0][0]
            put_label(frame, f"Keypoints+MLP: {shown}{conf_text}", 0,
                      GREEN if shown not in ("no hand", "?") else (0, 200, 255))

            if yolo is not None:
                r = yolo.predict(frame, imgsz=224, device=args.device, verbose=False)[0]
                name = r.names[int(r.probs.top1)]
                put_label(frame, f"YOLOv8n-cls: {name} ({float(r.probs.top1conf):.2f})", 1,
                          (255, 200, 0))

            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - last_t, 1e-6)
            last_t = now
            cv2.putText(frame, f"{fps:.0f} FPS   q: quit", (15, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2, cv2.LINE_AA)

            if args.record:
                if writer is None:
                    writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"),
                                             20, (w, h))
                writer.write(frame)

            cv2.imshow("Gesture demo", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
            print(f"影片已存到 {args.record}")
        cv2.destroyAllWindows()
        close()


if __name__ == "__main__":
    main()