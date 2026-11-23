"""筆電鏡頭即時手勢 demo:MediaPipe Tasks API + 方法 B 的 joblib 模型。

用法:
    python demo.py
    python demo.py --model results_keypoints/mlp.joblib
    python demo.py --source 0
    python demo.py --source data/val/ok/xxx.jpg --no-display --save demo_test.jpg
    python demo.py --yolo runs/classify/runs/classify/exp2_more_bg_aug/weights/best.pt

按鍵:
    q / Esc  離開
    s        儲存目前畫面
"""
import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import joblib
import numpy as np
from PIL import Image, ImageOps

from keypoints_common import features

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_detector(model_path: Path, min_conf: float, video_mode: bool):
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    if not model_path.exists():
        raise SystemExit(
            f"找不到 {model_path}。請先跑 extract_landmarks.py 讓它自動下載,"
            "或確認 hand_landmarker.task 在專案根目錄。"
        )

    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO if video_mode else vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=min_conf,
        min_hand_presence_confidence=min_conf,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options), mp


def detect_landmarks(detector, mp, rgb, ts_ms: int, video_mode: bool):
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = (detector.detect_for_video(image, ts_ms)
              if video_mode else detector.detect(image))
    if not result.hand_landmarks:
        return None
    h, w = rgb.shape[:2]
    return np.array([[p.x * w, p.y * h, p.z * w] for p in result.hand_landmarks[0]],
                    dtype=np.float32)


def draw_hand(frame, lm):
    pts = [(int(x), int(y)) for x, y, _ in lm]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
    for x, y in pts:
        cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)


def predict(model, classes, lm, history):
    feat = features(lm).reshape(1, -1)
    proba = model.predict_proba(feat)[0]
    if history is not None:
        history.append(proba)
        proba = np.mean(history, axis=0)
    idx = int(np.argmax(proba))
    return classes[idx], float(proba[idx]), proba


def draw_panel(frame, lines):
    x, y = 10, 10
    for text, color in lines:
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (x - 5, y - 5), (x + tw + 5, y + th + 5), (0, 0, 0), -1)
        cv2.putText(frame, text, (x, y + th), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, color, 2)
        y += th + 14


def resolve(path: str, base: Path) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else base / p


def load_image(path: Path, max_side: int = 1280):
    with Image.open(path) as im:
        img = ImageOps.exif_transpose(im).convert("RGB")
    if max(img.size) > max_side:
        scale = max_side / max(img.size)
        img = img.resize((int(img.width * scale), int(img.height * scale)))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="results_keypoints/mlp.joblib")
    ap.add_argument("--source", default="0", help="0/1=鏡頭,或影片/圖片路徑")
    ap.add_argument("--yolo", default=None, help="可選:YOLO best.pt 路徑,做對照")
    ap.add_argument("--yolo-every", type=int, default=5)
    ap.add_argument("--min-conf", type=float, default=0.5)
    ap.add_argument("--smooth", type=int, default=5, help="平滑幀數,0=關閉")
    ap.add_argument("--no-mirror", action="store_true")
    ap.add_argument("--record", default=None, help="輸出 mp4 路徑")
    ap.add_argument("--save", default=None, help="單張圖片輸出路徑")
    ap.add_argument("--no-display", action="store_true")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args()

    base = Path(__file__).resolve().parent
    model_path = resolve(args.model, base)
    if not model_path.exists():
        raise SystemExit(f"找不到模型: {model_path}")

    bundle = joblib.load(model_path)
    model = bundle["model"]
    classes = [str(c) for c in bundle["classes"]]

    source = args.source
    src_path = resolve(source, base) if not str(source).isdigit() else Path(source)
    is_image = src_path.suffix.lower() in IMAGE_EXTS and src_path.is_file()
    detector, mp = build_detector(base / "hand_landmarker.task",
                                  args.min_conf, video_mode=not is_image)

    yolo = None
    if args.yolo:
        from ultralytics import YOLO
        yolo = YOLO(str(resolve(args.yolo, base)))

    cap = None
    if is_image:
        frame = load_image(src_path)
        frames = [frame]
    else:
        cap = cv2.VideoCapture(int(source) if str(source).isdigit() else str(src_path))
        if not cap.isOpened():
            raise SystemExit(
                "無法開啟相機/影片。請確認終端機或 PyCharm 有相機權限,"
                "或把 --source 改成 0/1 或影片路徑。"
            )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        frames = None

    history = deque(maxlen=args.smooth) if args.smooth > 0 else None
    fps_hist = deque(maxlen=30)
    writer = None
    last_yolo = "?"
    frame_idx = 0
    start = time.monotonic()
    last_t = start
    label, conf = "no_hand", 0.0

    try:
        while True:
            if frames is not None:
                if frame_idx >= len(frames):
                    break
                frame = frames[frame_idx]
            else:
                ok, frame = cap.read()
                if not ok:
                    break

            if args.record and writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(
                    args.record, cv2.VideoWriter_fourcc(*"mp4v"),
                    args.fps, (w, h),
                )
                if not writer.isOpened():
                    print(f"[警告] 無法建立影片 {args.record},改用 QuickTime 錄影")
                    writer = None

            if not args.no_mirror:
                frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ts_ms = int((time.monotonic() - start) * 1000)
            lm = detect_landmarks(detector, mp, rgb, ts_ms, video_mode=not is_image)

            if lm is not None:
                draw_hand(frame, lm)
                label, conf, probs = predict(model, classes, lm, history)
                order = np.argsort(probs)[::-1][:2]
                prob_text = "  ".join(f"{classes[i]} {probs[i]:.2f}" for i in order)
                lines = [
                    (f"B: {label} {conf:.2f}", (0, 255, 0)),
                    (prob_text, (255, 255, 255)),
                ]
            else:
                if history is not None:
                    history.clear()
                label, conf = "no_hand", 0.0
                lines = [("B: no_hand", (0, 0, 255)),
                         ("move hand into frame", (255, 255, 255))]

            if yolo is not None and frame_idx % max(args.yolo_every, 1) == 0:
                result = yolo.predict(frame, imgsz=224, verbose=False)[0]
                last_yolo = str(yolo.names[int(result.probs.top1)])
            if yolo is not None:
                lines.append((f"YOLO: {last_yolo}", (0, 255, 255)))

            now = time.monotonic()
            dt = now - last_t
            last_t = now
            if dt > 0:
                fps_hist.append(1.0 / dt)
            fps = sum(fps_hist) / len(fps_hist) if fps_hist else 0.0
            lines.append((f"FPS {fps:.1f}", (255, 255, 255)))
            draw_panel(frame, lines)

            if writer is not None:
                writer.write(frame)
            if args.save and is_image:
                cv2.imwrite(args.save, frame)

            if not args.no_display:
                cv2.imshow("Gesture Demo", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord("s"):
                    shot = f"demo_shot_{int(time.time())}.jpg"
                    cv2.imwrite(shot, frame)
                    print(f"已儲存 {shot}")

            frame_idx += 1
            if args.max_frames and frame_idx >= args.max_frames:
                break
    finally:
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()
        detector.close()
        if not args.no_display:
            cv2.destroyAllWindows()

    if is_image:
        print(f"{source}: B={label} conf={conf:.3f}")


if __name__ == "__main__":
    main()
