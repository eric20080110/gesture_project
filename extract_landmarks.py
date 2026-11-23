"""用 MediaPipe 取出每張圖的 21 個手部關鍵點,存成 npz。
沒偵測到手的圖也會記錄(detected=False),之後評估時算作「判錯」,才能和 YOLO 公平比較。

同時支援新舊版 mediapipe:
  - 舊版(有 mp.solutions.hands)直接使用
  - 新版自動改用 Tasks API(第一次執行會下載約 7 MB 的 hand_landmarker.task 模型)

用法:
    python extract_landmarks.py --data /path/to/dataset
    python extract_landmarks.py --data /path/to/dataset --splits test_roommate
"""
import argparse
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_SIDE = 1280
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")
MODEL_PATH = Path("hand_landmarker.task")


def load_rgb(path: Path):
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)  # 手機照片的旋轉資訊,不處理的話手會是橫的
    img = img.convert("RGB")
    if max(img.size) > MAX_SIDE:
        s = MAX_SIDE / max(img.size)
        img = img.resize((int(img.width * s), int(img.height * s)))
    return np.ascontiguousarray(np.array(img))


def build_detector(min_conf: float):
    """回傳 (detect, close)。detect(rgb) -> 21 個 (x, y, z) 正規化座標的 list,沒偵測到回傳 None。"""
    import mediapipe as mp

    if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
        print("使用 mediapipe 舊版 solutions API")
        hands = mp.solutions.hands.Hands(
            static_image_mode=True, max_num_hands=1,
            min_detection_confidence=min_conf,
        )

        def detect(rgb):
            res = hands.process(rgb)
            if res.multi_hand_landmarks:
                return [(q.x, q.y, q.z) for q in res.multi_hand_landmarks[0].landmark]
            return None

        return detect, hands.close

    print("使用 mediapipe Tasks API")
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    if not MODEL_PATH.exists():
        print(f"下載手部模型到 {MODEL_PATH} ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=min_conf,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    def detect(rgb):
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = landmarker.detect(image)
        if res.hand_landmarks:
            return [(q.x, q.y, q.z) for q in res.hand_landmarks[0]]
        return None

    return detect, landmarker.close


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--out", default="keypoints")
    ap.add_argument("--min_conf", type=float, default=0.5)
    args = ap.parse_args()

    root = Path(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    classes = sorted(p.name for p in (root / "train").iterdir() if p.is_dir())
    print("類別:", classes)
    cls_idx = {c: i for i, c in enumerate(classes)}

    detect, close = build_detector(args.min_conf)
    try:
        for split in args.splits:
            lms, det, ys, paths = [], [], [], []
            for c in classes:
                folder = root / split / c
                if not folder.is_dir():
                    print(f"[警告] 找不到 {folder}")
                    continue
                files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in EXTS)
                n_det = 0
                for p in files:
                    rgb = load_rgb(p)
                    h, w = rgb.shape[:2]
                    pts = detect(rgb)
                    if pts is not None:
                        # 轉成像素尺度(x*w, y*h, z*w),避免非正方形圖片造成比例失真
                        arr = np.array([[x * w, y * h, z * w] for x, y, z in pts],
                                       dtype=np.float32)
                        lms.append(arr)
                        det.append(True)
                        n_det += 1
                    else:
                        lms.append(np.zeros((21, 3), dtype=np.float32))
                        det.append(False)
                    ys.append(cls_idx[c])
                    paths.append(str(p))
                print(f"{split:15s} {c:10s} 偵測到手: {n_det}/{len(files)}")

            np.savez(
                out / f"{split}.npz",
                lms=np.array(lms), detected=np.array(det),
                y=np.array(ys), paths=np.array(paths), classes=np.array(classes),
            )
            rate = np.mean(det) if det else 0
            print(f"=> {split}: 共 {len(det)} 張,偵測成功率 {rate:.1%}\n")
    finally:
        close()


if __name__ == "__main__":
    main()