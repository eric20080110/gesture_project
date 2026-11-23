import cv2, sys, os, time

if len(sys.argv) < 3:
    print("用法: python capture.py <split> <label> [condition]")
    print("例如: python capture.py train paper")
    print("      python capture.py test_roommate paper normal")
    sys.exit(1)

split, label = sys.argv[1], sys.argv[2]   # 例如: train paper
condition = sys.argv[3] if len(sys.argv) > 3 else ""
title = f"{split}/{label}" + (f"/{condition}" if condition else "")
folder = f"data/{title}"
os.makedirs(folder, exist_ok=True)
count = len(os.listdir(folder))

cap = cv2.VideoCapture(0)
auto, last = False, 0
print(f"存到 {folder}")
print("c=開始/暫停連拍, 空白鍵=單張, q=離開")
while True:
    ok, frame = cap.read()
    if not ok:
        break
    now = time.time()
    if auto and now - last > 0.2:
        cv2.imwrite(f"{folder}/{int(now*1000)}.jpg", frame)
        count += 1
        last = now
    shown = frame.copy()
    status = "AUTO" if auto else "PAUSED"
    cv2.putText(shown, f"{title}: {count} [{status}]", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("capture", shown)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("c"):
        auto = not auto
    elif key == ord(" "):
        cv2.imwrite(f"{folder}/{int(now*1000)}.jpg", frame)
        count += 1
    elif key == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()