"""方法 B 共用的特徵工程(擷取、訓練、之後的即時 demo 都會用到同一份,確保一致)。"""
import numpy as np

TIPS = [4, 8, 12, 16, 20]  # 拇指、食指、中指、無名指、小指指尖


def normalize(lm: np.ndarray) -> np.ndarray:
    """lm: (21, 3) 像素尺度座標。以手腕為原點,用「手腕到中指根部」的長度做尺度正規化,
    這樣特徵不受手在畫面中的位置與遠近影響。"""
    n = lm.astype(np.float64) - lm[0]
    scale = np.linalg.norm(n[9, :2])
    if scale < 1e-6:
        scale = 1.0
    return n / scale


def features(lm: np.ndarray, mirror: bool = False) -> np.ndarray:
    """回傳特徵向量:63 個正規化座標 + 15 個距離(指尖到手腕、指尖兩兩之間)。
    mirror=True 代表左右翻轉(用來做左右手的資料擴增)。"""
    n = normalize(lm)
    if mirror:
        n[:, 0] *= -1
    tips = n[TIPS]
    d = []
    for i in range(5):
        d.append(np.linalg.norm(tips[i] - n[0]))
        for j in range(i + 1, 5):
            d.append(np.linalg.norm(tips[i] - tips[j]))
    return np.concatenate([n.flatten(), np.array(d)])