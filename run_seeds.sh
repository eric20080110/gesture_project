#!/usr/bin/env bash
# 用種子 1、2 重複三個 YOLO 設定,並在四個資料集上評估,結果彙整到 seeds_summary.txt。
# 種子 0 就是你先前已經跑過的那三次(exp1_more_bg、crop_exp1、crop_m1),不需要重跑。
#
# 三個設定(和之前完全相同,只有種子不同):
#   full   : 整張圖,預設 augmentation          (對應 exp1_more_bg)
#   crop03 : 裁切 margin 0.3,預設 augmentation (對應 crop_exp1)
#   crop10 : 裁切 margin 1.0,預設 augmentation (對應 crop_m1)
#
# 用法(在 GestureProject 資料夾,且已啟用虛擬環境):
#   bash run_seeds.sh
# 預估時間:6 次訓練,每次約 3 分鐘,加上評估,總共約 20 到 25 分鐘。
# 中途失敗會停下來;訓練的詳細輸出在 logs/ 資料夾。

set -e
ROOT=/Users/air904/PycharmProjects/GestureProject
cd "$ROOT"

SEEDS="1 2"
SPLITS="val test_roommate test_dark test_cluttered"
OUT=seeds_summary.txt
mkdir -p logs results_seeds
: > "$OUT"

for s in $SEEDS; do
  echo "=== seed $s:整張圖 ==="
  name=full_seed$s
  python train_and_eval.py train --data "$ROOT/data" --epochs 20 --patience 8 \
    --seed "$s" --name "$name" --project "$ROOT/runs/classify" > "logs/$name.log" 2>&1
  for sp in $SPLITS; do
    python train_and_eval.py eval --data "$ROOT/data" \
      --weights "runs/classify/$name/weights/best.pt" --split "$sp" \
      --out "results_seeds/${name}_$sp" 2>&1 | grep "整體準確率" | sed "s/^/full seed=$s /" | tee -a "$OUT"
  done

  for cfg in "crop03 data_crop" "crop10 data_crop_m1"; do
    set -- $cfg
    tag=$1
    folder=$2
    echo "=== seed $s:$tag($folder) ==="
    name=${tag}_seed$s
    python train_and_eval.py train --data "$ROOT/$folder" --epochs 20 --patience 8 \
      --seed "$s" --name "$name" --project "$ROOT/runs/classify" > "logs/$name.log" 2>&1
    python crop_experiment.py eval --weights "runs/classify/$name/weights/best.pt" \
      --crop_data "$folder" --splits $SPLITS --out "results_seeds/${name}" 2>&1 \
      | grep -E "^\[|整體準確率" | sed "s/^/$tag seed=$s /" | tee -a "$OUT"
  done
done

echo ""
echo "完成。彙整結果在 $OUT"
