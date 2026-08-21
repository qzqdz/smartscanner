#!/usr/bin/env bash
# 直接加载 CCRNet 最优权重(SC_model_big_long_resnet_1d_24000/pytorch_model.bin)做评测
# 不开 do_train, 让 main.py 在 line 702 load_state_dict 后跑 eval/acc_eval/knn_eval
set -euo pipefail
cd "$(dirname "$0")"

MODEL_DIR=./SC_model_big_long_resnet_1d_24000       # 既是 model_path 又是权重来源
WEIGHT=$MODEL_DIR/pytorch_model.bin                 # CCRNet 训练好的 student 权重
DATA_DIR=./dataset/sc_big_local

echo "=== CCRNet 上游最优权重评测 ==="
echo "权重: $WEIGHT"
echo "  md5:  $(md5sum "$WEIGHT" | awk '{print $1}')"
echo "  size: $(stat -c%s "$WEIGHT") bytes"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits)"

exec /home/qzqdz/.conda/envs/trl_bw/bin/python main.py \
    --epochs 1 \
    --batch_size 4 \
    --lr 1e-3 \
    --maxlen 8192 \
    --pooling last-avg \
    --model_path "$MODEL_DIR" \
    --snli_train "$DATA_DIR/train_triplets.json" \
    --sts_dev   "$DATA_DIR/dev_pairs.json" \
    --sts_test  "$DATA_DIR/test_pairs.json" \
    --acc_train "$DATA_DIR/acc_train.json" \
    --acc_val   "$DATA_DIR/acc_val.json" \
    --acc_k 5 \
    --seed 3402 \
    --save_path "$WEIGHT" \
    "$@"
