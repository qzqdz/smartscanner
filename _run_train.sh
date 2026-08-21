#!/usr/bin/env bash
# GB10 (Blackwell) CCRNet 训练脚本
set -euo pipefail
cd "$(dirname "$0")"

MODEL_PATH=./SC_model_big_long_resnet_1d_24000
DATA_DIR=./dataset/sc_big_local
SAVE_DIR=./saved_model/ccrnet_gb10
mkdir -p "$SAVE_DIR"

echo "=== CCRNet training on GB10 ==="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits)"
echo "Data dir: $DATA_DIR"
ls "$DATA_DIR"

# 不开 teacher (rankcse soft-fail 后 teacher 模型本身也未提供), 不用 weight_decay 拆分, 单卡
exec /home/qzqdz/.conda/envs/trl_bw/bin/python main.py \
    --do_train \
    --epochs 1 \
    --batch_size 4 \
    --lr 1e-3 \
    --maxlen 8192 \
    --pooling last-avg \
    --model_path "$MODEL_PATH" \
    --snli_train "$DATA_DIR/train_triplets.json" \
    --sts_dev   "$DATA_DIR/dev_pairs.json" \
    --sts_test  "$DATA_DIR/test_pairs.json" \
    --acc_train "$DATA_DIR/acc_train.json" \
    --acc_val   "$DATA_DIR/acc_val.json" \
    --acc_k 5 \
    --seed 3402 \
    --save_path "$SAVE_DIR/pytorch_model.bin" \
    "$@"
