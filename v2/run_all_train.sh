#!/bin/bash
cd /home/qzqdz/Desktop/project/smart-contract-security/smartscanner/code_icde/v2
PY=/home/qzqdz/.conda/envs/trl_bw/bin/python
run(){ echo "===== $(date +%H:%M) START $1 ====="; $PY train_v2.py $2 2>&1 | grep -v -i "warning\|pynvml\|capsule"; 
       echo "----- eval $3 -----"; $PY eval_knn.py bench/emb_$3.npy $3 2>&1 | grep -v -i warning; }
run "d16 (R2 core)"      "--emb_dim 16 --epochs 4 --steps_per_epoch 500 --tag v2_d16"     v2_d16
run "d3 (ablation)"      "--emb_dim 3  --epochs 4 --steps_per_epoch 500 --tag v2_d3"      v2_d3
run "d8 (ablation)"      "--emb_dim 8  --epochs 4 --steps_per_epoch 500 --tag v2_d8"      v2_d8
run "d32 (ablation)"     "--emb_dim 32 --epochs 4 --steps_per_epoch 500 --tag v2_d32"     v2_d32
run "d16 no-contrastive" "--emb_dim 16 --epochs 4 --steps_per_epoch 500 --no_contrastive --tag v2_d16_noctr" v2_d16_noctr
echo "===== ALL TRAINING DONE $(date +%H:%M) ====="
