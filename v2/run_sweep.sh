#!/bin/bash
# Usage: run_sweep.sh <lr> <warmup> <suffix>   e.g. run_sweep.sh 2e-4 200 _w
# Full capacity curve d3/d8/d16/d32 + no-contrastive control, all at same LR.
LR=${1:-1e-3}; WU=${2:-0}; SUF=${3:-}
PY=/home/qzqdz/.conda/envs/trl_bw/bin/python
LOG=artifacts/sweep${SUF}.log
echo "=== $(date +%H:%M) SWEEP lr=$LR warmup=$WU suffix='$SUF' ===" > $LOG
for D in 3 8 16 32; do
  T=v2_d${D}${SUF}
  echo "----- $(date +%H:%M) train $T -----" >> $LOG
  $PY train_v2.py --emb_dim $D --lr $LR --warmup $WU --epochs 4 --steps_per_epoch 500 --tag $T >> $LOG 2>&1
  echo "----- eval $T -----" >> $LOG
  $PY eval_knn.py bench/emb_${T}.npy $T >> $LOG 2>&1
done
# no-contrastive control at d16
T=v2_d16_noctr${SUF}
echo "----- $(date +%H:%M) train $T -----" >> $LOG
$PY train_v2.py --emb_dim 16 --lr $LR --warmup $WU --no_contrastive --epochs 4 --steps_per_epoch 500 --tag $T >> $LOG 2>&1
echo "----- eval $T -----" >> $LOG
$PY eval_knn.py bench/emb_${T}.npy $T >> $LOG 2>&1
echo "=== $(date +%H:%M) SWEEP_DONE ===" >> $LOG
