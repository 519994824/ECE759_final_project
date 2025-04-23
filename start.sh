#!/usr/bin/env bash
set -e  # exit on any error

# 1) Run train_embedding.py
echo "[train_embedding.py] start:     $(date +"%Y-%m-%d %H:%M:%S")"
python train_embedding.py
echo "[train_embedding.py] finished:  $(date +"%Y-%m-%d %H:%M:%S")"

# 2) Serially run the other training scripts
for script in train_base.py train_c.py train_np.py; do
  echo "[$script] start:    $(date +"%Y-%m-%d %H:%M:%S")"
  python "$script"
  echo "[$script] finished: $(date +"%Y-%m-%d %H:%M:%S")"
done
