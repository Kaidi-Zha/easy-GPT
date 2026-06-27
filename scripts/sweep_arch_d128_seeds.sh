#!/usr/bin/env bash
# Repeat the d128 architectural ablation at two additional seeds.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$SCRIPT_DIR/../src"
COMMON="--cuda --epochs 10 --train_batch_size 16 --eval_batch_size 16 \
        --dropout 0.1 --grad_clip 1.0 --max_sql 256 --lr 1e-3 \
        --emb_dim 128 --num_layers 4 --num_heads 4 \
        --results_dir ../results/part_b"

for seed in 42 2024; do
    python3 "$SRC_DIR/train.py" $COMMON --seed "$seed" \
        --tag "arch_qk_d128_s${seed}" --qk_norm
    python3 "$SRC_DIR/train.py" $COMMON --seed "$seed" \
        --tag "arch_gate_d128_s${seed}" --attn_gate
    python3 "$SRC_DIR/train.py" $COMMON --seed "$seed" \
        --tag "arch_vemb_d128_s${seed}" --value_emb
    python3 "$SRC_DIR/train.py" $COMMON --seed "$seed" \
        --tag "arch_all3_d128_s${seed}" --qk_norm --attn_gate --value_emb
done
