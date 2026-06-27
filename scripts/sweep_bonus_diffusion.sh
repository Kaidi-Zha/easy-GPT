#!/usr/bin/env bash
set -e

COMMON="--cuda --epochs 10 --train_batch_size 16 --eval_batch_size 16 \
        --dropout 0.1 --grad_clip 1.0 --max_sql 256 \
        --results_dir ../results/bonus"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC_DIR="$SCRIPT_DIR/../src"
RESULT_DIR="$SCRIPT_DIR/../results/bonus"

run_diff() {
    local tag="$1"; shift
    echo ""
    echo "========== $tag =========="
    python3 "$SRC_DIR/train_diffusion.py" $COMMON --tag "$tag" "$@"
}

run_llada() {
    local tag="$1"; shift
    echo ""
    echo "========== $tag =========="
    python3 "$SRC_DIR/train_llada.py" $COMMON --tag "$tag" "$@"
}

while read -r dim layers heads lr; do

    SIZE_ARGS="--emb_dim $dim --num_layers $layers --num_heads $heads --lr $lr"
    suffix="d${dim}"

    # (a) MDLM
    run_diff "masked_diffusion_${suffix}" $SIZE_ARGS

    # (b) LLaDA
    run_llada "llada_${suffix}" $SIZE_ARGS

done <<EOF
64  2  2  1e-3
128 4  4  1e-3
192 6  4  3e-4
EOF
