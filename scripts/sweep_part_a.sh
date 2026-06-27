#!/usr/bin/env bash
set -e
# ═══════════════════════════════════════════════════════════════════════════════
# Part A — Hyperparameter Sweep
# ═══════════════════════════════════════════════════════════════════════════════
# Search space explored (one axis varied at a time from the expB reference):
#
#   emb_dim    : 64, 128, 256
#   num_layers : 2, 4, 6
#   num_heads  : 2, 4, 8        (must divide emb_dim evenly)
#   lr         : 1e-4, 3e-4, 1e-3
#   batch_size : 16, 32
#   dropout    : 0.1, 0.2
#
# Fixed across all runs: epochs=10, max_sql=256, grad_clip=1.0, AdamW+CosineAnnealingLR
# Metric for selection: best validation perplexity
#
# 5 configurations chosen to cover the Cartesian extremes:
#   expA — small model, high lr
#   expB — medium model, high lr  ← best
#   expC — large model, lower lr
#   expD — medium model, larger batch + more dropout
#   expE — large model, many heads, low lr, high dropout
# ═══════════════════════════════════════════════════════════════════════════════

COMMON="--cuda --epochs 10 --eval_batch_size 16 --results_dir ../results/part_a"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC_DIR="$SCRIPT_DIR/../src"
RESULT_DIR="$SCRIPT_DIR/../results/part_a"

run_exp() {
    local tag="$1"; shift
    echo ""
    echo "========== $tag =========="
    python3 "$SRC_DIR/train.py" $COMMON --tag "$tag" "$@"
}

echo "========== Experiment A: small baseline =========="
run_exp expA --emb_dim 64  --num_layers 2 --num_heads 2 --lr 1e-3 --train_batch_size 16 --dropout 0.1

echo "========== Experiment B: wider =========="
run_exp expB --emb_dim 128 --num_layers 4 --num_heads 4 --lr 1e-3 --train_batch_size 16 --dropout 0.1

echo "========== Experiment C: wider + lower lr =========="
run_exp expC --emb_dim 256 --num_layers 4 --num_heads 4 --lr 3e-4 --train_batch_size 16 --dropout 0.1

echo "========== Experiment D: bigger batch + more dropout =========="
run_exp expD --emb_dim 128 --num_layers 4 --num_heads 4 --lr 3e-4 --train_batch_size 32 --dropout 0.2

echo "========== Experiment E: large model =========="
run_exp expE --emb_dim 256 --num_layers 6 --num_heads 8 --lr 1e-4 --train_batch_size 16 --dropout 0.2

# ─── Summary ─────────────────────────────────────────────────────────────────
echo ""
echo "========== Part A sweep done =========="
echo ""
printf "%-8s %14s %14s\n" "Tag" "Best Valid PPL" "Best Epoch"
printf "%-8s %14s %14s\n" "---" "--------------" "----------"
for tag in expA expB expC expD expE; do
    f="$RESULT_DIR/results_${tag}.json"
    if [ -f "$f" ]; then
        python3 -c "
import json
d = json.load(open('$f'))
print(f\"$tag       {d['best_valid_ppl']:>14.2f}  {d['best_epoch']:>14d}\")
"
    else
        printf "%-8s %14s\n" "$tag" "(not found)"
    fi
done
