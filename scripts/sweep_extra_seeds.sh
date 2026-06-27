#!/usr/bin/env bash
# sweep_extra_seeds.sh — Run 2 extra seeds (42, 2024) for:
#   Part A  : expA-E          (5 configs × 2 seeds = 10 runs)
#   Part B  : scaling_s*      (10 configs × 2 seeds = 20 runs)
#   Bonus   : MDLM + LLaDA   (6 configs × 2 seeds = 12 runs)
# Total: 42 runs. Existing seed-1234 results are NOT re-run.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$SCRIPT_DIR/../src"

section() {
    echo ""
    echo "======================================================================"
    echo "  [$(date '+%Y-%m-%d %H:%M:%S')]  $*"
    echo "======================================================================"
}

# ── Part A ──────────────────────────────────────────────────────────────────
section "PART A — Extra Seeds"
A_COMMON="--cuda --epochs 10 --eval_batch_size 16 --results_dir ../results/part_a"
for seed in 42 2024; do
    python3 "$SRC_DIR/train.py" $A_COMMON --seed $seed \
        --tag "expA_s${seed}" --emb_dim 64  --num_layers 2 --num_heads 2 \
        --lr 1e-3 --train_batch_size 16 --dropout 0.1
    python3 "$SRC_DIR/train.py" $A_COMMON --seed $seed \
        --tag "expB_s${seed}" --emb_dim 128 --num_layers 4 --num_heads 4 \
        --lr 1e-3 --train_batch_size 16 --dropout 0.1
    python3 "$SRC_DIR/train.py" $A_COMMON --seed $seed \
        --tag "expC_s${seed}" --emb_dim 256 --num_layers 4 --num_heads 4 \
        --lr 3e-4 --train_batch_size 16 --dropout 0.1
    python3 "$SRC_DIR/train.py" $A_COMMON --seed $seed \
        --tag "expD_s${seed}" --emb_dim 128 --num_layers 4 --num_heads 4 \
        --lr 3e-4 --train_batch_size 32 --dropout 0.2
    python3 "$SRC_DIR/train.py" $A_COMMON --seed $seed \
        --tag "expE_s${seed}" --emb_dim 256 --num_layers 6 --num_heads 8 \
        --lr 1e-4 --train_batch_size 16 --dropout 0.2
done
echo "[Part A extra seeds done]"

# ── Part B baseline ──────────────────────────────────────────────────────────
section "PART B Baseline — Extra Seeds"
B_COMMON="--cuda --epochs 10 --train_batch_size 16 --eval_batch_size 16 \
          --dropout 0.1 --grad_clip 1.0 --max_sql 256 --results_dir ../results/part_b"
for seed in 42 2024; do
    python3 "$SRC_DIR/train.py" $B_COMMON --seed $seed --tag "scaling_s32_s${seed}"  --emb_dim 32  --num_layers 1  --num_heads 2 --lr 1e-3
    python3 "$SRC_DIR/train.py" $B_COMMON --seed $seed --tag "scaling_s48_s${seed}"  --emb_dim 48  --num_layers 2  --num_heads 4 --lr 1e-3
    python3 "$SRC_DIR/train.py" $B_COMMON --seed $seed --tag "scaling_s64_s${seed}"  --emb_dim 64  --num_layers 2  --num_heads 2 --lr 1e-3
    python3 "$SRC_DIR/train.py" $B_COMMON --seed $seed --tag "scaling_s96_s${seed}"  --emb_dim 96  --num_layers 3  --num_heads 3 --lr 8e-4
    python3 "$SRC_DIR/train.py" $B_COMMON --seed $seed --tag "scaling_s128_s${seed}" --emb_dim 128 --num_layers 4  --num_heads 4 --lr 1e-3
    python3 "$SRC_DIR/train.py" $B_COMMON --seed $seed --tag "scaling_s160_s${seed}" --emb_dim 160 --num_layers 5  --num_heads 4 --lr 5e-4
    python3 "$SRC_DIR/train.py" $B_COMMON --seed $seed --tag "scaling_s192_s${seed}" --emb_dim 192 --num_layers 6  --num_heads 4 --lr 3e-4
    python3 "$SRC_DIR/train.py" $B_COMMON --seed $seed --tag "scaling_s224_s${seed}" --emb_dim 224 --num_layers 7  --num_heads 4 --lr 3e-4
    python3 "$SRC_DIR/train.py" $B_COMMON --seed $seed --tag "scaling_s256_s${seed}" --emb_dim 256 --num_layers 8  --num_heads 8 --lr 2e-4
    python3 "$SRC_DIR/train.py" $B_COMMON --seed $seed --tag "scaling_s320_s${seed}" --emb_dim 320 --num_layers 10 --num_heads 8 --lr 1e-4
done
echo "[Part B baseline extra seeds done]"

# ── Bonus: MDLM + LLaDA ─────────────────────────────────────────────────────
section "BONUS — MDLM + LLaDA Extra Seeds"
DLM_COMMON="--cuda --epochs 10 --train_batch_size 16 --eval_batch_size 16 \
            --dropout 0.1 --grad_clip 1.0 --max_sql 256 --results_dir ../results/bonus"
for seed in 42 2024; do
    python3 "$SRC_DIR/train_diffusion.py" $DLM_COMMON --seed $seed \
        --tag "masked_diffusion_d64_s${seed}"  --emb_dim 64  --num_layers 2 --num_heads 2 --lr 1e-3
    python3 "$SRC_DIR/train_llada.py"     $DLM_COMMON --seed $seed \
        --tag "llada_d64_s${seed}"             --emb_dim 64  --num_layers 2 --num_heads 2 --lr 1e-3
    python3 "$SRC_DIR/train_diffusion.py" $DLM_COMMON --seed $seed \
        --tag "masked_diffusion_d128_s${seed}" --emb_dim 128 --num_layers 4 --num_heads 4 --lr 1e-3
    python3 "$SRC_DIR/train_llada.py"     $DLM_COMMON --seed $seed \
        --tag "llada_d128_s${seed}"            --emb_dim 128 --num_layers 4 --num_heads 4 --lr 1e-3
    python3 "$SRC_DIR/train_diffusion.py" $DLM_COMMON --seed $seed \
        --tag "masked_diffusion_d192_s${seed}" --emb_dim 192 --num_layers 6 --num_heads 4 --lr 3e-4
    python3 "$SRC_DIR/train_llada.py"     $DLM_COMMON --seed $seed \
        --tag "llada_d192_s${seed}"            --emb_dim 192 --num_layers 6 --num_heads 4 --lr 3e-4
done
echo "[Bonus extra seeds done]"

section "sweep_extra_seeds COMPLETE — 42 new runs finished"
