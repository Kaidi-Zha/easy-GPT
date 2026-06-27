#!/usr/bin/env bash
# sweep_b_extra.sh — Extra 5 model sizes for Part B §3.1 and §3.2
#
# Adds sizes d32/d48/d224/d256/d320 to fill out a 10-point scaling law curve.
# Existing results (d64–d192) are NOT re-run.
#
# New sizes (non-embedding params):
#   d32  : dim=32,  L=1,  H=2  → ~12K non-emb params
#   d48  : dim=48,  L=2,  H=4  → ~55K non-emb params
#   d224 : dim=224, L=7,  H=4  → ~4.2M non-emb params
#   d256 : dim=256, L=8,  H=8  → ~6.3M non-emb params
#   d320 : dim=320, L=10, H=8  → ~12.3M non-emb params

set -euo pipefail

COMMON="--cuda --epochs 10 --train_batch_size 16 --eval_batch_size 16 \
        --dropout 0.1 --grad_clip 1.0 --max_sql 256 \
        --results_dir ../results/part_b"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$SCRIPT_DIR/../src"

section() {
    echo ""
    echo "======================================================================"
    echo "  [$(date '+%Y-%m-%d %H:%M:%S')]  $*"
    echo "======================================================================"
}

run_exp() {
    local tag="$1"; shift
    echo ""
    echo "========== $tag =========="
    python3 "$SRC_DIR/train.py" $COMMON --tag "$tag" "$@"
}

# Extra 5 sizes (dim layers heads lr suffix)
EXTRA_SIZES=(
    "32  1   2  1e-3  d32"
    "48  2   4  1e-3  d48"
    "224 7   4  3e-4  d224"
    "256 8   8  2e-4  d256"
    "320 10  8  1e-4  d320"
)

# ── §3.1 Baseline (5 new sizes) ────────────────────────────────────────────────
section "PART B §3.1 — Baseline Scaling (5 extra sizes)"
for row in "${EXTRA_SIZES[@]}"; do
    read -r dim layers heads lr suffix <<< "$row"
    run_exp "scaling_s${dim}" \
        --emb_dim "$dim" --num_layers "$layers" --num_heads "$heads" --lr "$lr"
done
echo "[§3.1 extra done]"

# Helper: sweep 5 new sizes with given arch flags and tag prefix
sweep_extra() {
    local prefix="$1"; shift
    local arch_flags="$@"
    for row in "${EXTRA_SIZES[@]}"; do
        read -r dim layers heads lr suffix <<< "$row"
        run_exp "${prefix}_${suffix}" $arch_flags \
            --emb_dim "$dim" --num_layers "$layers" --num_heads "$heads" --lr "$lr"
    done
}

# ── §3.2 QK Norm (5 new sizes) ────────────────────────────────────────────────
section "PART B §3.2 — QK Norm (5 extra sizes)"
sweep_extra arch_qk --qk_norm
echo "[arch_qk extra done]"

# ── §3.2 Attention Gate (5 new sizes) ─────────────────────────────────────────
section "PART B §3.2 — Attention Gate (5 extra sizes)"
sweep_extra arch_gate --attn_gate
echo "[arch_gate extra done]"

# ── §3.2 Value Embedding (5 new sizes) ────────────────────────────────────────
section "PART B §3.2 — Value Embedding (5 extra sizes)"
sweep_extra arch_vemb --value_emb
echo "[arch_vemb extra done]"

# ── §3.2 All Three Combined (5 new sizes) ─────────────────────────────────────
section "PART B §3.2 — All Three Combined (5 extra sizes)"
sweep_extra arch_all3 --qk_norm --attn_gate --value_emb
echo "[arch_all3 extra done]"

section "sweep_b_extra COMPLETE — 25 new experiments finished"
echo "  New results in: results/part_b/"
echo "  Re-run analyze_partb.py to regenerate figures with all 10 sizes."
