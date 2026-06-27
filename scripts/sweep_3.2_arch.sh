#!/usr/bin/env bash
set -e
# ═══════════════════════════════════════════════════════════════════════════════
# Part B §3.2 — Architectural Variations
# ═══════════════════════════════════════════════════════════════════════════════
# Each modification is trained INDIVIDUALLY at 5 model sizes, then all three
# are trained together.  This lets us isolate each component's contribution.
#
# Variants:
#   arch_qk    — QK Norm only         (--qk_norm)
#   arch_gate  — Attention Gate only   (--attn_gate)
#   arch_vemb  — Value Embedding only  (--value_emb)
#   arch_all3  — All three combined    (--qk_norm --attn_gate --value_emb)
#
# Grid (same 5 sizes as §3.1 baseline):
#   dim  layers  heads  lr
#    64    2       2    1e-3
#    96    3       3    8e-4
#   128    4       4    1e-3
#   160    5       4    5e-4
#   192    6       4    3e-4
# ═══════════════════════════════════════════════════════════════════════════════

COMMON="--cuda --epochs 10 --train_batch_size 16 --eval_batch_size 16 \
        --dropout 0.1 --grad_clip 1.0 --max_sql 256 \
        --results_dir ../results/part_b"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC_DIR="$SCRIPT_DIR/../src"

run_exp() {
    local tag="$1"; shift
    echo ""
    echo "========== 3.2 $tag =========="
    python3 "$SRC_DIR/train.py" $COMMON --tag "$tag" "$@"
}

# Helper: sweep all 5 sizes with a given set of arch flags and tag prefix.
sweep_sizes() {
    local prefix="$1"; shift          # e.g.  arch_qk
    local arch_flags="$@"             # e.g.  --qk_norm

    while read -r dim layers heads lr suffix; do
        run_exp "${prefix}_${suffix}" $arch_flags \
            --emb_dim "$dim" --num_layers "$layers" --num_heads "$heads" --lr "$lr"
    done <<EOF
64  2  2  1e-3  d64
96  3  3  8e-4  d96
128 4  4  1e-3  d128
160 5  4  5e-4  d160
192 6  4  3e-4  d192
EOF
}

# ─── Individual modifications ─────────────────────────────────────────────────
echo ""; echo "##### QK Norm only #####"
sweep_sizes arch_qk   --qk_norm

echo ""; echo "##### Attention Gate only #####"
sweep_sizes arch_gate --attn_gate

echo ""; echo "##### Value Embedding only #####"
sweep_sizes arch_vemb --value_emb

# ─── Combined (all three) ─────────────────────────────────────────────────────
echo ""; echo "##### All three combined (QK Norm + Attention Gate + Value Emb) #####"
sweep_sizes arch_all3 --qk_norm --attn_gate --value_emb
