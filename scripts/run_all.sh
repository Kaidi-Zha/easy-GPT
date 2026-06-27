#!/usr/bin/env bash
# ==============================================================================
# run_all.sh — One-click runner for all Project 2 experiments
#
# GPU:  uses CUDA device 0 (set CUDA_VISIBLE_DEVICES=0)
# Log:  all output is tee'd to logs/run_all_<timestamp>.log
#
# Order:
#   1. Part A  — hyperparameter sweep (5 configs on PTB)
#   2. Part B §3.1 — scaling law sweep (5 model sizes)
#   3. Part B §3.2 — architectural variations (QK Norm / Attn Gate / Value Emb)
#   4. Part B analysis  — scaling-law plot + position-loss plot
#   5. Part C  — domain data download (arxiv) + Qwen2.5-0.5B LoRA fine-tuning
#   6. Bonus   — MaskedDiffusionLM + LLaDA sweep (3 sizes each)
#   7. Bonus analysis  — learning curves, scaling comparison, generation eval
# ==============================================================================

set -euo pipefail

# ─── GPU selection ────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=0

# ─── Path setup ───────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$SCRIPT_DIR/.."
SRC_DIR="$ROOT_DIR/src"
LOG_DIR="$ROOT_DIR/logs"
mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/run_all_$(date +%Y%m%d_%H%M%S).log"

# Tee all output (stdout + stderr) to log file while still printing to terminal
exec > >(tee -a "$LOG_FILE") 2>&1

# ─── Helper ───────────────────────────────────────────────────────────────────
section() {
    echo ""
    echo "======================================================================"
    echo "  [$(date '+%Y-%m-%d %H:%M:%S')]  $*"
    echo "======================================================================"
}

# ==============================================================================
# Part A: Hyperparameter Sweep
# ==============================================================================
section "PART A — Hyperparameter Sweep (5 configs, PTB)"
bash "$SCRIPT_DIR/sweep_part_a.sh"
echo "[Part A done]"

# ==============================================================================
# Part B §3.1: Scaling Laws
# ==============================================================================
section "PART B §3.1 — Scaling Laws (5 model sizes)"
bash "$SCRIPT_DIR/sweep_3.1_scaling.sh"
echo "[Part B §3.1 done]"

# ==============================================================================
# Part B §3.2: Architectural Variations
# ==============================================================================
section "PART B §3.2 — Architectural Variations"
echo "  Variants: QK Norm only | Attn Gate only | Value Emb only | All three"
bash "$SCRIPT_DIR/sweep_3.2_arch.sh"
echo "[Part B §3.2 done]"

# ==============================================================================
# Part B Analysis: Scaling-Law Plot + Position-Loss Plot
# ==============================================================================
section "PART B Analysis — Scaling Law + Position Loss Plots"
python3 "$SRC_DIR/analyze_partb.py" \
    --results_dir "$ROOT_DIR/results/part_b"
echo "[Part B analysis done]"

# ==============================================================================
# Part C: Fine-Tuning
# ==============================================================================
section "PART C — Domain Dataset Download (arxiv)"
DATA_DIR="$ROOT_DIR/data/domain"
if [ -f "$DATA_DIR/train.txt" ] && [ -f "$DATA_DIR/valid.txt" ]; then
    echo "  Domain data already present at $DATA_DIR — skipping download."
else
    python3 "$SRC_DIR/download.py" --dataset arxiv
fi

section "PART C — Fine-Tune Qwen2.5-0.5B with LoRA (r=8, alpha=16)"
python3 "$SRC_DIR/finetune.py" --cuda
echo "[Part C done]"

# ==============================================================================
# Bonus: MaskedDiffusionLM + LLaDA
# ==============================================================================
section "BONUS — MaskedDiffusionLM + LLaDA Sweep (3 sizes each)"
echo "  MaskedDiffusionLM: linear noise schedule, 32 sampling steps"
echo "  LLaDA:             cosine noise schedule, 128 steps with remasking"
bash "$SCRIPT_DIR/sweep_bonus_diffusion.sh"
echo "[Bonus sweep done]"

section "BONUS Analysis — Learning Curves + Scaling Comparison + Generation Eval"
python3 "$SRC_DIR/analyze_bonus.py" \
    --bonus_dir "$ROOT_DIR/results/bonus" \
    --partb_dir "$ROOT_DIR/results/part_b"

python3 "$SRC_DIR/eval_generation.py" \
    --bonus_dir "$ROOT_DIR/results/bonus"
echo "[Bonus analysis done]"

# ==============================================================================
# Summary
# ==============================================================================
section "ALL EXPERIMENTS COMPLETE"
echo ""
echo "  Results layout:"
echo "    results/part_a/   — 5 hyperparameter sweep runs"
echo "    results/part_b/   — 5 scaling + 20 arch variation runs + plots"
echo "    results/part_c/   — LoRA checkpoint + generation_samples.txt"
echo "    results/bonus/    — 6 diffusion runs + analysis plots"
echo ""
echo "  Full log: $LOG_FILE"
