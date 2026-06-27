# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
pip install -r requirements.txt
```

## Common Commands

```bash
# Part A: hyperparameter sweep on PTB
bash scripts/sweep_part_a.sh

# Part B: scaling laws and architectural variations
bash scripts/sweep_3.1_scaling.sh
bash scripts/sweep_3.2_arch.sh
python src/analyze_partb.py --results_dir ../results/part_b

# Part C: fine-tune Qwen2.5-0.5B with LoRA
python src/download.py --dataset arxiv
python src/finetune.py --cuda

# Bonus: diffusion language models
bash scripts/sweep_bonus_diffusion.sh
python src/analyze_bonus.py --bonus_dir ../results/bonus --partb_dir ../results/part_b
python src/eval_generation.py --bonus_dir ../results/bonus

# Run a single training experiment manually
python src/train.py --emb_dim 128 --num_layers 4 --num_heads 4 --lr 1e-3 --tag myexp --results_dir ../results/test
```

## Architecture

All source is in `src/`. The core model (`model.py`) is a GPT-style decoder shared across Parts A and B. The diffusion variants reuse the same backbone with bidirectional attention instead of causal attention.

### Tensor convention
Sequences are `(seq_len, B, dim)` — time-first, not batch-first. This applies throughout `model.py`, `diffusion_lm.py`, and `llada.py`. The attention modules transpose internally before linear projections and transpose back before returning.

### Model hierarchy
- `model.py`: `CausalLMM` → `TransformerBlock` → `CausalSelfAttention` + `SwiGLUFFN`
  - `CausalSelfAttention` supports optional architectural flags: `--qk_norm` (per-head LayerNorm on Q/K), `--attn_gate` (learnable sigmoid gate), `--value_emb` (separate token embedding added to V)
- `diffusion_lm.py`: `MaskedDiffusionLM` with `BidirectionalAttention` — same RoPE/SwiGLU backbone, no causal mask, linear noise schedule, 32 sampling steps
- `llada.py`: `LLaDA` — like `MaskedDiffusionLM` but uses 1/t-weighted loss, cosine noise schedule, and remasking during sampling (128 steps)

### Training scripts
- `train.py`: AR training for Parts A and B; writes `results_{tag}.json` and `loss_curves_{tag}.png` to `--results_dir`
- `train_diffusion.py` / `train_llada.py`: diffusion training; same output convention
- `finetune.py`: LoRA fine-tuning via HuggingFace PEFT on `Qwen/Qwen2.5-0.5B`; saves checkpoints to `results/part_c/`

### Data
- `data.py`: loads Penn Treebank from `data/ptb/`; returns word-level token tensors
- `download.py`: downloads `wikipedia-en` or `arxiv` via HuggingFace datasets into `data/domain/`

### Results layout
Experiment outputs go under `results/part_a/`, `results/part_b/`, `results/part_c/`, `results/bonus/`. Each training run writes a JSON metrics file and PNG loss curve named by the `--tag` argument.
