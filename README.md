# easy-GPT

PRML Assignment 2 --- Train Your Own GPT.

Hands-on project covering the full lifecycle of language model development: implementing a
Transformer decoder from scratch, studying scaling laws and architectural variations,
fine-tuning a pre-trained model, and exploring diffusion language models.

## Quickstart

```bash
pip install -r requirements.txt

# Part A
bash scripts/sweep_part_a.sh
# Part B 3.1
bash scripts/sweep_3.1_scaling.sh
# Part B 3.2
bash scripts/sweep_3.2_arch.sh
# Part B plotting
python src/analyze_partb.py --results_dir ../results/part_b

# Part C
python src/download.py --dataset arxiv
python src/finetune.py --cuda

# Bonus part
bash scripts/sweep_bonus_diffusion.sh
python src/analyze_bonus.py --bonus_dir ../results/bonus --partb_dir ../results/part_b
```

## Structure

```
easy-GPT/
├── src/                       # Python source (12 files)
│   ├── data.py                # PTB corpus loader
│   ├── model.py               # GPT decoder: RoPE, SwiGLU, Pre-Norm
│   ├── train.py               # AR training (Part A/B)
│   ├── diffusion_lm.py        # Masked diffusion LM (basic)
│   ├── llada.py               # LLaDA-style diffusion LM
│   ├── train_diffusion.py     # Train basic diffusion LM
│   ├── train_llada.py         # Train LLaDA
│   ├── download.py            # Domain dataset download (Part C)
│   ├── finetune.py            # LoRA fine-tuning (Part C)
│   ├── analyze_partb.py       # Part B plots
│   ├── analyze_bonus.py       # Bonus plots
│   └── eval_generation.py     # Generation metrics
├── scripts/                   # Experiment shell scripts (4 files)
│   ├── sweep_part_a.sh
│   ├── sweep_3.1_scaling.sh
│   ├── sweep_3.2_arch.sh
│   └── sweep_bonus_diffusion.sh
├── data/
│   ├── ptb/                   # Penn Treebank
│   └── domain/                # Part C fine-tuning corpus
├── results/
│   ├── part_a/                # 5 hyperparameter sweep runs
│   ├── part_b/                # 5 scaling + 5 architecture runs
│   ├── part_c/                # LoRA checkpoint, generation samples
│   └── bonus/                 # 6 DLM runs, comparison plots
├── bonus.md                   # Bonus experiment plan
├── requirements.txt
└── pyproject.toml
```

## Part A: Transformer from Scratch

Implement a GPT-style decoder with Rotary Position Embedding, causal multi-head
self-attention, SwiGLU FFN, and Pre-Norm residual blocks. Train on Penn Treebank
with next-token prediction.

| Experiment | dim | layers | heads | lr | batch | dropout |
|---|---:|---:|---:|---:|---:|---:|
| expA | 64 | 2 | 2 | 1e-3 | 16 | 0.1 |
| expB | 128 | 4 | 4 | 1e-3 | 16 | 0.1 |
| expC | 256 | 4 | 4 | 3e-4 | 16 | 0.1 |
| expD | 128 | 4 | 4 | 3e-4 | 32 | 0.2 |
| expE | 256 | 6 | 8 | 1e-4 | 16 | 0.2 |

Common: 10 epochs, `AdamW`, `CosineAnnealingLR`, `grad_clip` 1.0, `max_seq` 256.

```bash
bash scripts/sweep_part_a.sh
```

## Part B: Scaling Laws & Architectural Study

### 3.1 Scaling Laws

Five baseline models at increasing sizes with fixed training budget.

| Model | dim | layers | heads | lr | Non-Emb Params |
|---|---:|---:|---:|---:|---:|
| `s64` | 64 | 2 | 2 | 1e-3 | 0.77M |
| `s96` | 96 | 3 | 3 | 8e-4 | 1.40M |
| `s128` | 128 | 4 | 4 | 1e-3 | 2.33M |
| `s160` | 160 | 5 | 4 | 5e-4 | 3.65M |
| `s192` | 192 | 6 | 4 | 3e-4 | 5.46M |

Common: 10 epochs, `batch` 16, `dropout` 0.1, `grad_clip` 1.0, `max_seq` 256.

```bash
bash scripts/sweep_3.1_scaling.sh
```

### 3.2 Architectural Variations

Enhanced architecture with QK Norm, Attention Gate, and Value Embedding applied
simultaneously, trained at the same five model sizes as 3.1.

```bash
bash scripts/sweep_3.2_arch.sh
python src/analyze_partb.py --results_dir ../results/part_b
```

## Part C: Fine-Tuning

Fine-tune Qwen2.5-0.5B with LoRA (rank 8, alpha 16) on a domain corpus.
Supports datasets `wikipedia-en`, `arxiv` or custom text.

| Parameter | Default | Description |
|---|---:|---|
| `--epochs` | 3 | Max training epochs |
| `--batch_size` | 4 | Per-device batch size |
| `--lr` | 5e-5 | Learning rate |
| `--grad_accum` | 4 | Gradient accumulation steps |
| `--max_length` | 256 | Token chunk length |
| `--lora_r` | 8 | LoRA rank |
| `--lora_alpha` | 16 | LoRA scaling factor |
| `--patience` | 3 | Early stopping patience |
| `--eval_steps` | 200 | Validate every N optimizer steps |

```bash
python src/download.py --dataset arxiv
python src/finetune.py --cuda
python src/finetune.py --cuda --lr 1e-4 --epochs 5
```

## Bonus: Diffusion Language Models

Compare two masked absorbing-state diffusion LMs against the GPT baseline.
Same Transformer backbone, replacing causal attention with bidirectional attention
and next-token prediction with masked-token prediction.

| Model | Loss | Schedule | Steps |
|---|---|---|---|
| MDLM | Cross-entropy (unweighted) | Linear | 32 |
| LLaDA | 1/t-weighted cross-entropy | Cosine + remasking | 128 |

Three sizes matched to Part B: `d64` (2L/2H), `d128` (4L/4H), `d192` (6L/4H).

```bash
bash scripts/sweep_bonus_diffusion.sh
python src/analyze_bonus.py --bonus_dir ../results/bonus --partb_dir ../results/part_b
python src/eval_generation.py --bonus_dir ../results/bonus
```