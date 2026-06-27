"""Bonus analysis: learning curves, scaling comparison, generation quality, failure cases.

Produces:
  bonus_learning_curves.{svg,png}  — train/valid loss curves for all 6 diffusion models
  bonus_scaling_comparison.{svg,png} — GPT vs MaskedDiffusionLM vs LLaDA on log-log axes
  bonus_step_ablation.{svg,png}    — inference throughput and generation time vs seq length
  bonus_quality_vs_steps.{svg,png} — TTR diversity vs decoding steps
  generation_samples.txt           — qualitative generated text for each model/size
  failure_analysis.txt             — repetition loops, mask leakage, short-output statistics

Usage:
    python analyze_bonus.py --bonus_dir ../results/bonus --partb_dir ../results/part_b
"""

import argparse
import json
import math
import os
import sys
import time
from collections import Counter

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'

parser = argparse.ArgumentParser(description='Bonus analysis')
parser.add_argument('--bonus_dir', type=str, default='../results/bonus')
parser.add_argument('--partb_dir', type=str, default='../results/part_b')
args = parser.parse_args()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BONUS_DIR = os.path.join(SCRIPT_DIR, args.bonus_dir)
PARTB_DIR = os.path.join(SCRIPT_DIR, args.partb_dir)

MODEL_STYLES = {
    'GPT (AR)':          {'color': '#1f77b4', 'marker': 'o'},
    'MaskedDiffusionLM': {'color': '#ff7f0e', 'marker': 's'},
    'LLaDA':             {'color': '#d62728', 'marker': 'D'},
}

SIZE_COLORS = {'64': '#1f77b4', '128': '#ff7f0e', '192': '#2ca02c'}


def load_json_dir(directory):
    records = []
    if not os.path.isdir(directory):
        return records
    for fname in sorted(os.listdir(directory)):
        if fname.startswith('results_') and fname.endswith('.json'):
            with open(os.path.join(directory, fname)) as f:
                r = json.load(f)
                r['_file'] = fname
                r['_dir'] = directory
                records.append(r)
    return records


def get_non_emb_params(r):
    """Compute non-embedding params: total - vocab * dim."""
    total = r['config']['total_params']
    dim = r['config']['emb_dim']
    if 'llada' in r.get('tag', '') or 'diffusion' in r.get('tag', ''):
        vocab = 10001  # PTB ~10K + 1
    else:
        vocab = 10000
    return total - vocab * dim


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1: Learning curves — all 6 diffusion models
# ═══════════════════════════════════════════════════════════════════════════════

def plot_learning_curves():
    """Train loss and validation pseudo-loss curves for all bonus models."""
    all_bonus = load_json_dir(BONUS_DIR)
    diffusion_recs = sorted(
        [r for r in all_bonus if 'diffusion_d' in r['_file']],
        key=lambda r: r['config']['emb_dim'],
    )
    llada_recs = sorted(
        [r for r in all_bonus if 'llada_d' in r['_file']],
        key=lambda r: r['config']['emb_dim'],
    )

    if not diffusion_recs and not llada_recs:
        print("No bonus results found — skipping learning curves.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for col, (recs, title) in enumerate([
        (diffusion_recs, 'MaskedDiffusionLM'),
        (llada_recs, 'LLaDA'),
    ]):
        for r in recs:
            dim = str(r['config']['emb_dim'])
            color = SIZE_COLORS.get(dim, '#333333')
            label = f"d{dim}"
            epochs = range(1, len(r['train_epoch_loss']) + 1)
            axes[0, col].plot(epochs, r['train_epoch_loss'], '-o', color=color,
                              markersize=3, linewidth=1.5, label=label)
            axes[1, col].plot(epochs, r['valid_epoch_loss'], '-o', color=color,
                              markersize=3, linewidth=1.5, label=label)

        for row in range(2):
            axes[row, col].set_title(title, fontsize=12)
            axes[row, col].legend(fontsize=9)
            axes[row, col].grid(True, alpha=0.3)
            axes[row, col].set_xlabel('Epoch', fontsize=11)

    axes[0, 0].set_ylabel('Train Loss', fontsize=11)
    axes[0, 1].set_ylabel('Train Loss', fontsize=11)
    axes[1, 0].set_ylabel('Valid Pseudo-Loss', fontsize=11)
    axes[1, 1].set_ylabel('Valid Pseudo-Loss', fontsize=11)

    fig.suptitle('Diffusion LM Learning Curves', fontsize=14)
    plt.tight_layout()
    for fmt in ('svg', 'png'):
        path = os.path.join(BONUS_DIR, f'bonus_learning_curves.{fmt}')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Learning curves saved: {path}")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2: Scaling comparison (GPT vs Diffusion vs LLaDA)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_scaling_comparison():
    gpt_records = [r for r in load_json_dir(PARTB_DIR)
                   if r['config'].get('arch', 'baseline') == 'baseline'
                   and 'scaling_s' in r.get('_file', '')]

    diffusion_records = [r for r in load_json_dir(BONUS_DIR) if 'diffusion_d' in r.get('_file', '')]
    llada_records = [r for r in load_json_dir(BONUS_DIR) if 'llada_d' in r.get('_file', '')]

    fig, ax = plt.subplots(figsize=(10, 7))

    for label, records, default_best_epoch in [
        ('GPT (AR)', gpt_records, None),
        ('MaskedDiffusionLM', diffusion_records, 10),
        ('LLaDA', llada_records, 10),
    ]:
        points = []
        for r in records:
            non_emb = r['config'].get('non_embedding_params') or get_non_emb_params(r)
            best_epoch = r.get('best_epoch', default_best_epoch) or 10
            best_loss = r['valid_epoch_loss'][best_epoch - 1]
            points.append((non_emb, best_loss))
        points = sorted(points, key=lambda x: x[0])

        if not points:
            continue

        xs = np.array([p[0] for p in points])
        ys = np.array([p[1] for p in points])
        style = MODEL_STYLES[label]
        ax.scatter(xs, ys, color=style['color'], marker=style['marker'],
                   s=80, zorder=5, label=label, edgecolors='black', linewidth=0.5)

        if len(xs) >= 2:
            log_x, log_y = np.log10(xs), np.log10(ys)
            slope, intercept = np.polyfit(log_x, log_y, 1)
            x_fit = np.logspace(log_x.min(), log_x.max(), 50)
            y_fit = 10 ** (slope * np.log10(x_fit) + intercept)
            ax.plot(x_fit, y_fit, '--', color=style['color'], alpha=0.6, linewidth=1.5)
            mid_x = float(np.sqrt(x_fit[0] * x_fit[-1]))
            mid_y = float(np.sqrt(y_fit[0] * y_fit[-1]))
            ax.annotate(f'$\\alpha={slope:.3f}$', (mid_x, mid_y),
                        color=style['color'], fontsize=9,
                        bbox=dict(boxstyle='round,pad=0.15', fc='white', alpha=0.8))

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Non-Embedding Parameters', fontsize=13)
    ax.set_ylabel('Best Validation Loss / Pseudo-Loss', fontsize=13)
    ax.legend(loc='lower left', fontsize=9)
    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    for fmt in ('svg', 'png'):
        path = os.path.join(BONUS_DIR, f'bonus_scaling_comparison.{fmt}')
        plt.savefig(path, dpi=150)
        print(f"Scaling comparison saved: {path}")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Qualitative generation samples
# ═══════════════════════════════════════════════════════════════════════════════

def display_generation_samples():
    """Write qualitative generation examples to generation_samples.txt."""
    all_bonus = load_json_dir(BONUS_DIR)
    lines = [
        '=' * 70,
        'GENERATION SAMPLES — Bonus Diffusion LMs',
        '=' * 70,
        '',
    ]

    found_any = False
    for model_key, model_name in [('diffusion', 'MaskedDiffusionLM'), ('llada', 'LLaDA')]:
        recs = sorted(
            [r for r in all_bonus if f'{model_key}_d' in r['_file']],
            key=lambda r: r['config']['emb_dim'],
        )
        for r in recs:
            dim = r['config']['emb_dim']
            samples = r.get('samples', [])
            if not samples:
                lines.append(f"[{model_name} d{dim}] — no samples saved (re-run training)")
                lines.append('')
                continue
            found_any = True
            lines.append(f"Model: {model_name}  |  size: d{dim}")
            lines.append('-' * 50)
            for i, s in enumerate(samples[:3], 1):
                lines.append(f"  [{i}] {s}")
            lines.append('')

    out_path = os.path.join(BONUS_DIR, 'generation_samples.txt')
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Generation samples saved: {out_path}")

    if found_any:
        # Print a preview
        for line in lines[:40]:
            print(line)
    else:
        print("  No samples available yet — re-run sweep_bonus_diffusion.sh to populate.")


# ═══════════════════════════════════════════════════════════════════════════════
# Failure case analysis
# ═══════════════════════════════════════════════════════════════════════════════

def _is_repetitive(text, min_ngram=3, min_repeat=3):
    words = text.lower().split()
    for n in range(min_ngram, min(8, len(words) // 2 + 1)):
        for i in range(len(words) - n * min_repeat + 1):
            gram = tuple(words[i:i + n])
            if sum(1 for j in range(len(words) - n + 1)
                   if tuple(words[j:j + n]) == gram) >= min_repeat:
                return True
    return False


def _has_mask_leakage(text):
    lower = text.lower()
    return '<mask>' in lower or '[mask]' in lower


def analyze_failure_cases():
    """Detect repetition loops, mask token leakage, and degenerate outputs."""
    all_bonus = load_json_dir(BONUS_DIR)
    lines = [
        '=' * 70,
        'FAILURE CASE ANALYSIS — Bonus Diffusion LMs',
        '=' * 70,
        '',
        'Metrics:',
        '  Repetition loop : n-gram of length ≥3 repeated ≥3 times in one sample',
        '  [MASK] leakage  : generated text contains the literal <mask> token',
        '  Very short       : fewer than 10 tokens generated',
        '',
    ]

    any_data = False
    for model_key, model_name in [('diffusion', 'MaskedDiffusionLM'), ('llada', 'LLaDA')]:
        recs = sorted(
            [r for r in all_bonus if f'{model_key}_d' in r['_file']],
            key=lambda r: r['config']['emb_dim'],
        )
        for r in recs:
            dim = r['config']['emb_dim']
            samples = r.get('samples', [])
            if not samples:
                continue
            any_data = True
            n = len(samples)
            n_rep   = sum(1 for s in samples if _is_repetitive(s))
            n_mask  = sum(1 for s in samples if _has_mask_leakage(s))
            n_short = sum(1 for s in samples if len(s.split()) < 10)

            lines.append(f"{model_name} d{dim}  ({n} samples)")
            lines.append(f"  Repetition loops : {n_rep}/{n}  ({100*n_rep/n:.0f}%)")
            lines.append(f"  [MASK] leakage   : {n_mask}/{n}  ({100*n_mask/n:.0f}%)")
            lines.append(f"  Very short (<10)  : {n_short}/{n}  ({100*n_short/n:.0f}%)")

            rep_example = next((s for s in samples if _is_repetitive(s)), None)
            if rep_example:
                lines.append(f"  Example repetition: {rep_example[:150]}...")

            mask_example = next((s for s in samples if _has_mask_leakage(s)), None)
            if mask_example:
                lines.append(f"  Example mask leak : {mask_example[:150]}...")

            lines.append('')

    if not any_data:
        lines.append("No samples available yet — re-run sweep_bonus_diffusion.sh.")

    out_path = os.path.join(BONUS_DIR, 'failure_analysis.txt')
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Failure analysis saved: {out_path}")
    for line in lines:
        print(line)


# ═══════════════════════════════════════════════════════════════════════════════
# Inference comparison: throughput and generation time
# ═══════════════════════════════════════════════════════════════════════════════

def run_inference_comparison():
    """Compare inference dynamics: GPT (AR) vs LLaDA (diffusion)."""
    sys.path.insert(0, SCRIPT_DIR)
    import data as _data
    from llada import LLaDA
    from model import CausalLMM
    import torch

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Inference comparison device: {device}")
    dl = _data.Corpus(os.path.join(SCRIPT_DIR, '..', 'data', 'ptb'),
                      {'train': 1, 'valid': 1}, 256)
    vocab_size_ar = len(dl.vocabulary)
    vocab_size_dm = vocab_size_ar + 1

    print("Loading GPT (AR) checkpoint...")
    gpt = CausalLMM(vocab_size=vocab_size_ar, dim=128,
                    num_layers=4, num_heads=4,
                    max_seq_len=256, dropout=0.1).to(device)
    gpt_ckpt = os.path.join(PARTB_DIR, 'checkpoints', 'best_model_scaling_s128.pt')
    if not os.path.exists(gpt_ckpt):
        gpt_ckpt = os.path.join(PARTB_DIR, '..', 'checkpoints', 'best_model_scaling_s128.pt')
    if os.path.exists(gpt_ckpt):
        gpt.load_state_dict(torch.load(gpt_ckpt, map_location=device))
    else:
        print(f"  GPT checkpoint not found at {gpt_ckpt}, skipping AR comparison")
        gpt = None

    print("Loading LLaDA checkpoint...")
    llada_ckpt = os.path.join(BONUS_DIR, 'checkpoints', 'best_llada_llada_d128.pt')
    if not os.path.exists(llada_ckpt):
        print(f"  LLaDA checkpoint not found at {llada_ckpt}")
        return
    llada = LLaDA(vocab_size=vocab_size_dm, dim=128, num_layers=4, num_heads=4,
                  max_seq_len=256, dropout=0.1).to(device)
    llada.load_state_dict(torch.load(llada_ckpt, map_location=device))
    llada.eval()

    seq_lengths = [16, 32, 64, 128, 256]
    diffusion_steps = [4, 8, 16, 32, 64, 128]
    batch_size = 4
    n_warmup = 3
    n_repeat = 10
    results = {'seq_lengths': seq_lengths, 'diffusion_steps': diffusion_steps}

    if gpt is not None:
        gpt.eval()
        gpt_times = []
        gpt_tokens_per_sec = []
        print("\nGPT (AR) inference:")
        for L in seq_lengths:
            for _ in range(n_warmup):
                dummy = torch.randint(0, vocab_size_ar, (1, batch_size), device=device)
                for _ in range(L):
                    with torch.no_grad():
                        _ = gpt(dummy[-32:]) if dummy.size(0) > 32 else gpt(dummy)

            t0 = time.time()
            for _ in range(n_repeat):
                tokens = torch.randint(0, vocab_size_ar, (1, batch_size), device=device)
                generated = tokens.clone()
                for pos in range(L):
                    ctx = generated[-min(generated.size(0), 256):]
                    with torch.no_grad():
                        logits = gpt(ctx)
                    next_tok = logits[-1].argmax(dim=-1)
                    generated = torch.cat([generated, next_tok.unsqueeze(0)], dim=0)

            elapsed = (time.time() - t0) / n_repeat
            total_tokens = L * batch_size
            tps = total_tokens / elapsed
            gpt_times.append(elapsed)
            gpt_tokens_per_sec.append(tps)
            print(f"  seq_len={L:>3d}: {elapsed:.3f}s → {tps:.0f} tok/s")
        results['gpt_times'] = gpt_times
        results['gpt_tok_per_sec'] = gpt_tokens_per_sec

    llada_data = {s: {'times': [], 'ttr': []} for s in diffusion_steps}
    print("\nLLaDA (diffusion) inference:")
    for steps in diffusion_steps:
        for _ in range(n_warmup):
            _ = llada.generate(seq_len=64, batch_size=batch_size, steps=steps,
                               temperature=1.0, remask_low_conf=(steps > 8), device=device)
        if device == 'cuda':
            torch.cuda.synchronize()

        all_times = []
        all_ttr = []
        for L in seq_lengths:
            t0 = time.time()
            for _ in range(n_repeat):
                tokens = llada.generate(seq_len=L, batch_size=batch_size, steps=steps,
                                        temperature=1.0, remask_low_conf=(steps > 8),
                                        device=device)
            if device == 'cuda':
                torch.cuda.synchronize()
            elapsed = (time.time() - t0) / n_repeat
            total_tokens = L * batch_size
            tps = total_tokens / elapsed
            all_times.append(elapsed)
            tok = tokens.cpu().numpy().flatten()
            tok = tok[tok != vocab_size_dm - 1]
            ttr = len(set(tok)) / max(len(tok), 1)
            all_ttr.append(ttr)
            print(f"  steps={steps:>3d} seq={L:>3d}: {elapsed:.4f}s → {tps:.0f} tok/s  TTR={ttr:.3f}")
        llada_data[steps]['times'] = all_times
        llada_data[steps]['ttr'] = all_ttr
    results['llada'] = {str(k): v for k, v in llada_data.items()}

    # ── Figure: Tokens/sec vs Seq Length and Total Time vs Seq Length ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    if gpt is not None:
        ax1.plot(seq_lengths, gpt_tokens_per_sec, 'o-', color='#1f77b4',
                 linewidth=2, markersize=8, label='GPT (autoregressive)')
    for steps, style in [(4, '--'), (16, '-.'), (64, ':'), (128, '-')]:
        tps_vals = [
            (seq_lengths[i] * batch_size) / llada_data[steps]['times'][i]
            for i in range(len(seq_lengths))
        ]
        ax1.plot(seq_lengths, tps_vals, style + 's', color='#d62728',
                 linewidth=1.5, markersize=6, alpha=0.7,
                 label=f'LLaDA ({steps} steps)')

    ax1.set_xlabel('Sequence Length (tokens)', fontsize=13)
    ax1.set_ylabel('Tokens per Second', fontsize=13)
    ax1.legend(fontsize=8, loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log')

    if gpt is not None:
        ax2.plot(seq_lengths, gpt_times, 'o-', color='#1f77b4',
                 linewidth=2, markersize=8, label='GPT (AR, 1 step/token)')
    ax2.plot(seq_lengths, llada_data[8]['times'],   's--', color='#ff7f0e',
             linewidth=1.5, markersize=6, label='LLaDA (8 steps)')
    ax2.plot(seq_lengths, llada_data[32]['times'],  'D-',  color='#d62728',
             linewidth=2,   markersize=8, label='LLaDA (32 steps)')
    ax2.plot(seq_lengths, llada_data[128]['times'], 'v:',  color='#9467bd',
             linewidth=1.5, markersize=6, label='LLaDA (128 steps)')
    ax2.set_xlabel('Sequence Length (tokens)', fontsize=13)
    ax2.set_ylabel('Generation Time (s)', fontsize=13)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    for fmt in ('svg', 'png'):
        path = os.path.join(BONUS_DIR, f'bonus_step_ablation.{fmt}')
        plt.savefig(path, dpi=150)
        print(f"Step ablation saved: {path}")
    plt.close()

    # ── Figure: Quality (TTR) vs decoding steps ──
    fig, ax = plt.subplots(figsize=(8, 5))
    for L_idx, L in enumerate(seq_lengths):
        ttr_vals = [llada_data[s]['ttr'][L_idx] for s in diffusion_steps]
        ax.plot(diffusion_steps, ttr_vals, 'o-', markersize=5, linewidth=1.5,
                label=f'seq_len={L}', alpha=0.8)
    ax.set_xlabel('Diffusion Steps', fontsize=13)
    ax.set_ylabel('Type-Token Ratio (diversity)', fontsize=13)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    for fmt in ('svg', 'png'):
        path = os.path.join(BONUS_DIR, f'bonus_quality_vs_steps.{fmt}')
        plt.savefig(path, dpi=150)
        print(f"Quality vs steps saved: {path}")
    plt.close()

    with open(os.path.join(BONUS_DIR, 'step_ablation.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print("Step ablation data saved.")


if __name__ == '__main__':
    plot_learning_curves()
    plot_scaling_comparison()
    display_generation_samples()
    analyze_failure_cases()
    run_inference_comparison()
