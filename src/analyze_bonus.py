"""Bonus analysis: learning curves, scaling comparison, generation quality, failure cases.

Our two masked absorbing-state diffusion LMs:
  - MDLM  (Sahoo et al. 2024): uniform t~U(0,1), unweighted CE, 32-step greedy decoding
  - LLaDA (Nie   et al. 2025): 1/t-weighted CE, cosine schedule, 128-step remasking

Results are aggregated across seeds (1234, 42, 2024) and reported as mean +/- std.

Produces:
  bonus_learning_curves.{svg,png,pdf}   -- train/valid loss curves
  bonus_scaling_comparison.{svg,png,pdf} -- GPT vs MDLM vs LLaDA on log-log axes
  generation_samples.txt                 -- qualitative generated text
  failure_analysis.txt                   -- repetition, mask leakage stats
  generation_metrics_agg.json           -- mean +/- std of quality metrics across seeds
"""

import argparse
import json
import math
import os
import platform
import re
import sys
import time
from collections import Counter, defaultdict

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

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
    'GPT (AR)': {'color': '#1f77b4', 'marker': 'o', 'linestyle': '-'},
    'MDLM':     {'color': '#ff7f0e', 'marker': 's', 'linestyle': '--'},
    'LLaDA':    {'color': '#d62728', 'marker': 'D', 'linestyle': ':'},
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


def base_tag(tag):
    return re.sub(r'_s\d+$', '', tag)


def group_by_base_tag(records):
    groups = defaultdict(list)
    for r in records:
        groups[base_tag(r.get('tag', ''))].append(r)
    return groups


def agg_metric(records, key_path):
    vals = []
    for r in records:
        obj = r
        for k in key_path:
            obj = obj[k] if isinstance(obj, dict) else obj[int(k)]
        vals.append(float(obj))
    if not vals:
        return float('nan'), float('nan'), 0
    return float(np.mean(vals)), float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0), len(vals)


def get_architecture_params(r):
    total = r['config']['total_params']
    dim = r['config']['emb_dim']
    vocab = 10001 if ('llada' in r.get('tag', '') or 'diffusion' in r.get('tag', '')) else 10000
    return total - vocab * dim


def plot_learning_curves():
    """Train/valid loss curves with std shading where multiple seeds exist."""
    all_bonus = load_json_dir(BONUS_DIR)

    def get_grouped(prefix):
        recs = [r for r in all_bonus if base_tag(r.get('tag', '')).startswith(prefix)]
        by_dim = defaultdict(list)
        for r in recs:
            dim = str(r['config']['emb_dim'])
            by_dim[dim].append(r)
        return dict(sorted(by_dim.items(), key=lambda x: int(x[0])))

    mdlm_by_dim = get_grouped('masked_diffusion_d')
    llada_by_dim = get_grouped('llada_d')

    if not mdlm_by_dim and not llada_by_dim:
        print("No bonus results found -- skipping learning curves.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for col, (by_dim, title) in enumerate([
        (mdlm_by_dim, 'MDLM (Sahoo et al. 2024)'),
        (llada_by_dim, 'LLaDA'),
    ]):
        for dim_str, recs in by_dim.items():
            color = SIZE_COLORS.get(dim_str, '#333333')
            label = f"d{dim_str}"
            n_epochs = max(len(r['train_epoch_loss']) for r in recs)
            epochs = range(1, n_epochs + 1)

            for row, loss_key in enumerate(['train_epoch_loss', 'valid_epoch_loss']):
                all_curves = [r[loss_key] for r in recs if loss_key in r]
                if not all_curves:
                    continue
                min_len = min(len(c) for c in all_curves)
                arr = np.array([c[:min_len] for c in all_curves])
                mu = arr.mean(axis=0)
                ep = list(range(1, min_len + 1))
                axes[row, col].plot(ep, mu, '-o', color=color,
                                    markersize=3, linewidth=1.5, label=label)
                if len(all_curves) > 1:
                    sd = arr.std(axis=0, ddof=1)
                    axes[row, col].fill_between(ep, mu - sd, mu + sd,
                                                color=color, alpha=0.15)

        for row in range(2):
            axes[row, col].set_title(title, fontsize=12)
            axes[row, col].legend(fontsize=9)
            axes[row, col].grid(True, alpha=0.3)
            axes[row, col].set_xlabel('Epoch', fontsize=11)

    axes[0, 0].set_ylabel('Train Loss', fontsize=11)
    # LLaDA training loss is 1/t-weighted (E[1/t]~6.9) -- inflated vs eval loss
    axes[0, 1].set_ylabel('Train Loss (1/t-weighted)', fontsize=11)
    axes[1, 0].set_ylabel('Valid Pseudo-Loss', fontsize=11)
    axes[1, 1].set_ylabel('Valid Pseudo-Loss (unweighted)', fontsize=11)

    fig.suptitle(
        'Diffusion LM Learning Curves (shaded band = +/-1 std across 3 seeds)\n'
        'LLaDA train loss uses 1/t weighting per paper; valid uses unweighted pseudo-loss',
        fontsize=11,
    )
    plt.tight_layout()
    for fmt in ('svg', 'png', 'pdf'):
        path = os.path.join(BONUS_DIR, f'bonus_learning_curves.{fmt}')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Learning curves saved: {path}")
    plt.close()


def plot_scaling_comparison():
    """Log-log plot of validation objective vs comparable parameter count."""
    gpt_records = [r for r in load_json_dir(PARTB_DIR)
                   if r['config'].get('arch', 'baseline') == 'baseline'
                   and 'scaling_s' in r.get('_file', '')
                   and r['config'].get('emb_dim') in (64, 128, 192)]

    all_bonus = load_json_dir(BONUS_DIR)

    fig, ax = plt.subplots(figsize=(10, 7))

    for label, prefix, gpt_recs in [
        ('GPT (AR)',  None,                gpt_records),
        ('MDLM',     'masked_diffusion_d', None),
        ('LLaDA',    'llada_d',            None),
    ]:
        style = MODEL_STYLES[label]

        if gpt_recs is not None:
            by_dim = defaultdict(list)
            for r in gpt_recs:
                by_dim[r['config']['emb_dim']].append(r)
            pts = []
            for dim in sorted(by_dim):
                group = by_dim[dim]
                params = [r['config'].get(
                    'architecture_params_excluding_input_embedding',
                    r['config'].get('non_embedding_params') or get_architecture_params(r),
                ) for r in group]
                losses = [r['valid_epoch_loss'][(r.get('best_epoch', 10) or 10) - 1]
                          for r in group]
                pts.append((float(np.mean(params)), float(np.mean(losses)),
                            float(np.std(losses, ddof=1)) if len(losses) > 1 else 0.0))
        else:
            recs = [r for r in all_bonus if base_tag(r.get('tag', '')).startswith(prefix)]
            by_dim = defaultdict(list)
            for r in recs:
                by_dim[r['config']['emb_dim']].append(r)
            pts = []
            for dim in sorted(by_dim):
                group = by_dim[dim]
                ne_vals = [r['config'].get(
                    'architecture_params_excluding_input_embedding',
                    r['config'].get('non_embedding_params') or get_architecture_params(r),
                ) for r in group]
                ne = float(np.mean(ne_vals))
                best_losses = []
                for r in group:
                    be = r.get('best_epoch', 10) or 10
                    best_losses.append(r['valid_epoch_loss'][be - 1])
                mu = float(np.mean(best_losses))
                sd = float(np.std(best_losses, ddof=1)) if len(best_losses) > 1 else 0.0
                pts.append((ne, mu, sd))

        if not pts:
            continue

        xs = np.array([p[0] for p in pts])
        ys = np.array([p[1] for p in pts])

        sds = np.array([p[2] for p in pts])
        ax.errorbar(xs, ys, yerr=sds, fmt='none',
                    ecolor=style['color'], elinewidth=1.2, capsize=4, zorder=4)
        ax.scatter(xs, ys, color=style['color'], marker=style['marker'],
                   s=80, zorder=5, label=label, edgecolors='black', linewidth=0.5)

        if len(xs) >= 2:
            log_x, log_y = np.log10(xs), np.log10(ys)
            slope, intercept = np.polyfit(log_x, log_y, 1)
            x_fit = np.logspace(log_x.min(), log_x.max(), 50)
            y_fit = 10 ** (slope * np.log10(x_fit) + intercept)
            ax.plot(x_fit, y_fit, color=style['color'], linestyle=style['linestyle'],
                    alpha=0.7, linewidth=1.8)
            mid_x = float(np.sqrt(x_fit[0] * x_fit[-1]))
            mid_y = float(np.sqrt(y_fit[0] * y_fit[-1]))
            ax.annotate(f'$\\alpha={slope:.3f}$', (mid_x, mid_y),
                        color=style['color'], fontsize=9,
                        bbox=dict(boxstyle='round,pad=0.15', fc='white', alpha=0.8))

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Trainable Parameters Excluding Shared Input Embedding', fontsize=13)
    ax.set_ylabel('Best Validation Loss / Pseudo-Loss', fontsize=13)
    ax.legend(loc='lower left', fontsize=9)
    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    for fmt in ('svg', 'png', 'pdf'):
        path = os.path.join(BONUS_DIR, f'bonus_scaling_comparison.{fmt}')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Scaling comparison saved: {path}")
    plt.close()


def display_generation_samples():
    all_bonus = load_json_dir(BONUS_DIR)
    lines = ['=' * 70, 'GENERATION SAMPLES -- Bonus Diffusion LMs', '=' * 70, '']
    found_any = False
    for model_key, model_name in [('masked_diffusion', 'MDLM'), ('llada', 'LLaDA')]:
        recs = sorted(
            [r for r in all_bonus
             if base_tag(r.get('tag', '')) == f'{model_key}_d{r["config"]["emb_dim"]}'
             and not re.search(r'_s\d+$', r.get('tag', ''))],
            key=lambda r: r['config']['emb_dim'],
        )
        for r in recs:
            dim = r['config']['emb_dim']
            samples = r.get('samples', [])
            if not samples:
                lines.append(f"[{model_name} d{dim}] -- no samples saved")
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
        for line in lines[:40]:
            print(line)


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
    all_bonus = load_json_dir(BONUS_DIR)
    lines = ['=' * 70, 'FAILURE CASE ANALYSIS -- Bonus Diffusion LMs', '=' * 70, '',
             'Metrics:', '  Repetition loop : n-gram >=3 repeated >=3 times',
             '  [MASK] leakage  : literal <mask> in output', '  Very short : <10 tokens', '']

    any_data = False
    for model_key, model_name in [('masked_diffusion', 'MDLM'), ('llada', 'LLaDA')]:
        recs = sorted(
            [r for r in all_bonus
             if base_tag(r.get('tag', '')) == f'{model_key}_d{r["config"]["emb_dim"]}'
             and not re.search(r'_s\d+$', r.get('tag', ''))],
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
            rep_ex = next((s for s in samples if _is_repetitive(s)), None)
            if rep_ex:
                lines.append(f"  Example: {rep_ex[:150]}...")
            lines.append('')

    if not any_data:
        lines.append("No samples available.")

    out_path = os.path.join(BONUS_DIR, 'failure_analysis.txt')
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Failure analysis saved: {out_path}")
    for line in lines:
        print(line)


def compute_sample_metrics(samples):
    texts = [s.strip() for s in samples if s.strip()]
    if not texts:
        return {}
    all_tokens, doc_tokens = [], []
    for text in texts:
        toks = text.lower().split()
        doc_tokens.append(toks)
        all_tokens.extend(toks)
    ttr = len(set(all_tokens)) / max(len(all_tokens), 1)
    per_doc_ttr = [len(set(t)) / max(len(t), 1) for t in doc_tokens if t]
    mean_ttr = sum(per_doc_ttr) / max(len(per_doc_ttr), 1)

    def ngrams(toks, n):
        return Counter(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))

    total_bi, n_pairs = 0.0, 0
    for i in range(len(doc_tokens)):
        for j in range(i + 1, len(doc_tokens)):
            bi, bj = ngrams(doc_tokens[i], 2), ngrams(doc_tokens[j], 2)
            inter = sum((bi & bj).values())
            union = max(sum((bi | bj).values()), 1)
            total_bi += inter / union
            n_pairs += 1
    mean_bigram = total_bi / max(n_pairs, 1)

    def has_rep(text, min_ngram=3, min_repeat=3):
        words = text.lower().split()
        for n in range(min_ngram, min(6, len(words) // 2)):
            for i in range(len(words) - n * min_repeat):
                gram = tuple(words[i:i + n])
                if sum(1 for j in range(len(words) - n + 1)
                       if tuple(words[j:j + n]) == gram) >= min_repeat:
                    return True
        return False

    rep_rate = sum(1 for t in texts if has_rep(t)) / len(texts)
    return {
        'ttr_overall': ttr,
        'ttr_mean_per_doc': mean_ttr,
        'pairwise_bigram_overlap': mean_bigram,
        'repetition_rate': rep_rate,
    }


def aggregate_generation_metrics():
    """Compute mean +/- std of generation quality across seeds, print and save."""
    all_bonus = load_json_dir(BONUS_DIR)
    agg = {}
    for model_key, model_name in [('masked_diffusion', 'MDLM'), ('llada', 'LLaDA')]:
        model_agg = {}
        for size in ['d64', 'd128', 'd192']:
            recs = [r for r in all_bonus
                    if base_tag(r.get('tag', '')) == f'{model_key}_{size}']
            if not recs:
                continue
            seed_metrics = []
            for r in recs:
                s = r.get('samples', [])
                if s:
                    seed_metrics.append(compute_sample_metrics(s))
            if not seed_metrics:
                continue
            keys = list(seed_metrics[0].keys())
            size_agg = {}
            for k in keys:
                vals = [m[k] for m in seed_metrics if k in m]
                mu = float(np.mean(vals))
                sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                size_agg[k] = {'mean': round(mu, 4), 'std': round(sd, 4), 'n': len(vals)}
            model_agg[size] = size_agg

            print(f"\n{model_name} {size} (n_seeds={len(recs)}):")
            for k, v in size_agg.items():
                print(f"  {k}: {v['mean']:.4f} +/- {v['std']:.4f}")
        agg[model_name] = model_agg

    print(f"\n{'='*60}")
    print("Comparison at M size (d128) -- mean +/- std")
    for mname in ['MDLM', 'LLaDA']:
        m = agg.get(mname, {}).get('d128', {})
        if m:
            print(f"\n{mname}:")
            for k, v in m.items():
                print(f"  {k}: {v['mean']:.4f} +/- {v['std']:.4f}  (n={v['n']})")

    out = os.path.join(BONUS_DIR, 'generation_metrics_agg.json')
    with open(out, 'w') as f:
        json.dump(agg, f, indent=2)
    print(f"\nAggregated metrics saved to {out}")
    return agg


def run_inference_comparison():
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

    gpt = CausalLMM(vocab_size=vocab_size_ar, dim=128, num_layers=4, num_heads=4,
                    max_seq_len=256, dropout=0.1).to(device)
    gpt_ckpt = os.path.join(PARTB_DIR, 'checkpoints', 'best_model_scaling_s128.pt')
    if os.path.exists(gpt_ckpt):
        gpt.load_state_dict(torch.load(gpt_ckpt, map_location=device))
    else:
        print(f"  GPT checkpoint not found at {gpt_ckpt}")
        gpt = None

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
    n_warmup, n_repeat = 3, 10
    results = {
        'seq_lengths': seq_lengths,
        'diffusion_steps': diffusion_steps,
        'environment': {
            'device': torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU',
            'precision': 'float32',
            'batch_size': batch_size,
            'gpt_kv_cache': False,
            'context_window': 256,
            'warmup_runs': n_warmup,
            'timed_repetitions': n_repeat,
            'torch_version': torch.__version__,
            'python_version': platform.python_version(),
        },
    }

    if gpt is not None:
        gpt.eval()
        gpt_times, gpt_tps = [], []
        print("\nGPT (AR) inference:")
        for L in seq_lengths:
            for _ in range(n_warmup):
                dummy = torch.randint(0, vocab_size_ar, (1, batch_size), device=device)
                for _ in range(L):
                    with torch.no_grad():
                        _ = gpt(dummy[-32:] if dummy.size(0) > 32 else dummy)
            if device == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n_repeat):
                tokens = torch.randint(0, vocab_size_ar, (1, batch_size), device=device)
                generated = tokens.clone()
                for pos in range(L):
                    ctx = generated[-min(generated.size(0), 256):]
                    with torch.no_grad():
                        logits = gpt(ctx)
                    next_tok = logits[-1].argmax(dim=-1)
                    generated = torch.cat([generated, next_tok.unsqueeze(0)], dim=0)
            if device == 'cuda':
                torch.cuda.synchronize()
            elapsed = (time.perf_counter() - t0) / n_repeat
            tps = L * batch_size / elapsed
            gpt_times.append(elapsed)
            gpt_tps.append(tps)
            print(f"  seq={L:>3d}: {elapsed:.3f}s -> {tps:.0f} tok/s")
        results['gpt_times'] = gpt_times
        results['gpt_tok_per_sec'] = gpt_tps

    llada_data = {s: {'times': [], 'ttr': []} for s in diffusion_steps}
    print("\nLLaDA (diffusion) inference:")
    for steps in diffusion_steps:
        for _ in range(n_warmup):
            _ = llada.generate(seq_len=64, batch_size=batch_size, steps=steps,
                               temperature=1.0, remask_low_conf=(steps > 8), device=device)
        if device == 'cuda':
            torch.cuda.synchronize()
        all_times, all_ttr = [], []
        for L in seq_lengths:
            if device == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n_repeat):
                tokens = llada.generate(seq_len=L, batch_size=batch_size, steps=steps,
                                        temperature=1.0, remask_low_conf=(steps > 8),
                                        device=device)
            if device == 'cuda':
                torch.cuda.synchronize()
            elapsed = (time.perf_counter() - t0) / n_repeat
            tps = L * batch_size / elapsed
            all_times.append(elapsed)
            tok = tokens.cpu().numpy().flatten()
            tok = tok[tok != vocab_size_dm - 1]
            ttr = len(set(tok)) / max(len(tok), 1)
            all_ttr.append(ttr)
            print(f"  steps={steps:>3d} seq={L:>3d}: {elapsed:.4f}s -> {tps:.0f} tok/s  TTR={ttr:.3f}")
        llada_data[steps]['times'] = all_times
        llada_data[steps]['ttr'] = all_ttr
    results['llada'] = {str(k): v for k, v in llada_data.items()}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    if gpt is not None:
        ax1.plot(seq_lengths, gpt_tps, 'o-', color='#1f77b4',
                 linewidth=2, markersize=8, label='GPT (autoregressive)')
    for steps, style in [(4, '--'), (16, '-.'), (64, ':'), (128, '-')]:
        tps_vals = [(seq_lengths[i] * batch_size) / llada_data[steps]['times'][i]
                    for i in range(len(seq_lengths))]
        ax1.plot(seq_lengths, tps_vals, style + 's', color='#d62728',
                 linewidth=1.5, markersize=6, alpha=0.7, label=f'LLaDA ({steps} steps)')
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
    for fmt in ('svg', 'png', 'pdf'):
        path = os.path.join(BONUS_DIR, f'bonus_step_ablation.{fmt}')
        plt.savefig(path, dpi=150)
        print(f"Step ablation saved: {path}")
    plt.close()

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
    aggregate_generation_metrics()
    run_inference_comparison()
