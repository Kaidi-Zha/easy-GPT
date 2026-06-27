"""Part B analysis: scaling laws (3.1) and architectural comparison (3.2).

Produces two figures:
  Figure 1 — Loss vs Non-Embedding Parameters (log-log, linear fit).
  Figure 2 — Loss vs Token Position (context length effect).

Usage:
    python analyze_partb.py --results_dir ../results/part_b
"""

import argparse
import json
import os
import re
from collections import defaultdict

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'

parser = argparse.ArgumentParser(description='Part B scaling law analysis')
parser.add_argument('--results_dir', type=str, default='../results/part_b',
                    help='directory containing Part B results JSON files')
parser.add_argument('--part_a_results', type=str, default='../results',
                    help='directory containing Part A baseline results (optional)')
args = parser.parse_args()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, args.results_dir)
PART_A_DIR = os.path.join(SCRIPT_DIR, args.part_a_results)
ARCH_STYLES = {
    'baseline':                    {'color': '#1f77b4', 'marker': 'o', 'linestyle': '-'},
    'qk_norm':                     {'color': '#ff7f0e', 'marker': 's', 'linestyle': '--'},
    'attn_gate':                   {'color': '#2ca02c', 'marker': '^', 'linestyle': '-.'},
    'value_emb':                   {'color': '#d62728', 'marker': 'D', 'linestyle': ':'},
    'qk_norm+attn_gate':          {'color': '#9467bd', 'marker': 'v', 'linestyle': (0, (5, 1))},
    'qk_norm+value_emb':          {'color': '#8c564b', 'marker': 'p', 'linestyle': (0, (3, 1, 1, 1))},
    'attn_gate+value_emb':         {'color': '#17becf', 'marker': 'P', 'linestyle': (0, (5, 2, 1, 2))},
    'qk_norm+attn_gate+value_emb': {'color': '#e377c2', 'marker': '*', 'linestyle': (0, (3, 1, 1, 1, 1, 1))},
}


def load_results(json_dir):
    records = []
    if not os.path.isdir(json_dir):
        return records
    for fname in sorted(os.listdir(json_dir)):
        if not fname.endswith('.json'):
            continue
        if fname.startswith('results_'):
            with open(os.path.join(json_dir, fname)) as f:
                r = json.load(f)
                r['_file'] = fname
                r['_dir'] = json_dir
                records.append(r)
    return records


def base_tag(tag):
    return re.sub(r'_s\d+$', '', tag)


def aggregate_by_config(records):
    """Group by (arch, non_emb_params) across seeds; return list of {arch, non_emb, mean, std, n}."""
    groups = defaultdict(list)
    for r in records:
        arch = r['config'].get('arch', 'baseline')
        ne = r['config'].get(
            'architecture_params_excluding_input_embedding',
            r['config'].get('non_embedding_params'),
        )
        if ne is None:
            continue
        be = r.get('best_epoch', 10) or 10
        loss = r['valid_epoch_loss'][be - 1]
        groups[(arch, ne)].append(loss)

    result = []
    for (arch, ne), losses in groups.items():
        mu = float(np.mean(losses))
        sd = float(np.std(losses, ddof=1)) if len(losses) > 1 else 0.0
        result.append({'arch': arch, 'non_emb': ne, 'mean': mu, 'std': sd, 'n': len(losses)})
    return result


def fit_log_log(points):
    xs = np.asarray([p['non_emb'] for p in points], dtype=float)
    ys = np.asarray([p['mean'] for p in points], dtype=float)
    log_x, log_y = np.log10(xs), np.log10(ys)
    slope, intercept = np.polyfit(log_x, log_y, 1)
    fitted = slope * log_x + intercept
    residual = float(np.sum((log_y - fitted) ** 2))
    total = float(np.sum((log_y - log_y.mean()) ** 2))
    return {
        'alpha': float(slope),
        'r_squared': 1.0 - residual / total if total > 0 else 1.0,
        'n_points': len(points),
        'parameter_range': [int(xs.min()), int(xs.max())],
    }


def write_baseline_fit_statistics(agg):
    baseline = sorted(
        (p for p in agg if p['arch'] == 'baseline'),
        key=lambda p: p['non_emb'],
    )
    core = [p for p in baseline if 700_000 <= p['non_emb'] <= 5_600_000]
    if len(core) < 5:
        print("Fewer than five core baseline points; skipping fit statistics.")
        return
    core = core[:5]
    stats = {
        'parameter_definition': 'trainable parameters excluding shared input token embedding',
        'all_five_models': fit_log_log(core),
        'capacity_limited_first_three': fit_log_log(core[:3]),
        'points': core,
    }
    path = os.path.join(RESULTS_DIR, 'scaling_fit_stats.json')
    with open(path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Scaling fit statistics saved: {path}")


def write_architecture_ablation(records, dim=128):
    groups = defaultdict(list)
    for r in records:
        config = r.get('config', {})
        if config.get('emb_dim') != dim:
            continue
        arch = config.get('arch', 'baseline')
        if arch == 'baseline' and not r.get('tag', '').startswith('scaling_s'):
            continue
        groups[arch].append(r)

    summary = {
        'dimension': dim,
        'parameter_definition': 'trainable parameters excluding shared input token embedding',
        'architectures': {},
    }
    for arch, group in sorted(groups.items()):
        losses = [r['valid_epoch_loss'][(r.get('best_epoch', 10) or 10) - 1]
                  for r in group]
        ppls = [r['best_valid_ppl'] for r in group]
        params = [r['config'].get(
            'architecture_params_excluding_input_embedding',
            r['config'].get('non_embedding_params'),
        ) for r in group]
        summary['architectures'][arch] = {
            'parameter_count': int(round(float(np.mean(params)))),
            'valid_loss_mean': float(np.mean(losses)),
            'valid_loss_std': float(np.std(losses, ddof=1)) if len(losses) > 1 else 0.0,
            'valid_ppl_mean': float(np.mean(ppls)),
            'valid_ppl_std': float(np.std(ppls, ddof=1)) if len(ppls) > 1 else 0.0,
            'n_seeds': len(group),
        }
    path = os.path.join(RESULTS_DIR, f'architecture_ablation_d{dim}.json')
    with open(path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Architecture ablation summary saved: {path}")


def plot_scaling_laws(records):
    """All architectures on one log-log plot with per-arch linear fits and std errorbars."""
    agg = aggregate_by_config(records)
    write_baseline_fit_statistics(agg)

    arch_data = defaultdict(list)
    for pt in agg:
        arch_data[pt['arch']].append(pt)

    fig, ax = plt.subplots(figsize=(12, 8))
    for arch, points in sorted(arch_data.items()):
        points = sorted(points, key=lambda p: p['non_emb'])
        xs  = np.array([p['non_emb'] for p in points])
        ys  = np.array([p['mean']    for p in points])
        sds = np.array([p['std']     for p in points])
        style = ARCH_STYLES.get(arch, {'color': '#333333', 'marker': 'x', 'linestyle': '--'})
        color  = style['color']
        marker = style['marker']

        if any(s > 0 for s in sds):
            ax.errorbar(xs, ys, yerr=sds, fmt='none',
                        ecolor=color, elinewidth=1.0, capsize=3, zorder=4)
        ax.scatter(xs, ys, color=color, marker=marker, s=80, zorder=5,
                   label=arch, edgecolors='black', linewidth=0.5)

        if len(xs) >= 2:
            log_x = np.log10(xs)
            log_y = np.log10(ys)
            slope, intercept = np.polyfit(log_x, log_y, 1)
            x_fit = np.logspace(log_x.min(), log_x.max(), 50)
            y_fit = 10 ** (slope * np.log10(x_fit) + intercept)
            ls = style.get('linestyle', '--')
            ax.plot(x_fit, y_fit, color=color, linestyle=ls, alpha=0.8, linewidth=2.0)
            mid_x = float(np.sqrt(x_fit[0] * x_fit[-1]))
            mid_y = float(np.sqrt(y_fit[0] * y_fit[-1]))
            ax.annotate(f'$\\alpha={slope:.3f}$', (mid_x, mid_y),
                        color=color, fontsize=8,
                        bbox=dict(boxstyle='round,pad=0.15', fc='white', alpha=0.8))

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Trainable Parameters Excluding Shared Input Embedding', fontsize=13)
    ax.set_ylabel('Best Validation Loss (mean $\\pm$ std across seeds)', fontsize=13)
    ax.legend(loc='lower left', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    for fmt in ('svg', 'png', 'pdf'):
        path = os.path.join(RESULTS_DIR, f'partb_fig1_scaling_laws.{fmt}')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Figure 1 saved: {path}")
    plt.close()


def plot_position_loss(record):
    """Average loss vs token position group from the best-performing baseline model."""
    pos_groups = record.get('position_groups', [])
    if not pos_groups:
        print("No position_groups in record — train.py may need re-run.")
        return

    labels = [g['range'] for g in pos_groups]
    losses = [g['loss'] for g in pos_groups]
    positions = []
    for label in labels:
        lo, hi = label.split('-')
        positions.append((int(lo) + int(hi)) / 2)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(positions, losses, 'b-o', markersize=6, linewidth=1.5)
    ax.set_xlabel('Token Position', fontsize=13)
    ax.set_ylabel('Average Loss', fontsize=13)
    ax.grid(True, alpha=0.3)
    for i in [0, len(positions) // 4, len(positions) // 2, -1]:
        ax.annotate(labels[i], (positions[i], losses[i]),
                    textcoords="offset points", xytext=(0, 12),
                    fontsize=8, ha='center',
                    bbox=dict(boxstyle='round,pad=0.1', fc='lightyellow', alpha=0.8))

    plt.tight_layout()
    for fmt in ('svg', 'png', 'pdf'):
        path = os.path.join(RESULTS_DIR, f'partb_fig2_position_loss.{fmt}')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Figure 2 saved: {path}")
    plt.close()

def main():
    records = load_results(RESULTS_DIR)
    if os.path.isdir(PART_A_DIR) and PART_A_DIR != RESULTS_DIR:
        part_a_records = load_results(PART_A_DIR)
        for r in part_a_records:
            if r['config'].get('arch', 'baseline') == 'baseline':
                r['_file'] = f"[Part A] {r['_file']}"
                records.append(r)

    if not records:
        print(f"No results_*.json files found in {RESULTS_DIR}")
        print("Run scripts/sweep_3.1_scaling.sh first.")
        return

    print(f"Loaded {len(records)} result files")
    write_architecture_ablation(records)
    plot_scaling_laws(records)

    baseline_records = [r for r in records
                        if r['config'].get('arch', 'baseline') == 'baseline'
                        and r.get('position_groups')]
    if baseline_records:
        best = min(baseline_records, key=lambda r: r['best_valid_ppl'])
        print(f"\nPosition-loss source: {best['_file']} (PPL={best['best_valid_ppl']:.2f})")
        plot_position_loss(best)
    else:
        print("No baseline results — cannot produce position-loss plot.")

if __name__ == '__main__':
    main()
