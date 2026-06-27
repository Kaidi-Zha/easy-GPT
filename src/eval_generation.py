"""Reproducible generation comparison for GPT, MDLM, and LLaDA.

The diffusion samples are read from their three seeded training runs. Matching
GPT samples are generated from the corresponding d128 checkpoints. All metrics
use fixed-length, 64-token continuations and are aggregated across seeds.
"""

import argparse
import json
import os
import re
import sys
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import Corpus
from model import CausalLMM


parser = argparse.ArgumentParser(description='Compare GPT and diffusion generation')
parser.add_argument('--bonus_dir', default='../results/bonus')
parser.add_argument('--partb_dir', default='../results/part_b')
parser.add_argument('--seq_len', type=int, default=64)
parser.add_argument('--samples_per_seed', type=int, default=5)
parser.add_argument('--temperature', type=float, default=1.0)
parser.add_argument('--top_p', type=float, default=0.9)
parser.add_argument('--seeds', type=int, nargs='+', default=[1234, 42, 2024])
parser.add_argument('--cuda', action='store_true')
args = parser.parse_args()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BONUS_DIR = os.path.join(SCRIPT_DIR, args.bonus_dir)
PARTB_DIR = os.path.join(SCRIPT_DIR, args.partb_dir)
device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')


def compute_metrics(samples):
    texts = [s.strip() for s in samples if s.strip()]
    docs = [text.lower().split() for text in texts]
    all_tokens = [token for doc in docs for token in doc]

    def ngrams(tokens, n):
        return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))

    overlaps = []
    for i in range(len(docs)):
        for j in range(i + 1, len(docs)):
            left, right = ngrams(docs[i], 2), ngrams(docs[j], 2)
            overlaps.append(sum((left & right).values()) / max(sum((left | right).values()), 1))

    def repetitive(tokens):
        for n in range(3, min(6, len(tokens) // 2)):
            grams = ngrams(tokens, n)
            if grams and max(grams.values()) >= 3:
                return True
        return False

    return {
        'n_samples': len(docs),
        'mean_length_tokens': float(np.mean([len(d) for d in docs])),
        'ttr_overall': len(set(all_tokens)) / max(len(all_tokens), 1),
        'ttr_mean_per_sample': float(np.mean([len(set(d)) / max(len(d), 1) for d in docs])),
        'pairwise_bigram_overlap': float(np.mean(overlaps)) if overlaps else 0.0,
        'repetition_rate': float(np.mean([repetitive(d) for d in docs])),
    }


def nucleus_sample(logits):
    logits = logits / args.temperature
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    remove = cumulative > args.top_p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    sorted_logits[remove] = -float('inf')
    sampled_rank = torch.multinomial(F.softmax(sorted_logits, dim=-1), 1)
    return sorted_indices.gather(-1, sampled_rank).squeeze(-1)


@torch.inference_mode()
def generate_gpt_samples(corpus, seed):
    suffix = '' if seed == 1234 else f'_s{seed}'
    checkpoint = os.path.join(
        PARTB_DIR, 'checkpoints', f'best_model_scaling_s128{suffix}.pt'
    )
    model = CausalLMM(
        vocab_size=len(corpus.vocabulary), dim=128, num_layers=4,
        num_heads=4, max_seq_len=256, dropout=0.1,
    ).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    eos_id = corpus.word_id['<eos>']
    samples = []
    for sample_index in range(args.samples_per_seed):
        torch.manual_seed(seed + sample_index)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(seed + sample_index)
        generated = torch.tensor([[eos_id]], dtype=torch.long, device=device)
        output_ids = []
        for _ in range(args.seq_len):
            logits = model(generated[-256:])[-1]
            next_token = nucleus_sample(logits)
            generated = torch.cat([generated, next_token.view(1, 1)], dim=0)
            output_ids.append(int(next_token.item()))
        samples.append(' '.join(corpus.vocabulary[token] for token in output_ids))
    return samples


def result_path(model, seed):
    suffix = '' if seed == 1234 else f'_s{seed}'
    return os.path.join(BONUS_DIR, f'results_{model}_d128{suffix}.json')


def aggregate(per_seed):
    keys = next(iter(per_seed.values())).keys()
    output = {}
    for key in keys:
        values = [metrics[key] for metrics in per_seed.values()]
        output[key] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            'n_seeds': len(values),
        }
    return output


def main():
    print(f'Generation evaluation device: {device}')
    corpus = Corpus(
        os.path.join(SCRIPT_DIR, '..', 'data', 'ptb'),
        {'train': 1, 'valid': 1}, 256,
    )
    samples_by_model = {'GPT (AR)': {}, 'MDLM': {}, 'LLaDA': {}}
    for seed in args.seeds:
        samples_by_model['GPT (AR)'][seed] = generate_gpt_samples(corpus, seed)
        for label, filename_model in [('MDLM', 'masked_diffusion'), ('LLaDA', 'llada')]:
            with open(result_path(filename_model, seed)) as f:
                samples = json.load(f).get('samples', [])
            if len(samples) != args.samples_per_seed:
                raise ValueError(
                    f'{label} seed {seed} has {len(samples)} samples; '
                    f'expected {args.samples_per_seed}'
                )
            samples_by_model[label][seed] = samples

    per_seed = {
        model: {str(seed): compute_metrics(samples) for seed, samples in runs.items()}
        for model, runs in samples_by_model.items()
    }
    output = {
        'config': {
            'device': str(device),
            'torch_version': torch.__version__,
            'model_size': 'd128 (4 layers, 4 heads)',
            'seeds': args.seeds,
            'samples_per_seed': args.samples_per_seed,
            'total_samples_per_model': len(args.seeds) * args.samples_per_seed,
            'sequence_length_tokens': args.seq_len,
            'gpt_sampling': {
                'prompt': '<eos>', 'temperature': args.temperature,
                'top_p': args.top_p, 'kv_cache': False,
            },
            'mdlm_sampling': {'steps': 32, 'schedule': 'linear', 'sampling': 'multinomial'},
            'llada_sampling': {
                'steps': 128, 'temperature': 1.0,
                'schedule': 'cosine', 'low_confidence_remasking': True,
            },
        },
        'per_seed': per_seed,
        'aggregate': {model: aggregate(runs) for model, runs in per_seed.items()},
        'gpt_samples': samples_by_model['GPT (AR)'],
    }
    path = os.path.join(BONUS_DIR, 'generation_comparison.json')
    with open(path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'Generation comparison saved: {path}')
    for model, metrics in output['aggregate'].items():
        print(model)
        for key in ('ttr_overall', 'ttr_mean_per_sample', 'pairwise_bigram_overlap', 'repetition_rate'):
            value = metrics[key]
            print(f"  {key}: {value['mean']:.4f} +/- {value['std']:.4f}")


if __name__ == '__main__':
    main()
