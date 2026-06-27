import argparse
import json
import math
import os
import sys
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data
from diffusion_lm import MaskedDiffusionLM

parser = argparse.ArgumentParser(description='Train Masked Diffusion LM on PTB')
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--train_batch_size', type=int, default=16)
parser.add_argument('--eval_batch_size', type=int, default=16)
parser.add_argument('--max_sql', type=int, default=256, help='sequence length')
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--num_layers', type=int, default=4)
parser.add_argument('--num_heads', type=int, default=4)
parser.add_argument('--emb_dim', type=int, default=128)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--grad_clip', type=float, default=1.0)
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--tag', type=str, default='diffusion', help='experiment tag')
parser.add_argument('--results_dir', type=str, default='../results/bonus')
parser.add_argument('--gen_steps', type=int, default=32,
                    help='sampling steps for iterative unmasking')
args = parser.parse_args()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(SCRIPT_DIR, args.results_dir)
CKPT_DIR = os.path.join(RESULT_DIR, 'checkpoints')
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

torch.manual_seed(args.seed)
device = torch.device(f'cuda:{args.gpu_id}' if args.cuda and torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
print(f"Config: layers={args.num_layers}, heads={args.num_heads}, dim={args.emb_dim}, "
      f"lr={args.lr}, batch={args.train_batch_size}, max_sql={args.max_sql}")

batch_size = {'train': args.train_batch_size, 'valid': args.eval_batch_size}
data_loader = data.Corpus(
    os.path.join(SCRIPT_DIR, "..", "data", "ptb"), batch_size, args.max_sql)

orig_vocab_size = len(data_loader.vocabulary)
vocab_size = orig_vocab_size + 1
mask_id = orig_vocab_size
data_loader.vocabulary.append('<mask>')
data_loader.word_id['<mask>'] = mask_id
print(f"Vocabulary: {orig_vocab_size:,} + [MASK] → {vocab_size:,}")

model = MaskedDiffusionLM(
    vocab_size=vocab_size, dim=args.emb_dim, num_layers=args.num_layers,
    num_heads=args.num_heads, max_seq_len=args.max_sql, dropout=args.dropout,
)
model = model.to(device)
print(f"Model params: {model.num_params():,}")

optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

def forward_process(input_ids):
    """Randomly mask tokens at ratio t ~ U(0, 1). Returns (masked_ids, mask, targets)."""
    B = input_ids.size(1)
    t = torch.rand(B, device=device)
    mask = torch.rand_like(input_ids, dtype=torch.float, device=device) < t.unsqueeze(0)
    masked_ids = input_ids.clone()
    masked_ids[mask] = mask_id
    targets = input_ids.clone()
    targets[~mask] = -100
    return masked_ids, mask, targets


@torch.no_grad()
def evaluate_diffusion():
    data_loader.set_valid()
    model.eval()
    total_loss = 0.0
    total_masked = 0
    while True:
        input_ids, _target, end_flag = data_loader.get_batch()
        input_ids = input_ids.to(device)
        masked_ids, mask, targets = forward_process(input_ids)
        logits = model(masked_ids)
        loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1), ignore_index=-100)
        total_loss += loss.item() * mask.sum().item()
        total_masked += mask.sum().item()
        if end_flag:
            break
    avg_loss = total_loss / max(total_masked, 1)
    ppl = math.exp(avg_loss)
    print(f"  Valid diffusion loss: {avg_loss:.4f}, pseudo-ppl: {ppl:.2f}")
    return ppl, avg_loss

def train_epoch():
    data_loader.set_train()
    model.train()
    losses = []
    idx = 0
    while True:
        input_ids, _target, end_flag = data_loader.get_batch()
        input_ids = input_ids.to(device)

        masked_ids, mask, targets = forward_process(input_ids)

        optimizer.zero_grad()
        logits = model(masked_ids)
        loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1), ignore_index=-100)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        losses.append(loss.item())
        if (idx + 1) % 50 == 0:
            print(f"  Step {idx + 1}, loss: {loss.item():.4f}")
        idx += 1
        if end_flag:
            break
    avg = sum(losses) / len(losses)
    return avg, losses

@torch.no_grad()
def generate_samples(prompts, max_len=64):
    model.eval()
    samples = []
    for _ in range(5):
        tokens = model.generate(seq_len=max_len, batch_size=1, steps=args.gen_steps, device=device)
        text = ' '.join([data_loader.vocabulary[t.item()]
                         for t in tokens[:, 0] if t.item() != mask_id])
        samples.append(text)
    return samples

all_step_losses = []
train_epoch_loss = []
valid_epoch_ppl = []
valid_epoch_loss = []
best_valid_ppl = float('inf')
for epoch in range(1, args.epochs + 1):
    print(f"\n{'='*50}")
    print(f"Epoch {epoch}/{args.epochs}  (lr: {scheduler.get_last_lr()[0]:.2e})")
    print(f"{'='*50}")

    avg_loss, step_losses = train_epoch()
    train_epoch_loss.append(avg_loss)
    all_step_losses.extend(step_losses)

    v_ppl, v_loss = evaluate_diffusion()
    valid_epoch_ppl.append(v_ppl)
    valid_epoch_loss.append(v_loss)

    if v_ppl < best_valid_ppl:
        best_valid_ppl = v_ppl
        best_epoch = epoch
        torch.save(model.state_dict(), os.path.join(CKPT_DIR, f"best_diffusion_{args.tag}.pt"))
        print(f"  Best model (ppl={best_valid_ppl:.2f})")

    scheduler.step()

print(f"Best valid pseudo-ppl: {best_valid_ppl:.2f} @ epoch {best_epoch}")
model.load_state_dict(
    torch.load(
        os.path.join(CKPT_DIR, f"best_diffusion_{args.tag}.pt"),
        map_location=device
    )
)
print("\nGenerating unconditional samples...")
samples = generate_samples([])
for i, s in enumerate(samples):
    print(f"\n  Sample {i + 1}: {s[:200]}...")

results = {
    'tag': args.tag,
    'config': {
        'emb_dim': args.emb_dim, 'num_layers': args.num_layers,
        'num_heads': args.num_heads, 'lr': args.lr, 'dropout': args.dropout,
        'batch_size': args.train_batch_size, 'max_sql': args.max_sql,
        'epochs': args.epochs, 'gen_steps': args.gen_steps,
        'seed': args.seed,
        'total_params': model.num_params(),
    },
    'train_epoch_loss': train_epoch_loss,
    'valid_epoch_ppl': valid_epoch_ppl,
    'valid_epoch_loss': valid_epoch_loss,
    'best_valid_ppl': best_valid_ppl,
    'best_epoch': best_epoch,
    'samples': samples,
    'generation_config': {
        'n_samples': 5, 'sequence_length': 64,
        'steps': args.gen_steps, 'sampling': 'multinomial',
        'schedule': 'linear', 'seed': args.seed,
    },
}
with open(os.path.join(RESULT_DIR, f"results_{args.tag}.json"), 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {os.path.join(RESULT_DIR, f'results_{args.tag}.json')}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(all_step_losses, alpha=0.6, linewidth=0.5)
if len(all_step_losses) > 50:
    w = min(50, len(all_step_losses) // 10)
    smoothed = [sum(all_step_losses[max(0, i-w):i+1])/(i-max(0, i-w)+1)
                for i in range(len(all_step_losses))]
    axes[0].plot(smoothed, 'r', linewidth=1.5, label=f'avg({w})')
    axes[0].legend()
axes[0].set_xlabel('Step')
axes[0].set_ylabel('Loss')
axes[0].grid(True, alpha=0.3)

epochs_range = range(1, args.epochs + 1)
axes[1].plot(epochs_range, valid_epoch_ppl, 'r-o', label='Valid pseudo-PPL', markersize=4)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Pseudo-Perplexity')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
path = os.path.join(RESULT_DIR, f"loss_curves_{args.tag}.png")
plt.savefig(path, dpi=150)
print(f"Loss curves: {path}")
