import argparse
import json
import math
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data
import model

parser = argparse.ArgumentParser(description='PyTorch PTB Language Model')
parser.add_argument('--epochs', type=int, default=10, help='upper epoch limit')
parser.add_argument('--train_batch_size', type=int, default=16, metavar='N')
parser.add_argument('--eval_batch_size', type=int, default=16, metavar='N')
parser.add_argument('--max_sql', type=int, default=256, help='sequence length')
parser.add_argument('--seed', type=int, default=1234, help='set random seed')
parser.add_argument('--num_layers', type=int, default=4)
parser.add_argument('--num_heads', type=int, default=4)
parser.add_argument('--emb_dim', type=int, default=128)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
parser.add_argument('--grad_clip', type=float, default=1.0, help='gradient clipping max norm')
parser.add_argument('--cuda', action='store_true', help='use CUDA device')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU device id used')
parser.add_argument('--qk_norm', action='store_true', help='apply LayerNorm to Q and K')
parser.add_argument('--attn_gate', action='store_true', help='add learnable sigmoid gate to attention')
parser.add_argument('--value_emb', action='store_true', help='add separate value embedding')
parser.add_argument('--fixed_steps', type=int, default=0,
                    help='fixed training steps per epoch for scaling study')
parser.add_argument('--position_split', type=str, default='train',
                    choices=['train', 'valid'],
                    help='data split for position-dependent loss analysis')
parser.add_argument('--tag', type=str, default='default', help='experiment tag for output files')
parser.add_argument('--results_dir', type=str, default='../results',
                    help='directory for results output')

args = parser.parse_args()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(SCRIPT_DIR, args.results_dir)
CKPT_DIR = os.path.join(RESULT_DIR, 'checkpoints')
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

torch.manual_seed(args.seed)
use_gpu = args.cuda
if use_gpu:
    torch.cuda.set_device(args.gpu_id)
    device = torch.device(args.gpu_id)
else:
    device = torch.device("cpu")

arch_flags = []
if args.qk_norm: arch_flags.append('qk_norm')
if args.attn_gate: arch_flags.append('attn_gate')
if args.value_emb: arch_flags.append('value_emb')
arch_name = '+'.join(arch_flags) if arch_flags else 'baseline'
print(f"Device: {device}")
print(f"Config: layers={args.num_layers}, heads={args.num_heads}, dim={args.emb_dim}, "
      f"lr={args.lr}, dropout={args.dropout}, batch={args.train_batch_size}, "
      f"max_sql={args.max_sql}, epochs={args.epochs}, arch={arch_name}")

batch_size = {'train': args.train_batch_size, 'valid': args.eval_batch_size}
data_loader = data.Corpus(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "ptb"),
                          batch_size, args.max_sql)
lm = model.CausalLMM(
    vocab_size=len(data_loader.vocabulary),
    dim=args.emb_dim,
    num_layers=args.num_layers,
    num_heads=args.num_heads,
    max_seq_len=args.max_sql,
    dropout=args.dropout,
    use_qk_norm=args.qk_norm,
    use_attn_gate=args.attn_gate,
    use_value_emb=args.value_emb,
)
lm = lm.to(device)
total_params = sum(p.numel() for p in lm.parameters())
architecture_params = lm.num_architecture_params()
print(f"Model params: {total_params:,} "
      f"(excluding shared input embedding: {architecture_params:,})")

optimizer = optim.AdamW(lm.parameters(), lr=args.lr, weight_decay=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
criterion = nn.CrossEntropyLoss()

def evaluate():
    data_loader.set_valid()
    lm.eval()
    total_loss = 0.0
    total_tokens = 0
    idx = 0
    print("Validating...")
    while True:
        with torch.no_grad():
            data, target, end_flag = data_loader.get_batch()
            data = data.to(device)
            target = target.to(device)
            logits = lm(data)
            loss = criterion(logits.view(-1, logits.size(-1)), target)
            total_loss += loss.item() * target.size(0)
            total_tokens += target.size(0)
            idx += 1
            if end_flag:
                break
    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    print(f"  Valid loss: {avg_loss:.4f}, perplexity: {ppl:.2f}")
    return ppl, avg_loss

def evaluate_by_position(group_size=32, split='train'):
    """Average loss per token-position group. Uses train split by default (§3.1 Study 2)."""
    if split == 'train':
        data_loader.set_train()
    else:
        data_loader.set_valid()
    lm.eval()
    max_len = args.max_sql
    pos_loss_sum = torch.zeros(max_len)
    pos_count = torch.zeros(max_len)
    with torch.no_grad():
        while True:
            data, target, end_flag = data_loader.get_batch()
            data = data.to(device)
            target = target.to(device)
            logits = lm(data)
            seq_len = logits.size(0)
            loss_per_token = F.cross_entropy(
                logits.view(-1, logits.size(-1)), target, reduction='none'
            ).view(seq_len, -1)
            pos_loss_sum[:seq_len] += loss_per_token.sum(dim=1).cpu()
            pos_count[:seq_len] += loss_per_token.size(1)
            if end_flag:
                break
    avg_by_pos = pos_loss_sum / pos_count.clamp(min=1)
    groups = []
    for start in range(0, max_len, group_size):
        end = min(start + group_size, max_len)
        group_avg = avg_by_pos[start:end].mean().item()
        groups.append((f"{start + 1}-{end}", group_avg))
    return groups

def train():
    data_loader.set_train()
    lm.train()
    step_losses = []
    idx = 0
    while True:
        data, target, end_flag = data_loader.get_batch()
        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        logits = lm(data)
        loss = criterion(logits.view(-1, logits.size(-1)), target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lm.parameters(), args.grad_clip)
        optimizer.step()

        step_losses.append(loss.item())
        if (idx + 1) % 50 == 0:
            print(f"  Step {idx + 1}, loss: {loss.item():.4f}")
        idx += 1
        if end_flag:
            break
        if args.fixed_steps > 0 and idx >= args.fixed_steps:
            break

    avg_loss = sum(step_losses) / len(step_losses)
    ppl = math.exp(avg_loss)
    print(f"  Train avg loss: {avg_loss:.4f}, perplexity: {ppl:.2f}")
    return ppl, avg_loss, step_losses

all_step_losses = []
train_epoch_ppl = []
train_epoch_loss = []
valid_epoch_ppl = []
valid_epoch_loss = []
best_valid_ppl = float('inf')

for epoch in range(1, args.epochs + 1):
    print(f"\n{'='*50}")
    print(f"Epoch {epoch}/{args.epochs}  (lr: {scheduler.get_last_lr()[0]:.2e})")
    print(f"{'='*50}")

    t_ppl, t_loss, step_losses = train()
    train_epoch_ppl.append(t_ppl)
    train_epoch_loss.append(t_loss)
    all_step_losses.extend(step_losses)

    v_ppl, v_loss = evaluate()
    valid_epoch_ppl.append(v_ppl)
    valid_epoch_loss.append(v_loss)

    if v_ppl < best_valid_ppl:
        best_valid_ppl = v_ppl
        best_epoch = epoch
        torch.save(lm.state_dict(), os.path.join(CKPT_DIR, f"best_model_{args.tag}.pt"))
        print(f"New best valid PPL: {best_valid_ppl:.2f} (epoch {best_epoch})")

    scheduler.step()

print(f"Training complete. Best valid PPL: {best_valid_ppl:.2f} @ epoch {best_epoch}")
print(f"\nAnalyzing loss by position group (split={args.position_split})...")
lm.load_state_dict(torch.load(os.path.join(CKPT_DIR, f"best_model_{args.tag}.pt"),
                              map_location=device))
pos_groups = evaluate_by_position(group_size=32, split=args.position_split)
print(f"Position groups ({args.position_split} loss per group):")
for label, avg_loss in pos_groups:
    print(f"pos {label:>10s}: {avg_loss:.4f}")

results = {
    'tag': args.tag,
    'config': {
        'emb_dim': args.emb_dim, 'num_layers': args.num_layers,
        'num_heads': args.num_heads, 'lr': args.lr, 'dropout': args.dropout,
        'batch_size': args.train_batch_size, 'max_sql': args.max_sql,
        'epochs': args.epochs, 'grad_clip': args.grad_clip,
        'seed': args.seed,
        'arch': arch_name,
        'total_params': total_params,
        'architecture_params_excluding_input_embedding': architecture_params,
        'non_embedding_params': architecture_params,
    },
    'train_epoch_ppl': train_epoch_ppl,
    'train_epoch_loss': train_epoch_loss,
    'valid_epoch_ppl': valid_epoch_ppl,
    'valid_epoch_loss': valid_epoch_loss,
    'best_valid_ppl': best_valid_ppl,
    'best_epoch': best_epoch,
    'position_groups': [{'range': label, 'loss': avg_loss} for label, avg_loss in pos_groups],
    'position_split': args.position_split,
}

with open(os.path.join(RESULT_DIR, f"results_{args.tag}.json"), 'w') as f:
    json.dump(results, f, indent=2)
print(f"Results saved to {os.path.join(RESULT_DIR, f'results_{args.tag}.json')}" )
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].plot(all_step_losses, alpha=0.6, linewidth=0.5)
window = min(50, len(all_step_losses) // 10) if len(all_step_losses) > 50 else 1
if window > 1:
    smoothed = [
        sum(all_step_losses[max(0,i-window):i+1]) / (i-max(0,i-window)+1)
        for i in range(len(all_step_losses))
    ]
    axes[0].plot(smoothed, color='red', linewidth=1.5, label=f'avg({window})')
    axes[0].legend()
axes[0].set_xlabel('Step')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training Loss (per step)')
axes[0].grid(True, alpha=0.3)

epochs_range = range(1, args.epochs + 1)
axes[1].plot(epochs_range, train_epoch_ppl, 'b-o', label='Train PPL', markersize=4)
axes[1].plot(epochs_range, valid_epoch_ppl, 'r-o', label='Valid PPL', markersize=4)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Perplexity')
axes[1].set_title('Train & Valid Perplexity')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].plot(epochs_range, train_epoch_loss, 'b-o', label='Train Loss', markersize=4)
axes[2].plot(epochs_range, valid_epoch_loss, 'r-o', label='Valid Loss', markersize=4)
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('Loss')
axes[2].set_title('Train & Valid Loss')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
save_path = os.path.join(RESULT_DIR, f"loss_curves_{args.tag}.png")
plt.savefig(save_path, dpi=150)
print(f"Loss curves saved to {save_path}")
