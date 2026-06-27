"""Part C: Fine-tune Qwen2.5-0.5B on a domain-specific dataset.

Uses LoRA for parameter-efficient fine-tuning with next-token prediction.
After training, generates sample texts comparing the base and fine-tuned models.

Usage:
    python finetune.py                                          # default settings
    python finetune.py --data_dir ../data/domain                # custom dataset
    python finetune.py --epochs 3 --lr 5e-5 --batch_size 4     # hyperparams
    python finetune.py --generate_only                           # skip training
"""

import argparse
import json
import math
import os
import re
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
)
from datasets import load_dataset, concatenate_datasets, Dataset as HFDataset
from peft import LoraConfig, PeftModel, get_peft_model, TaskType

parser = argparse.ArgumentParser(description='Fine-tune Qwen2.5-0.5B with LoRA')
parser.add_argument('--model_id', type=str, default='Qwen/Qwen2.5-0.5B',
                    help='HuggingFace model ID')
parser.add_argument('--data_dir', type=str, default='../data/domain',
                    help='directory with train.txt and valid.txt')
parser.add_argument('--output_dir', type=str, default='../results/part_c',
                    help='output directory for model and results')
parser.add_argument('--epochs', type=int, default=3,
                    help='number of fine-tuning epochs')
parser.add_argument('--batch_size', type=int, default=4,
                    help='training batch size')
parser.add_argument('--eval_batch_size', type=int, default=16,
                    help='validation batch size (no gradients)')
parser.add_argument('--lr', type=float, default=5e-5,
                    help='learning rate (LoRA: 5e-5 ~ 1e-4)')
parser.add_argument('--max_length', type=int, default=256,
                    help='max token length for training')
parser.add_argument('--grad_accum', type=int, default=4,
                    help='gradient accumulation steps')
parser.add_argument('--lora_r', type=int, default=8,
                    help='LoRA rank')
parser.add_argument('--lora_alpha', type=int, default=16,
                    help='LoRA alpha')
parser.add_argument('--patience', type=int, default=3,
                    help='early stopping patience (validations without improvement)')
parser.add_argument('--eval_steps', type=int, default=200,
                    help='validate every N optimizer steps (0 = per epoch only)')
parser.add_argument('--generate_only', action='store_true',
                    help='load output_dir/best_lora and only generate samples')
parser.add_argument('--seed', type=int, default=1234,
                    help='training and generation random seed')
parser.add_argument('--samples_per_prompt', type=int, default=3,
                    help='number of sampled continuations per prompt and model')
parser.add_argument('--max_new_tokens', type=int, default=80)
parser.add_argument('--temperature', type=float, default=0.8)
parser.add_argument('--top_p', type=float, default=0.9)
parser.add_argument('--num_workers', type=int, default=None,
                    help='CPU workers for tokenization (default: auto-detect)')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA (auto-detected if available)')
args = parser.parse_args()
set_seed(args.seed)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, args.data_dir)
OUTPUT_DIR = os.path.join(SCRIPT_DIR, args.output_dir)
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device('cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu')
print(f"Device: {device}")
print(f"\nLoading tokenizer: {args.model_id}")
tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Loading model: {args.model_id}")
if device.type == 'cuda' and torch.cuda.is_bf16_supported():
    compute_dtype = torch.bfloat16
    print("  Using bfloat16")
else:
    compute_dtype = torch.float32
    print("  Using float32 (bf16 not supported)")
model = AutoModelForCausalLM.from_pretrained(
    args.model_id,
    dtype=compute_dtype,
    trust_remote_code=True,
)
model = model.to(device)
model.config.use_cache = False
use_amp = (compute_dtype == torch.float16)
scaler = torch.amp.GradScaler('cuda', enabled=use_amp) if device.type == 'cuda' else None

total_params = sum(p.numel() for p in model.parameters())
print(f"Model params: {total_params:,}")

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=args.lora_r,
    lora_alpha=args.lora_alpha,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
)

if args.generate_only:
    adapter_dir = os.path.join(OUTPUT_DIR, 'best_lora')
    if not os.path.isdir(adapter_dir):
        raise FileNotFoundError(
            f"--generate_only requires a trained adapter at {adapter_dir}"
        )
    model = PeftModel.from_pretrained(model, adapter_dir)
    print(f"Loaded trained LoRA adapter: {adapter_dir}")
else:
    model = get_peft_model(model, lora_config)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"LoRA trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

num_workers = args.num_workers or min(32, os.cpu_count() or 1)
print(f"\nData loading: using {num_workers} CPU workers for tokenization")
print(f"Loading dataset from: {DATA_DIR}")


def build_causal_dataset(file_path, tokenizer, max_length):
    raw = load_dataset("text", data_files={"data": file_path}, split="data",
                       streaming=False, cache_dir=os.path.join(OUTPUT_DIR, ".cache"))

    def tokenize_fn(examples):
        return tokenizer(examples["text"], add_special_tokens=False, truncation=False)

    tokenized = raw.map(
        tokenize_fn, batched=True, batch_size=1000,
        num_proc=num_workers,
        remove_columns=["text"],
        desc="Tokenizing",
    )
    all_ids = []
    for row in tokenized:
        all_ids.extend(row["input_ids"])
        all_ids.append(tokenizer.eos_token_id)

    stride = max_length // 2
    chunks = []
    for i in range(0, len(all_ids) - max_length, stride):
        chunk = all_ids[i:i + max_length]
        if len(chunk) == max_length:
            chunks.append({"input_ids": chunk, "labels": chunk,
                           "attention_mask": [1] * max_length})

    return HFDataset.from_list(chunks)


t0 = time.time()

train_path = os.path.join(DATA_DIR, 'train.txt')
valid_path = os.path.join(DATA_DIR, 'valid.txt')
train_ds = build_causal_dataset(train_path, tokenizer, args.max_length)
valid_ds = build_causal_dataset(valid_path, tokenizer, args.max_length)

print(f"  Train chunks: {len(train_ds):,}")
print(f"  Valid chunks: {len(valid_ds):,}")
print(f"  Tokenization took {time.time() - t0:.1f}s")
train_ds.set_format(type="torch", columns=["input_ids", "labels", "attention_mask"])
valid_ds.set_format(type="torch", columns=["input_ids", "labels", "attention_mask"])
train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=4, pin_memory=(device.type == 'cuda'))
valid_loader = DataLoader(valid_ds, batch_size=args.eval_batch_size, shuffle=False,
                          num_workers=2, pin_memory=(device.type == 'cuda'))

@torch.no_grad()
def generate_samples(model, tokenizer, prompts):
    model.eval()
    results = []
    for prompt_index, prompt in enumerate(prompts):
        for sample_index in range(args.samples_per_prompt):
            sample_seed = args.seed + prompt_index * 1000 + sample_index
            set_seed(sample_seed)
            inputs = tokenizer(prompt, return_tensors='pt').to(device)
            if inputs['input_ids'].size(1) > args.max_length // 2:
                inputs['input_ids'] = inputs['input_ids'][:, :args.max_length // 2]
                inputs['attention_mask'] = inputs['attention_mask'][:, :args.max_length // 2]
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                pad_token_id=tokenizer.pad_token_id,
            )
            generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            results.append({
                'prompt': prompt,
                'sample_index': sample_index,
                'seed': sample_seed,
                'generation': generated,
            })
    model.train()
    return results

if not args.generate_only:
    def validate():
        model.eval()
        total = 0.0
        total_examples = 0
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=(compute_dtype != torch.float32)):
            for batch in valid_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                batch_examples = input_ids.size(0)
                total += outputs.loss.item() * batch_examples
                total_examples += batch_examples
        model.train()
        return total / max(total_examples, 1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    optimizer_steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps = optimizer_steps_per_epoch * args.epochs
    warmup_steps = max(10, int(0.1 * total_steps))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    eval_every = max(1, args.eval_steps) if args.eval_steps > 0 else 0

    print(f"\n{'='*50}")
    print(f"Fine-tuning: {args.epochs} epochs, lr={args.lr}, batch={args.batch_size}")
    print(f"  Gradient accumulation: {args.grad_accum}, total steps: {total_steps}")
    print(f"  Warmup: {warmup_steps}, eval every: {eval_every or 'epoch'}, patience: {args.patience}")
    print(f"  dtype: {compute_dtype}")
    print(f"{'='*50}")

    train_losses = []
    valid_losses = []
    valid_steps = []
    best_valid_loss = float('inf')
    best_step = 0
    best_adapter_state = None
    patience_counter = 0
    global_step = 0
    stopped_early = False

    for epoch in range(1, args.epochs + 1):
        if stopped_early:
            break

        model.train()
        epoch_loss = 0.0
        step_losses = []
        start_time = time.time()
        for step, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            with torch.amp.autocast('cuda', enabled=(compute_dtype != torch.float32)):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / args.grad_accum

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  ⚠ NaN/Inf loss at step {step + 1}, skipping batch")
                optimizer.zero_grad()
                continue
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            step_losses.append(loss.item() * args.grad_accum)
            epoch_loss += loss.item() * args.grad_accum
            if (step + 1) % args.grad_accum == 0 or (step + 1) == len(train_loader):
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                should_validate = (
                    eval_every > 0
                    and global_step % eval_every == 0
                    and (global_step == eval_every or global_step >= warmup_steps)
                )
                if should_validate:
                    avg_train = sum(step_losses[-50:]) / min(50, len(step_losses)) if step_losses else 0
                    avg_valid = validate()
                    valid_losses.append(avg_valid)
                    valid_steps.append(global_step)
                    print(f"step {global_step:>5d} | train_loss: {avg_train:.4f} | valid_loss: {avg_valid:.4f} | lr: {scheduler.get_last_lr()[0]:.2e}")
                    if avg_valid < best_valid_loss:
                        best_valid_loss = avg_valid
                        best_step = global_step
                        patience_counter = 0
                        best_adapter_state = {
                            name: parameter.detach().cpu().clone()
                            for name, parameter in model.named_parameters()
                            if parameter.requires_grad
                        }
                        model.save_pretrained(os.path.join(OUTPUT_DIR, 'best_lora'))
                        tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, 'best_lora'))
                        print(f"Best model @ step {global_step} (valid_loss={best_valid_loss:.4f})")
                    elif global_step >= warmup_steps:
                        patience_counter += 1
                        if patience_counter >= args.patience:
                            print(f"\n  ══ Early stopping @ step {global_step} "
                                  f"(no improvement for {args.patience} validations) ══")
                            stopped_early = True
                            break
                    else:
                        print("  Warm-up in progress; early-stopping patience not incremented")

            if (step + 1) % 100 == 0:
                avg = sum(step_losses[-100:]) / min(100, len(step_losses))
                print(f"  micro-batch {step + 1:>5d}/{len(train_loader):<5d}  "
                      f"train_loss: {avg:.4f}")

        avg_train_loss = epoch_loss / max(len(step_losses), 1)
        train_losses.append(avg_train_loss)
        elapsed = time.time() - start_time
        if eval_every == 0:
            avg_valid = validate()
            valid_losses.append(avg_valid)
            valid_steps.append(global_step)
            print(f"Epoch {epoch} done ({elapsed:.1f}s)")
            print(f"train_loss: {avg_train_loss:.4f}, valid_loss: {avg_valid:.4f}")

            if avg_valid < best_valid_loss:
                best_valid_loss = avg_valid
                best_step = global_step
                patience_counter = 0
                best_adapter_state = {
                    name: parameter.detach().cpu().clone()
                    for name, parameter in model.named_parameters()
                    if parameter.requires_grad
                }
                model.save_pretrained(os.path.join(OUTPUT_DIR, 'best_lora'))
                tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, 'best_lora'))
                print(f"Best model saved (valid_loss={best_valid_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"\n  ══ Early stopping @ epoch {epoch} ══")
                    stopped_early = True
                    break
        else:
            print(f"Epoch {epoch} done ({elapsed:.1f}s), train_loss: {avg_train_loss:.4f}")

    if best_adapter_state is not None:
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name in best_adapter_state:
                    parameter.copy_(best_adapter_state[name].to(parameter.device))
        print(f"Restored best adapter from optimizer step {best_step} for generation")

    metrics = {
        'train_losses': train_losses,
        'valid_losses': valid_losses,
        'valid_steps': valid_steps,
        'best_valid_loss': best_valid_loss,
        'best_step': best_step,
        'stopped_early': stopped_early,
        'total_global_steps': global_step,
        'optimizer_steps_per_epoch': optimizer_steps_per_epoch,
        'planned_optimizer_steps': total_steps,
        'train_chunks': len(train_ds),
        'valid_chunks': len(valid_ds),
        'micro_batch_size': args.batch_size,
        'eval_batch_size': args.eval_batch_size,
        'gradient_accumulation_steps': args.grad_accum,
        'effective_batch_size': args.batch_size * args.grad_accum,
        'seed': args.seed,
        'generation_checkpoint_step': best_step,
        'epochs': args.epochs,
        'lora_r': args.lora_r,
        'lora_alpha': args.lora_alpha,
        'lr': args.lr,
        'model_id': args.model_id,
    }
    with open(os.path.join(OUTPUT_DIR, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {os.path.join(OUTPUT_DIR, 'metrics.json')}")

print(f"\n{'='*50}")
print("Generating comparison samples")
print(f"{'='*50}")

PROMPTS = [
    "The history of artificial intelligence begins with",
    "In recent years, researchers have discovered that",
    "The fundamental principle behind",
    "A major challenge in the field is",
    "One of the most important developments has been",
]

print("\n--- Base Model ---")
model.disable_adapter_layers()
base_samples = generate_samples(model, tokenizer, PROMPTS)
for sample in base_samples:
    print(f"\nPrompt: {sample['prompt']} [sample {sample['sample_index']}, seed {sample['seed']}]")
    print(f"Base: {sample['generation']}")

print("\n--- Fine-tuned Model ---")
model.enable_adapter_layers()
finetuned_samples = generate_samples(model, tokenizer, PROMPTS)
for sample in finetuned_samples:
    print(f"\nPrompt: {sample['prompt']} [sample {sample['sample_index']}, seed {sample['seed']}]")
    print(f"FT: {sample['generation']}")

with open(os.path.join(OUTPUT_DIR, 'generation_samples.txt'), 'w', encoding='utf-8') as f:
    f.write("=== Base Model Generations ===\n\n")
    for sample in base_samples:
        f.write(f"Prompt: {sample['prompt']}\nSeed: {sample['seed']}\n"
                f"Generated: {sample['generation']}\n\n")
    f.write("\n=== Fine-tuned Model Generations ===\n\n")
    for sample in finetuned_samples:
        f.write(f"Prompt: {sample['prompt']}\nSeed: {sample['seed']}\n"
                f"Generated: {sample['generation']}\n\n")

def summarize_generations(samples):
    """Count transparent format markers; these are not quality scores."""
    texts = [sample['generation'].lower() for sample in samples]
    exam_pattern = re.compile(r'\banswer\s*:|\na[.)].*\nb[.)]', re.DOTALL)
    abstract_markers = ('in this paper', 'in this work', 'we present',
                        'we propose', 'our results', 'we study')
    return {
        'n_samples': len(texts),
        'exam_format_count': sum(bool(exam_pattern.search(text)) for text in texts),
        'abstract_marker_count': sum(
            any(marker in text for marker in abstract_markers) for text in texts
        ),
        'mean_word_count': sum(len(text.split()) for text in texts) / max(len(texts), 1),
    }


generation_record = {
    'config': {
        'seed': args.seed,
        'samples_per_prompt': args.samples_per_prompt,
        'max_new_tokens': args.max_new_tokens,
        'temperature': args.temperature,
        'top_p': args.top_p,
        'do_sample': True,
    },
    'base': base_samples,
    'fine_tuned': finetuned_samples,
    'diagnostics': {
        'base': summarize_generations(base_samples),
        'fine_tuned': summarize_generations(finetuned_samples),
    },
}
with open(os.path.join(OUTPUT_DIR, 'generation_samples.json'), 'w', encoding='utf-8') as f:
    json.dump(generation_record, f, indent=2, ensure_ascii=False)

print(f"\nSamples saved to {os.path.join(OUTPUT_DIR, 'generation_samples.txt')}")
print("Done!")
