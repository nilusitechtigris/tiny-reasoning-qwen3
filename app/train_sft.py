"""LoRA SFT on Qwen3-1.7B-Base using Unsloth.

Expects: data/train.jsonl produced by prepare_data.py
Outputs: outputs/qwen3-reasoning-1.7b/ (LoRA adapter + tokenizer)

Usage:
    python train_sft.py --smoke           # 50-example sanity check first!
    python train_sft.py                   # full run, ~2-4h on a single A40/4090
    python train_sft.py --learning-rate 1e-4 --epochs 3   # iterate hyperparams
"""

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel


def load_jsonl_dataset(path):
    examples = []
    with open(path) as f:
        for line in f:
            examples.append(json.loads(line))
    return Dataset.from_list(examples)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="Qwen/Qwen3-1.7B-Base")
    ap.add_argument("--data", default="data/train.jsonl")
    ap.add_argument("--output", default="outputs/qwen3-reasoning-1.7b")
    ap.add_argument("--max-seq-length", type=int, default=4096)

    # LoRA
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)

    # Optimization
    ap.add_argument("--learning-rate", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--warmup-steps", type=int, default=50)

    ap.add_argument("--smoke", action="store_true",
                    help="Train on 50 examples for 1 epoch — sanity check the pipeline before the real run.")
    args = ap.parse_args()

    # ---- Load base model (4-bit) + attach LoRA adapters ----
    print(f"Loading {args.base_model} in 4-bit...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        dtype=None,  # auto: bf16 on Ampere+, fp16 otherwise
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        use_gradient_checkpointing="unsloth",  # ~30% VRAM savings
        random_state=42,
    )

    # ---- Load dataset ----
    dataset = load_jsonl_dataset(args.data)
    if args.smoke:
        dataset = dataset.select(range(min(50, len(dataset))))
        print(f"\n[SMOKE MODE] Training on {len(dataset)} examples for 1 epoch.")

    print(f"\nTraining on {len(dataset):,} examples")
    effective_batch = args.batch_size * args.grad_accum
    print(f"Effective batch size: {effective_batch}")
    steps_per_epoch = max(1, len(dataset) // effective_batch)
    print(f"Approx steps per epoch: {steps_per_epoch}")

    # ---- Trainer ----
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        args=TrainingArguments(
            output_dir=args.output,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            warmup_steps=args.warmup_steps,
            num_train_epochs=1 if args.smoke else args.epochs,
            learning_rate=args.learning_rate,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=10,
            save_steps=500,
            save_total_limit=2,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=42,
            report_to="none",  # set to "wandb" if you want live tracking
        ),
    )

    trainer.train()

    # ---- Save final adapter ----
    Path(args.output).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"\n✓ Saved adapter to {args.output}")
    print(f"\nNext: python eval.py --model {args.output} --limit 100   # quick eval first")


if __name__ == "__main__":
    main()