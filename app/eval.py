"""Evaluate trained model + baselines on GSM8K and MATH-500.

Models compared:
  - Your trained model (LoRA adapter or merged checkpoint)
  - Qwen3-1.7B-Base                 (control — same family, no training)
  - Llama-3.1-8B-Base                (via Together API)
  - Llama-3.1-70B-Base               (via Together API)
  - GPT-4o-mini                      (via OpenAI API)

Set env vars before running:
    export TOGETHER_API_KEY=...
    export OPENAI_API_KEY=...
(Either can be omitted — that baseline is skipped.)

Outputs: results/results.csv

Usage:
    python eval.py --model outputs/qwen3-reasoning-1.7b-merged --limit 100    # quick check
    python eval.py --model outputs/qwen3-reasoning-1.7b-merged                # full run
    python eval.py --skip-apis --model outputs/qwen3-reasoning-1.7b-merged    # local-only
"""

import argparse
import csv
import os
import re
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


SYSTEM_PROMPT = (
    "You are a careful reasoner. Think step by step inside <think>...</think> "
    "tags, then give the final answer in \\boxed{}."
)


# ---------- Answer extraction & comparison ----------

def extract_boxed(text):
    """Pull the FIRST \\boxed{...} from text.

    R1-style models often produce the real answer once, then ramble or loop
    after. Taking the first match captures the model's actual answer and
    ignores any post-answer noise.
    """
    matches = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if matches:
        return matches[0].strip()
    # Fallback: last number in text (handles models that don't use \boxed)
    nums = re.findall(r"-?\d+\.?\d*", text)
    return nums[-1] if nums else ""


def normalize(s):
    s = str(s).strip().replace(",", "").replace("$", "").rstrip(".")
    try:
        return str(float(s))
    except ValueError:
        return s.lower()


def is_correct(pred, gold):
    return normalize(pred) == normalize(gold)


# ---------- Benchmarks ----------

def load_gsm8k(limit=None):
    ds = load_dataset("gsm8k", "main", split="test")
    problems = [{
        "problem": row["question"],
        "answer": row["answer"].split("####")[-1].strip(),
    } for row in ds]
    return problems[:limit] if limit else problems


def load_math500(limit=None):
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    problems = [{
        "problem": row["problem"],
        "answer": row["answer"],
    } for row in ds]
    return problems[:limit] if limit else problems


# ---------- Prompt formatting ----------

def build_chatml_prompt(problem):
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{problem}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def build_messages(problem):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem},
    ]


# ---------- Generators ----------

def make_openai_compat_gen(model_name, base_url, api_key):
    from openai import OpenAI
    client = OpenAI(base_url=base_url, api_key=api_key)

    def gen(problem):
        resp = client.chat.completions.create(
            model=model_name,
            messages=build_messages(problem),
            max_tokens=2048,
            temperature=0.0,
        )
        return resp.choices[0].message.content or ""
    return gen


def make_local_gen(model_path, max_model_len=4096):
    """Load a model with vLLM, return a generator that builds prompts manually."""
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    llm = LLM(model=model_path, dtype="bfloat16", max_model_len=max_model_len,
              gpu_memory_utilization=0.85)

    # Resolve stop token IDs from the tokenizer — works even when the
    # tokenizer's eos_token isn't properly set after a merge.
    tok = AutoTokenizer.from_pretrained(model_path)
    stop_ids = []
    try:
        im_end_id = tok.convert_tokens_to_ids("<|im_end|>")
        if im_end_id is not None and im_end_id != tok.unk_token_id:
            stop_ids.append(im_end_id)
    except Exception:
        pass
    if tok.eos_token_id is not None and tok.eos_token_id not in stop_ids:
        stop_ids.append(tok.eos_token_id)

    sampling = SamplingParams(
        max_tokens=3500,           # bumped from 2048 — give harder problems room
        temperature=0.0,
        stop=["<|im_end|>"],       # string match (catches if model emits as text)
        stop_token_ids=stop_ids or None,  # token-id match (catches when emitted as a token)
    )

    def gen(problem):
        prompt = build_chatml_prompt(problem)
        out = llm.generate([prompt], sampling, use_tqdm=False)
        return out[0].outputs[0].text
    return gen, llm


# ---------- Eval loop ----------

def evaluate(gen, problems, label):
    correct = 0
    errors = 0
    for p in tqdm(problems, desc=label):
        try:
            response = gen(p["problem"])
            pred = extract_boxed(response)
            if is_correct(pred, p["answer"]):
                correct += 1
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"\n  error on problem ({errors}): {e}")
    if errors:
        print(f"  total errors in {label}: {errors}/{len(problems)}")
    return correct / len(problems) if problems else 0.0


# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="Path to your trained model directory")
    ap.add_argument("--base", default="Qwen/Qwen3-1.7B-Base",
                    help="Control model. Pass empty string '' to skip.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Limit problems per benchmark. Use 100 for quick iteration, None for full run.")
    ap.add_argument("--skip-apis", action="store_true",
                    help="Skip Together + OpenAI baselines (e.g. when iterating).")
    ap.add_argument("--output", default="results/results.csv")
    args = ap.parse_args()

    gsm8k = load_gsm8k(args.limit)
    math500 = load_math500(args.limit)
    print(f"GSM8K:    {len(gsm8k):,} problems")
    print(f"MATH-500: {len(math500):,} problems")

    results = []

    def run_pair(label, gen):
        results.append({"model": label, "benchmark": "GSM8K",
                        "accuracy": evaluate(gen, gsm8k, f"{label}/gsm8k")})
        results.append({"model": label, "benchmark": "MATH-500",
                        "accuracy": evaluate(gen, math500, f"{label}/math500")})

    # 1) Your trained model
    if args.model:
        print(f"\n--- Loading trained model: {args.model} ---")
        gen, llm = make_local_gen(args.model)
        run_pair("qwen3-1.7b-reasoning (ours)", gen)
        del llm

    # 2) Base control
    if args.base:
        print(f"\n--- Loading base control: {args.base} ---")
        gen, llm = make_local_gen(args.base)
        run_pair("qwen3-1.7b-base (control)", gen)
        del llm

    # 3) API baselines
    if not args.skip_apis:
        together_key = os.environ.get("TOGETHER_API_KEY")
        openai_key = os.environ.get("OPENAI_API_KEY")

        if together_key:
            print("\n--- Llama baselines via Together ---")
            for model_id, label in [
                ("meta-llama/Meta-Llama-3.1-8B", "llama-3.1-8b-base"),
                ("meta-llama/Meta-Llama-3.1-70B", "llama-3.1-70b-base"),
            ]:
                gen = make_openai_compat_gen(
                    model_id, "https://api.together.xyz/v1", together_key
                )
                run_pair(label, gen)
        else:
            print("\nTOGETHER_API_KEY not set — skipping Llama baselines")

        if openai_key:
            print("\n--- GPT-4o-mini via OpenAI ---")
            gen = make_openai_compat_gen("gpt-4o-mini", None, openai_key)
            run_pair("gpt-4o-mini", gen)
        else:
            print("OPENAI_API_KEY not set — skipping GPT-4o-mini")

    # ---- Write CSV + print summary ----
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "benchmark", "accuracy"])
        writer.writeheader()
        for r in results:
            writer.writerow({**r, "accuracy": f"{r['accuracy']:.4f}"})

    print("\n" + "=" * 60)
    print(f"{'Model':<35} {'Benchmark':<12} {'Accuracy':<8}")
    print("-" * 60)
    for r in results:
        print(f"{r['model']:<35} {r['benchmark']:<12} {r['accuracy']:.4f}")
    print("=" * 60)
    print(f"\n✓ Written to {out}")


if __name__ == "__main__":
    main()