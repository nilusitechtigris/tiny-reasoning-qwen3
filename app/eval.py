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
    python eval.py --model outputs/qwen3-reasoning-1.7b --limit 100    # quick check
    python eval.py --model outputs/qwen3-reasoning-1.7b                 # full run
    python eval.py --skip-apis --model outputs/qwen3-reasoning-1.7b     # local-only
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
    """Pull the answer out of \\boxed{...}, falling back to the last number."""
    matches = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if matches:
        return matches[-1].strip()
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


# ---------- Generators ----------

def build_messages(problem):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem},
    ]


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
    """Load a model with vLLM and return a generator function."""
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    llm = LLM(model=model_path, dtype="bfloat16", max_model_len=max_model_len,
              gpu_memory_utilization=0.85)
    tok = AutoTokenizer.from_pretrained(model_path)
    sampling = SamplingParams(max_tokens=2048, temperature=0.0)

    def gen(problem):
        text = tok.apply_chat_template(
            build_messages(problem), tokenize=False, add_generation_prompt=True
        )
        out = llm.generate([text], sampling, use_tqdm=False)
        return out[0].outputs[0].text
    return gen, llm  # return llm so we can free it later


# ---------- Eval loop ----------

def evaluate(gen, problems, label):
    correct = 0
    for p in tqdm(problems, desc=label):
        try:
            response = gen(p["problem"])
            pred = extract_boxed(response)
            if is_correct(pred, p["answer"]):
                correct += 1
        except Exception as e:
            print(f"\n  error on problem: {e}")
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
        del llm  # free VRAM before loading base

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