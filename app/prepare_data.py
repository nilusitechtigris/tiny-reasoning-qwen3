"""Download and filter OpenThoughts-114k for math reasoning SFT.

Outputs a JSONL file with one {"text": "..."} record per training example,
ready for Unsloth's SFTTrainer.

Usage:
    python prepare_data.py                  # default: 15k math examples
    python prepare_data.py --inspect        # print first raw row, exit
    python prepare_data.py --max-examples 5000 --domain-filter math
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset

# Wraps each example in Qwen3's chatml template. The base model has never seen
# this format, so SFT will teach it: "when you see this template, produce
# <think>...</think> reasoning then a final boxed answer."
SYSTEM_PROMPT = (
    "You are a careful reasoner. Think step by step inside <think>...</think> "
    "tags, then give the final answer in \\boxed{}."
)


def format_example(row):
    """Convert one OpenThoughts row into a single training string.

    OpenThoughts-114k uses a `conversations` list of {from, value} dicts.
    Fallbacks handle related datasets if you swap in a different one.
    """
    if "conversations" in row and isinstance(row["conversations"], list):
        convs = row["conversations"]
        user_msgs = [c.get("value", "") for c in convs if c.get("from") in ("human", "user")]
        asst_msgs = [c.get("value", "") for c in convs if c.get("from") in ("gpt", "assistant")]
        problem = user_msgs[0] if user_msgs else ""
        solution = asst_msgs[0] if asst_msgs else ""
    else:
        problem = row.get("problem") or row.get("question") or ""
        solution = row.get("solution") or row.get("response") or ""

    if not problem or not solution:
        return None

    text = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{problem}<|im_end|>\n"
        f"<|im_start|>assistant\n{solution}<|im_end|>"
    )
    return {"text": text}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="open-thoughts/OpenThoughts-114k")
    ap.add_argument("--domain-filter", default="math",
                    help="Substring to match against the 'domain' column. Empty string = no filter.")
    ap.add_argument("--max-examples", type=int, default=15000)
    ap.add_argument("--max-length-chars", type=int, default=12000,
                    help="Drop examples whose formatted text exceeds this (proxy for token length).")
    ap.add_argument("--output", default="data/train.jsonl")
    ap.add_argument("--inspect", action="store_true",
                    help="Print the first raw row from the dataset and exit (for schema sanity-check).")
    args = ap.parse_args()

    print(f"Loading {args.dataset} ...")
    ds = load_dataset(args.dataset, split="train")
    print(f"Initial size: {len(ds):,}")
    print(f"Columns: {ds.column_names}")

    if args.inspect:
        print("\n=== First raw row ===")
        print(json.dumps(ds[0], indent=2, default=str)[:2000])
        return

    # Optional domain filter (OpenThoughts has a `domain` column with values like "math", "code", "science")
    if args.domain_filter and "domain" in ds.column_names:
        before = len(ds)
        ds = ds.filter(lambda r: args.domain_filter.lower() in str(r.get("domain", "")).lower())
        print(f"After domain filter ('{args.domain_filter}'): {len(ds):,} (was {before:,})")

    # Format + length filter + cap
    formatted = []
    skipped_empty = 0
    skipped_long = 0
    for row in ds:
        ex = format_example(row)
        if ex is None:
            skipped_empty += 1
            continue
        if len(ex["text"]) > args.max_length_chars:
            skipped_long += 1
            continue
        formatted.append(ex)
        if len(formatted) >= args.max_examples:
            break

    print(f"\nKept:    {len(formatted):,}")
    print(f"Skipped (empty fields): {skipped_empty:,}")
    print(f"Skipped (too long):     {skipped_long:,}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for ex in formatted:
            f.write(json.dumps(ex) + "\n")
    print(f"\nWrote {len(formatted):,} examples to {out}")

    print("\n=== Preview of first formatted example ===")
    print(formatted[0]["text"][:800] + ("..." if len(formatted[0]["text"]) > 800 else ""))


if __name__ == "__main__":
    main()