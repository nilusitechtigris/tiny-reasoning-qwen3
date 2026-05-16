"""Download and prep OpenR1-Math-220k for math reasoning SFT.

Outputs a JSONL file with one {"text": "..."} record per training example,
ready for Unsloth's SFTTrainer.

Usage:
    python prepare_data.py --inspect        # print first raw row, exit
    python prepare_data.py                  # default: 15k examples from curated subset
    python prepare_data.py --max-examples 5000
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset

# OpenR1-Math-220k uses <think>...</think> for reasoning and \boxed{} for answers
# (verified by their Math Verify pipeline). We match that format.
SYSTEM_PROMPT = (
    "You are a careful reasoner. Think step by step inside <think>...</think> "
    "tags, then give the final answer in \\boxed{}."
)


def format_example(row):
    """Convert one OpenR1-Math row into a training string.

    OpenR1-Math-220k typically has:
      - 'problem'    : the math problem text
      - 'solution'   : original NuminaMath solution (not the R1 trace)
      - 'messages'   : list of {role, content} including R1's <think>...</think> reasoning
      - 'generations': sometimes a list of raw R1 generations
    """
    problem = row.get("problem", "")

    # Prefer the R1 reasoning trace from the messages column
    completion = ""
    if "messages" in row and isinstance(row["messages"], list) and row["messages"]:
        for msg in row["messages"]:
            if msg.get("role") == "assistant":
                completion = msg.get("content", "")
                break
    elif "generations" in row and row["generations"]:
        gens = row["generations"]
        completion = gens[0] if isinstance(gens, list) else str(gens)
    elif "solution" in row:
        completion = row.get("solution", "")

    if not problem or not completion:
        return None

    text = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{problem}<|im_end|>\n"
        f"<|im_start|>assistant\n{completion}<|im_end|>"
    )
    return {"text": text}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="open-r1/OpenR1-Math-220k")
    ap.add_argument("--subset", default="default",
                    help="'default' (94k, curated, best for SFT) or 'extended' (131k more).")
    ap.add_argument("--max-examples", type=int, default=15000)
    ap.add_argument("--max-length-chars", type=int, default=12000,
                    help="Drop examples whose formatted text exceeds this (proxy for token length).")
    ap.add_argument("--output", default="data/train.jsonl")
    ap.add_argument("--inspect", action="store_true",
                    help="Print first raw row's columns and a content snippet, then exit.")
    args = ap.parse_args()

    print(f"Loading {args.dataset} ({args.subset}) ...")
    ds = load_dataset(args.dataset, args.subset, split="train")
    print(f"Initial size: {len(ds):,}")
    print(f"Columns: {ds.column_names}")

    if args.inspect:
        print("\n=== First raw row ===")
        row = ds[0]
        for k, v in row.items():
            if isinstance(v, (list, dict)):
                length = len(v) if hasattr(v, "__len__") else "?"
                print(f"\n{k} (type={type(v).__name__}, len={length}):")
                print(str(v)[:1200])
            else:
                print(f"\n{k}: {str(v)[:1200]}")
        return

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