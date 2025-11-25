#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Detokenization.py
-----------------
Detokenize tokenized Indonesian YouTube comments.

Input  : CSV with a `text` column containing token lists (string repr).
Output : Clean CSV with:
         - tokens (list)
         - token_len (int)
         - text_raw (detokenized full text)
         - text_trunc (detokenized, truncated to max_tokens)

Default paths (relative to project root):
- input : data/vibe_coding_yt_comments_clean.csv
- output: data/vibe_coding_dataset_ready.csv

Usage:
python model/Detokenization.py \
    --input data/vibe_coding_yt_comments_clean.csv \
    --output data/vibe_coding_dataset_ready.csv \
    --max_tokens 256
"""

import os
import re
import ast
import sys
from pathlib import Path
import argparse
from typing import List, Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.setting import PROJECT_ROOT


def parse_tokens(x: Any) -> List[str]:
    """
    Convert a cell into list of tokens.
    Handles:
    - already-a-list
    - string repr like "['a','b']"
    - NaN / empty
    """
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return []
    s = str(x).strip()
    if not s:
        return []
    try:
        val = ast.literal_eval(s)
        if isinstance(val, list):
            return [str(t).strip() for t in val if str(t).strip()]
        # if it's a single string or something else
        return [str(val).strip()] if str(val).strip() else []
    except Exception:
        # fallback: best-effort parse
        s2 = re.sub(r"^\[|\]$", "", s)
        parts = [p.strip(" '\"") for p in s2.split(",") if p.strip()]
        return parts


def detokenize(tokens: List[str]) -> str:
    """Join tokens into a single normalized string."""
    text = " ".join(tokens)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    parser = argparse.ArgumentParser(description="Detokenize token-list comments CSV.")
    parser.add_argument(
        "--input",
        type=str,
        default=str(PROJECT_ROOT / "data" / "vibe_coding_yt_comments_clean.csv"),
        help="Path to input CSV (with tokenized `text` column).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "data" / "vibe_coding_dataset_ready.csv"),
        help="Path to output CSV.",
    )
    parser.add_argument(
        "--text_col",
        type=str,
        default="text",
        help="Name of column containing tokens.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=256,
        help="Max tokens to keep for truncated text.",
    )
    args = parser.parse_args()

    in_path = args.input
    out_path = args.output

    if not os.path.exists(in_path):
        raise FileNotFoundError(f"Input file not found: {in_path}")

    # Load
    df = pd.read_csv(in_path)

    if args.text_col not in df.columns:
        raise ValueError(f"Column `{args.text_col}` not found. Columns: {list(df.columns)}")

    # Parse tokens
    df["tokens"] = df[args.text_col].apply(parse_tokens)
    df["token_len"] = df["tokens"].apply(len)

    # Detokenize full
    df["text_raw"] = df["tokens"].apply(detokenize)

    # Drop empty detokenized rows
    df_clean = df[df["text_raw"].str.len() > 0].copy()

    # Truncate tokens for modeling
    df_clean["tokens_trunc"] = df_clean["tokens"].apply(lambda t: t[: args.max_tokens])
    df_clean["text_trunc"] = df_clean["tokens_trunc"].apply(detokenize)

    # Optional: choose output columns (keep original + new)
    # If you want to keep everything, comment this block out.
    keep_cols = [
        c for c in df_clean.columns
        if c not in ["tokens_trunc"]  # drop helper col
    ]
    df_clean = df_clean[keep_cols]

    # Ensure output dir exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Save
    df_clean.to_csv(out_path, index=False, encoding="utf-8-sig")

    # Simple report
    print("Detokenization complete!")
    print(f"Input rows : {len(df)}")
    print(f"Empty rows : {(df['text_raw'].str.len() == 0).sum()}")
    print(f"Output rows: {len(df_clean)}")
    print(f"Saved to   : {out_path}")


if __name__ == "__main__":
    main()