#!/usr/bin/env python3
"""
Shuffle a large JSONL file without loading everything into memory.

This script performs a deterministic shuffle by:
1. Creating an index of line offsets
2. Shuffling the index
3. Reading lines in shuffled order

Usage:
    python scripts/shuffle_jsonl.py \\
        --input data/piles_packed_32768.jsonl \\
        --output data/piles_packed_32768_shuffled.jsonl \\
        --seed 42
"""
import json
import os
import sys
import time
import random
from typing import List, Tuple


def build_line_offsets(input_path: str) -> List[int]:
    """
    Build an index of line offsets for the input file.
    
    Args:
        input_path: Path to input JSONL file
        
    Returns:
        List of byte offsets for each line
    """
    print(f"📋 Building line offsets for {input_path}...", flush=True)
    offsets = []
    start_time = time.time()
    
    with open(input_path, 'r', encoding='utf-8') as f:
        offset = f.tell()
        line = f.readline()
        while line:
            offsets.append(offset)
            offset = f.tell()
            line = f.readline()
    
    elapsed = time.time() - start_time
    print(f"✅ Built {len(offsets):,} line offsets in {elapsed:.0f} seconds", flush=True)
    return offsets


def shuffle_offsets(offsets: List[int], seed: int) -> List[Tuple[int, int]]:
    """
    Shuffle line offsets deterministically and assign new positions.
    
    Args:
        offsets: List of original byte offsets
        seed: Random seed for deterministic shuffling
        
    Returns:
        List of (original_position, original_offset) tuples in shuffled order
    """
    print(f"🔀 Shuffling offsets with seed {seed}...", flush=True)
    
    # Create list of (position, offset) pairs
    indexed_offsets = list(enumerate(offsets))
    
    # Deterministic shuffle
    random.Random(seed).shuffle(indexed_offsets)
    
    print(f"✅ Shuffled {len(indexed_offsets):,} offsets", flush=True)
    return indexed_offsets


def write_shuffled_data(
    input_path: str, 
    output_path: str, 
    shuffled_offsets: List[Tuple[int, int]]
):
    """
    Write shuffled data to output file.
    
    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file
        shuffled_offsets: List of (original_position, original_offset) in shuffled order
    """
    print(f"📝 Writing shuffled data to {output_path}...", flush=True)
    start_time = time.time()
    
    # Open input file for reading
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        
        # Write lines in shuffled order
        for i, (original_pos, offset) in enumerate(shuffled_offsets):
            # Seek to the original offset
            fin.seek(offset)
            # Read the line
            line = fin.readline()
            # Write to output
            fout.write(line)
            
            # Report progress
            if (i + 1) % 10000 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                print(f"  Written {i+1:,} lines, {rate:.0f} lines/sec", flush=True)
    
    elapsed = time.time() - start_time
    total_lines = len(shuffled_offsets)
    print(f"✅ Wrote {total_lines:,} lines in {elapsed:.0f} seconds", flush=True)


def shuffle_jsonl(input_path: str, output_path: str, seed: int = 42):
    """
    Main function to shuffle a JSONL file.
    
    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file
        seed: Random seed for deterministic shuffling
    """
    # Check if output already exists
    if os.path.exists(output_path):
        print(f"⚠️  Output file {output_path} already exists. Please remove it first.", flush=True)
        response = input("Continue and overwrite? (y/N): ")
        if response.lower() != 'y':
            print("❌ Aborted.", flush=True)
            return
    
    # Build line offsets
    offsets = build_line_offsets(input_path)
    
    # Shuffle offsets
    shuffled_offsets = shuffle_offsets(offsets, seed)
    
    # Write shuffled data
    write_shuffled_data(input_path, output_path, shuffled_offsets)
    
    print(f"\n🎉 Shuffle complete! Output: {output_path}", flush=True)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Shuffle a large JSONL file without loading everything into memory'
    )
    parser.add_argument('--input', default='data/piles_packed_32768.jsonl',
                        help='Input JSONL path')
    parser.add_argument('--output', default='data/piles_packed_32768_shuffled.jsonl',
                        help='Output JSONL path')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for deterministic shuffling')
    
    args = parser.parse_args()
    
    shuffle_jsonl(args.input, args.output, args.seed)