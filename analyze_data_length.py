#!/usr/bin/env python3
"""
Analyze sequence lengths in the parquet dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
import seaborn as sns

def analyze_sequence_lengths(
    data_file: str = "final_prompt_output_pairs.parquet",
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    max_length_threshold: int = 2048
):
    """
    Analyze sequence lengths in the dataset
    
    Args:
        data_file: Path to parquet file
        model_name: Model name for tokenizer
        max_length_threshold: Maximum length threshold to check
    """
    
    print(f"Loading data from {data_file}...")
    df = pd.read_parquet(data_file)
    print(f"Loaded {len(df)} samples")
    
    print(f"Loading tokenizer from {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print("Using approximation based on character count...")
        tokenizer = None
    
    # Analyze lengths
    prompt_lengths = []
    output_lengths = []
    total_lengths = []
    
    print("Analyzing sequence lengths...")
    
    for i, row in df.iterrows():
        prompt = str(row['prompt'])
        output = str(row['output'])
        
        if tokenizer:
            # Tokenize prompt and output
            prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
            output_tokens = len(tokenizer.encode(output, add_special_tokens=False))
            total_tokens = prompt_tokens + output_tokens + 3  # Add special tokens
        else:
            # Rough approximation: 1 token ≈ 3-4 characters for Chinese/English mixed text
            prompt_tokens = len(prompt) // 3
            output_tokens = len(output) // 3
            total_tokens = prompt_tokens + output_tokens
        
        prompt_lengths.append(prompt_tokens)
        output_lengths.append(output_tokens)
        total_lengths.append(total_tokens)
        
        # Progress indicator
        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1}/{len(df)} samples")
    
    # Add lengths to dataframe
    df['prompt_length'] = prompt_lengths
    df['output_length'] = output_lengths
    df['total_length'] = total_lengths
    
    # Statistics
    print(f"\n{'='*60}")
    print("SEQUENCE LENGTH ANALYSIS")
    print('='*60)
    
    print(f"\nPrompt lengths:")
    print(f"  Mean: {np.mean(prompt_lengths):.1f}")
    print(f"  Median: {np.median(prompt_lengths):.1f}")
    print(f"  Min: {np.min(prompt_lengths)}")
    print(f"  Max: {np.max(prompt_lengths)}")
    print(f"  95th percentile: {np.percentile(prompt_lengths, 95):.1f}")
    print(f"  99th percentile: {np.percentile(prompt_lengths, 99):.1f}")
    
    print(f"\nOutput lengths:")
    print(f"  Mean: {np.mean(output_lengths):.1f}")
    print(f"  Median: {np.median(output_lengths):.1f}")
    print(f"  Min: {np.min(output_lengths)}")
    print(f"  Max: {np.max(output_lengths)}")
    print(f"  95th percentile: {np.percentile(output_lengths, 95):.1f}")
    print(f"  99th percentile: {np.percentile(output_lengths, 99):.1f}")
    
    print(f"\nTotal lengths (prompt + output):")
    print(f"  Mean: {np.mean(total_lengths):.1f}")
    print(f"  Median: {np.median(total_lengths):.1f}")
    print(f"  Min: {np.min(total_lengths)}")
    print(f"  Max: {np.max(total_lengths)}")
    print(f"  95th percentile: {np.percentile(total_lengths, 95):.1f}")
    print(f"  99th percentile: {np.percentile(total_lengths, 99):.1f}")
    
    # Check different length thresholds
    thresholds = [512, 1024, 1536, 2048, 3072, 4096]
    print(f"\nSamples within different length thresholds:")
    for threshold in thresholds:
        within_threshold = sum(1 for length in total_lengths if length <= threshold)
        percentage = (within_threshold / len(total_lengths)) * 100
        print(f"  ≤ {threshold}: {within_threshold}/{len(total_lengths)} ({percentage:.1f}%)")
    
    # Find problematic samples
    long_samples = df[df['total_length'] > max_length_threshold]
    print(f"\nSamples exceeding {max_length_threshold} tokens: {len(long_samples)}")
    
    if len(long_samples) > 0:
        print(f"\nTop 5 longest samples:")
        top_long = long_samples.nlargest(5, 'total_length')
        for i, (idx, row) in enumerate(top_long.iterrows()):
            print(f"  {i+1}. Index {idx}: {row['total_length']} tokens")
            print(f"     Prompt: {row['prompt_length']} tokens")
            print(f"     Output: {row['output_length']} tokens")
            print(f"     Preview: {str(row['output'])[:100]}...")
            print()
    
    # Recommendations
    print(f"\n{'='*60}")
    print("RECOMMENDATIONS")
    print('='*60)
    
    # Find optimal max_length
    percentiles = [90, 95, 98, 99, 99.5]
    print(f"\nSuggested max_length based on percentiles:")
    for p in percentiles:
        suggested_length = int(np.percentile(total_lengths, p))
        within_threshold = sum(1 for length in total_lengths if length <= suggested_length)
        percentage = (within_threshold / len(total_lengths)) * 100
        print(f"  {p}th percentile: {suggested_length} (keeps {percentage:.1f}% of data)")
    
    # Memory considerations
    max_length_suggestions = [1536, 2048, 3072, 4096]
    print(f"\nMemory impact of different max_length settings:")
    base_memory = 1024  # Baseline
    for length in max_length_suggestions:
        relative_memory = (length / base_memory) ** 2  # Quadratic relationship
        within_threshold = sum(1 for l in total_lengths if l <= length)
        percentage = (within_threshold / len(total_lengths)) * 100
        print(f"  max_length={length}: ~{relative_memory:.1f}x memory, keeps {percentage:.1f}% data")
    
    return df

def filter_long_sequences(
    input_file: str = "final_prompt_output_pairs.parquet",
    output_file: str = "filtered_prompt_output_pairs.parquet",
    max_length: int = 2048,
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
):
    """
    Filter out sequences that are too long
    
    Args:
        input_file: Input parquet file
        output_file: Output parquet file
        max_length: Maximum allowed sequence length
        model_name: Model name for tokenizer
    """
    
    print(f"Filtering sequences longer than {max_length} tokens...")
    
    # Analyze first
    df = analyze_sequence_lengths(input_file, model_name, max_length)
    
    # Filter
    filtered_df = df[df['total_length'] <= max_length].copy()
    
    # Remove length columns for final dataset
    filtered_df = filtered_df[['prompt', 'output']]
    
    # Save filtered dataset
    filtered_df.to_parquet(output_file, index=False)
    
    print(f"\nFiltered dataset saved to {output_file}")
    print(f"Original samples: {len(df)}")
    print(f"Filtered samples: {len(filtered_df)}")
    print(f"Removed samples: {len(df) - len(filtered_df)}")
    print(f"Retention rate: {len(filtered_df)/len(df)*100:.1f}%")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze sequence lengths in dataset")
    parser.add_argument("--data_file", default="final_prompt_output_pairs.parquet", help="Data file to analyze")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-7B-Instruct", help="Model name for tokenizer")
    parser.add_argument("--max_length", type=int, default=2048, help="Max length threshold")
    parser.add_argument("--filter", action="store_true", help="Filter long sequences")
    parser.add_argument("--output_file", default="filtered_prompt_output_pairs.parquet", help="Output file for filtered data")
    
    args = parser.parse_args()
    
    if args.filter:
        filter_long_sequences(args.data_file, args.output_file, args.max_length, args.model_name)
    else:
        analyze_sequence_lengths(args.data_file, args.model_name, args.max_length) 