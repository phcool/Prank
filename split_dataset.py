#!/usr/bin/env python3
"""
Split parquet dataset into training and validation sets
"""

import pandas as pd
import os
from sklearn.model_selection import train_test_split
import argparse

def split_parquet_dataset(
    input_file: str = "final_prompt_output_pairs.parquet",
    train_file: str = "train_data.parquet", 
    val_file: str = "val_data.parquet",
    test_size: float = 0.1,
    random_state: int = 42
):
    """
    Split parquet dataset into training and validation sets
    
    Args:
        input_file: Path to input parquet file
        train_file: Path to output training parquet file
        val_file: Path to output validation parquet file
        test_size: Fraction of data to use for validation (0.1 = 10%)
        random_state: Random seed for reproducible splits
    """
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found!")
        print("Available files in current directory:")
        for f in os.listdir("."):
            if f.endswith(".parquet"):
                print(f"  - {f}")
        return False
    
    print(f"Loading dataset from {input_file}...")
    
    try:
        # Load the dataset
        df = pd.read_parquet(input_file)
        print(f"Loaded dataset with {len(df)} samples")
        print(f"Columns: {list(df.columns)}")
        
        # Display basic statistics
        print(f"\nDataset statistics:")
        print(f"  Total samples: {len(df)}")
        if 'prompt' in df.columns:
            avg_prompt_len = df['prompt'].apply(lambda x: len(str(x))).mean()
            print(f"  Average prompt length: {avg_prompt_len:.0f} characters")
        if 'output' in df.columns:
            avg_output_len = df['output'].apply(len).mean()
            print(f"  Average output length: {avg_output_len:.0f} characters")
        
        # Split the dataset
        print(f"\nSplitting dataset...")
        print(f"  Training set: {100 * (1 - test_size):.1f}%")
        print(f"  Validation set: {100 * test_size:.1f}%")
        
        train_df, val_df = train_test_split(
            df, 
            test_size=test_size, 
            random_state=random_state,
            shuffle=True
        )
        
        print(f"\nSplit results:")
        print(f"  Training samples: {len(train_df)}")
        print(f"  Validation samples: {len(val_df)}")
        
        # Save the split datasets
        print(f"\nSaving split datasets...")
        train_df.to_parquet(train_file, index=False)
        print(f"  Training set saved to: {train_file}")
        
        val_df.to_parquet(val_file, index=False)
        print(f"  Validation set saved to: {val_file}")
        
        # Verify the saved files
        print(f"\nVerification:")
        train_size = os.path.getsize(train_file) / (1024 * 1024)  # MB
        val_size = os.path.getsize(val_file) / (1024 * 1024)      # MB
        print(f"  {train_file}: {train_size:.2f} MB")
        print(f"  {val_file}: {val_size:.2f} MB")
        
        # Display sample data from each split
        print(f"\n" + "="*60)
        print("SAMPLE FROM TRAINING SET:")
        print("="*60)
        if len(train_df) > 0:
            sample_idx = 0
            if 'prompt' in train_df.columns:
                sample_prompt = str(train_df.iloc[sample_idx]['prompt'])
                print(f"Sample prompt: {sample_prompt[:200]}...")
            if 'output' in train_df.columns:
                sample_output = train_df.iloc[sample_idx]['output']
                print(f"Sample output: {sample_output[:200]}...")
        
        print(f"\n" + "="*60)
        print("SAMPLE FROM VALIDATION SET:")
        print("="*60)
        if len(val_df) > 0:
            sample_idx = 0
            if 'prompt' in val_df.columns:
                sample_prompt = str(val_df.iloc[sample_idx]['prompt'])
                print(f"Sample prompt: {sample_prompt[:200]}...")
            if 'output' in val_df.columns:
                sample_output = val_df.iloc[sample_idx]['output']
                print(f"Sample output: {sample_output[:200]}...")
        
        print(f"\n" + "="*60)
        print("SUCCESS! Dataset split completed.")
        print("You can now use these files in your training script:")
        print(f"  data.train_files={train_file}")
        print(f"  data.val_files={val_file}")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"Error processing dataset: {e}")
        return False

def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(description="Split parquet dataset into train/val sets")
    
    parser.add_argument(
        "--input_file",
        type=str,
        default="final_prompt_output_pairs.parquet",
        help="Input parquet file path"
    )
    
    parser.add_argument(
        "--train_file", 
        type=str,
        default="train_data.parquet",
        help="Output training set file path"
    )
    
    parser.add_argument(
        "--val_file",
        type=str, 
        default="val_data.parquet",
        help="Output validation set file path"
    )
    
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Validation set ratio (default: 0.1 for 10%%)"
    )
    
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not 0 < args.val_ratio < 1:
        print("Error: val_ratio must be between 0 and 1")
        return
    
    print("Dataset Splitting Configuration:")
    print(f"  Input file: {args.input_file}")
    print(f"  Training file: {args.train_file}")
    print(f"  Validation file: {args.val_file}")
    print(f"  Validation ratio: {args.val_ratio} ({args.val_ratio*100:.1f}%)")
    print(f"  Random seed: {args.random_seed}")
    print()
    
    # Perform the split
    success = split_parquet_dataset(
        input_file=args.input_file,
        train_file=args.train_file,
        val_file=args.val_file,
        test_size=args.val_ratio,
        random_state=args.random_seed
    )
    
    if not success:
        exit(1)

if __name__ == "__main__":
    main() 