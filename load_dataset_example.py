#!/usr/bin/env python3
"""
Script to load and display sample data from le723z/rearank_12k dataset
"""

from datasets import load_dataset
import json

def load_and_show_sample():
    """Load the dataset and print a sample data"""
    try:
        print("Loading le723z/rearank_12k dataset from Hugging Face...")
        
        # Load the dataset
        dataset = load_dataset("le723z/rearank_12k")
        
        print(f"Dataset loaded successfully!")
        print(f"Available splits: {list(dataset.keys())}")
        
        # Get the first split (usually 'train')
        split_name = list(dataset.keys())[0]
        data_split = dataset[split_name]
        
        print(f"\nDataset info for '{split_name}' split:")
        print(f"Number of examples: {len(data_split)}")
        print(f"Features: {data_split.features}")
        
        # Print first example
        if len(data_split) > 0:
            print(f"\n=== Sample Data (First Example) ===")
            sample = data_split[0]
            
            # Pretty print the sample
            for key, value in sample.items():
                print(f"{key}: {value}")
                print("-" * 50)
            
            # Also print as JSON for better formatting
            print(f"\n=== Sample Data (JSON Format) ===")
            print(json.dumps(sample, indent=2, ensure_ascii=False))
        else:
            print("Dataset is empty!")
            
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Make sure you have the 'datasets' library installed:")
        print("pip install datasets")

if __name__ == "__main__":
    load_and_show_sample() 