#!/usr/bin/env python3
"""
Fix data format issues in the parquet dataset
Handle cases where prompt/output fields contain dicts instead of strings
"""

import pandas as pd
import json
from typing import Any, List, Dict
import os

def check_data_types(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Check data types in the dataframe
    
    Args:
        df: Input dataframe
        
    Returns:
        Dictionary with type analysis
    """
    analysis = {
        'total_samples': len(df),
        'prompt_types': {},
        'output_types': {},
        'problematic_samples': []
    }
    
    print("Analyzing data types...")
    
    for i, row in df.iterrows():
        # Check prompt type
        prompt = row['prompt']
        prompt_type = type(prompt).__name__
        if prompt_type not in analysis['prompt_types']:
            analysis['prompt_types'][prompt_type] = 0
        analysis['prompt_types'][prompt_type] += 1
        
        # Check output type
        output = row['output']
        output_type = type(output).__name__
        if output_type not in analysis['output_types']:
            analysis['output_types'][output_type] = 0
        analysis['output_types'][output_type] += 1
        
        # Identify problematic samples
        if not isinstance(prompt, str) or not isinstance(output, str):
            analysis['problematic_samples'].append({
                'index': i,
                'prompt_type': prompt_type,
                'output_type': output_type,
                'prompt_preview': str(prompt)[:100] if prompt is not None else 'None',
                'output_preview': str(output)[:100] if output is not None else 'None'
            })
        
        # Progress indicator
        if (i + 1) % 1000 == 0:
            print(f"Analyzed {i + 1}/{len(df)} samples")
    
    return analysis

def convert_to_string(data: Any) -> str:
    """
    Convert various data types to string format suitable for training
    
    Args:
        data: Input data of any type
        
    Returns:
        String representation
    """
    if isinstance(data, str):
        return data
    elif isinstance(data, list):
        # If it's a list of chat messages, convert to chat format
        if len(data) > 0 and isinstance(data[0], dict) and 'role' in data[0]:
            # This looks like a chat conversation
            chat_text = ""
            for message in data:
                role = message.get('role', 'unknown')
                content = message.get('content', '')
                if role == 'system':
                    chat_text += f"<|im_start|>system\n{content}<|im_end|>\n"
                elif role == 'user':
                    chat_text += f"<|im_start|>user\n{content}<|im_end|>\n"
                elif role == 'assistant':
                    chat_text += f"<|im_start|>assistant\n{content}<|im_end|>\n"
                else:
                    chat_text += f"{role}: {content}\n"
            return chat_text.strip()
        else:
            # Regular list, join with newlines
            return '\n'.join(str(item) for item in data)
    elif isinstance(data, dict):
        # Convert dict to JSON string or extract meaningful content
        if 'content' in data:
            # Looks like a single message
            return str(data.get('content', ''))
        elif 'text' in data:
            # Alternative text field
            return str(data.get('text', ''))
        else:
            # Fallback to JSON
            return json.dumps(data, ensure_ascii=False)
    elif data is None:
        return ""
    else:
        return str(data)

def fix_data_format(
    input_file: str = "final_prompt_output_pairs.parquet",
    output_file: str = "fixed_prompt_output_pairs.parquet"
) -> bool:
    """
    Fix data format issues in the dataset
    
    Args:
        input_file: Input parquet file
        output_file: Output parquet file with fixed format
        
    Returns:
        True if successful, False otherwise
    """
    
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found!")
        return False
    
    print(f"Loading data from {input_file}...")
    try:
        df = pd.read_parquet(input_file)
        print(f"Loaded {len(df)} samples")
    except Exception as e:
        print(f"Error loading data: {e}")
        return False
    
    # Analyze current data types
    print("\n" + "="*60)
    print("DATA TYPE ANALYSIS")
    print("="*60)
    
    analysis = check_data_types(df)
    
    print(f"\nTotal samples: {analysis['total_samples']}")
    
    print(f"\nPrompt field types:")
    for dtype, count in analysis['prompt_types'].items():
        percentage = (count / analysis['total_samples']) * 100
        print(f"  {dtype}: {count} ({percentage:.1f}%)")
    
    print(f"\nOutput field types:")
    for dtype, count in analysis['output_types'].items():
        percentage = (count / analysis['total_samples']) * 100
        print(f"  {dtype}: {count} ({percentage:.1f}%)")
    
    if analysis['problematic_samples']:
        print(f"\nProblematic samples: {len(analysis['problematic_samples'])}")
        print("Sample problematic entries:")
        for i, sample in enumerate(analysis['problematic_samples'][:5]):
            print(f"  {i+1}. Index {sample['index']}:")
            print(f"     Prompt type: {sample['prompt_type']}")
            print(f"     Output type: {sample['output_type']}")
            print(f"     Prompt preview: {sample['prompt_preview']}")
            print(f"     Output preview: {sample['output_preview']}")
            print()
    
    # Fix the data
    print("\n" + "="*60)
    print("FIXING DATA FORMAT")
    print("="*60)
    
    print("Converting all fields to string format...")
    
    fixed_prompts = []
    fixed_outputs = []
    
    for i, row in df.iterrows():
        # Convert prompt to string
        fixed_prompt = convert_to_string(row['prompt'])
        fixed_prompts.append(fixed_prompt)
        
        # Convert output to string
        fixed_output = convert_to_string(row['output'])
        fixed_outputs.append(fixed_output)
        
        # Progress indicator
        if (i + 1) % 1000 == 0:
            print(f"Fixed {i + 1}/{len(df)} samples")
    
    # Create new dataframe with fixed data
    fixed_df = pd.DataFrame({
        'prompt': fixed_prompts,
        'output': fixed_outputs
    })
    
    # Verify the fix
    print("\nVerifying fixed data...")
    all_prompts_str = all(isinstance(p, str) for p in fixed_df['prompt'])
    all_outputs_str = all(isinstance(o, str) for o in fixed_df['output'])
    
    print(f"All prompts are strings: {all_prompts_str}")
    print(f"All outputs are strings: {all_outputs_str}")
    
    if not (all_prompts_str and all_outputs_str):
        print("Warning: Some data could not be converted to strings!")
        return False
    
    # Save fixed data
    print(f"\nSaving fixed data to {output_file}...")
    try:
        fixed_df.to_parquet(output_file, index=False)
        print(f"Successfully saved {len(fixed_df)} samples")
    except Exception as e:
        print(f"Error saving data: {e}")
        return False
    
    # Show before/after examples
    print("\n" + "="*60)
    print("BEFORE/AFTER EXAMPLES")
    print("="*60)
    
    if analysis['problematic_samples']:
        for i, sample in enumerate(analysis['problematic_samples'][:3]):
            idx = sample['index']
            print(f"\nExample {i+1} (Index {idx}):")
            print("BEFORE:")
            print(f"  Prompt type: {sample['prompt_type']}")
            print(f"  Prompt: {sample['prompt_preview']}")
            print(f"  Output type: {sample['output_type']}")
            print(f"  Output: {sample['output_preview']}")
            
            print("AFTER:")
            print(f"  Prompt: {fixed_df.iloc[idx]['prompt'][:100]}...")
            print(f"  Output: {fixed_df.iloc[idx]['output'][:100]}...")
    
    # Final statistics
    print(f"\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Original samples: {len(df)}")
    print(f"Fixed samples: {len(fixed_df)}")
    print(f"Problematic samples fixed: {len(analysis['problematic_samples'])}")
    
    avg_prompt_len = fixed_df['prompt'].str.len().mean()
    avg_output_len = fixed_df['output'].str.len().mean()
    print(f"Average prompt length: {avg_prompt_len:.0f} characters")
    print(f"Average output length: {avg_output_len:.0f} characters")
    
    print(f"\nFixed dataset saved to: {output_file}")
    print("You can now use this file for training!")
    
    return True

def inspect_sample_data(
    file_path: str = "final_prompt_output_pairs.parquet",
    sample_indices: List[int] = None
):
    """
    Inspect specific samples in detail
    
    Args:
        file_path: Path to parquet file
        sample_indices: List of indices to inspect (None for first 3)
    """
    
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found!")
        return
    
    df = pd.read_parquet(file_path)
    
    if sample_indices is None:
        sample_indices = [0, 1, 2]
    
    print(f"\n" + "="*60)
    print(f"DETAILED SAMPLE INSPECTION: {file_path}")
    print("="*60)
    
    for i, idx in enumerate(sample_indices):
        if idx >= len(df):
            print(f"Index {idx} is out of range (max: {len(df)-1})")
            continue
            
        print(f"\nSample {i+1} (Index {idx}):")
        print("-" * 40)
        
        prompt = df.iloc[idx]['prompt']
        output = df.iloc[idx]['output']
        
        print(f"Prompt type: {type(prompt).__name__}")
        print(f"Prompt content: {repr(prompt)}")
        
        print(f"\nOutput type: {type(output).__name__}")
        print(f"Output content: {repr(output)}")
        
        if isinstance(prompt, (list, dict)):
            print(f"Prompt structure: {prompt}")
        if isinstance(output, (list, dict)):
            print(f"Output structure: {output}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fix data format issues in parquet dataset")
    parser.add_argument("--input", default="final_prompt_output_pairs.parquet", help="Input parquet file")
    parser.add_argument("--output", default="fixed_prompt_output_pairs.parquet", help="Output parquet file")
    parser.add_argument("--inspect", action="store_true", help="Inspect sample data without fixing")
    parser.add_argument("--indices", nargs="+", type=int, help="Sample indices to inspect")
    
    args = parser.parse_args()
    
    if args.inspect:
        inspect_sample_data(args.input, args.indices)
    else:
        success = fix_data_format(args.input, args.output)
        if success:
            print("\nData format fixing completed successfully!")
        else:
            print("\nData format fixing failed!")
            exit(1) 