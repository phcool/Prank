#!/usr/bin/env python3
"""
Process model outputs by removing <think> tags and merge with prompts into parquet format
"""

import json
import re
import pandas as pd
from typing import List, Dict, Any

def load_json_file(file_path: str) -> List[Any]:
    """
    Load data from JSON file
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Loaded data
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} items from {file_path}")
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []

def remove_think_tags(text: str) -> str:
    """
    Remove <think></think> content from text while preserving other tags
    
    Args:
        text: Input text with potential <think> tags
        
    Returns:
        Text with <think> content removed
    """
    # Remove <think>...</think> content (including nested tags and multiline)
    # Use re.DOTALL to match newlines within the tags
    cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Clean up any extra whitespace that might be left
    cleaned_text = re.sub(r'\n\s*\n', '\n', cleaned_text)  # Remove multiple empty lines
    cleaned_text = cleaned_text.strip()  # Remove leading/trailing whitespace
    
    return cleaned_text

def process_outputs(outputs: List[str]) -> List[str]:
    """
    Process all outputs by removing <think> tags
    
    Args:
        outputs: List of model outputs
        
    Returns:
        List of processed outputs
    """
    processed_outputs = []
    
    print("Processing outputs to remove <think> tags...")
    
    for i, output in enumerate(outputs):
        processed_output = remove_think_tags(output)
        processed_outputs.append(processed_output)
        
        # Print progress for every 1000 items
        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1}/{len(outputs)} outputs")
    
    print(f"Finished processing {len(processed_outputs)} outputs")
    return processed_outputs

def save_processed_outputs(processed_outputs: List[str], output_file: str):
    """
    Save processed outputs to JSON file
    
    Args:
        processed_outputs: List of processed outputs
        output_file: Output file path
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_outputs, f, indent=2, ensure_ascii=False)
        
        print(f"Processed outputs saved to {output_file}")
        
    except Exception as e:
        print(f"Error saving processed outputs: {e}")

def create_final_parquet(
    prompts: List[List[Dict[str, str]]], 
    processed_outputs: List[str], 
    parquet_file: str
):
    """
    Create final parquet file with prompts and processed outputs
    
    Args:
        prompts: Original prompts
        processed_outputs: Processed outputs without <think> tags
        parquet_file: Output parquet file path
    """
    try:
        # Ensure both lists have the same length
        min_length = min(len(prompts), len(processed_outputs))
        if len(prompts) != len(processed_outputs):
            print(f"Warning: Length mismatch - prompts: {len(prompts)}, outputs: {len(processed_outputs)}")
            print(f"Using first {min_length} items from both")
            prompts = prompts[:min_length]
            processed_outputs = processed_outputs[:min_length]
        
        # Create DataFrame
        data = {
            'prompt': prompts,
            'output': processed_outputs
        }
        
        df = pd.DataFrame(data)
        
        # Save as parquet
        df.to_parquet(parquet_file, index=False)
        
        print(f"Final parquet file saved to {parquet_file}")
        print(f"DataFrame shape: {df.shape}")
        
        # Display some statistics
        avg_prompt_length = df['prompt'].apply(lambda x: len(str(x))).mean()
        avg_output_length = df['output'].apply(len).mean()
        
        print(f"Average prompt length: {avg_prompt_length:.0f} characters")
        print(f"Average processed output length: {avg_output_length:.0f} characters")
        
        # Count how many outputs had <think> tags removed
        original_outputs = load_json_file("outputs.json")
        if original_outputs:
            think_count = sum(1 for output in original_outputs if '<think>' in output.lower())
            print(f"Removed <think> tags from {think_count}/{len(original_outputs)} outputs")
        
    except Exception as e:
        print(f"Error creating final parquet file: {e}")

def analyze_output_format(outputs: List[str]) -> Dict[str, int]:
    """
    Analyze the format of outputs to understand tag usage
    
    Args:
        outputs: List of outputs to analyze
        
    Returns:
        Dictionary with format statistics
    """
    stats = {
        'total_outputs': len(outputs),
        'has_think_tags': 0,
        'has_reason_tags': 0,
        'has_answer_tags': 0,
        'complete_format': 0  # Has all three tags
    }
    
    for output in outputs:
        output_lower = output.lower()
        has_think = '<think>' in output_lower and '</think>' in output_lower
        has_reason = '<reason>' in output_lower and '</reason>' in output_lower
        has_answer = '<answer>' in output_lower and '</answer>' in output_lower
        
        if has_think:
            stats['has_think_tags'] += 1
        if has_reason:
            stats['has_reason_tags'] += 1
        if has_answer:
            stats['has_answer_tags'] += 1
        if has_think and has_reason and has_answer:
            stats['complete_format'] += 1
    
    return stats

def main():
    """Main function"""
    print("Starting output processing pipeline...")
    
    # File paths
    prompts_file = "prompts_only.json"
    outputs_file = "outputs.json"
    processed_outputs_file = "processed_outputs.json"
    final_parquet_file = "final_prompt_output_pairs.parquet"
    
    # Load prompts and outputs
    prompts = load_json_file(prompts_file)
    outputs = load_json_file(outputs_file)
    
    if not prompts or not outputs:
        print("Failed to load required files. Make sure prompts_only.json and outputs.json exist.")
        return
    
    # Analyze output format
    print("\nAnalyzing output format...")
    format_stats = analyze_output_format(outputs)
    print("Format analysis:")
    for key, value in format_stats.items():
        if key == 'total_outputs':
            print(f"  {key}: {value}")
        else:
            percentage = (value / format_stats['total_outputs']) * 100
            print(f"  {key}: {value} ({percentage:.1f}%)")
    
    # Process outputs to remove <think> tags
    processed_outputs = process_outputs(outputs)
    
    # Save processed outputs
    save_processed_outputs(processed_outputs, processed_outputs_file)
    
    # Create final parquet file
    create_final_parquet(prompts, processed_outputs, final_parquet_file)
    
    print("\nProcessing completed successfully!")
    print("Files created:")
    print(f"  - Processed outputs: {processed_outputs_file}")
    print(f"  - Final parquet: {final_parquet_file}")
    
    # Show a sample of before and after
    if outputs and processed_outputs:
        print("\n" + "="*80)
        print("SAMPLE COMPARISON (first output):")
        print("="*80)
        print("ORIGINAL OUTPUT:")
        print(outputs[0][:500] + "..." if len(outputs[0]) > 500 else outputs[0])
        print("\n" + "-"*40)
        print("PROCESSED OUTPUT (without <think>):")
        print(processed_outputs[0][:500] + "..." if len(processed_outputs[0]) > 500 else processed_outputs[0])
        print("="*80)

if __name__ == "__main__":
    main() 