#!/usr/bin/env python3
"""
Batch inference script using vLLM to process prompts with Qwen model
Supports GPU parallelization and sample count control
"""

import json
import os
import argparse
from typing import List, Dict, Any
from tqdm import tqdm
import pandas as pd
from vllm import LLM, SamplingParams

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Batch inference with vLLM for reranking tasks")
    
    parser.add_argument(
        "--input_file", 
        type=str, 
        default="prompts_only.json",
        help="Input JSON file containing prompts"
    )
    
    parser.add_argument(
        "--output_file", 
        type=str, 
        default="outputs.json",
        help="Output JSON file for model responses"
    )
    
    parser.add_argument(
        "--parquet_file", 
        type=str, 
        default="prompt_output_pairs.parquet",
        help="Final parquet file with prompt-output pairs"
    )
    
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="Qwen/Qwen3-235B-A22B-Instruct-2507-FP8",
        help="Model name to use for inference"
    )
    
    parser.add_argument(
        "--gpu_count", 
        type=int, 
        default=1,
        help="Number of GPUs to use for inference"
    )
    
    parser.add_argument(
        "--sample_count", 
        type=int, 
        default=None,
        help="Number of samples to process (None for all)"
    )
    
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=32,
        help="Batch size for inference"
    )
    
    parser.add_argument(
        "--max_tokens", 
        type=int, 
        default=4096,
        help="Maximum number of tokens to generate"
    )
    
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.7,
        help="Sampling temperature"
    )
    
    parser.add_argument(
        "--top_p", 
        type=float, 
        default=0.9,
        help="Top-p sampling parameter"
    )
    
    return parser.parse_args()

def load_prompts(file_path: str, sample_count: int = None) -> List[List[Dict[str, str]]]:
    """
    Load prompts from JSON file
    
    Args:
        file_path: Path to the JSON file
        sample_count: Number of samples to load (None for all)
        
    Returns:
        List of prompt conversations
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            prompts = json.load(f)
        
        if sample_count is not None:
            prompts = prompts[:sample_count]
            
        print(f"Loaded {len(prompts)} prompts from {file_path}")
        return prompts
        
    except Exception as e:
        print(f"Error loading prompts: {e}")
        return []

def format_prompts_for_vllm(prompts: List[List[Dict[str, str]]]) -> List[str]:
    """
    Format prompts for vLLM chat completion
    
    Args:
        prompts: List of conversation prompts
        
    Returns:
        List of formatted prompt strings
    """
    formatted_prompts = []
    
    for prompt_conversation in prompts:
        # Convert conversation to a single string format
        formatted_prompt = ""
        
        for message in prompt_conversation:
            role = message.get('role', '')
            content = message.get('content', '')
            
            if role == 'system':
                formatted_prompt += f"<|im_start|>system\n{content}<|im_end|>\n"
            elif role == 'user':
                formatted_prompt += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == 'assistant':
                formatted_prompt += f"<|im_start|>assistant\n{content}<|im_end|>\n"
        
        # Add the final assistant prompt
        formatted_prompt += "<|im_start|>assistant\n"
        formatted_prompts.append(formatted_prompt)
    
    return formatted_prompts

def setup_vllm_model(model_name: str, gpu_count: int) -> LLM:
    """
    Setup vLLM model
    
    Args:
        model_name: Name of the model to load
        gpu_count: Number of GPUs to use
        
    Returns:
        LLM instance
    """
    try:
        print(f"Loading model {model_name} with {gpu_count} GPU(s)...")
        
        llm = LLM(
            model=model_name,
            tensor_parallel_size=gpu_count,
            trust_remote_code=True,
            dtype="float16",
            max_model_len=32768,  # Adjust based on model capacity
            gpu_memory_utilization=0.9,
        )
        
        print("Model loaded successfully!")
        return llm
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure you have:")
        print("1. Sufficient GPU memory")
        print("2. vLLM installed: pip install vllm")
        print("3. Access to the model")
        return None

def run_batch_inference(
    llm: LLM, 
    prompts: List[str], 
    batch_size: int,
    max_tokens: int,
    temperature: float,
    top_p: float
) -> List[str]:
    """
    Run batch inference on prompts
    
    Args:
        llm: vLLM model instance
        prompts: List of formatted prompts
        batch_size: Batch size for processing
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        
    Returns:
        List of generated responses
    """
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stop=["<|im_end|>"]
    )
    
    responses = []
    
    print(f"Running inference on {len(prompts)} prompts...")
    
    # Process in batches
    for i in tqdm(range(0, len(prompts), batch_size), desc="Processing batches"):
        batch_prompts = prompts[i:i + batch_size]
        
        # Generate responses for the batch
        outputs = llm.generate(batch_prompts, sampling_params)
        
        # Extract generated text
        for output in outputs:
            generated_text = output.outputs[0].text.strip()
            responses.append(generated_text)
    
    return responses

def save_outputs(outputs: List[str], output_file: str):
    """
    Save outputs to JSON file
    
    Args:
        outputs: List of generated responses
        output_file: Output file path
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(outputs, f, indent=2, ensure_ascii=False)
        
        print(f"Outputs saved to {output_file}")
        
    except Exception as e:
        print(f"Error saving outputs: {e}")

def create_parquet_file(
    prompts: List[List[Dict[str, str]]], 
    outputs: List[str], 
    parquet_file: str
):
    """
    Create parquet file with prompt-output pairs
    
    Args:
        prompts: Original prompts
        outputs: Generated outputs
        parquet_file: Output parquet file path
    """
    try:
        # Create DataFrame
        data = {
            'prompt': prompts,
            'output': outputs
        }
        
        df = pd.DataFrame(data)
        
        # Save as parquet
        df.to_parquet(parquet_file, index=False)
        
        print(f"Parquet file saved to {parquet_file}")
        print(f"DataFrame shape: {df.shape}")
        
        # Display some statistics
        avg_prompt_length = df['prompt'].apply(lambda x: len(str(x))).mean()
        avg_output_length = df['output'].apply(len).mean()
        
        print(f"Average prompt length: {avg_prompt_length:.0f} characters")
        print(f"Average output length: {avg_output_length:.0f} characters")
        
    except Exception as e:
        print(f"Error creating parquet file: {e}")

def main():
    """Main function"""
    args = parse_arguments()
    
    print("Starting batch inference pipeline...")
    print(f"Model: {args.model_name}")
    print(f"GPU count: {args.gpu_count}")
    print(f"Sample count: {args.sample_count}")
    print(f"Batch size: {args.batch_size}")
    
    # Load prompts
    prompts = load_prompts(args.input_file, args.sample_count)
    if not prompts:
        print("No prompts loaded. Exiting.")
        return
    
    # Setup model
    llm = setup_vllm_model(args.model_name, args.gpu_count)
    if llm is None:
        print("Failed to load model. Exiting.")
        return
    
    # Format prompts
    formatted_prompts = format_prompts_for_vllm(prompts)
    
    # Run inference
    outputs = run_batch_inference(
        llm=llm,
        prompts=formatted_prompts,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p
    )
    
    # Save outputs
    save_outputs(outputs, args.output_file)
    
    # Create parquet file
    create_parquet_file(prompts, outputs, args.parquet_file)
    
    print("\nBatch inference completed successfully!")
    print(f"Generated {len(outputs)} responses")
    print(f"Files created:")
    print(f"  - JSON outputs: {args.output_file}")
    print(f"  - Parquet file: {args.parquet_file}")

if __name__ == "__main__":
    main() 