#!/usr/bin/env python3
"""
Test LLM output on a single prompt from prompts_only.json
"""

import json
import os
import argparse
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import torch

def load_single_prompt(json_file="prompts_only.json", index=0):
    """
    Load a single prompt from the JSON file
    
    Args:
        json_file: Path to the JSON file containing prompts
        index: Index of the prompt to load
        
    Returns:
        str: The selected prompt string
    """
    try:
        if not os.path.exists(json_file):
            print(f"Error: {json_file} not found. Please run extract_prompts.py first.")
            return None
            
        with open(json_file, 'r', encoding='utf-8') as f:
            prompts = json.load(f)
        
        if not prompts:
            print("Error: No prompts found in the JSON file.")
            return None
            
        if index >= len(prompts):
            print(f"Error: Index {index} is out of range. Available prompts: 0-{len(prompts)-1}")
            return None
            
        selected_prompt = prompts[index]
        print(f"Loaded prompt {index} from {json_file}")
        print(f"Prompt length: {len(selected_prompt)} characters")
        return selected_prompt
        
    except Exception as e:
        print(f"Error loading prompt: {e}")
        return None

def setup_vllm_model(model_name, gpu_count=1, lora_path=None):
    """
    Setup vLLM model with optional LoRA adapter
    
    Args:
        model_name: Base model name or path
        gpu_count: Number of GPUs to use
        lora_path: Path to LoRA adapter (optional)
        
    Returns:
        LLM instance or None if failed
    """
    try:
        if lora_path:
            print(f"Loading base model with vLLM: {model_name}")
            print(f"Loading LoRA adapter: {lora_path}")
        else:
            print(f"Loading model with vLLM: {model_name}")
        print("This may take a while...")
        
        # Configure vLLM parameters
        vllm_kwargs = {
            "model": model_name,
            "tensor_parallel_size": gpu_count,
            "trust_remote_code": True,
            "dtype": "float16",
            "max_model_len": 8192,  # Adjust based on model capacity
            "gpu_memory_utilization": 0.9,
        }
        
        # Add LoRA configuration if provided
        if lora_path:
            vllm_kwargs["enable_lora"] = True
            vllm_kwargs["max_lora_rank"] = 64  # Adjust based on your LoRA configuration
            vllm_kwargs["max_loras"] = 1
        
        llm = LLM(**vllm_kwargs)
        
        print("Model loaded successfully with vLLM!")
        
        # Load LoRA adapter if provided
        if lora_path:
            print(f"Adding LoRA adapter: {lora_path}")
            # The LoRA adapter will be specified during generation
        
        return llm
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure you have:")
        print("1. Sufficient GPU memory")
        print("2. vLLM installed with LoRA support: pip install vllm")
        print("3. Access to the base model and LoRA adapter")
        if lora_path:
            print(f"4. LoRA adapter path is correct: {lora_path}")
        return None

def generate_response_vllm(llm, prompt, max_tokens=2048, temperature=0.7, top_p=0.9, lora_request=None):
    """
    Generate response using vLLM with optional LoRA adapter
    
    Args:
        llm: vLLM model instance
        prompt: Input prompt string
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        lora_request: LoRA request object (optional)
        
    Returns:
        str: Generated response
    """
    try:
        if lora_request:
            print("Generating response with vLLM + LoRA adapter...")
        else:
            print("Generating response with vLLM...")
        
        # Set up sampling parameters
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=["<|im_end|>", "</s>"]  # Common stop tokens
        )
        
        print(f"Input prompt length: {len(prompt)} characters")
        
        # Generate response
        if lora_request:
            outputs = llm.generate([prompt], sampling_params, lora_request=lora_request)
        else:
            outputs = llm.generate([prompt], sampling_params)
        
        # Extract generated text
        generated_response = outputs[0].outputs[0].text.strip()
        
        print(f"Generated response length: {len(generated_response)} characters")
        
        return generated_response
        
    except Exception as e:
        print(f"Error during generation: {e}")
        return None

def analyze_response(response):
    """
    Analyze the generated response format
    
    Args:
        response: Generated response string
        
    Returns:
        dict: Analysis results
    """
    analysis = {
        'total_length': len(response),
        'has_reason_tags': '<reason>' in response.lower() and '</reason>' in response.lower(),
        'has_answer_tags': '<answer>' in response.lower() and '</answer>' in response.lower(),
        'complete_format': False
    }
    
    # Check if response follows the expected format
    analysis['complete_format'] = analysis['has_reason_tags'] and analysis['has_answer_tags']
    
    # Extract reason content
    if analysis['has_reason_tags']:
        try:
            reason_start = response.lower().find('<reason>')
            reason_end = response.lower().find('</reason>')
            if reason_start != -1 and reason_end != -1:
                reason_content = response[reason_start+8:reason_end].strip()
                analysis['reason_length'] = len(reason_content)
            else:
                analysis['reason_length'] = 0
        except:
            analysis['reason_length'] = 0
    else:
        analysis['reason_length'] = 0
    
    # Extract answer content
    if analysis['has_answer_tags']:
        try:
            answer_start = response.lower().find('<answer>')
            answer_end = response.lower().find('</answer>')
            if answer_start != -1 and answer_end != -1:
                answer_content = response[answer_start+8:answer_end].strip()
                analysis['answer_content'] = answer_content
                analysis['answer_length'] = len(answer_content)
            else:
                analysis['answer_content'] = ""
                analysis['answer_length'] = 0
        except:
            analysis['answer_content'] = ""
            analysis['answer_length'] = 0
    else:
        analysis['answer_content'] = ""
        analysis['answer_length'] = 0
    
    return analysis

def save_results(prompt, response, analysis, output_file="test_result.json"):
    """
    Save test results to file
    
    Args:
        prompt: Input prompt
        response: Generated response
        analysis: Response analysis
        output_file: Output file path
    """
    results = {
        "prompt": prompt,
        "response": response,
        "analysis": analysis,
        "prompt_length": len(prompt),
        "response_length": len(response)
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to {output_file}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Test LLM on a single prompt")
    
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Model name or path"
    )
    
    parser.add_argument(
        "--prompts_file",
        type=str,
        default="prompts_only.json",
        help="Path to prompts JSON file"
    )
    
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Index of prompt to test (default: 0)"
    )
    
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=2048,
        help="Maximum tokens to generate"
    )
    
    parser.add_argument(
        "--gpu_count",
        type=int,
        default=1,
        help="Number of GPUs to use for vLLM"
    )
    
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="Path to LoRA adapter directory (optional)"
    )
    
    parser.add_argument(
        "--lora_name",
        type=str,
        default="default_lora",
        help="Name for the LoRA adapter"
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
    
    parser.add_argument(
        "--output_file",
        type=str,
        default="test_result.json",
        help="Output file for results"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("vLLM SINGLE PROMPT TEST")
    print("="*80)
    print(f"Model: {args.model_name}")
    print(f"GPU count: {args.gpu_count}")
    if args.lora_path:
        print(f"LoRA adapter: {args.lora_path}")
        print(f"LoRA name: {args.lora_name}")
    else:
        print("LoRA adapter: None (using base model)")
    print(f"Prompts file: {args.prompts_file}")
    print(f"Prompt index: {args.index}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Temperature: {args.temperature}")
    print(f"Top-p: {args.top_p}")
    print("="*80)
    
    # Load prompt
    prompt = load_single_prompt(args.prompts_file, args.index)
    if prompt is None:
        return
    
    print(f"\n{'='*80}")
    print("SELECTED PROMPT (first 500 characters):")
    print('='*80)
    print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
    print('='*80)
    
    # Setup vLLM model
    llm = setup_vllm_model(args.model_name, args.gpu_count, args.lora_path)
    if llm is None:
        return
    
    # Create LoRA request if LoRA path is provided
    lora_request = None
    if args.lora_path:
        try:
            print(f"Creating LoRA request for: {args.lora_name}")
            lora_request = LoRARequest(args.lora_name, 1, args.lora_path)
        except Exception as e:
            print(f"Error creating LoRA request: {e}")
            print("Continuing with base model...")
            lora_request = None
    
    # Generate response
    response = generate_response_vllm(
        llm, prompt, 
        args.max_tokens, args.temperature, args.top_p, lora_request
    )
    
    if response is None:
        print("Failed to generate response")
        return
    
    # Analyze response
    analysis = analyze_response(response)
    
    # Display results
    print(f"\n{'='*80}")
    print("GENERATED RESPONSE:")
    print('='*80)
    print(response)
    print('='*80)
    
    print(f"\n{'='*80}")
    print("RESPONSE ANALYSIS:")
    print('='*80)
    print(f"Response length: {analysis['total_length']} characters")
    print(f"Has <reason> tags: {analysis['has_reason_tags']}")
    print(f"Has <answer> tags: {analysis['has_answer_tags']}")
    print(f"Complete format: {analysis['complete_format']}")
    print(f"Reason section length: {analysis['reason_length']} characters")
    print(f"Answer section length: {analysis['answer_length']} characters")
    
    if analysis['answer_content']:
        print(f"\nExtracted answer: {analysis['answer_content']}")
    
    print('='*80)
    
    # Save results with LoRA info
    results_data = {
        "prompt": prompt,
        "response": response,
        "analysis": analysis,
        "prompt_length": len(prompt),
        "response_length": len(response),
        "model_config": {
            "base_model": args.model_name,
            "lora_path": args.lora_path,
            "lora_name": args.lora_name if args.lora_path else None,
            "gpu_count": args.gpu_count,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p
        }
    }
    
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to {args.output_file}")
    
    print(f"\nTest completed successfully!")
    if args.lora_path:
        print(f"Used LoRA adapter: {args.lora_path}")
    else:
        print("Used base model without LoRA")

if __name__ == "__main__":
    main() 