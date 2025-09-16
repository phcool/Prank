#!/usr/bin/env python3
"""
Test script to run Qwen/Qwen3-235B-A22B-Thinking-2507-FP8 model on the first prompt from prompts_only.json
"""

import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def load_first_prompt(json_file="prompts_only.json"):
    """
    Load the first prompt from the JSON file
    
    Args:
        json_file (str): Path to the JSON file containing prompts
        
    Returns:
        list: First prompt conversation
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
            
        first_prompt = prompts[0]
        print(f"Loaded first prompt with {len(first_prompt)} messages")
        return first_prompt
        
    except Exception as e:
        print(f"Error loading prompt: {e}")
        return None

def setup_qwen_model():
    """
    Setup Qwen model and tokenizer
    
    Returns:
        tuple: (model, tokenizer) or (None, None) if failed
    """
    try:
        model_name = "Qwen/Qwen3-235B-A22B-Thinking-2507-FP8"
        print(f"Loading model: {model_name}")
        print("This may take a while for the first time...")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # Load model with appropriate settings
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        print("Model loaded successfully!")
        return model, tokenizer
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure you have:")
        print("1. Sufficient GPU memory (this is a very large model)")
        print("2. Access to the model (may require Hugging Face authentication)")
        print("3. Required packages: pip install transformers torch accelerate")
        return None, None

def run_inference(model, tokenizer, prompt_messages):
    """
    Run inference on the prompt messages
    
    Args:
        model: The loaded model
        tokenizer: The tokenizer
        prompt_messages (list): List of message dictionaries
        
    Returns:
        str: Model output
    """
    try:
        # Apply chat template
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        print("=== Input Prompt (last 500 chars) ===")
        print(prompt_text[-500:])
        print("\n" + "="*50 + "\n")
        
        # Tokenize input
        inputs = tokenizer(prompt_text, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        print("Generating response...")
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=2048,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode response
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the new generated part
        generated_response = full_response[len(prompt_text):].strip()
        
        return generated_response
        
    except Exception as e:
        print(f"Error during inference: {e}")
        return None

def test_qwen_on_first_prompt():
    """
    Main function to test Qwen model on the first prompt
    """
    print("Starting Qwen model test...")
    
    # Load the first prompt
    first_prompt = load_first_prompt()
    if first_prompt is None:
        return
    
    print(f"\n=== First Prompt Structure ===")
    for i, msg in enumerate(first_prompt):
        print(f"Message {i+1}: {msg['role']} - {len(msg['content'])} characters")
    
    # Setup model
    model, tokenizer = setup_qwen_model()
    if model is None or tokenizer is None:
        return
    
    # Run inference
    print(f"\n=== Running Inference ===")
    response = run_inference(model, tokenizer, first_prompt)
    
    if response:
        print(f"\n=== Model Response ===")
        print(response)
        
        # Save response to file
        output_file = "qwen_response.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=== Qwen Model Response ===\n\n")
            f.write(response)
        
        print(f"\nResponse saved to {output_file}")
        
        # Basic analysis
        print(f"\n=== Response Analysis ===")
        print(f"Response length: {len(response)} characters")
        print(f"Contains <think> tags: {'<think>' in response}")
        print(f"Contains <answer> tags: {'<answer>' in response}")
        
    else:
        print("Failed to generate response")

if __name__ == "__main__":
    test_qwen_on_first_prompt() 