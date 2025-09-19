#!/usr/bin/env python3
"""
Script to extract prompt fields from le723z/rearank_12k dataset and save to JSON file
"""

from datasets import load_dataset
import json
import os
from tqdm import tqdm



def extract_prompts_only(output_file="prompts_only.json"):
    """
    Extract only the prompt conversations with modified final instruction
    
    Args:
        output_file (str): Output JSON file name
    """
    try:
        print("Loading dataset for prompt-only extraction...")
        
        # Load the dataset
        dataset = load_dataset("le723z/rearank_12k")
        train_data = dataset['train']
        total_examples = len(train_data)
        
        print(f"Extracting and modifying prompt conversations from {total_examples} examples...")
        
        # New instruction text
        new_instruction = '''IMPORTANT：Please rank these passages according to their relevance to the search query:
            Follow these steps exactly:

            1) Within <reason> tags, you must think and do a raw rerank task based on you understanding of the passagses first. You need to analyze each passages to understand what they are talking about and analyze the query to learn what types of passage is more relevant. Then you need to output the reranked order based on your analyze. Your format should be:
            - Query: <analyzing>
            - passgase[i] <analyzing>
            - raw anwser <your output order>
            2) After that, you must conduct multi-round pairwise comparisons BETWEEN passages as needed to derive a reliable order. In each round, you can do many comparisons as you need. After each round, you need to output the current order.:
               - Round [I]:
               - Comparison line: "Compare [i] vs [j]: [i|j] is more relevant because <brief reason>"
               - Comparison line: ….
               - …
               - current order line:"[i] > [j] > [k]> ...."(full order)
               Guidance for <reason>:
               - You do NOT need to compare every pair; you just need to do comparisons to make the order more reasonable
               - Keep reasons concise and focused on query relevance

            3) Then, you need to do a final check, refine the final order as needed and make sure the end order contains all passages without duplication 

            4) In the end , within <answer> tags, output ONLY the final ranking in descending order of relevance using the exact format:
               [X] > [Y] > [Z] > ... (include ALL identifiers exactly once, no omissions, no duplicates)
               Do NOT include any extra text besides the chain inside <answer>.

IMPORTANT: You must strictly follow the <reason> and <answer>format   
               '''
        
        # Extract and modify prompts
        prompts_only = []
        
        for i in tqdm(range(total_examples), desc="Processing prompts"):
            example = train_data[i]
            original_prompt = example.get('prompt', [])
            
            # Convert conversation to string format, excluding assistant messages
            prompt_text = ""
            
            for message in original_prompt:
                role = message.get('role', '')
                content = message.get('content', '')
                
                # Skip assistant messages completely
                if role == 'assistant':
                    continue
                
                # For system and user messages, only keep the content
                if role == 'system':
                    prompt_text += content + "\n\n"
                elif role == 'user':
                    # Check if this is the last user message that needs modification
                    if 'Please rank these passages' in content:
                        # Split at "Please rank these passages" and keep everything before it
                        parts = content.split('Please rank these passages', 1)
                        # Rebuild with new instruction
                        prompt_text += parts[0] + new_instruction
                    else:
                        prompt_text += content + "\n\n"
            
            # Clean up extra whitespace and newlines
            prompt_text = prompt_text.strip()
            
            prompts_only.append(prompt_text)
        
        # Save to JSON file
        print(f"\nSaving {len(prompts_only)} modified prompt strings to {output_file}...")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(prompts_only, f, indent=2, ensure_ascii=False)
        
        print(f"Successfully saved prompt strings to {output_file}")
        print(f"Output file size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")
        
        # Show a sample of the converted format
        if prompts_only:
            print(f"\n{'='*80}")
            print("SAMPLE CONVERTED PROMPT (first 500 characters):")
            print('='*80)
            sample_prompt = prompts_only[0]
            print(sample_prompt[:500] + "..." if len(sample_prompt) > 500 else sample_prompt)
            print('='*80)
        
    except Exception as e:
        print(f"Error processing dataset: {e}")

if __name__ == "__main__":
    # Extract only prompt conversations with modified instructions
    extract_prompts_only("prompts_only.json")
    
    print("\nExtraction completed! Generated file:")
    print("prompts_only.json - Contains modified prompt strings (pure text format) with new ranking instructions") 