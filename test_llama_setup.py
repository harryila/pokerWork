#!/usr/bin/env python3
"""
Test script to download and verify local LLM model setup
Using Microsoft Phi-3 mini model as it's open and doesn't require gated access
"""

import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def test_llm_setup():
    """Download and test Microsoft Phi-3 mini model"""
    print("Setting up Microsoft Phi-3 mini model for local testing...")
    
    # Using Phi-3 mini as it's open-source and doesn't require gated access
    model_name = "microsoft/Phi-3-mini-4k-instruct"
    
    try:
        # Download tokenizer
        print("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # Download model
        print("Downloading model (this may take a while)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Test the model
        print("\nTesting model with a simple prompt...")
        test_prompt = "You are playing poker. Your hand is Ace King. What should you do?"
        
        inputs = tokenizer(test_prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\nPrompt: {test_prompt}")
        print(f"Response: {response}")
        
        print("\n✅ Local LLM model setup successful!")
        
        # Save model path for future use
        model_cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        print(f"\nModel cached at: {model_cache_dir}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error setting up local LLM model: {e}")
        return False

if __name__ == "__main__":
    test_llm_setup()