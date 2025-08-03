#!/usr/bin/env python3
"""
Download and set up TinyLlama model for local testing
"""

import os
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def download_llama_model():
    """Download and save the TinyLlama model locally"""
    
    # Model name - using TinyLlama which is open and doesn't require special access
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    # Local directory to save the model
    model_dir = "./model/tinyllama-1.1b-chat"
    
    print(f"Downloading {model_name}...")
    print(f"This may take a few minutes depending on your internet connection...")
    
    try:
        # Download tokenizer
        print("\nDownloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir="./model/cache",
            local_files_only=False
        )
        
        # Save tokenizer locally
        tokenizer.save_pretrained(model_dir)
        print("✓ Tokenizer downloaded and saved")
        
        # Download model
        print("\nDownloading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir="./model/cache",
            local_files_only=False,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # Save model locally
        model.save_pretrained(model_dir)
        print("✓ Model downloaded and saved")
        
        print(f"\nModel successfully saved to: {model_dir}")
        print(f"Total size: ~2.2GB")
        
        # Test the model
        print("\nTesting model...")
        
        # Create a chat-style prompt
        chat = [
            {"role": "user", "content": "Hello! How are you today?"}
        ]
        
        # Apply chat template
        prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\nTest prompt: Hello! How are you today?")
        print(f"Model response: {response}")
        
        print("\n✓ Model is working correctly!")
        print("\nNote: This is TinyLlama, a smaller model suitable for testing.")
        print("When ready for production, you can switch to GPT-4 or other larger models.")
        
    except Exception as e:
        print(f"\nError downloading model: {e}")
        print("\nPlease ensure you have:")
        print("1. Sufficient disk space (~3GB)")
        print("2. Internet connection")
        sys.exit(1)

if __name__ == "__main__":
    download_llama_model()