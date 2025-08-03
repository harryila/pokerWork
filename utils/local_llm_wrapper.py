"""
Local LLM Wrapper for testing communication experiments
Supports switching between local models (for testing) and OpenAI GPT-4 (for production)
"""

import os
import json
from typing import Dict, Any, Optional, List
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class LocalLLMWrapper:
    """Wrapper class to handle both local and OpenAI models"""
    
    def __init__(self, model_type: str = "local", local_model_name: Optional[str] = None):
        """
        Initialize the LLM wrapper
        
        Args:
            model_type: "local" for local models, "openai" for GPT-4
            local_model_name: Name of the local model to use (if model_type is "local")
        """
        self.model_type = model_type
        
        if model_type == "local":
            # Default to a small, open model for testing
            self.model_name = local_model_name or "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            
            # Check if model is already downloaded locally
            local_model_path = "./model/tinyllama-1.1b-chat"
            if os.path.exists(local_model_path) and not local_model_name:
                self.model_name = local_model_path
                print(f"Using locally downloaded model: {self.model_name}")
            else:
                print(f"Initializing model: {self.model_name}")
            
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
                
                # Set up the pipeline for easier generation
                self.pipeline = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device_map="auto"
                )
                
                print(f"✅ Local model loaded successfully")
                
            except Exception as e:
                print(f"❌ Error loading local model: {e}")
                print("Falling back to mock mode for testing")
                self.model_type = "mock"
                
        elif model_type == "openai":
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.model_name = "gpt-4"
            print(f"✅ OpenAI client initialized with model: {self.model_name}")
            
        else:  # mock mode for testing without actual models
            self.model_type = "mock"
            print("✅ Mock mode enabled for testing")
    
    def generate_response(self, prompt: str, max_tokens: int = 150, temperature: float = 0.7) -> str:
        """
        Generate a response from the model
        
        Args:
            prompt: The input prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated text response
        """
        if self.model_type == "local":
            try:
                # Format prompt for chat models
                if "chat" in self.model_name.lower() or "instruct" in self.model_name.lower():
                    formatted_prompt = f"<|system|>You are a helpful assistant playing poker.<|user|>{prompt}<|assistant|>"
                else:
                    formatted_prompt = prompt
                
                # Generate response
                outputs = self.pipeline(
                    formatted_prompt,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
                # Extract generated text
                generated_text = outputs[0]['generated_text']
                
                # Remove the prompt from the response
                if formatted_prompt in generated_text:
                    response = generated_text.replace(formatted_prompt, "").strip()
                else:
                    response = generated_text.strip()
                
                return response
                
            except Exception as e:
                print(f"Error generating response: {e}")
                return self._mock_response(prompt)
                
        elif self.model_type == "openai":
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant playing poker."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return response.choices[0].message.content
                
            except Exception as e:
                print(f"Error with OpenAI API: {e}")
                return self._mock_response(prompt)
                
        else:  # mock mode
            return self._mock_response(prompt)
    
    def generate_json_response(self, prompt: str, max_tokens: int = 150) -> Dict[str, Any]:
        """
        Generate a JSON response from the model
        
        Args:
            prompt: The input prompt (should request JSON format)
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Parsed JSON response
        """
        # Add JSON instruction to prompt if not present
        if "json" not in prompt.lower():
            prompt += "\n\nPlease respond in valid JSON format."
        
        response = self.generate_response(prompt, max_tokens, temperature=0.3)
        
        # Try to extract and parse JSON
        try:
            # Look for JSON in the response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
            else:
                # If no JSON found, create a simple response
                return {"response": response, "error": "No valid JSON found"}
                
        except json.JSONDecodeError:
            return {"response": response, "error": "Invalid JSON format"}
    
    def _mock_response(self, prompt: str) -> str:
        """Generate mock responses for testing"""
        if "action" in prompt.lower() and "poker" in prompt.lower():
            return '{"action": "call", "amount": 10, "reasoning": "Mock response for testing"}'
        elif "message" in prompt.lower() or "chat" in prompt.lower():
            return "Nice hand! Good luck everyone."
        else:
            return "Mock response for testing purposes."
    
    def switch_model(self, model_type: str, model_name: Optional[str] = None):
        """Switch between local and OpenAI models"""
        self.__init__(model_type, model_name)


# Convenience functions for easy switching
def get_test_llm() -> LocalLLMWrapper:
    """Get a local LLM for testing"""
    return LocalLLMWrapper(model_type="local")


def get_production_llm() -> LocalLLMWrapper:
    """Get OpenAI GPT-4 for production"""
    return LocalLLMWrapper(model_type="openai")


def get_mock_llm() -> LocalLLMWrapper:
    """Get mock LLM for unit testing"""
    return LocalLLMWrapper(model_type="mock")


if __name__ == "__main__":
    # Test the wrapper
    print("Testing Local LLM Wrapper...")
    
    # Test mock mode
    print("\n1. Testing Mock Mode:")
    mock_llm = get_mock_llm()
    response = mock_llm.generate_response("What should I do with pocket aces in poker?")
    print(f"Response: {response}")
    
    # Test local mode (will download model if not cached)
    print("\n2. Testing Local Model Mode:")
    local_llm = get_test_llm()
    response = local_llm.generate_response("I have King Queen suited. Should I raise or fold?")
    print(f"Response: {response}")
    
    # Test JSON generation
    print("\n3. Testing JSON Response:")
    json_prompt = """
    You are playing poker. Your hand is Ace King.
    Respond with your action in this JSON format:
    {"action": "raise|call|fold", "amount": 0, "reasoning": "your reasoning"}
    """
    json_response = local_llm.generate_json_response(json_prompt)
    print(f"JSON Response: {json_response}")