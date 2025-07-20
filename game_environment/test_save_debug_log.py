from transformers import AutoTokenizer
from collusion_llm_agent import CollusionLLMAgent

api_key = "OPEN-AI-KEY-HERE"  # Replace this with your actual API key

agent = CollusionLLMAgent(
    model="gpt-3.5-turbo",
    tokenizer=AutoTokenizer.from_pretrained("gpt2"),
    api_key=api_key,
    teammate_id=1,
)

print("[DEBUG] Calling save log...")

agent._save_llm_response(
    response_type="manual_test",
    raw_response="This is the raw response",
    processed_response="This is the cleaned response",
    error="Test error message",
    player_id=0,
)

print("✅ Test complete. Check debug_logs folder.")
