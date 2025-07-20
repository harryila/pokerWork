from transformers import AutoTokenizer
from game_environment.collusion_llm_agent import CollusionLLMAgent

api_key = "OPENAI_API_KEY"  # Replace with your actual OpenAI API key

agent = CollusionLLMAgent(
    model="gpt-4o",
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
