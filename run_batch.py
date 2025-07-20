# run_batch.py
import subprocess

NUM_BATCHES = 93
HANDS_PER_BATCH = 10

for i in range(NUM_BATCHES):
    print(f"Running batch {i + 1}/{NUM_BATCHES}...")

    result = subprocess.run([
        "python", "run_game.py",
        "--buyin", "500",
        "--big-blind", "1",
        "--small-blind", "2",
        "--max-players", "3",
        "--llm-players", "0,1,2",
        "--collusion-llm-players", "1,2",
        "--model", "gpt-3.5-turbo",
        "--api-key", "OPEN-AI-KEY-HERE"  # your key
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[!] Batch {i+1} failed:\n", result.stderr)
    else:
        print(f"[✓] Batch {i+1} complete.")
