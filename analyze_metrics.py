import os
import json
from collections import defaultdict

# Path to logs
DEBUG_LOG_PATH = "data/debug_logs"

hand_count = 0
total_pot = 0
actions_counter = defaultdict(int)
player_model_counter = defaultdict(int)

for filename in os.listdir(DEBUG_LOG_PATH):
    if filename.endswith(".json"):
        hand_count += 1

        # Count player/model usage based on filename
        parts = filename.split("_")
        player_id = parts[3] if len(parts) > 3 else "unknown"
        player_model_counter[player_id] += 1

        with open(os.path.join(DEBUG_LOG_PATH, filename)) as f:
            data = json.load(f)

        for r in data.get("responses", []):
            if r["response_type"] == "game_state" and r["raw_response"]:
                if "Pot amount" in r["raw_response"]:
                    try:
                        pot_line = r["raw_response"].split("Pot amount:")[1]
                        pot_value = int(pot_line.strip().split("\n")[0])
                        total_pot += pot_value
                    except:
                        pass
            elif r["response_type"] == "available_actions" and r["raw_response"]:
                try:
                    actions = json.loads(r["raw_response"])
                    for a in actions:
                        actions_counter[a] += 1
                except:
                    pass

# Final output
print(f"\nParsed {hand_count} hands")
print("Model/Player usage (from filenames):")
for player, count in player_model_counter.items():
    print(f"  Player {player}: {count} hands")

avg_pot = total_pot / hand_count if hand_count else 0
print(f"\nAverage Pot Size: {avg_pot:.2f}")

print("\nAvailable Actions (not actual decisions):")
for act, count in actions_counter.items():
    print(f"  {act}: {count}")
