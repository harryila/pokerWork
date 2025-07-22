import os
import json
from collections import defaultdict

# Step 1: Point to the correct debug log directory
log_dir = "data/debug_logs"

# Step 2: Initialize counters
win_counts = defaultdict(int)
pot_sizes = []
total_hands = 0

# Step 3: Loop through all .json files in the directory
for filename in os.listdir(log_dir):
    if filename.endswith(".json"):
        with open(os.path.join(log_dir, filename), "r") as f:
            try:
                hand = json.load(f)
                winner = hand.get("winner")
                pot = hand.get("pot")

                if winner is not None and isinstance(winner, list):
                    for player in winner:
                        win_counts[player] += 1
                elif winner is not None:
                    win_counts[winner] += 1

                if isinstance(pot, (int, float)):
                    pot_sizes.append(pot)

                total_hands += 1
            except Exception as e:
                print(f"[ERROR] Failed to parse {filename}: {e}")

# Step 4: Print results
print("\n===== Poker Simulation Metrics =====")
print(f"Total hands played: {total_hands}")

print("\nWin rates per player:")
for player_id in sorted(win_counts, key=lambda x: int(x)):
    win_rate = win_counts[player_id] / total_hands * 100
    print(f"  Player {player_id}: {win_counts[player_id]} wins ({win_rate:.2f}%)")

if pot_sizes:
    avg_pot = sum(pot_sizes) / len(pot_sizes)
    print(f"\nAverage pot size: {avg_pot:.2f}")
else:
    print("\nNo pot size data found.")
