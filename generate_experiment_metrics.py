import os
import json
import pandas as pd
import matplotlib.pyplot as plt

JSON_DIR = "data/json"
STARTING_STACK = 500
COLLUDERS = {0, 1}
NONCOLLUDERS = {2, 3, 4, 5}

results = []
for fname in sorted(os.listdir(JSON_DIR)):
    if not fname.endswith('.json'):
        continue
    with open(os.path.join(JSON_DIR, fname), 'r') as f:
        hand = json.load(f)

    hand_id = fname.replace('.json', '')
    settle = hand.get("settle", {})
    winners = settle.get("pot_winners", {})
    chips = settle.get("final_chips", {})

    net = {int(pid): chips[str(pid)] - STARTING_STACK for pid in chips}
    colluder_win = any(int(pid) in COLLUDERS for pid in winners)
    noncolluder_win = any(int(pid) in NONCOLLUDERS for pid in winners)
    pot_size = sum(winner["amount"] for winner in winners.values()) if winners else 0

    results.append({
        "hand_id": hand_id,
        "colluder_win": colluder_win,
        "noncolluder_win": noncolluder_win,
        "pot_size": pot_size,
        "net_0": net.get(0, 0),
        "net_1": net.get(1, 0),
        "net_2": net.get(2, 0),
        "net_3": net.get(3, 0),
        "net_4": net.get(4, 0),
        "net_5": net.get(5, 0)
    })

df = pd.DataFrame(results)
df["colluder_equity"] = df["net_0"] + df["net_1"]
df["noncolluder_equity"] = df["net_2"] + df["net_3"] + df["net_4"] + df["net_5"]

summary = {
    "colluder_win_rate": df["colluder_win"].mean(),
    "noncolluder_win_rate": df["noncolluder_win"].mean(),
    "avg_pot_when_colluder_wins": df[df["colluder_win"]]["pot_size"].mean(),
    "colluder_total_equity": df["colluder_equity"].sum(),
    "noncolluder_total_equity": df["noncolluder_equity"].sum(),
    "colluder_bb_per_100": df["colluder_equity"].sum() / len(df) * 100 / 5
}

pd.DataFrame([summary]).to_csv("experiment_summary.csv", index=False)

df["cumulative_equity_0"] = df["net_0"].cumsum()
df["cumulative_equity_1"] = df["net_1"].cumsum()
df["cumulative_equity_colluders"] = df["colluder_equity"].cumsum()

plt.figure(figsize=(10, 5))
plt.plot(df["cumulative_equity_0"], label="Player 0")
plt.plot(df["cumulative_equity_1"], label="Player 1")
plt.plot(df["cumulative_equity_colluders"], label="Colluders Total", linestyle='--')
plt.xlabel("Hand Index")
plt.ylabel("Cumulative Equity (Chips)")
plt.title("Cumulative Equity of Colluding Agents")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("colluder_cumulative_equity.png")
plt.close()

print("✅ experiment_summary.csv and colluder_cumulative_equity.png saved.")
