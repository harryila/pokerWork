import os
import json
import pandas as pd

# Set JSON directory path
json_dir = "data/json"

rows = []
for fname in sorted(os.listdir(json_dir)):
    if not fname.endswith('.json'):
        continue

    with open(os.path.join(json_dir, fname), 'r') as f:
        hand = json.load(f)

    hand_id = fname.replace('.json', '')
    stages = ['preflop', 'flop', 'turn', 'river']
    actions_0 = []
    actions_1 = []

    for stage in stages:
        if hand.get(stage) and hand[stage].get('actions'):
            for act in hand[stage]['actions']:
                if act['player_id'] == 0:
                    actions_0.append(act)
                elif act['player_id'] == 1:
                    actions_1.append(act)

    if actions_0 and actions_1:
        def get_first_action(actions, pid):
            for a in actions:
                if a['player_id'] == pid:
                    return a['action_type'], a.get('value') or 0
            return None, 0

        a0, v0 = get_first_action(actions_0, 0)
        a1, v1 = get_first_action(actions_1, 1)

        rows.append({
            'hand_id': hand_id,
            'player_0_action': a0,
            'player_0_amount': v0,
            'player_1_action': a1,
            'player_1_amount': v1
        })

df = pd.DataFrame(rows)
df.to_csv("collusion_actions.csv", index=False)
print("✅ Saved: collusion_actions.csv")
