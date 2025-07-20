import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mutual_info_score
import numpy as np

# Load saved actions
df = pd.read_csv("collusion_actions.csv")

batch_size = 100
batch_data = []

for i in range(0, len(df), batch_size):
    batch = df.iloc[i:i+batch_size].copy()
    if len(batch) < batch_size:
        continue
    try:
        a_mi = mutual_info_score(batch['player_0_action'], batch['player_1_action'])
        batch['b0'] = pd.qcut(batch['player_0_amount'].astype(float), q=5, duplicates='drop')
        batch['b1'] = pd.qcut(batch['player_1_amount'].astype(float), q=5, duplicates='drop')
        amt_mi = mutual_info_score(batch['b0'], batch['b1'])
    except Exception:
        a_mi = mutual_info_score(batch['player_0_action'], batch['player_1_action'])
        amt_mi = np.nan

    batch_data.append({
        'batch_index': i // batch_size,
        'action_MI': a_mi,
        'amount_MI': amt_mi
    })

batch_df = pd.DataFrame(batch_data)
batch_df.to_csv("collusion_mi_batches.csv", index=False)

plt.figure(figsize=(10, 5))
plt.plot(batch_df['action_MI'], label='Action MI', marker='o')
plt.plot(batch_df['amount_MI'], label='Amount MI', marker='x')
plt.xlabel('Batch Index (100 hands per batch)')
plt.ylabel('Mutual Information')
plt.title('Mutual Information Between Colluding Players')
plt.legend()
plt.grid(True)
plt.savefig("collusion_mi_batches_plot.png")
plt.close()

print("✅ Files generated successfully!")
