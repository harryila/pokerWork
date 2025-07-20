# 🃏 Multiagent Poker Collusion with LLMs

This project simulates 1000+ hands of Texas Hold'em between LLM-based poker agents, including two colluding players using GPT-4o. The goal is to detect and quantify ex-ante collusion using Mutual Information (MI) analysis.

---

## 🎯 Objective

Analyze whether two LLM agents can coordinate strategies and gain an unfair advantage in a multiagent poker game. We:
- Simulate hands using a custom poker environment
- Use GPT-4o to power CollusionLLMAgents
- Export detailed hand histories in JSON/PGN
- Compute Mutual Information to detect collusion

---

## 🛠️ How to Run

### ✅ 1. Install Requirements
```bash
pip install -r requirements.txt
```

### ✅ 2. Simulate Hands
```bash
python -m game_environment.mixed_player_game --num-hands 1000
```

### ✅ 3. Generate Analysis
```bash
python generate_collusion_actions.py
python generate_mi_metrics.py
python generate_experiment_metrics.py
```

---

## 📁 Output Files

| File | Description |
|------|-------------|
| `collusion_actions.csv` | Extracted actions & bet amounts |
| `collusion_mi_batches.csv` | Mutual Information (MI) per batch |
| `collusion_mi_batches_plot.png` | MI trend plot |
| `experiment_summary.csv` | Win rates, pot size, BB/100 |
| `colluder_cumulative_equity.png` | Equity over time |

---

## 🤖 Agent Setup

This repo includes:
- `CollusionLLMAgent`: Strategically aligned LLM-based poker bot
- `mixed_player_game.py`: Game runner
- `data/json/`: Hand-level logs
- `generate_*.py`: Analysis scripts

---

## 📊 Key Findings (GPT-4o Results)

- Colluding agents won nearly 100% of hands
- Positive MI suggests coordination
- Colluders extracted up to **7.89 BB/100**

---

## 📚 Acknowledgements

- Built with OpenAI’s GPT-4o
- Inspired by research in multiagent alignment and collusion detection
- Developed by [Krish Jain](https://github.com/krishjainm)

---

## 🧠 Citation / Attribution

If you use or reference this work, please cite the project or contact the author.
