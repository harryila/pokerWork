# 🔍 Logging Analysis & Fixes

## 📊 **Issues Identified**

### **1. Dual Logging System (Confusing)**
- **System A**: Custom `HandHistoryLogger` → `data/hand_history/`
- **System B**: texasholdem library export → `data/json/`
- **Problem**: No real-time per-action logging, only hand summaries

### **2. Card Rank Display Issue**
- **Raw JSON**: Shows integer ranks (0-12) instead of card names
- **LLM Input**: Properly formatted (2, 3, 4, 5, 6, 7, 8, 9, T, J, Q, K, A)
- **Problem**: Confusion between raw data and formatted display

### **3. Missing Per-Action Logging**
- **Current**: Only logs at hand completion
- **Need**: Real-time logging of each action and game state
- **Problem**: Can't analyze decision-making process during games

---

## 🎯 **What's Passed to LLMs**

### **Game State Format**:
```python
Current game state:
- Your position: 0 (SB)
- Small blind position: 0 (SB) 
- Big blind position: 1 (BB)
- Your hand: 7h, 0h  # ← 7h = 7♥, 0h = 2♥ (deuce of hearts)
- Community cards: Kh, 4c, 8h  # ← K♥, 4♣, 8♥
- Current phase: FLOP
- Pot amount: 15
- Your chips: 495
- Chips to call: 0
- Minimum raise: 5

Player positions and chips:
Position 0 (SB): 495 chips (Active)
Position 1 (BB): 510 chips (Active)

Betting history:
PREFLOP:
Position 3 (UTG): FOLD 
Position 4 (MP): FOLD
Position 5 (CO): FOLD
Position 0 (SB): CALL
Position 1 (BB): CALL
```

### **Available Actions**:
```python
{
    "CHECK": "Check (pass the action without betting)",
    "CALL": "Call (match the current bet of 0 chips)", 
    "RAISE": "33% of pot (5 chips), 50% of pot (7 chips), 66% of pot (10 chips), 125% of pot (18 chips), 2.5x previous bet (5 chips)"
}
```

---

## 🃏 **Card Rank System Explained**

### **Rank Mapping**:
```python
STR_RANKS = "23456789TJQKA"
# Index:    0 1 2 3 4 5 6 7 8 9 10 11 12
# Card:     2 3 4 5 6 7 8 9 T J  Q  K  A
```

### **Examples**:
```json
// Raw JSON data
{
  "rank": 0,   // = 2 (deuce)
  "suit": "hearts"
}
{
  "rank": 12,  // = A (ace) 
  "suit": "diamonds"
}
{
  "rank": 10,  // = Q (queen)
  "suit": "clubs" 
}

// Formatted for LLMs
"7h, 0h"  // 7♥, 2♥ (seven of hearts, deuce of hearts)
"Kh, 4c, 8h"  // K♥, 4♣, 8♥ (king of hearts, four of clubs, eight of hearts)
```

---

## 🛠️ **Fixes Implemented**

### **1. Enhanced Logging System**
```python
# New methods in HandHistoryLogger
def log_action(self, action_log: dict, hand_id: Union[int, str], action_num: int):
    """Log individual actions during the game for real-time analysis."""
    
def log_game_state(self, game_state: dict, hand_id: Union[int, str], phase: str):
    """Log the current game state before each action."""
```

### **2. Per-Action Logging in Game**
```python
# Before each action:
game_state = {
    "hand_id": self.game.get_hand_id(),
    "current_player": current_player,
    "phase": self.game.hand_phase.name,
    "pot_amount": self.game._get_last_pot().get_total_amount(),
    "player_chips": {p.player_id: p.chips for p in self.game.players},
    "community_cards": [...],  # With both raw and formatted data
    "current_player_hand": [...]  # With both raw and formatted data
}

# After each action:
action_log = {
    "hand_id": self.game.get_hand_id(),
    "action_num": len(hand_log["actions"]),
    "player_id": current_player,
    "action_type": action_type.name,
    "amount": total if total is not None else 0,
    "reason": reason,
    "phase": self.game.hand_phase.name,
    "timestamp": datetime.now().isoformat()
}
```

### **3. Enhanced Card Debugging**
```python
# Added to _format_game_state():
debug_data = {
    "player_id": player_id,
    "hand_cards": [
        {"rank": card.rank, "suit": card.suit, "str_rank": Card.STR_RANKS[card.rank], "str_suit": Card.INT_SUIT_TO_CHAR_SUIT[card.suit]}
        for card in hand
    ],
    "community_cards": [...],
    "formatted_hand": hand_str,
    "formatted_community": community_str
}
```

---

## 📁 **New File Structure**

### **Before**:
```
data/
├── json/           # texasholdem library exports (hand completion only)
│   ├── texas(1).json
│   ├── texas(2).json
│   └── ...
└── hand_history/   # Custom logger (summary only)
    ├── hand_1_summary_20250101_120000.json
    └── ...
```

### **After**:
```
data/
├── json/           # texasholdem library exports (unchanged)
├── hand_history/   # Enhanced custom logger
│   ├── hand_1_action_000_20250101_120000.json    # Per-action logs
│   ├── hand_1_action_001_20250101_120001.json
│   ├── hand_1_state_preflop_20250101_120000.json # Per-state logs
│   ├── hand_1_state_flop_20250101_120002.json
│   └── hand_1_summary_20250101_120005.json       # Hand summary
└── test_logs/      # Test directory for verification
```

---

## 🧪 **Testing**

### **Run Test**:
```bash
python3 test_improved_logging.py
```

### **Expected Output**:
```
✅ Testing improved logging system...
📁 Log directory: data/test_logs
✅ Game completed successfully!
📄 Created 15 log files:
   - hand_1_action_000_20250101_120000.json
   - hand_1_action_001_20250101_120001.json
   - hand_1_state_preflop_20250101_120000.json
   - hand_1_state_flop_20250101_120002.json
   - hand_1_summary_20250101_120005.json
   - ...
```

---

## 🎯 **Benefits of New System**

### **1. Real-Time Analysis**
- Can analyze decision-making process step-by-step
- See how LLMs react to different game states
- Track collusion patterns in real-time

### **2. Better Debugging**
- Both raw card data and formatted display
- Complete game state before each action
- Timestamps for temporal analysis

### **3. Enhanced Research**
- Per-action mutual information analysis
- Real-time collusion detection
- Detailed decision tree analysis

### **4. Improved Data Quality**
- No more confusion about card ranks
- Complete action context
- Better error tracking and debugging 