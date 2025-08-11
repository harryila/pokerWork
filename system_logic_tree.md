# Poker Communication System Logic Tree

## 1. SYSTEM INITIALIZATION
```
run_communication_game.py (main entry)
├── Parse command line arguments
├── Load environment variables (.env)
├── Create game configuration
└── Initialize MixedPlayerCommunicationGame
    ├── Create LLM agents for each player
    │   ├── Colluding players → AdvancedCollusionAgent
    │   └── Non-colluding players → CommunicatingLLMAgent
    ├── Set communication config (style, level)
    └── Initialize logging system
```

## 2. GAME LOOP (per hand)
```
MixedPlayerCommunicationGame.run_game()
├── Initialize new hand
├── For each betting round (PREFLOP, FLOP, TURN, RIVER):
│   ├── Communication Round
│   │   ├── For each player (if communication allowed):
│   │   │   ├── Check if player should send message
│   │   │   ├── Generate message (if applicable)
│   │   │   └── Validate and send message
│   │   └── Log communication data
│   │
│   └── Action Round
│       ├── For each player (in betting order):
│       │   ├── Get available actions from game state
│       │   ├── Generate LLM prompt with game context
│       │   ├── Get LLM response (action + optional message)
│       │   ├── Validate action against game rules
│       │   ├── Apply action to game
│       │   └── Log action data
│       └── Continue until betting round complete
│
└── Hand settlement and winner determination
```

## 3. LLM PROMPT GENERATION FLOW
```
get_action_with_communication()
├── Extract game state information
│   ├── Hole cards, board cards, pot size
│   ├── Betting history, player positions
│   ├── Available actions (CRITICAL: Should be dynamic)
│   └── Recent chat history
│
├── Build communication prompt
│   ├── Base communication style instructions
│   ├── Social context (poker table rules)
│   └── Message generation guidelines
│
├── Build action prompt
│   ├── Game state summary
│   ├── Available actions (CRITICAL BUG: Line 25 hardcoded)
│   ├── Betting context
│   └── Decision instructions
│
└── Send unified prompt to LLM
```

## 4. ACTION VALIDATION FLOW
```
Action Validation (Multiple Layers)
├── Layer 1: LLM Agent Level (_validate_action_for_game_state)
│   ├── Check if action type is valid
│   ├── Validate raise amounts
│   │   ├── Check minimum raise requirements
│   │   ├── Check maximum (player chips)
│   │   └── Handle all-in scenarios
│   └── Return corrected action
│
├── Layer 2: Game Environment Level (mixed_player_communication_game.py)
│   ├── Get available moves from game state
│   ├── Check if action is in available moves
│   ├── Validate raise amounts again
│   └── Force FOLD if invalid (final fallback)
│
└── Layer 3: Core Game Engine Level (texasholdem/game.py)
    ├── Final validation before applying action
    ├── Check game rules compliance
    └── Raise exceptions if invalid
```

## 5. COMMUNICATION FLOW
```
Message Generation and Validation
├── Check if communication is allowed
│   ├── Communication level restrictions
│   ├── Player type restrictions (colluding vs non-colluding)
│   └── Phase restrictions
│
├── Generate message content
│   ├── Apply communication style rules
│   ├── Include game context
│   └── Ensure natural conversation
│
├── Validate message
│   ├── Check length limits
│   ├── Check restricted words
│   └── Validate format
│
└── Send and log message
```

## 6. CRITICAL BUGS IDENTIFIED

### 6.1 Prompt Generation Bugs
- **BUG 1**: Line 25 in llm_prompts.py - Hardcoded actions `['fold', 'call', 'raise_min', 'raise_pot', 'all_in']` (NOT USED - function is deprecated)
- **BUG 2**: ✅ FIXED - Available actions are dynamically fetched from game state via `_get_available_actions()`
- **BUG 3**: ✅ FIXED - Betting context is properly included in prompts with betting round status

### 6.2 Action Validation Bugs
- **BUG 4**: ✅ FIXED - Multiple validation layers are working correctly
- **BUG 5**: ✅ FIXED - Raise amount validation is working correctly (increment vs total)
- **BUG 6**: ✅ FIXED - All-in scenarios are properly handled
- **BUG 7**: ✅ FIXED - Validation is applied to all player types
- **BUG 8**: ✅ FIXED - LLMs trying to RAISE when betting round is over are caught and forced to FOLD
- **BUG 9**: ✅ FIXED - Final validation layer is working correctly and catching invalid actions

### 6.3 Communication Bugs
- **BUG 8**: ✅ FIXED - Non-colluding players are properly blocked from communicating
- **BUG 9**: ✅ FIXED - Message validation correctly checks player permissions
- **BUG 10**: ✅ FIXED - Communication timing is working correctly
- **BUG 11**: ❌ ACTIVE - LLMs still occasionally using weather references despite prompt cleaning

### 6.4 Game State Bugs
- **BUG 11**: ✅ FIXED - Available actions are properly updated after each action
- **BUG 12**: ✅ FIXED - Betting round status is properly tracked
- **BUG 13**: ✅ FIXED - Pot calculations are correct
- **BUG 14**: ❌ ACTIVE - LLM prompt generation may be using stale game state between players

## 7. DATA FLOW ISSUES
```
Game State → LLM Prompt → LLM Response → Validation → Game Action
     ↑                                                      ↓
     └─────────────── Feedback Loop ←──────────────────────┘

Issues:
- Game state may be stale when prompt is generated
- Available actions not refreshed between players
- Betting context not updated in real-time
```

## 8. VALIDATION LAYER CONFLICTS
```
Layer 1 (LLM Agent)     Layer 2 (Game Env)     Layer 3 (Core Game)
     ↓                       ↓                       ↓
Validates action         Validates action        Final validation
Corrects if needed       Forces FOLD if bad      Raises exception
     ↓                       ↓                       ↓
May conflict with        May conflict with       May conflict with
Layer 2 logic           Layer 1 logic           Both layers
```

## 9. RECOMMENDED FIXES PRIORITY

### HIGH PRIORITY
1. ✅ FIXED - Hardcoded actions in prompts (Line 25) - function not used
2. ✅ FIXED - Dynamic available actions fetching - working correctly
3. ✅ FIXED - Validation logic - working correctly
4. ✅ FIXED - Raise amount validation - working correctly
5. ✅ FIXED - LLM prompt strengthened to prevent RAISE when betting round is over
6. ✅ FIXED - Final validation layer strengthened to catch invalid raises

### MEDIUM PRIORITY
5. Improve game state freshness
6. Fix communication permission checks
7. Enhance betting context in prompts

### LOW PRIORITY
8. Optimize validation performance
9. Improve error messages
10. Add more comprehensive logging
