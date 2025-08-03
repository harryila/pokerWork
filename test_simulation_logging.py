#!/usr/bin/env python3
"""
Test script for the new simulation logging system.
This will test the hierarchical logging with dummy agents.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.simulation_logger import SimulationLogger
from utils.game_state_extractor import extract_complete_game_state
import os
import json

# Create a dummy agent class for testing
class DummyAgent:
    def __init__(self, name="dummy"):
        self.name = name
        self.model = "dummy-agent"
    
    def get_action(self, game, player_id):
        """Simple dummy agent that always calls."""
        from texasholdem.texasholdem.game.action_type import ActionType
        return ActionType.CALL, None, "Dummy agent always calls"

def test_simulation_logging():
    """Test the new simulation logging system."""
    
    print("🎮 Testing simulation logging system...")
    
    # Import the game engine
    try:
        from texasholdem.texasholdem.game.game import TexasHoldEm
        from texasholdem.texasholdem.game.action_type import ActionType
        from texasholdem.texasholdem.game.hand_phase import HandPhase
        
        print("✅ Successfully imported Texas Hold'em game engine")
        
        # Create simulation logger
        logger = SimulationLogger()
        simulation_id = logger.start_simulation()
        
        print(f"📁 Simulation directory: {logger.get_simulation_path()}")
        
        # Create a simple game
        game = TexasHoldEm(
            buyin=500,
            big_blind=5,
            small_blind=2,
            max_players=2
        )
        
        print("✅ Created game instance")
        
        # Create dummy agents
        agents = {
            0: DummyAgent("dummy-1"),
            1: DummyAgent("dummy-2")
        }
        
        # Run a few hands to test logging
        for hand_num in range(2):
            print(f"\n🃏 Playing hand {hand_num + 1}...")
            
            # Start hand
            game.start_hand()
            hand_id = game.get_hand_id()
            
            print(f"   Hand ID: {hand_id}")
            print(f"   Current phase: {game.hand_phase.name}")
            
            # Play through the hand
            action_count = 0
            while game.is_hand_running():
                current_player = game.current_player
                print(f"   Player {current_player} to act...")
                
                # Extract complete game state
                game_state = extract_complete_game_state(game, current_player)
                
                # Get action from dummy agent
                agent = agents.get(current_player, agents[0])
                action_type, total, reason = agent.get_action(game, current_player)
                
                print(f"   Action: {action_type.name}, Amount: {total}, Reason: {reason}")
                
                # Log the action with complete game state
                logger.log_action(
                    hand_id=hand_id,
                    phase=game_state["phase"],
                    player_id=current_player,
                    action_type=action_type.name,
                    amount=total,
                    reason=reason,
                    game_state=game_state
                )
                
                # Take the action
                try:
                    if action_type == ActionType.RAISE and total is not None:
                        game.take_action(action_type, total=total)
                    else:
                        game.take_action(action_type)
                    print(f"   ✅ Action taken successfully")
                except Exception as e:
                    print(f"   ❌ Action failed: {e}")
                    game.take_action(ActionType.FOLD)
                
                action_count += 1
            
            # Hand is complete, log summary
            try:
                winner = game.get_winner()
                pot_size = game._get_last_pot().get_total_amount()
                
                hand_summary = {
                    "winner": winner,
                    "pot": pot_size,
                    "final_chips": {p.player_id: p.chips for p in game.players}
                }
                
                logger.log_hand_summary(hand_id, hand_summary)
                print(f"   ✅ Hand complete. Winner: {winner}, Pot: {pot_size}")
                
            except Exception as e:
                print(f"   ❌ Error getting hand result: {e}")
        
        # End simulation
        final_stats = {
            "total_hands": 2,
            "final_chips": {p.player_id: p.chips for p in game.players},
            "test_mode": True
        }
        
        logger.end_simulation(final_stats)
        
        print("\n✅ Simulation logging test completed!")
        
        # Check created files
        print("\n📄 Checking created files...")
        simulation_path = logger.get_simulation_path()
        if simulation_path and simulation_path.exists():
            game_logs_path = simulation_path / "game_logs"
            if game_logs_path.exists():
                files = list(game_logs_path.glob("*.json"))
                print(f"✅ Created {len(files)} log files in {game_logs_path}:")
                for file in sorted(files):
                    size = file.stat().st_size
                    print(f"   - {file.name} ({size} bytes)")
                    
                    # Show a sample of the content structure
                    try:
                        with open(file, 'r') as f:
                            content = json.load(f)
                            if isinstance(content, dict):
                                if "game_state" in content:
                                    print(f"     Contains complete game state with {len(content['game_state'].get('players', {}))} players")
                                elif "winner" in content:
                                    print(f"     Hand summary - Winner: {content.get('winner')}, Pot: {content.get('pot')}")
                    except Exception as e:
                        print(f"     Error reading file: {e}")
            else:
                print("❌ No game_logs directory created!")
        else:
            print("❌ No simulation directory created!")
            
    except Exception as e:
        print(f"❌ Error during simulation test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_simulation_logging() 