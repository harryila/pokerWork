#!/usr/bin/env python3
"""
Test script for improved logging system.
This will run a few hands and verify that per-action logging works.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from game_environment.mixed_player_game import MixedPlayerGame
from utils.logging_utils import HandHistoryLogger
import os

def test_improved_logging():
    """Test the improved logging system with a few hands."""
    
    # Create logger with custom directory
    log_dir = "data/test_logs"
    logger = HandHistoryLogger(log_dir=log_dir)
    
    print(f"✅ Testing improved logging system...")
    print(f"📁 Log directory: {log_dir}")
    
    # Run a small game with 3 hands
    game = MixedPlayerGame(
        buyin=500,
        big_blind=5,
        small_blind=2,
        max_players=3,
        llm_player_ids=[0, 1, 2],  # All AI players
        collusion_llm_player_ids=[0, 1],  # Players 0 and 1 collude
        openai_model="gpt-4o",
        openai_api_key=None,  # Will use .env file
        num_hands=3,  # Just 3 hands for testing
        logger=logger
    )
    
    try:
        game.run_game()
        print("✅ Game completed successfully!")
        
        # Check what files were created
        if os.path.exists(log_dir):
            files = os.listdir(log_dir)
            print(f"📄 Created {len(files)} log files:")
            for file in sorted(files):
                print(f"   - {file}")
        else:
            print("❌ No log directory created!")
            
    except Exception as e:
        print(f"❌ Error during game: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_improved_logging() 