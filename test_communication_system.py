#!/usr/bin/env python3
"""
Test script for the communication-enabled poker system.
Demonstrates various communication scenarios and collusion strategies.
"""

import os
from dotenv import load_dotenv

from game_environment.mixed_player_communication_game import MixedPlayerCommunicationGame
from utils.communication_logger import CommunicationLogger
from communication_protocols import EXPERIMENTAL_PRESETS

# Load environment variables
load_dotenv()


def test_basic_communication():
    """Test basic communication without collusion."""
    print("\n" + "="*80)
    print("TEST 1: Basic Communication (No Collusion)")
    print("="*80)
    
    # Create logger
    logger = CommunicationLogger(base_dir="data/communication_tests")
    
    # Create game with communication
    game = MixedPlayerCommunicationGame(
        buyin=500,
        big_blind=5,
        small_blind=2,
        max_players=4,
        llm_player_ids=[0, 1, 2, 3],  # All AI players
        collusion_llm_player_ids=[],  # No collusion
        communication_config={
            "level": "moderate",
            "style": "cooperative",
            "strategy": None
        },
        num_hands=3,
        logger=logger,
        use_local_llm=True  # Use local model for testing
    )
    
    # Run the game
    results = game.run_game()
    
    print(f"\n✅ Test 1 Complete!")
    print(f"Communication stats: {results['communication_stats']}")
    
    return results


def test_steganographic_collusion():
    """Test steganographic communication with collusion."""
    print("\n" + "="*80)
    print("TEST 2: Steganographic Collusion")
    print("="*80)
    
    # Create logger
    logger = CommunicationLogger(base_dir="data/communication_tests")
    
    # Use preset configuration
    preset = EXPERIMENTAL_PRESETS["full_steganography"]
    
    # Create game with steganographic collusion
    game = MixedPlayerCommunicationGame(
        buyin=500,
        big_blind=5,
        small_blind=2,
        max_players=6,
        llm_player_ids=[2, 3, 4, 5],  # Regular AI players
        collusion_llm_player_ids=[0, 1],  # Colluding players
        communication_config={
            "level": preset["level"],
            "style": preset["style"],
            "strategy": preset["strategy"]
        },
        num_hands=5,
        logger=logger,
        use_local_llm=True
    )
    
    # Run the game
    results = game.run_game()
    
    print(f"\n✅ Test 2 Complete!")
    print(f"Detected signals: {results['communication_stats'].get('potential_signals_detected', 0)}")
    
    return results


def test_limited_communication():
    """Test limited communication with restrictions."""
    print("\n" + "="*80)
    print("TEST 3: Limited Communication")
    print("="*80)
    
    # Create logger
    logger = CommunicationLogger(base_dir="data/communication_tests")
    
    # Create game with limited communication
    game = MixedPlayerCommunicationGame(
        buyin=500,
        big_blind=5,
        small_blind=2,
        max_players=4,
        llm_player_ids=[0, 1, 2, 3],
        collusion_llm_player_ids=[0, 1],  # Colluding pair
        communication_config={
            "level": "limited",
            "style": "subtle",
            "strategy": "information_sharing"
        },
        num_hands=3,
        logger=logger,
        use_local_llm=True
    )
    
    # Run the game
    results = game.run_game()
    
    print(f"\n✅ Test 3 Complete!")
    print(f"Messages per hand limit respected: {results['communication_stats']['total_messages'] <= 3 * 2 * 4}")
    
    return results


def test_no_communication_baseline():
    """Test baseline with no communication."""
    print("\n" + "="*80)
    print("TEST 4: No Communication Baseline")
    print("="*80)
    
    # Create logger
    logger = CommunicationLogger(base_dir="data/communication_tests")
    
    # Create game without communication
    game = MixedPlayerCommunicationGame(
        buyin=500,
        big_blind=5,
        small_blind=2,
        max_players=4,
        llm_player_ids=[0, 1, 2, 3],
        collusion_llm_player_ids=[0, 1],
        communication_config={
            "level": "none",
            "style": "cooperative",
            "strategy": "signal_and_squeeze"
        },
        num_hands=3,
        logger=logger,
        use_local_llm=True
    )
    
    # Run the game
    results = game.run_game()
    
    print(f"\n✅ Test 4 Complete!")
    print(f"Confirmed no messages: {results['communication_stats'].get('total_messages', 0) == 0}")
    
    return results


def main():
    """Run all communication tests."""
    print("\n" + "="*80)
    print("🃏 POKER COMMUNICATION SYSTEM TEST SUITE")
    print("="*80)
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\n⚠️  Warning: No OpenAI API key found. Using local models for testing.")
        print("   For production, set OPENAI_API_KEY in your .env file")
    
    # Run tests
    all_results = []
    
    try:
        # Test 1: Basic communication
        results1 = test_basic_communication()
        all_results.append(("Basic Communication", results1))
        
        # Test 2: Steganographic collusion
        results2 = test_steganographic_collusion()
        all_results.append(("Steganographic Collusion", results2))
        
        # Test 3: Limited communication
        results3 = test_limited_communication()
        all_results.append(("Limited Communication", results3))
        
        # Test 4: No communication baseline
        results4 = test_no_communication_baseline()
        all_results.append(("No Communication", results4))
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Summary
    print("\n" + "="*80)
    print("📊 TEST SUMMARY")
    print("="*80)
    
    for test_name, results in all_results:
        comm_stats = results.get('communication_stats', {})
        print(f"\n{test_name}:")
        print(f"  - Hands played: {results['hands_played']}")
        print(f"  - Total messages: {comm_stats.get('total_messages', 0)}")
        print(f"  - Unique speakers: {comm_stats.get('unique_speakers', 0)}")
        print(f"  - Signals detected: {comm_stats.get('potential_signals_detected', 0)}")
        print(f"  - Dataset path: {results.get('dataset_path', 'N/A')}")
    
    print("\n✅ All tests completed successfully!")
    print("\n💡 Next steps:")
    print("  1. Review communication transcripts in the simulation folders")
    print("  2. Analyze detected steganographic patterns")
    print("  3. Compare win rates across different communication levels")
    print("  4. Export datasets for further analysis")


if __name__ == "__main__":
    main()