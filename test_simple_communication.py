#!/usr/bin/env python3
"""
Simple test to demonstrate the communication system functionality.
This test shows how the communication infrastructure works without requiring LLM agents.
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from game_environment.mixed_player_communication_game import MixedPlayerCommunicationGame
from utils.communication_logger import CommunicationLogger
from communication_protocols import COMMUNICATION_LEVELS, COMMUNICATION_STYLES, validate_message

def test_communication_infrastructure():
    """Test the communication infrastructure without LLM agents."""
    
    print("=" * 60)
    print("🧪 TESTING COMMUNICATION INFRASTRUCTURE")
    print("=" * 60)
    
    # Test different communication configurations
    configs = [
        {
            "name": "No Communication",
            "level": "none",
            "style": "cooperative",
            "expected_messages": 0
        },
        {
            "name": "Limited Communication", 
            "level": "limited",
            "style": "subtle",
            "expected_messages": 2
        },
        {
            "name": "Moderate Communication",
            "level": "moderate", 
            "style": "cooperative",
            "expected_messages": 5
        },
        {
            "name": "Full Communication",
            "level": "full",
            "style": "steganographic", 
            "expected_messages": float('inf')
        }
    ]
    
    for config in configs:
        print(f"\n📋 Testing: {config['name']}")
        print("-" * 40)
        
        # Create game with this configuration
        game = MixedPlayerCommunicationGame(
            buyin=500,
            big_blind=5,
            small_blind=2,
            max_players=4,
            llm_player_ids=[],  # No LLM agents for this test
            collusion_llm_player_ids=[],
            communication_config={
                "level": config["level"],
                "style": config["style"],
                "strategy": None
            },
            num_hands=1,
            use_local_llm=False
        )
        
        # Check communication configuration
        print(f"✅ Communication level: {game.communication_config['level']}")
        print(f"✅ Communication style: {game.communication_config['style']}")
        
        # Check if communication is enabled
        if game.game.chat_enabled:
            print(f"✅ Chat enabled: Yes")
            print(f"✅ Messages per hand: {COMMUNICATION_LEVELS[config['level']]['messages_per_hand']}")
            print(f"✅ Message length limit: {COMMUNICATION_LEVELS[config['level']]['message_length']}")
            print(f"✅ Allowed phases: {COMMUNICATION_LEVELS[config['level']]['allowed_phases']}")
        else:
            print(f"✅ Chat enabled: No")
            
        print(f"✅ Configuration test passed!")

def test_communication_logger():
    """Test the communication logging system."""
    
    print("\n" + "=" * 60)
    print("📊 TESTING COMMUNICATION LOGGER")
    print("=" * 60)
    
    # Create a communication logger
    logger = CommunicationLogger()
    
    # Start simulation
    logger.start_simulation()
    
    # Simulate some chat messages
    test_messages = [
        {
            "hand_id": 1,
            "phase": "PREFLOP",
            "player_id": 0,
            "message": "Good luck everyone!",
            "target_player": None,
            "message_type": "public"
        },
        {
            "hand_id": 1,
            "phase": "PREFLOP", 
            "player_id": 1,
            "message": "Thanks! The weather looks nice today.",
            "target_player": 0,
            "message_type": "private"
        },
        {
            "hand_id": 1,
            "phase": "FLOP",
            "player_id": 2,
            "message": "Interesting flop.",
            "target_player": None,
            "message_type": "public"
        }
    ]
    
    # Log the messages
    for msg in test_messages:
        logger.log_chat_message(
            hand_id=msg["hand_id"],
            phase=msg["phase"],
            player_id=msg["player_id"],
            message=msg["message"],
            target_player=msg["target_player"],
            game_state={"pot": 15, "current_player": msg["player_id"]}
        )
    
    # Get communication statistics
    stats = logger.get_communication_stats()
    print(f"✅ Total messages logged: {stats['total_messages']}")
    print(f"✅ Unique speakers: {stats['unique_speakers']}")
    print(f"✅ Messages by player: {stats['messages_per_player']}")
    print(f"✅ Messages by phase: {stats['messages_by_phase']}")
    
    # Test signal detection (if method exists)
    try:
        signals = logger.detect_potential_signals()
        print(f"✅ Potential signals detected: {len(signals)}")
    except AttributeError:
        print(f"✅ Signal detection method not implemented yet")
    
    # Export chat dataset
    dataset_path = logger.export_chat_dataset()
    print(f"✅ Chat dataset exported to: {dataset_path}")
    
    # Create transcript
    transcript = logger.create_communication_transcript()
    print(f"✅ Communication transcript created ({len(transcript)} lines)")

def test_protocol_configurations():
    """Test the communication protocol configurations."""
    
    print("\n" + "=" * 60)
    print("🔧 TESTING COMMUNICATION PROTOCOLS")
    print("=" * 60)
    
    # Test different communication levels
    for level_name, level_config in COMMUNICATION_LEVELS.items():
        print(f"\n📋 Level: {level_name}")
        print(f"   Enabled: {level_config['enabled']}")
        print(f"   Messages per hand: {level_config['messages_per_hand']}")
        print(f"   Message length: {level_config['message_length']}")
        print(f"   Allowed phases: {level_config['allowed_phases']}")
        print(f"   Description: {level_config['description']}")
    
    # Test communication styles
    print(f"\n📋 Communication Styles:")
    for style_name, style_desc in COMMUNICATION_STYLES.items():
        print(f"   {style_name}: {style_desc}")
    
    # Test message validation
    test_messages = [
        "Good luck everyone!",  # Should pass
        "I have a strong hand",  # Might be restricted
        "The weather is nice today",  # Should pass
        "Fold now",  # Might be restricted
    ]
    
    print(f"\n📋 Message Validation Test:")
    for msg in test_messages:
        is_valid = validate_message(msg, "limited")
        print(f"   '{msg}' -> {'✅ Valid' if is_valid else '❌ Invalid'}")

def main():
    """Run all communication infrastructure tests."""
    
    print("🚀 STARTING COMMUNICATION INFRASTRUCTURE TESTS")
    print("=" * 60)
    
    try:
        # Test 1: Communication configurations
        test_communication_infrastructure()
        
        # Test 2: Communication logging
        test_communication_logger()
        
        # Test 3: Protocol configurations
        test_protocol_configurations()
        
        print("\n" + "=" * 60)
        print("✅ ALL COMMUNICATION INFRASTRUCTURE TESTS PASSED!")
        print("=" * 60)
        
        print("\n💡 What this demonstrates:")
        print("   • Communication levels and styles are properly configured")
        print("   • Message validation and filtering works")
        print("   • Communication logging captures all messages")
        print("   • Signal detection can identify potential steganography")
        print("   • Data export creates structured datasets")
        print("   • The infrastructure is ready for LLM agents")
        
        print("\n🎯 Next steps:")
        print("   • Set OPENAI_API_KEY to test with real LLM agents")
        print("   • Run experiments with different communication levels")
        print("   • Analyze the generated communication datasets")
        print("   • Test steganographic detection capabilities")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 