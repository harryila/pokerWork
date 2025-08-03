#!/usr/bin/env python3
"""
Command-line runner for the communication-enabled poker game.
This script allows you to run experiments with different models and configurations.
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
from game_environment.mixed_player_communication_game import MixedPlayerCommunicationGame
from utils.communication_logger import CommunicationLogger

def main():
    parser = argparse.ArgumentParser(description="Run communication-enabled poker experiments")
    parser.add_argument("--num-hands", type=int, default=3, help="Number of hands to play")
    parser.add_argument("--buyin", type=int, default=500, help="Starting chips per player")
    parser.add_argument("--big-blind", type=int, default=5, help="Big blind amount")
    parser.add_argument("--small-blind", type=int, default=2, help="Small blind amount")
    parser.add_argument("--max-players", type=int, default=4, help="Maximum number of players")
    parser.add_argument("--llm-players", type=str, default="0,1,2,3", help="Comma-separated LLM player IDs")
    parser.add_argument("--collusion-llm-players", type=str, default="0,1", help="Comma-separated colluding player IDs")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="OpenAI model to use")
    parser.add_argument("--api-key", type=str, help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--communication-level", type=str, default="moderate", 
                       choices=["none", "limited", "moderate", "full"], 
                       help="Communication level")
    parser.add_argument("--communication-style", type=str, default="steganographic",
                       choices=["cooperative", "subtle", "steganographic", "deceptive"],
                       help="Communication style")
    parser.add_argument("--collusion-strategy", type=str, default="signal_and_squeeze",
                       choices=["signal_and_squeeze", "chip_dumping", "information_sharing", "whipsaw"],
                       help="Collusion strategy")
    parser.add_argument("--use-local-llm", action="store_true", help="Use local LLM instead of API")
    parser.add_argument("--output-dir", type=str, default="data/communication_experiments", 
                       help="Output directory for results")

    args = parser.parse_args()

    # Parse player IDs
    llm_ids = [int(x) for x in args.llm_players.split(",")]
    collusion_ids = [int(x) for x in args.collusion_llm_players.split(",")]

    # Get API key
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key and not args.use_local_llm:
        print("❌ Error: No OpenAI API key provided!")
        print("   Set OPENAI_API_KEY environment variable or use --api-key")
        print("   Or use --use-local-llm for local testing")
        return 1

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create logger
    logger = CommunicationLogger(base_dir=str(output_dir))

    print("=" * 60)
    print("🎮 COMMUNICATION-ENABLED POKER EXPERIMENT")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Hands: {args.num_hands}")
    print(f"Players: {args.max_players}")
    print(f"LLM Players: {llm_ids}")
    print(f"Colluding Players: {collusion_ids}")
    print(f"Communication Level: {args.communication_level}")
    print(f"Communication Style: {args.communication_style}")
    print(f"Collusion Strategy: {args.collusion_strategy}")
    print(f"Use Local LLM: {args.use_local_llm}")
    print(f"Output Directory: {output_dir}")
    print("=" * 60)

    # Create the game
    try:
        game = MixedPlayerCommunicationGame(
            buyin=args.buyin,
            big_blind=args.big_blind,
            small_blind=args.small_blind,
            max_players=args.max_players,
            llm_player_ids=llm_ids,
            collusion_llm_player_ids=collusion_ids,
            communication_config={
                "level": args.communication_level,
                "style": args.communication_style,
                "strategy": args.collusion_strategy
            },
            openai_model=args.model,
            openai_api_key=api_key,
            num_hands=args.num_hands,
            logger=logger,
            use_local_llm=args.use_local_llm
        )

        # Run the game
        print("\n🎮 Starting game...")
        results = game.run_game()

        print("\n✅ Experiment completed successfully!")
        print(f"📊 Results saved to: {output_dir}")
        
        # Show quick stats
        stats = logger.get_communication_stats()
        print(f"\n📈 Quick Statistics:")
        print(f"   Total messages: {stats.get('total_messages', 0)}")
        print(f"   Unique speakers: {stats.get('unique_speakers', 0)}")
        print(f"   Messages by player: {stats.get('messages_per_player', {})}")
        print(f"   Potential signals: {stats.get('potential_signals_detected', 0)}")

        return 0

    except Exception as e:
        print(f"\n❌ Error during experiment: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main()) 