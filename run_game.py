# run_game.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
from game_environment.mixed_player_game import MixedPlayerGame
from utils.logging_utils import HandHistoryLogger
import inspect
print("[DEBUG] Logger loaded from:", inspect.getfile(HandHistoryLogger))
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--buyin", type=int, default=500)
parser.add_argument("--big-blind", type=int, default=1)
parser.add_argument("--small-blind", type=int, default=2)
parser.add_argument("--max-players", type=int, default=3)
parser.add_argument("--llm-players", type=str, default="0,1,2")
parser.add_argument("--collusion-llm-players", type=str, default="1,2")
parser.add_argument("--model", type=str, default="gpt-3.5-turbo")
parser.add_argument("--api-key", type=str, required=True)

args = parser.parse_args()

llm_ids = [int(x) for x in args.llm_players.split(",")]
collusion_ids = [int(x) for x in args.collusion_llm_players.split(",")]

base_dir = Path(__file__).resolve().parent
logger = HandHistoryLogger(log_dir=base_dir / "data" / "debug_logs")

game = MixedPlayerGame(
    buyin=args.buyin,
    big_blind=args.big_blind,
    small_blind=args.small_blind,
    max_players=args.max_players,
    llm_player_ids=llm_ids,
    collusion_llm_player_ids=collusion_ids,
    openai_model=args.model,
    openai_api_key=args.api_key,
    logger=logger  # ✅ REQUIRED
)

game.run_game()
