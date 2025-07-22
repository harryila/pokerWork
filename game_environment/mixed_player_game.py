print("✅ USING official-llm-poker-collusion-main version")

"""
Mixed player game implementation for Texas Hold'em poker.
This module provides a game where some players are controlled by LLMs and others are human-controlled.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
print("[DEBUG] Forced path:", Path(__file__).resolve().parent.parent)
import os
import time
from typing import List, Dict, Optional, Tuple, Set, Union
from utils.logging_utils import HandHistoryLogger  # add it here
from dotenv import load_dotenv
from texasholdem.texasholdem.game.game import TexasHoldEm
#from texasholdem.texasholdem.gui.text_gui import TextGUI
from texasholdem.texasholdem.game.action_type import ActionType
from game_environment.llm_agent import LLMAgent
from game_environment.collusion_llm_agent import CollusionLLMAgent
from game_environment.preflop_strategy import load_preflop_chart, lookup_action
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import accelerate
import torch
from transformers.utils import logging
logging.set_verbosity_debug()
import traceback
import json
from datetime import datetime

class MixedPlayerGame:
    """
    A Texas Hold'em game where some players are controlled by LLMs and others are human-controlled.
    """

    def __init__(
    self,
    buyin: int = 500,
    big_blind: int = 5,
    small_blind: int = 2,
    max_players: int = 6,
    llm_player_ids: Optional[List[int]] = None,
    collusion_llm_player_ids: Optional[List[int]] = None,
    openai_model: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    num_hands: int = 10,
    logger: Optional[HandHistoryLogger] = None  # ✅ add this line
):

        """
        Initialize the mixed player game.

        Args:
            buyin: The amount of chips each player starts with
            big_blind: The big blind amount
            small_blind: The small blind amount
            max_players: The maximum number of players
            llm_player_ids: The IDs of players controlled by regular LLM. If None, players 0 and 1 will be LLM-controlled.
            collusion_llm_player_ids: The IDs of players controlled by collusion LLM. If None, no players will be collusion LLM-controlled.
            openai_model: The model name to use. If None, will try to get from .env file
            openai_api_key: The API key. If None, will try to get from .env file
        """
        # Load environment variables from .env file
        load_dotenv()

        # Dynamically resolve model path relative to the current project
        project_root = Path(__file__).resolve().parent.parent
        model_path = (project_root / "workspace" / "models" / "Llama-3.2-3B-Instruct").as_posix()
        print(f"[DEBUG] Using model path from current repo: {model_path}")

        from transformers import AutoModelForCausalLM

        self.hf_model = "gpt-4o"  # or "gpt-4" if available
        self.hf_tokenizer = None  # not needed for OpenAI

        print("[DEBUG] Successfully loaded LLM model & tokenizer.")

        # No tokenizer is created here; each agent will load its own tokenizer on demand

        self.game = TexasHoldEm(
            buyin=buyin,
            big_blind=big_blind,
            small_blind=small_blind,
            max_players=max_players,
        )
        self.gui = None

        # Set up AI players
        if llm_player_ids is None:
            llm_player_ids = [0, 1, 2, 3, 4, 5]  # Make all players LLM-controlled

        self.llm_player_ids = set(llm_player_ids)
        self.collusion_llm_player_ids = set(collusion_llm_player_ids)
        self.human_player_ids = (
            set(range(max_players))
            - self.llm_player_ids
            - self.collusion_llm_player_ids
        )

        self.num_hands = num_hands
        self.logger = logger or HandHistoryLogger(log_dir="C:/Users/Krish Jain/Downloads/official-llm-poker-collusion-main/data/debug_logs")

        # Load the preflop strategy table
        self.preflop_strategy = load_preflop_chart('preflop_chart.csv')


        # Initialize AI agents
        self.ai_agents = {}

        # Initialize regular LLM agents with the shared Hugging Face model
        for player_id in self.llm_player_ids:
            self.ai_agents[player_id] = LLMAgent(model=self.hf_model, tokenizer=self.hf_tokenizer)

        # Initialize collusion LLM agents
        if len(collusion_llm_player_ids) == 2:
            player1, player2 = sorted(collusion_llm_player_ids)
            self.ai_agents[player1] = CollusionLLMAgent(
            model=self.hf_model,
            tokenizer=self.hf_tokenizer,
            api_key=openai_api_key,
            teammate_id=player2
        )
        self.ai_agents[player2] = CollusionLLMAgent(
            model=self.hf_model,
            tokenizer=self.hf_tokenizer,
            api_key=openai_api_key,
            teammate_id=player1
)

    def _is_ai_player(self, player_id: int) -> bool:
        """
        Check if a player is controlled by AI.

        Args:
            player_id: The ID of the player to check

        Returns:
            True if the player is controlled by AI, False otherwise
        """
        return (
            player_id in self.llm_player_ids
            or player_id in self.collusion_llm_player_ids
        )

    def _get_ai_action(self, player_id: int) -> Tuple[ActionType, Optional[int], str]:
        print(f"[DEBUG] Calling _get_ai_action for player_id={player_id}")
        # Check if we're in the preflop round
        # TEMPORARILY DISABLED PREFLOP STRATEGY TO FORCE LLM EXECUTION
        if player_id in self.ai_agents:
            agent = self.ai_agents[player_id]
            print("[DEBUG] Forcing LLM get_action even in preflop...")
            return agent.get_action(self.game, player_id)

        # Normal postflop or other phase: call LLM or CFR
        if (
            player_id not in self.llm_player_ids
            and player_id not in self.collusion_llm_player_ids
        ):
            raise ValueError(f"Player {player_id} is not an LLM player")

        agent = self.ai_agents[player_id]
        return agent.get_action(self.game, player_id)


    def _get_human_action(self) -> Tuple[ActionType, Optional[int]]:
        print("[DEBUG] Auto-folding human player to avoid loop.")
        self.game.take_action(ActionType.FOLD)
        return ActionType.FOLD, None
        # Use the GUI to get the action from the human player
        #self.gui.run_step()

        # The action is already taken by the GUI, so we just return None
        return None, None

    def run_game(self):
        """
        Run the game until it's over.
        """
        error_message = None
        try:
            num_hands_played = 0
            while self.game.is_game_running() and num_hands_played < self.num_hands:

                print("[DEBUG] Starting new hand...")
                self.game.start_hand()
                hand_log = {
                    "hand_id": self.game.get_hand_id(),
                    "actions": [],
                    "winner": None,
                    "pot": None
                }

                print(f"[DEBUG] Hand running? {self.game.is_hand_running()}")

                while self.game.is_hand_running():
                    current_player = self.game.current_player
                    print(f"[DEBUG] Current player: {current_player}")

                    if self._is_ai_player(current_player):
                        # Get action from AI
                        result = self._get_ai_action(current_player)
                        
                        if result is None:
                            print(f"[ERROR] Agent returned None. Forcing fold.")
                            action_type, total, reason = ActionType.FOLD, None, None
                        else:
                            action_type, total, reason = result
                            print(f"[DEBUG] ActionType: {action_type}, Total: {total}, Reason: {reason}")


                        # Take the action
                        try:
                            if action_type == ActionType.RAISE and total is not None:
                                print(f"[DEBUG] Taking RAISE action with total={total}")
                                self.game.take_action(action_type, total=total)
                            else:
                                print(f"[DEBUG] Taking action: {action_type}")
                                self.game.take_action(action_type)
                            model_name = getattr(self.ai_agents[current_player], "model", "unknown")
                            hand_log["actions"].append({
                                "player_id": current_player,
                                "model": str(model_name),
                                "action": action_type.name.lower(),
                                "amount": total if total is not None else 0
                                })

                        except Exception as e:
                            print(f"[ERROR] Action failed: {e}. Forcing fold.")
                            self.game.take_action(ActionType.FOLD)

                    else:
                        # Get action from human
                        self._get_human_action()
                
                    try:
                        hand_log["winner"] = self.game.get_winner()
                        hand_log["pot"] = self.game._get_last_pot().get_total_amount()
                    except Exception as e:
                        print(f"[WARNING] Could not extract winner or pot size: {e}")

                # Export and replay the hand history
                try:
                    pgn_path = self.game.export_history("./data/pgns")
                    json_path = self.game.hand_history.export_history_json("./data/json")
                    try:
                        winning_player = self.game.get_winner()
                        hand_log["winner"] = winning_player
                        hand_log["pot"] = self.game._get_last_pot().get_total_amount()
                    except Exception as e:
                        print(f"[WARNING] Could not determine winner or pot: {e}")

                    print(f"\nExported hand history to:")
                    print(f"PGN: {pgn_path}")
                    print(f"JSON: {json_path}")
                except Exception as e:
                    print(f"[WARNING] Failed to export hand history: {e}")
            # ✅ Force settlement if needed
            if self.game.hand_phase != HandPhase.SETTLE:
                print("[WARNING] Forcing hand settlement.")
                self.game.settle_hand()

            # ✅ Extract winner and pot again after forced settlement
            try:
                winning_player = self.game.get_winner()
                hand_log["winner"] = winning_player
                hand_log["pot"] = self.game._get_last_pot().get_total_amount()
            except Exception as e:
                print(f"[WARNING] Could not determine winner or pot: {e}")

            try:
                winning_player = self.game.get_winner()
                hand_log["winner"] = winning_player

                pot_total = self.game._get_last_pot().get_total_amount()
                if isinstance(pot_total, (int, float)):
                    hand_log["pot"] = pot_total
                else:
                    print(f"[WARNING] Pot total is not a number: {pot_total}")
                    hand_log["pot"] = None

            except Exception as e:
                print(f"[WARNING] Could not determine winner or pot: {e}")
                hand_log["pot"] = None

            self.logger.log_hand(hand_log, hand_log["hand_id"])

            num_hands_played += 1
            time.sleep(1)

        except Exception as e:
            # Save the error message and include full traceback
            error_message = f"\nError occurred: {str(e)}\n{traceback.format_exc()}"
        else:
            # No error occurred
            error_message = None
        finally:

            # Always clean up the curses session
            #self.gui.hide()
            # Reset the terminal
            # os.system("reset")  # Commented out because 'reset' is for Unix, not Windows

            # Display the error message after cleanup if there was one
            if error_message:
                print(error_message)


if __name__ == "__main__":
    import argparse
    from utils.logging_utils import HandHistoryLogger

    parser = argparse.ArgumentParser()
    parser.add_argument("--num-hands", type=int, default=10)
    args = parser.parse_args()

    logger = HandHistoryLogger(log_dir="C:/Users/Krish Jain/Downloads/official-llm-poker-collusion-main/data/debug_logs")

    game = MixedPlayerGame(
        buyin=500,
        big_blind=1,
        small_blind=2,
        max_players=2,
        llm_player_ids=[],
        collusion_llm_player_ids=[0, 1],
        openai_model="gpt-4o",
        openai_api_key=None,
        num_hands=args.num_hands,
        logger=logger 
    )

    game.run_game()
