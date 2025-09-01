"""
Communicating LLM Agent for Texas Hold'em Poker.
Adds communication capabilities on top of a standard LLM agent.
"""

from typing import Tuple, Optional, Dict, Any, List
import random

from texasholdem.texasholdem.game.game import TexasHoldEm
from texasholdem.texasholdem.game.action_type import ActionType
from texasholdem.texasholdem.game.hand_phase import HandPhase
from texasholdem.texasholdem.game.player_state import PlayerState

from utils.safe_json_parse import safe_json_parse
from llm_prompts import build_communication_game_prompt


class CommunicatingLLMAgent:
    """
    Base class for LLM agents that can both act and communicate.
    Returns a 4-tuple (ActionType, amount, reasoning, message).
    """

    def __init__(
        self,
        model,
        tokenizer=None,
        api_key: Optional[str] = None,
        communication_style: str = "emergent",
        teammate_ids: Optional[List[int]] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.api_key = api_key
        self.communication_style = communication_style
        self.teammate_ids = teammate_ids or []
        self.is_hf = tokenizer is not None
        self.client = None

        # Lazy init OpenAI API (non-HF path)
        if not self.is_hf and api_key:
            import openai
            openai.api_key = api_key
            self.client = openai  # Use the module directly

    # -----------------------------------------------------------
    # Communication Helpers
    # -----------------------------------------------------------

    def should_send_message(self, game: TexasHoldEm, player_id: int) -> bool:
        """Check whether this agent should send a message this phase."""
        if not hasattr(game, "allow_communication") or not game.allow_communication():
            return False
        try:
            if game.get_messages_this_hand(player_id) >= getattr(
                game, "max_messages_per_hand", 0
            ):
                return False
        except Exception:
            pass
        return True

    def generate_message(self, game: TexasHoldEm, player_id: int, stage: Optional[str] = None) -> str:
        """
        Default chat line for communication rounds.
        Short, natural, non-revealing; respects any message length limits.
        Subclasses can override.
        """
        # Determine current street
        phase = stage or getattr(game, "hand_phase", None) or "PREFLOP"
        phase_str = getattr(phase, "name", str(phase)).upper()

        templates = {
            "PREFLOP": [
                "Let’s see a flop and feel it out.",
                "Keeping it light preflop.",
                "Small ball sounds fine.",
                "No need to go big yet.",
            ],
            "FLOP": [
                "Let’s keep it steady and see the turn.",
                "Curious to see their reaction here.",
                "Staying measured on this board.",
                "We can take a small stab.",
            ],
            "TURN": [
                "Let’s not overcommit on the turn.",
                "Still keeping it disciplined.",
                "We’ll see what the river brings.",
                "Measured pressure could work.",
            ],
            "RIVER": [
                "Let’s see how they respond first.",
                "Value spot maybe—staying precise.",
                "Keeping it tight on the river.",
                "No rush here.",
            ],
        }
        lines = templates.get(phase_str, templates["PREFLOP"])

        # Deterministic but varied pick
        try:
            idx_source = getattr(game, "num_hands", None) or getattr(game, "hand_number", None)
            if isinstance(idx_source, int):
                msg = lines[(idx_source + player_id) % len(lines)]
            else:
                msg = random.choice(lines)
        except Exception:
            msg = random.choice(lines)

        # Respect any message length limit the engine may enforce
        try:
            limit = getattr(game, "message_length_limit", None)
            if isinstance(limit, int) and limit > 0 and len(msg) > limit:
                msg = msg[: max(0, limit - 3)] + "..."
        except Exception:
            pass

        return msg

    # -----------------------------------------------------------
    # Action + Communication
    # -----------------------------------------------------------

    def get_action_with_communication(
        self, game: TexasHoldEm, player_id: int
    ) -> Tuple[ActionType, Optional[int], Optional[str], Optional[str]]:
        """
        Unified decision: action + optional message.
        Always returns a 4-tuple: (action_type, amount, reasoning, message)
        """
        try:
            # recent history
            recent_messages = game.get_chat_history(player_id, hand_id=game.num_hands)[-10:]

            # format state
            hole_cards = self._format_hole_cards(game, player_id)
            board_cards = self._format_board_cards(game)
            betting_history = self._format_betting_history(game)

            # build prompt
            prompt = build_communication_game_prompt(
                hole_cards=hole_cards,
                board_cards=board_cards,
                betting_history=betting_history,
                chat_history=recent_messages,
                teammate_ids=self.teammate_ids,
                communication_style=self.communication_style,
            )

            # run LLM
            if not self.is_hf:
                response = self.client.ChatCompletion.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a poker agent. Respond ONLY in JSON."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.7,
                    max_tokens=200,
                )
                content = response.choices[0].message.content.strip()
                # Extract JSON
                json_start = content.find("{")
                json_end = content.rfind("}") + 1
                if json_start >= 0 and json_end > json_start:
                    content = content[json_start:json_end]
                response = safe_json_parse(content)
            else:
                # If you add a HF path, implement _generate_llm_response in subclasses.
                response_text = self._generate_llm_response(prompt, max_tokens=200)  # type: ignore[attr-defined]
                response = safe_json_parse(response_text)

            # parse
            if isinstance(response, dict):
                action = response.get("action", "fold").lower()
                amount = response.get("amount", 0)
                reasoning = response.get("reasoning", "")
                message = None
                if response.get("send_message", False):
                    message = response.get("message", "")

                action_type = self._string_to_action_type(action)
                return action_type, amount, reasoning, message
            else:
                return ActionType.FOLD, None, "Failed to parse response", None

        except Exception as e:
            print(f"[ERROR] CommunicatingLLMAgent.get_action_with_communication failed for player {player_id}: {e}")
            import traceback
            traceback.print_exc()
            return ActionType.FOLD, None, f"CommunicatingLLMAgent error: {str(e)}", None

    # -----------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------

    def _string_to_action_type(self, action: str) -> ActionType:
        mapping = {
            "fold": ActionType.FOLD,
            "check": ActionType.CHECK,
            "call": ActionType.CALL,
            "raise": ActionType.RAISE,
        }
        return mapping.get(action.lower(), ActionType.FOLD)

    def _format_hole_cards(self, game: TexasHoldEm, player_id: int) -> str:
        try:
            hole_cards = game.get_hand(player_id)
            return f"{hole_cards[0]} {hole_cards[1]}"
        except Exception:
            return "Unknown"

    def _format_board_cards(self, game: TexasHoldEm) -> str:
        if not game.board:
            return "No cards yet"
        return " ".join(str(c) for c in game.board)

    def _format_betting_history(self, game: TexasHoldEm) -> str:
        history = []
        if game.hand_history:
            for phase in [
                HandPhase.PREFLOP,
                HandPhase.FLOP,
                HandPhase.TURN,
                HandPhase.RIVER,
            ]:
                if phase in game.hand_history and game.hand_history[phase]:
                    for action in game.hand_history[phase].actions:
                        action_name = (
                            action.action_type.name
                            if hasattr(action.action_type, "name")
                            else str(action.action_type)
                        )
                        amount_str = f" ${action.total}" if getattr(action, "total", None) else ""
                        history.append(
                            f"Player {action.player_id}: {action_name.lower()}{amount_str}"
                        )
        return " | ".join(history[-5:]) if history else ""
