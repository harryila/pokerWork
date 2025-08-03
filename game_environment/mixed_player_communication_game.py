"""
Mixed player game with communication support for Texas Hold'em experiments.
Extends the base MixedPlayerGame with chat functionality and communication rounds.
"""

import os
from typing import List, Optional, Dict, Tuple, Any
from datetime import datetime

from texasholdem.texasholdem.game.game import TexasHoldEm
from texasholdem.texasholdem.game.action_type import ActionType
from texasholdem.texasholdem.game.player_state import PlayerState
from texasholdem.texasholdem.game.hand_phase import HandPhase

from game_environment.mixed_player_game import MixedPlayerGame
from game_environment.communicating_llm_agent import CommunicatingLLMAgent
from game_environment.advanced_collusion_agent import AdvancedCollusionAgent
from game_environment.llm_agent import LLMAgent

from utils.communication_logger import CommunicationLogger
from communication_protocols import (
    COMMUNICATION_LEVELS,
    get_communication_config,
    validate_message,
    get_signal_meaning
)


class MixedPlayerCommunicationGame(MixedPlayerGame):
    """
    Extended poker game that supports natural language communication between players.
    """
    
    def __init__(
        self,
        buyin: int = 500,
        big_blind: int = 5,
        small_blind: int = 2,
        max_players: int = 6,
        llm_player_ids: Optional[List[int]] = None,
        collusion_llm_player_ids: Optional[List[int]] = None,
        communication_config: Optional[Dict[str, Any]] = None,
        openai_model: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        num_hands: int = 10,
        logger: Optional[CommunicationLogger] = None,
        use_local_llm: bool = False
    ):
        """
        Initialize the communication-enabled game.
        
        Args:
            buyin: Starting chips for each player
            big_blind: Big blind amount
            small_blind: Small blind amount
            max_players: Maximum number of players
            llm_player_ids: IDs of regular LLM players
            collusion_llm_player_ids: IDs of colluding LLM players
            communication_config: Communication configuration dict
            openai_model: OpenAI model name
            openai_api_key: OpenAI API key
            num_hands: Number of hands to play
            logger: Communication logger instance
            use_local_llm: Whether to use local LLM for testing
        """
        # Initialize base game
        super().__init__(
            buyin=buyin,
            big_blind=big_blind,
            small_blind=small_blind,
            max_players=max_players,
            llm_player_ids=llm_player_ids,
            collusion_llm_player_ids=collusion_llm_player_ids,
            openai_model=openai_model,
            openai_api_key=openai_api_key,
            num_hands=num_hands,
            logger=logger
        )
        
        # Communication configuration
        self.communication_config = communication_config or {
            "level": "none",
            "style": "cooperative",
            "strategy": None
        }
        
        # Use communication logger if provided
        if logger and isinstance(logger, CommunicationLogger):
            self.logger = logger
        else:
            self.logger = CommunicationLogger()
        
        # Enable communication in the game
        self._setup_communication()
        
        # Replace agents with communication-enabled versions
        self._upgrade_agents_to_communication(use_local_llm)
        
        # Track communication rounds
        self.communication_round_messages = []
        
    def _get_game_state_for_logging(self):
        """Extract game state for logging purposes."""
        from utils.game_state_extractor import extract_complete_game_state
        return extract_complete_game_state(self.game, self.game.current_player)
        
    def _create_hand_summary(self):
        """Create a summary of the current hand."""
        try:
            winning_player = self.game.get_winner()
            pot_size = self.game._get_last_pot().get_total_amount()
            
            # Calculate chip differences
            player_chips_after = {p.player_id: p.chips for p in self.game.players}
            
            hand_summary = {
                "winner": winning_player,
                "pot": pot_size,
                "final_chips": player_chips_after
            }
            
            return hand_summary
        except Exception as e:
            print(f"[WARNING] Could not create hand summary: {e}")
            return {"error": str(e)}
            
    def _calculate_final_statistics(self):
        """Calculate final statistics for the simulation."""
        return {
            "total_hands": self.game.num_hands,
            "final_chips": {p.player_id: p.chips for p in self.game.players},
            "collusion_players": list(self.collusion_llm_player_ids),
            "llm_players": list(self.llm_player_ids),
            "human_players": list(self.human_player_ids),
            "communication_config": self.communication_config
        }
        
    def _setup_communication(self):
        """Configure communication in the game based on config."""
        config = get_communication_config(
            level=self.communication_config.get("level", "none"),
            style=self.communication_config.get("style", "cooperative"),
            strategy=self.communication_config.get("strategy")
        )
        
        level_config = config["level"]
        
        # Enable communication in the game
        if level_config["enabled"]:
            self.game.enable_communication(
                level=self.communication_config["level"],
                max_messages_per_hand=level_config.get("messages_per_hand"),
                message_length_limit=level_config.get("message_length"),
                allowed_phases=level_config.get("allowed_phases", [])
            )
            
            print(f"🗣️ Communication enabled: {level_config['description']}")
        else:
            print("🔇 Communication disabled for this game")
    
    def _upgrade_agents_to_communication(self, use_local_llm: bool = False):
        """Upgrade existing agents to communication-enabled versions."""
        new_agents = {}
        
        for player_id, agent in self.ai_agents.items():
            # Determine teammate IDs for this agent
            if player_id in self.collusion_llm_player_ids:
                teammate_ids = [
                    pid for pid in self.collusion_llm_player_ids 
                    if pid != player_id
                ]
            else:
                teammate_ids = []
            
            # Create appropriate communication agent
            if player_id in self.collusion_llm_player_ids:
                # Create advanced collusion agent
                new_agent = AdvancedCollusionAgent(
                    model=agent.model if hasattr(agent, 'model') else self.openai_model,
                    tokenizer=agent.tokenizer if hasattr(agent, 'tokenizer') else None,
                    api_key=self.openai_api_key,
                    communication_style=self.communication_config.get("style", "steganographic"),
                    teammate_ids=teammate_ids,
                    use_local_llm=use_local_llm,
                    collusion_strategy=self.communication_config.get("strategy", "signal_and_squeeze")
                )
                print(f"🤝 Upgraded player {player_id} to AdvancedCollusionAgent")
            else:
                # Create regular communicating agent
                new_agent = CommunicatingLLMAgent(
                    model=agent.model if hasattr(agent, 'model') else self.openai_model,
                    tokenizer=agent.tokenizer if hasattr(agent, 'tokenizer') else None,
                    api_key=self.openai_api_key,
                    communication_style="cooperative",
                    teammate_ids=[],
                    use_local_llm=use_local_llm
                )
                print(f"💬 Upgraded player {player_id} to CommunicatingLLMAgent")
            
            new_agents[player_id] = new_agent
        
        self.ai_agents = new_agents
    
    def _handle_communication_round(self):
        """Handle a round of communication between players."""
        if not self.game.allow_communication():
            return
        
        print(f"\n💬 Communication Round - {self.game.hand_phase.name}")
        
        # Clear round messages
        self.communication_round_messages = []
        
        # Get list of active players who can communicate
        active_players = list(self.game.in_pot_iter())
        
        # Each active AI player gets a chance to send a message
        for player_id in active_players:
            if self._is_ai_player(player_id):
                agent = self.ai_agents[player_id]
                
                # Check if agent wants to send a message
                if isinstance(agent, (CommunicatingLLMAgent, AdvancedCollusionAgent)):
                    if agent.should_send_message(self.game, player_id):
                        # Generate and send message
                        message = agent.generate_message(self.game, player_id)
                        
                        # Validate message
                        is_valid, reason = validate_message(
                            message, 
                            self.communication_config["level"]
                        )
                        
                        if is_valid:
                            # Add message to game
                            success = self.game.add_chat_message(
                                player_id=player_id,
                                message=message
                            )
                            
                            if success:
                                print(f"  Player {player_id}: \"{message}\"")
                                
                                # Check for signals
                                signals = get_signal_meaning(
                                    message,
                                    self.communication_config["style"]
                                )
                                
                                # Log the message
                                game_state = self._get_game_state_for_logging()
                                self.logger.log_chat_message(
                                    hand_id=self.game.num_hands,
                                    phase=self.game.hand_phase.name,
                                    player_id=player_id,
                                    message=message,
                                    target_player=None,
                                    game_state=game_state,
                                    contains_signal=signals is not None
                                )
                                
                                # Track for round logging
                                self.communication_round_messages.append({
                                    "player_id": player_id,
                                    "message": message,
                                    "signals": signals
                                })
                                
                                # Log signal detection if found
                                if signals:
                                    for category, signal_info in signals.items():
                                        self.logger.log_steganographic_detection(
                                            hand_id=self.game.num_hands,
                                            player_id=player_id,
                                            message=message,
                                            detected_signal=signal_info["meaning"],
                                            confidence=signal_info["confidence"]
                                        )
                        else:
                            print(f"  Player {player_id}: [Message rejected - {reason}]")
        
        # Log the complete communication round
        if self.communication_round_messages:
            self.logger.log_communication_round(
                hand_id=self.game.num_hands,
                phase=self.game.hand_phase.name,
                all_messages=self.communication_round_messages,
                game_state=self._get_game_state_for_logging()
            )
    
    def _get_ai_action_with_communication(
        self, 
        player_id: int
    ) -> Tuple[ActionType, Optional[int], str, Optional[str]]:
        """
        Get action from AI agent with optional communication.
        
        Returns:
            Tuple of (action_type, amount, reasoning, message)
        """
        if player_id not in self.ai_agents:
            print(f"[ERROR] No AI agent found for player {player_id}")
            return ActionType.FOLD, None, "No AI agent", None
        
        try:
            agent = self.ai_agents[player_id]
            
            # Use unified action + communication method if available
            if isinstance(agent, (CommunicatingLLMAgent, AdvancedCollusionAgent)):
                action_type, total, reason, message = agent.get_action_with_communication(
                    self.game, player_id
                )
                
                # Handle message if provided
                if message and self.game.allow_communication():
                    # Validate message
                    is_valid, reject_reason = validate_message(
                        message,
                        self.communication_config["level"]
                    )
                    
                    if is_valid:
                        success = self.game.add_chat_message(
                            player_id=player_id,
                            message=message
                        )
                        
                        if success:
                            print(f"  💬 Player {player_id}: \"{message}\"")
                            
                            # Log the message
                            game_state = self._get_game_state_for_logging()
                            signals = get_signal_meaning(
                                message,
                                self.communication_config["style"]
                            )
                            
                            self.logger.log_chat_message(
                                hand_id=self.game.num_hands,
                                phase=self.game.hand_phase.name,
                                player_id=player_id,
                                message=message,
                                target_player=None,
                                game_state=game_state,
                                contains_signal=signals is not None
                            )
                
                return action_type, total, reason or "AI decision", message
            else:
                # Fallback to regular action
                action_type, total, reason = agent.get_action(self.game, player_id)
                return action_type, total, reason or "AI decision", None
                
        except Exception as e:
            print(f"[ERROR] AI agent error for player {player_id}: {e}")
            return ActionType.FOLD, None, f"AI error: {str(e)}", None
    
    def run_game(self):
        """
        Run the game with communication support.
        """
        print(f"\n{'='*60}")
        print(f"🎮 STARTING COMMUNICATION-ENABLED POKER GAME")
        print(f"{'='*60}")
        print(f"Players: {self.game.max_players}")
        print(f"Regular LLM players: {self.llm_player_ids}")
        print(f"Collusion LLM players: {self.collusion_llm_player_ids}")
        print(f"Communication level: {self.communication_config['level']}")
        print(f"Communication style: {self.communication_config['style']}")
        print(f"Collusion strategy: {self.communication_config.get('strategy', 'None')}")
        print(f"Target hands: {self.num_hands}")
        print(f"{'='*60}\n")
        
        # Start logging
        self.logger.start_simulation()
        
        hands_played = 0
        
        while hands_played < self.num_hands and self.game.game_state == self.game.game_state.RUNNING:
            hands_played += 1
            print(f"\n{'='*50}")
            print(f"HAND {hands_played}")
            print(f"{'='*50}")
            
            # Start new hand
            self.game.start_hand()
            
            # Communication before preflop
            if self.game.hand_phase == HandPhase.PREFLOP:
                self._handle_communication_round()
            
            # Run the hand with communication
            while self.game.is_hand_running():
                current_player = self.game.current_player
                
                # Get game state for logging
                game_state = self._get_game_state_for_logging()
                
                if self._is_ai_player(current_player):
                    # Get action with possible communication
                    action_type, total, reason, message = self._get_ai_action_with_communication(current_player)
                    
                    # Log the action
                    self.logger.log_action(
                        hand_id=self.game.num_hands,
                        phase=self.game.hand_phase.name,
                        player_id=current_player,
                        action_type=action_type.name,
                        amount=total,
                        reason=reason,
                        game_state=game_state
                    )
                else:
                    # Human player (placeholder)
                    action_type, total = self._get_human_action()
                    reason = "Human decision"
                
                # Take the action
                self.game.take_action(action_type, total=total)
                
                # Check if we've moved to a new phase and allow communication
                if self.game.hand_phase != HandPhase.PREHAND and self.game.is_hand_running():
                    # Only communicate at phase transitions
                    last_phase = game_state.get("phase")
                    if last_phase != self.game.hand_phase.name:
                        self._handle_communication_round()
            
            # Log hand summary
            hand_summary = self._create_hand_summary()
            self.logger.log_hand_summary(self.game.num_hands, hand_summary)
            
            # Print results
            print(f"\nHand {hands_played} complete!")
            if self.game.hand_history and HandPhase.SETTLE in self.game.hand_history:
                settle_history = self.game.hand_history[HandPhase.SETTLE]
                print("\nWinners:")
                for pot_id, (pot_amount, winner_ids, pot_winners) in settle_history.pot_winners.items():
                    print(f"  Pot {pot_id} (${pot_amount}): Players {pot_winners}")
        
        # End simulation
        final_stats = self._calculate_final_statistics()
        self.logger.end_simulation(final_stats)
        
        # Export communication dataset
        dataset_path = self.logger.export_chat_dataset()
        
        # Create transcript
        transcript = self.logger.create_communication_transcript()
        
        # Print communication statistics
        comm_stats = self.logger.get_communication_stats()
        if comm_stats:
            print(f"\n{'='*50}")
            print("COMMUNICATION STATISTICS")
            print(f"{'='*50}")
            print(f"Total messages: {comm_stats['total_messages']}")
            print(f"Unique speakers: {comm_stats['unique_speakers']}")
            print(f"Average message length: {comm_stats['avg_message_length']:.1f}")
            print(f"Potential signals detected: {comm_stats['potential_signals_detected']}")
            print(f"Messages by player: {comm_stats['messages_per_player']}")
        
        print(f"\n✅ Game complete! Played {hands_played} hands")
        print(f"📊 Results saved to: {self.logger.get_simulation_path()}")
        print(f"💬 Chat dataset exported to: {dataset_path}")
        
        return {
            "hands_played": hands_played,
            "final_stats": final_stats,
            "communication_stats": comm_stats,
            "simulation_path": str(self.logger.get_simulation_path()),
            "dataset_path": dataset_path
        }