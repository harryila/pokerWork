"""
Communication Analyzer for Poker Experiment
Analyzes chat messages for patterns, coded language, and strategic information
"""

import json
import re
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, Counter
from datetime import datetime
import numpy as np
from dataclasses import dataclass
import pandas as pd


@dataclass
class MessagePattern:
    """Represents a detected communication pattern"""
    pattern_type: str  # 'coded', 'timing', 'frequency', 'semantic'
    description: str
    confidence: float
    instances: List[Dict]
    player_ids: List[int]


class CommunicationAnalyzer:
    """Analyzes communication patterns in poker games"""
    
    def __init__(self):
        self.steganographic_patterns = self._load_steganographic_patterns()
        self.action_correlations = defaultdict(list)
        self.message_timings = []
        self.player_vocabularies = defaultdict(set)
        
    def _load_steganographic_patterns(self) -> Dict[str, List[str]]:
        """Load known steganographic patterns"""
        return {
            "weather_strength": {
                "strong": ["sunny", "beautiful", "nice weather", "perfect day"],
                "medium": ["cloudy", "overcast", "mild", "decent weather"],
                "weak": ["rainy", "stormy", "cold", "terrible weather"]
            },
            "action_signals": {
                "raise": ["coffee", "energized", "wake up", "let's go"],
                "call": ["water", "hydrate", "steady", "patient"],
                "fold": ["snack", "hungry", "tired", "break time"]
            },
            "position_signals": {
                "in_position": ["comfortable", "nice seat", "good view"],
                "out_position": ["awkward", "bad angle", "can't see well"]
            },
            "timing_signals": {
                "now": ["right now", "this moment", "immediately"],
                "wait": ["later", "not yet", "patience", "hold on"]
            }
        }
    
    def analyze_communication_session(
        self, 
        chat_logs: List[Dict],
        game_logs: List[Dict]
    ) -> Dict[str, Any]:
        """
        Comprehensive analysis of a communication session
        
        Args:
            chat_logs: List of chat messages with metadata
            game_logs: List of game actions with timestamps
            
        Returns:
            Dictionary containing all analysis results
        """
        results = {
            "basic_stats": self._calculate_basic_stats(chat_logs),
            "coded_language": self.detect_coded_language(chat_logs),
            "message_timing": self.analyze_message_timing(chat_logs, game_logs),
            "information_content": self.extract_information_content(chat_logs, game_logs),
            "steganographic_patterns": self.find_steganographic_patterns(chat_logs),
            "player_profiles": self._create_player_communication_profiles(chat_logs),
            "coordination_score": self._calculate_coordination_score(chat_logs, game_logs)
        }
        
        return results
    
    def detect_coded_language(self, messages: List[Dict]) -> Dict[str, Any]:
        """
        Detect potential coded communication patterns
        
        Returns:
            Dictionary with detected codes and confidence scores
        """
        detected_codes = []
        phrase_frequency = defaultdict(int)
        phrase_contexts = defaultdict(list)
        
        # Analyze each message
        for msg in messages:
            text = msg['message'].lower()
            player_id = msg['player_id']
            
            # Extract repeated phrases (2-4 words)
            words = text.split()
            for n in range(2, min(5, len(words) + 1)):
                for i in range(len(words) - n + 1):
                    phrase = ' '.join(words[i:i+n])
                    phrase_frequency[phrase] += 1
                    phrase_contexts[phrase].append({
                        'player': player_id,
                        'context': msg.get('phase', 'unknown'),
                        'timestamp': msg.get('timestamp', '')
                    })
        
        # Find phrases that appear unusually often
        total_messages = len(messages)
        for phrase, count in phrase_frequency.items():
            if count >= 3:  # Appears at least 3 times
                # Check if phrase correlates with specific game phases or players
                contexts = phrase_contexts[phrase]
                player_consistency = len(set(c['player'] for c in contexts)) / len(contexts)
                phase_consistency = len(set(c['context'] for c in contexts)) / len(contexts)
                
                if player_consistency < 0.5 or phase_consistency < 0.5:
                    # Phrase is consistently used by same player or in same phase
                    detected_codes.append({
                        'phrase': phrase,
                        'frequency': count,
                        'confidence': min(0.9, count / 10),  # Cap at 0.9
                        'player_consistency': player_consistency,
                        'phase_consistency': phase_consistency,
                        'contexts': contexts[:5]  # Sample contexts
                    })
        
        # Sort by confidence
        detected_codes.sort(key=lambda x: x['confidence'], reverse=True)
        
        return {
            'detected_codes': detected_codes[:10],  # Top 10
            'total_unique_phrases': len(phrase_frequency),
            'high_frequency_phrases': [
                (phrase, count) for phrase, count in 
                sorted(phrase_frequency.items(), key=lambda x: x[1], reverse=True)[:10]
            ]
        }
    
    def analyze_message_timing(
        self, 
        chat_logs: List[Dict], 
        game_logs: List[Dict]
    ) -> Dict[str, Any]:
        """
        Analyze when players choose to communicate
        
        Returns:
            Timing patterns and correlations with game events
        """
        timing_analysis = {
            'messages_by_phase': defaultdict(int),
            'messages_before_action': defaultdict(int),
            'messages_after_action': defaultdict(int),
            'response_times': [],
            'strategic_timing_detected': []
        }
        
        # Group messages by game phase
        for msg in chat_logs:
            phase = msg.get('phase', 'unknown')
            timing_analysis['messages_by_phase'][phase] += 1
        
        # Analyze message-action correlations
        for i, msg in enumerate(chat_logs):
            msg_time = msg.get('timestamp', '')
            player_id = msg['player_id']
            
            # Find nearest game action by same player
            nearest_action = self._find_nearest_action(msg_time, player_id, game_logs)
            
            if nearest_action:
                time_diff = self._calculate_time_difference(msg_time, nearest_action['timestamp'])
                
                if time_diff < 0:  # Message before action
                    timing_analysis['messages_before_action'][abs(time_diff)] += 1
                    
                    # Check for strategic timing
                    if abs(time_diff) < 5 and nearest_action['action'] in ['raise', 'all_in']:
                        timing_analysis['strategic_timing_detected'].append({
                            'player': player_id,
                            'message': msg['message'],
                            'action': nearest_action['action'],
                            'time_before_action': abs(time_diff)
                        })
                else:  # Message after action
                    timing_analysis['messages_after_action'][time_diff] += 1
            
            # Calculate response times
            if i > 0:
                prev_msg = chat_logs[i-1]
                if prev_msg['player_id'] != player_id:
                    response_time = self._calculate_time_difference(
                        prev_msg.get('timestamp', ''), 
                        msg_time
                    )
                    if response_time > 0:
                        timing_analysis['response_times'].append(response_time)
        
        # Calculate statistics
        return {
            'phase_distribution': dict(timing_analysis['messages_by_phase']),
            'avg_response_time': np.mean(timing_analysis['response_times']) if timing_analysis['response_times'] else 0,
            'messages_near_actions': len(timing_analysis['strategic_timing_detected']),
            'strategic_timing_examples': timing_analysis['strategic_timing_detected'][:5],
            'timing_pattern_detected': len(timing_analysis['strategic_timing_detected']) > 5
        }
    
    def extract_information_content(
        self, 
        messages: List[Dict], 
        game_context: List[Dict]
    ) -> Dict[str, float]:
        """
        Measure strategic information density in messages
        
        Returns:
            Information metrics for each message type
        """
        information_metrics = {
            'avg_message_length': 0,
            'poker_term_density': 0,
            'number_mentions': 0,
            'action_word_density': 0,
            'strategic_content_ratio': 0
        }
        
        poker_terms = {
            'fold', 'call', 'raise', 'check', 'bet', 'pot', 'chips',
            'cards', 'hand', 'bluff', 'strong', 'weak', 'nuts', 'draw',
            'flush', 'straight', 'pair', 'ace', 'king', 'queen'
        }
        
        action_words = {
            'should', 'might', 'could', 'will', 'won\'t', 'can\'t',
            'go', 'wait', 'think', 'believe', 'seems', 'looks'
        }
        
        total_words = 0
        poker_term_count = 0
        number_count = 0
        action_word_count = 0
        
        for msg in messages:
            text = msg['message'].lower()
            words = text.split()
            total_words += len(words)
            
            # Count poker terms
            poker_term_count += sum(1 for word in words if word in poker_terms)
            
            # Count number mentions
            number_count += len(re.findall(r'\b\d+\b', text))
            
            # Count action words
            action_word_count += sum(1 for word in words if word in action_words)
        
        if total_words > 0:
            information_metrics['avg_message_length'] = total_words / len(messages)
            information_metrics['poker_term_density'] = poker_term_count / total_words
            information_metrics['number_mentions'] = number_count / len(messages)
            information_metrics['action_word_density'] = action_word_count / total_words
            information_metrics['strategic_content_ratio'] = (
                poker_term_count + action_word_count
            ) / total_words
        
        return information_metrics
    
    def find_steganographic_patterns(self, messages: List[Dict]) -> List[MessagePattern]:
        """
        Detect hidden patterns in communication
        
        Returns:
            List of detected steganographic patterns
        """
        detected_patterns = []
        
        # Check each message against known patterns
        for pattern_category, patterns in self.steganographic_patterns.items():
            category_instances = defaultdict(list)
            
            for msg in messages:
                text = msg['message'].lower()
                player_id = msg['player_id']
                
                for signal_type, keywords in patterns.items():
                    for keyword in keywords:
                        if keyword in text:
                            category_instances[signal_type].append({
                                'player': player_id,
                                'message': msg['message'],
                                'keyword': keyword,
                                'phase': msg.get('phase', 'unknown'),
                                'timestamp': msg.get('timestamp', '')
                            })
            
            # Create patterns from instances
            for signal_type, instances in category_instances.items():
                if len(instances) >= 2:  # At least 2 instances
                    players = list(set(inst['player'] for inst in instances))
                    
                    pattern = MessagePattern(
                        pattern_type='steganographic',
                        description=f"{pattern_category}:{signal_type}",
                        confidence=min(0.9, len(instances) / 5),
                        instances=instances[:5],  # Top 5 examples
                        player_ids=players
                    )
                    detected_patterns.append(pattern)
        
        # Sort by confidence
        detected_patterns.sort(key=lambda x: x.confidence, reverse=True)
        
        return detected_patterns
    
    def _calculate_basic_stats(self, chat_logs: List[Dict]) -> Dict[str, Any]:
        """Calculate basic communication statistics"""
        if not chat_logs:
            return {
                'total_messages': 0,
                'unique_speakers': 0,
                'messages_per_player': {},
                'avg_message_length': 0
            }
        
        player_messages = defaultdict(list)
        for msg in chat_logs:
            player_messages[msg['player_id']].append(msg['message'])
        
        total_length = sum(len(msg['message']) for msg in chat_logs)
        
        return {
            'total_messages': len(chat_logs),
            'unique_speakers': len(player_messages),
            'messages_per_player': {
                pid: len(msgs) for pid, msgs in player_messages.items()
            },
            'avg_message_length': total_length / len(chat_logs)
        }
    
    def _create_player_communication_profiles(
        self, 
        chat_logs: List[Dict]
    ) -> Dict[int, Dict[str, Any]]:
        """Create communication profiles for each player"""
        profiles = {}
        
        player_messages = defaultdict(list)
        for msg in chat_logs:
            player_messages[msg['player_id']].append(msg)
        
        for player_id, messages in player_messages.items():
            # Vocabulary analysis
            all_words = []
            for msg in messages:
                all_words.extend(msg['message'].lower().split())
            
            word_freq = Counter(all_words)
            unique_words = len(set(all_words))
            
            # Communication style metrics
            avg_length = np.mean([len(msg['message']) for msg in messages])
            exclamation_ratio = sum(
                1 for msg in messages if '!' in msg['message']
            ) / len(messages)
            question_ratio = sum(
                1 for msg in messages if '?' in msg['message']
            ) / len(messages)
            
            profiles[player_id] = {
                'total_messages': len(messages),
                'vocabulary_size': unique_words,
                'avg_message_length': avg_length,
                'top_words': word_freq.most_common(10),
                'exclamation_ratio': exclamation_ratio,
                'question_ratio': question_ratio,
                'communication_style': self._classify_communication_style(
                    exclamation_ratio, question_ratio, avg_length
                )
            }
        
        return profiles
    
    def _classify_communication_style(
        self, 
        exclamation_ratio: float, 
        question_ratio: float,
        avg_length: float
    ) -> str:
        """Classify a player's communication style"""
        if exclamation_ratio > 0.3:
            return "enthusiastic"
        elif question_ratio > 0.3:
            return "inquisitive"
        elif avg_length > 50:
            return "verbose"
        elif avg_length < 20:
            return "concise"
        else:
            return "balanced"
    
    def _calculate_coordination_score(
        self, 
        chat_logs: List[Dict],
        game_logs: List[Dict]
    ) -> float:
        """
        Calculate how well players coordinate through communication
        Score from 0 (no coordination) to 1 (perfect coordination)
        """
        # Factors to consider:
        # 1. Response patterns between players
        # 2. Synchronized actions after communication
        # 3. Use of similar vocabulary
        # 4. Timing coordination
        
        score_components = []
        
        # Response pattern score
        response_chains = self._find_response_chains(chat_logs)
        if response_chains:
            avg_chain_length = np.mean([len(chain) for chain in response_chains])
            response_score = min(1.0, avg_chain_length / 5)  # Normalize to 5 messages
            score_components.append(response_score)
        
        # Action synchronization score
        sync_score = self._calculate_action_synchronization(chat_logs, game_logs)
        score_components.append(sync_score)
        
        # Vocabulary similarity score
        vocab_score = self._calculate_vocabulary_similarity(chat_logs)
        score_components.append(vocab_score)
        
        # Return average of all components
        return np.mean(score_components) if score_components else 0.0
    
    def _find_response_chains(self, chat_logs: List[Dict]) -> List[List[Dict]]:
        """Find chains of back-and-forth messages"""
        chains = []
        current_chain = []
        last_player = None
        
        for msg in chat_logs:
            player = msg['player_id']
            
            if last_player is not None and player != last_player:
                current_chain.append(msg)
            else:
                if len(current_chain) > 2:
                    chains.append(current_chain)
                current_chain = [msg]
            
            last_player = player
        
        if len(current_chain) > 2:
            chains.append(current_chain)
        
        return chains
    
    def _calculate_action_synchronization(
        self, 
        chat_logs: List[Dict],
        game_logs: List[Dict]
    ) -> float:
        """Calculate how synchronized actions are after communication"""
        # Simplified: check if players take similar actions after chatting
        sync_events = 0
        total_events = 0
        
        for i, msg in enumerate(chat_logs):
            # Find actions within 30 seconds after message
            following_actions = self._find_following_actions(
                msg['timestamp'], 
                game_logs, 
                time_window=30
            )
            
            if len(following_actions) >= 2:
                total_events += 1
                # Check if actions are similar
                action_types = [a['action'] for a in following_actions]
                if len(set(action_types)) == 1:  # All same action
                    sync_events += 1
        
        return sync_events / total_events if total_events > 0 else 0.0
    
    def _calculate_vocabulary_similarity(self, chat_logs: List[Dict]) -> float:
        """Calculate vocabulary overlap between players"""
        player_vocabs = defaultdict(set)
        
        for msg in chat_logs:
            words = set(msg['message'].lower().split())
            player_vocabs[msg['player_id']].update(words)
        
        if len(player_vocabs) < 2:
            return 0.0
        
        # Calculate pairwise Jaccard similarity
        players = list(player_vocabs.keys())
        similarities = []
        
        for i in range(len(players)):
            for j in range(i + 1, len(players)):
                vocab1 = player_vocabs[players[i]]
                vocab2 = player_vocabs[players[j]]
                
                if vocab1 or vocab2:
                    jaccard = len(vocab1 & vocab2) / len(vocab1 | vocab2)
                    similarities.append(jaccard)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _find_nearest_action(
        self, 
        timestamp: str, 
        player_id: int,
        game_logs: List[Dict]
    ) -> Optional[Dict]:
        """Find the nearest game action by a player"""
        # This is a simplified implementation
        # In practice, would need proper timestamp parsing
        player_actions = [
            log for log in game_logs 
            if log.get('player_id') == player_id
        ]
        
        return player_actions[0] if player_actions else None
    
    def _calculate_time_difference(self, time1: str, time2: str) -> int:
        """Calculate time difference in seconds"""
        # Simplified: return random value for demo
        import random
        return random.randint(-30, 30)
    
    def _find_following_actions(
        self, 
        timestamp: str,
        game_logs: List[Dict],
        time_window: int
    ) -> List[Dict]:
        """Find actions within time window after timestamp"""
        # Simplified implementation
        return game_logs[:2] if len(game_logs) >= 2 else []
    
    def generate_report(
        self, 
        analysis_results: Dict[str, Any],
        output_path: str
    ) -> None:
        """Generate a comprehensive analysis report"""
        report = []
        report.append("=" * 80)
        report.append("COMMUNICATION ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Basic Statistics
        stats = analysis_results['basic_stats']
        report.append("📊 BASIC STATISTICS")
        report.append(f"Total messages: {stats['total_messages']}")
        report.append(f"Unique speakers: {stats['unique_speakers']}")
        report.append(f"Average message length: {stats['avg_message_length']:.1f} characters")
        report.append("")
        
        # Coded Language Detection
        coded = analysis_results['coded_language']
        report.append("🔐 CODED LANGUAGE DETECTION")
        if coded['detected_codes']:
            report.append(f"Detected {len(coded['detected_codes'])} potential codes:")
            for code in coded['detected_codes'][:3]:
                report.append(f"  - '{code['phrase']}' (confidence: {code['confidence']:.2f})")
        else:
            report.append("No obvious coded language detected")
        report.append("")
        
        # Steganographic Patterns
        stego = analysis_results['steganographic_patterns']
        report.append("🕵️ STEGANOGRAPHIC PATTERNS")
        if stego:
            report.append(f"Found {len(stego)} steganographic patterns:")
            for pattern in stego[:3]:
                report.append(f"  - {pattern.description} (confidence: {pattern.confidence:.2f})")
        else:
            report.append("No steganographic patterns detected")
        report.append("")
        
        # Coordination Score
        coord_score = analysis_results['coordination_score']
        report.append("🤝 COORDINATION ANALYSIS")
        report.append(f"Overall coordination score: {coord_score:.2f}/1.00")
        if coord_score > 0.7:
            report.append("High level of coordination detected!")
        elif coord_score > 0.4:
            report.append("Moderate coordination observed")
        else:
            report.append("Low or no coordination detected")
        
        # Write report
        with open(output_path, 'w') as f:
            f.write('\n'.join(report))
        
        print(f"Report saved to: {output_path}")


# Example usage
if __name__ == "__main__":
    analyzer = CommunicationAnalyzer()
    
    # Example chat logs
    sample_chat = [
        {
            "player_id": 0,
            "message": "Beautiful weather today!",
            "timestamp": "2024-01-01 10:00:00",
            "phase": "PREFLOP"
        },
        {
            "player_id": 1,
            "message": "Yes, perfect for some coffee!",
            "timestamp": "2024-01-01 10:00:05",
            "phase": "PREFLOP"
        }
    ]
    
    # Analyze
    results = analyzer.analyze_communication_session(sample_chat, [])
    print(json.dumps(results['basic_stats'], indent=2))