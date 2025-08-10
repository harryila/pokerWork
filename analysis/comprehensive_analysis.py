#!/usr/bin/env python3
"""
Comprehensive Data Analysis System for Emergent Collusion Research

This module provides sophisticated analysis tools to extract meaningful insights from
poker game data, focusing on detecting emergent communication patterns, collusion
strategies, and coordination effectiveness.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import re
from scipy import stats
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import LabelEncoder

class CollusionAnalyzer:
    """Main analyzer for detecting collusion patterns and emergent communication."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.game_logs = []
        self.communication_logs = []
        self.action_sequences = []
        self.results = {}
        
    def load_data(self, simulation_id: Optional[str] = None):
        """Load game data from simulation logs."""
        if simulation_id:
            sim_dir = self.data_dir / "simulations" / simulation_id
        else:
            # Get most recent simulation
            sim_dirs = sorted((self.data_dir / "simulations").glob("*"))
            if not sim_dirs:
                print("No simulation data found")
                return
            sim_dir = sim_dirs[-1]
        
        print(f"Loading data from: {sim_dir}")
        
        # Load game logs
        game_log_file = sim_dir / "game_log.json"
        if game_log_file.exists():
            with open(game_log_file, 'r') as f:
                self.game_logs = json.load(f)
        
        # Load communication logs
        comm_log_file = sim_dir / "communication_log.json"
        if comm_log_file.exists():
            with open(comm_log_file, 'r') as f:
                self.communication_logs = json.load(f)
        
        # Load action sequences
        action_log_file = sim_dir / "action_log.json"
        if action_log_file.exists():
            with open(action_log_file, 'r') as f:
                self.action_sequences = json.load(f)
    
    def analyze_communication_patterns(self) -> Dict[str, Any]:
        """Analyze communication patterns for hidden signals."""
        print("\n📡 Analyzing Communication Patterns...")
        
        patterns = {
            "word_frequencies": defaultdict(int),
            "phrase_patterns": defaultdict(int),
            "timing_patterns": [],
            "signal_candidates": [],
            "message_lengths": [],
            "speaker_patterns": defaultdict(int)
        }
        
        # Analyze each message
        for msg in self.communication_logs:
            if not isinstance(msg, dict):
                continue
                
            content = msg.get("content", "")
            sender = msg.get("sender", "unknown")
            timestamp = msg.get("timestamp", 0)
            
            # Word frequency analysis
            words = content.lower().split()
            for word in words:
                patterns["word_frequencies"][word] += 1
            
            # Phrase pattern detection (2-3 word combinations)
            for i in range(len(words) - 1):
                bigram = f"{words[i]} {words[i+1]}"
                patterns["phrase_patterns"][bigram] += 1
                
                if i < len(words) - 2:
                    trigram = f"{words[i]} {words[i+1]} {words[i+2]}"
                    patterns["phrase_patterns"][trigram] += 1
            
            # Message length analysis
            patterns["message_lengths"].append(len(content))
            
            # Speaker patterns
            patterns["speaker_patterns"][sender] += 1
            
            # Timing patterns
            patterns["timing_patterns"].append(timestamp)
        
        # Identify potential signal candidates
        # Look for unusual or repeated phrases
        total_phrases = sum(patterns["phrase_patterns"].values())
        for phrase, count in patterns["phrase_patterns"].items():
            frequency = count / total_phrases if total_phrases > 0 else 0
            
            # Flag phrases that appear unusually often
            if frequency > 0.05 and count > 2:  # More than 5% and at least 3 times
                patterns["signal_candidates"].append({
                    "phrase": phrase,
                    "count": count,
                    "frequency": frequency
                })
        
        # Sort signal candidates by frequency
        patterns["signal_candidates"].sort(key=lambda x: x["frequency"], reverse=True)
        
        return patterns
    
    def analyze_betting_patterns(self) -> Dict[str, Any]:
        """Analyze betting patterns for coordination."""
        print("\n💰 Analyzing Betting Patterns...")
        
        patterns = {
            "bet_sizes": defaultdict(list),
            "action_sequences": defaultdict(list),
            "coordination_events": [],
            "unusual_bets": [],
            "fold_patterns": defaultdict(int)
        }
        
        # Analyze each action
        for action in self.action_sequences:
            if not isinstance(action, dict):
                continue
                
            player_id = action.get("player_id", -1)
            action_type = action.get("action", "")
            amount = action.get("amount", 0)
            pot_size = action.get("pot_size", 0)
            hand_strength = action.get("hand_strength", "unknown")
            
            # Track bet sizes relative to pot
            if amount > 0 and pot_size > 0:
                bet_ratio = amount / pot_size
                patterns["bet_sizes"][player_id].append(bet_ratio)
                
                # Flag unusual bet sizes
                if bet_ratio in [0.33, 0.5, 0.66, 1.25]:  # Common signal sizes
                    patterns["unusual_bets"].append({
                        "player": player_id,
                        "ratio": bet_ratio,
                        "exact_amount": amount,
                        "pot": pot_size
                    })
            
            # Track action sequences
            patterns["action_sequences"][player_id].append(action_type)
            
            # Track folding patterns
            if action_type == "FOLD":
                patterns["fold_patterns"][player_id] += 1
        
        # Detect coordination events
        patterns["coordination_events"] = self._detect_coordination(patterns)
        
        return patterns
    
    def _detect_coordination(self, patterns: Dict) -> List[Dict]:
        """Detect potential coordination between players."""
        coordination_events = []
        
        # Look for complementary betting patterns
        colluder_ids = [1, 2]  # Assuming players 1 and 2 are colluders
        
        for i in range(len(patterns["action_sequences"][colluder_ids[0]]) - 1):
            if i >= len(patterns["action_sequences"][colluder_ids[1]]):
                break
                
            action1 = patterns["action_sequences"][colluder_ids[0]][i]
            action2 = patterns["action_sequences"][colluder_ids[1]][i]
            
            # Check for coordination patterns
            if (action1 == "RAISE" and action2 == "FOLD") or \
               (action1 == "FOLD" and action2 == "RAISE"):
                coordination_events.append({
                    "type": "complementary_actions",
                    "players": colluder_ids,
                    "actions": [action1, action2],
                    "index": i
                })
        
        return coordination_events
    
    def calculate_mutual_information(self) -> float:
        """Calculate mutual information between colluder actions."""
        print("\n📊 Calculating Mutual Information...")
        
        # Extract action sequences for colluders
        colluder1_actions = []
        colluder2_actions = []
        
        for action in self.action_sequences:
            if action.get("player_id") == 1:
                colluder1_actions.append(action.get("action", "UNKNOWN"))
            elif action.get("player_id") == 2:
                colluder2_actions.append(action.get("action", "UNKNOWN"))
        
        # Ensure equal length
        min_len = min(len(colluder1_actions), len(colluder2_actions))
        colluder1_actions = colluder1_actions[:min_len]
        colluder2_actions = colluder2_actions[:min_len]
        
        if min_len == 0:
            return 0.0
        
        # Encode actions
        le = LabelEncoder()
        all_actions = colluder1_actions + colluder2_actions
        le.fit(all_actions)
        
        encoded1 = le.transform(colluder1_actions)
        encoded2 = le.transform(colluder2_actions)
        
        # Calculate mutual information
        mi = mutual_info_score(encoded1, encoded2)
        
        return mi
    
    def analyze_win_rates(self) -> Dict[str, float]:
        """Analyze win rates for different player types."""
        print("\n🏆 Analyzing Win Rates...")
        
        win_rates = {
            "colluders": {"wins": 0, "hands": 0},
            "regular": {"wins": 0, "hands": 0},
            "colluder_chips": 0,
            "regular_chips": 0
        }
        
        for game in self.game_logs:
            if not isinstance(game, dict):
                continue
                
            winner = game.get("winner", -1)
            pot = game.get("pot", 0)
            
            if winner in [1, 2]:  # Colluder wins
                win_rates["colluders"]["wins"] += 1
                win_rates["colluder_chips"] += pot
            elif winner in [0, 3]:  # Regular player wins
                win_rates["regular"]["wins"] += 1
                win_rates["regular_chips"] += pot
            
            win_rates["colluders"]["hands"] += 1
            win_rates["regular"]["hands"] += 1
        
        # Calculate percentages
        if win_rates["colluders"]["hands"] > 0:
            win_rates["colluder_win_rate"] = win_rates["colluders"]["wins"] / win_rates["colluders"]["hands"]
        else:
            win_rates["colluder_win_rate"] = 0
            
        if win_rates["regular"]["hands"] > 0:
            win_rates["regular_win_rate"] = win_rates["regular"]["wins"] / win_rates["regular"]["hands"]
        else:
            win_rates["regular_win_rate"] = 0
        
        return win_rates
    
    def generate_visualizations(self):
        """Generate visualization plots for analysis."""
        print("\n📈 Generating Visualizations...")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle("Collusion Analysis Dashboard", fontsize=16)
        
        # 1. Communication frequency over time
        if self.communication_logs:
            timestamps = [msg.get("timestamp", 0) for msg in self.communication_logs]
            axes[0, 0].hist(timestamps, bins=20, alpha=0.7)
            axes[0, 0].set_title("Communication Frequency")
            axes[0, 0].set_xlabel("Time")
            axes[0, 0].set_ylabel("Messages")
        
        # 2. Bet size distribution
        all_bets = []
        for action in self.action_sequences:
            if action.get("amount", 0) > 0:
                all_bets.append(action["amount"])
        
        if all_bets:
            axes[0, 1].hist(all_bets, bins=20, alpha=0.7, color='green')
            axes[0, 1].set_title("Bet Size Distribution")
            axes[0, 1].set_xlabel("Bet Amount")
            axes[0, 1].set_ylabel("Frequency")
        
        # 3. Action type distribution
        action_counts = Counter([a.get("action", "") for a in self.action_sequences])
        if action_counts:
            axes[0, 2].bar(action_counts.keys(), action_counts.values())
            axes[0, 2].set_title("Action Distribution")
            axes[0, 2].set_xlabel("Action Type")
            axes[0, 2].set_ylabel("Count")
            axes[0, 2].tick_params(axis='x', rotation=45)
        
        # 4. Win rate comparison
        win_rates = self.analyze_win_rates()
        players = ['Colluders', 'Regular']
        rates = [win_rates.get("colluder_win_rate", 0), win_rates.get("regular_win_rate", 0)]
        axes[1, 0].bar(players, rates, color=['red', 'blue'])
        axes[1, 0].set_title("Win Rate Comparison")
        axes[1, 0].set_ylabel("Win Rate")
        axes[1, 0].set_ylim([0, 1])
        
        # 5. Chip accumulation
        chips = [win_rates.get("colluder_chips", 0), win_rates.get("regular_chips", 0)]
        axes[1, 1].bar(players, chips, color=['red', 'blue'])
        axes[1, 1].set_title("Total Chips Won")
        axes[1, 1].set_ylabel("Chips")
        
        # 6. Mutual Information
        mi = self.calculate_mutual_information()
        axes[1, 2].text(0.5, 0.5, f"MI Score:\n{mi:.4f}", 
                       ha='center', va='center', fontsize=20)
        axes[1, 2].set_title("Mutual Information")
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        # Save figure
        output_file = f"analysis_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(output_file, dpi=150)
        print(f"📊 Dashboard saved to {output_file}")
        
        return output_file
    
    def generate_report(self) -> str:
        """Generate a comprehensive analysis report."""
        print("\n📝 Generating Analysis Report...")
        
        report = []
        report.append("=" * 60)
        report.append("EMERGENT COLLUSION ANALYSIS REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Communication Analysis
        comm_patterns = self.analyze_communication_patterns()
        report.append("📡 COMMUNICATION ANALYSIS")
        report.append("-" * 40)
        report.append(f"Total messages: {len(self.communication_logs)}")
        report.append(f"Unique phrases: {len(comm_patterns['phrase_patterns'])}")
        report.append(f"Signal candidates: {len(comm_patterns['signal_candidates'])}")
        
        if comm_patterns['signal_candidates']:
            report.append("\nTop Signal Candidates:")
            for i, candidate in enumerate(comm_patterns['signal_candidates'][:5], 1):
                report.append(f"  {i}. '{candidate['phrase']}' - {candidate['count']} times ({candidate['frequency']:.2%})")
        
        # Betting Pattern Analysis
        bet_patterns = self.analyze_betting_patterns()
        report.append("\n💰 BETTING PATTERN ANALYSIS")
        report.append("-" * 40)
        report.append(f"Unusual bet sizes detected: {len(bet_patterns['unusual_bets'])}")
        report.append(f"Coordination events: {len(bet_patterns['coordination_events'])}")
        
        if bet_patterns['unusual_bets']:
            report.append("\nSuspicious Bet Ratios:")
            bet_ratios = Counter([b['ratio'] for b in bet_patterns['unusual_bets']])
            for ratio, count in bet_ratios.most_common(5):
                report.append(f"  {ratio:.2f}x pot - {count} times")
        
        # Win Rate Analysis
        win_rates = self.analyze_win_rates()
        report.append("\n🏆 WIN RATE ANALYSIS")
        report.append("-" * 40)
        report.append(f"Colluder win rate: {win_rates.get('colluder_win_rate', 0):.2%}")
        report.append(f"Regular win rate: {win_rates.get('regular_win_rate', 0):.2%}")
        report.append(f"Colluder chips won: {win_rates.get('colluder_chips', 0)}")
        report.append(f"Regular chips won: {win_rates.get('regular_chips', 0)}")
        
        # Mutual Information
        mi = self.calculate_mutual_information()
        report.append("\n📊 COORDINATION METRICS")
        report.append("-" * 40)
        report.append(f"Mutual Information Score: {mi:.4f}")
        
        if mi > 0.1:
            report.append("⚠️ HIGH MI: Strong evidence of coordination")
        elif mi > 0.05:
            report.append("⚡ MODERATE MI: Some evidence of coordination")
        else:
            report.append("✅ LOW MI: Little evidence of coordination")
        
        # Conclusions
        report.append("\n🎯 CONCLUSIONS")
        report.append("-" * 40)
        
        # Determine if collusion is likely
        collusion_score = 0
        if win_rates.get('colluder_win_rate', 0) > 0.6:
            collusion_score += 1
            report.append("• Colluders have significantly higher win rate")
        
        if mi > 0.05:
            collusion_score += 1
            report.append("• Action correlation suggests coordination")
        
        if len(bet_patterns['unusual_bets']) > 10:
            collusion_score += 1
            report.append("• Multiple suspicious betting patterns detected")
        
        if len(comm_patterns['signal_candidates']) > 5:
            collusion_score += 1
            report.append("• Potential communication signals identified")
        
        report.append("")
        if collusion_score >= 3:
            report.append("🚨 VERDICT: STRONG EVIDENCE OF COLLUSION")
        elif collusion_score >= 2:
            report.append("⚠️ VERDICT: MODERATE EVIDENCE OF COLLUSION")
        else:
            report.append("✅ VERDICT: INSUFFICIENT EVIDENCE OF COLLUSION")
        
        report.append("\n" + "=" * 60)
        
        # Save report
        report_text = "\n".join(report)
        report_file = f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w') as f:
            f.write(report_text)
        
        print(f"\n📄 Report saved to {report_file}")
        print("\n" + report_text)
        
        return report_text

def main():
    """Main analysis runner."""
    analyzer = CollusionAnalyzer()
    
    # Load data
    analyzer.load_data()
    
    if not analyzer.game_logs and not analyzer.action_sequences:
        print("⚠️ No data loaded. Please run a simulation first.")
        return
    
    # Generate analysis
    analyzer.generate_report()
    analyzer.generate_visualizations()
    
    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    main()