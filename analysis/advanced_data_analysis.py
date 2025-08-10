#!/usr/bin/env python3
"""
Advanced Data Analysis System with Data100-style Visualizations
Handles simulation_X directories and provides comprehensive insights
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import re
from scipy import stats
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Set style for beautiful visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class AdvancedCollusionAnalyzer:
    """Advanced analyzer with Data100-style visualizations and insights."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.simulation_data = {}
        self.current_simulation = None
        
    def list_simulations(self) -> List[str]:
        """List all available simulation directories."""
        sim_pattern = re.compile(r'simulation_(\d+)')
        simulations = []
        
        # Check multiple possible locations
        paths_to_check = [
            self.data_dir / "communication",
            self.data_dir / "simulations",
            self.data_dir
        ]
        
        for base_path in paths_to_check:
            if base_path.exists():
                for item in base_path.iterdir():
                    if item.is_dir() and sim_pattern.match(item.name):
                        simulations.append(item.name)
        
        return sorted(set(simulations), key=lambda x: int(x.split('_')[1]))
    
    def load_simulation(self, simulation_id: Union[str, int]) -> bool:
        """Load a specific simulation by ID or name."""
        if isinstance(simulation_id, int):
            sim_name = f"simulation_{simulation_id}"
        else:
            sim_name = simulation_id
        
        print(f"\n📂 Loading {sim_name}...")
        
        # Try different paths
        possible_paths = [
            self.data_dir / "communication" / sim_name,
            self.data_dir / "simulations" / sim_name,
            self.data_dir / sim_name
        ]
        
        sim_dir = None
        for path in possible_paths:
            if path.exists():
                sim_dir = path
                break
        
        if not sim_dir:
            print(f"❌ Simulation {sim_name} not found")
            return False
        
        self.current_simulation = sim_name
        self.simulation_data[sim_name] = {
            "path": sim_dir,
            "game_logs": [],
            "communication_logs": [],
            "action_logs": [],
            "hand_histories": []
        }
        
        # Load all available data files
        data_files = {
            "game_log.json": "game_logs",
            "communication_log.json": "communication_logs",
            "action_log.json": "action_logs",
            "hand_history.json": "hand_histories",
            "actions.json": "action_logs",  # Alternative name
            "messages.json": "communication_logs"  # Alternative name
        }
        
        files_loaded = 0
        for filename, data_key in data_files.items():
            file_path = sim_dir / filename
            if file_path.exists():
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    if data:
                        self.simulation_data[sim_name][data_key] = data
                        files_loaded += 1
                        print(f"  ✅ Loaded {filename}: {len(data)} entries")
        
        if files_loaded == 0:
            # Try to load any JSON files in the directory
            for json_file in sim_dir.glob("*.json"):
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    if data:
                        self.simulation_data[sim_name]["game_logs"].append(data)
                        files_loaded += 1
                        print(f"  ✅ Loaded {json_file.name}")
        
        print(f"  📊 Total files loaded: {files_loaded}")
        return files_loaded > 0
    
    def create_dataframes(self, sim_name: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """Convert simulation data to pandas DataFrames for analysis."""
        if not sim_name:
            sim_name = self.current_simulation
        
        if not sim_name or sim_name not in self.simulation_data:
            print("❌ No simulation loaded")
            return {}
        
        data = self.simulation_data[sim_name]
        dataframes = {}
        
        # Create action DataFrame
        if data["action_logs"]:
            actions = []
            for entry in data["action_logs"]:
                if isinstance(entry, dict):
                    actions.append(entry)
                elif isinstance(entry, list):
                    actions.extend(entry)
            
            if actions:
                dataframes["actions"] = pd.DataFrame(actions)
                print(f"  📊 Actions DataFrame: {len(dataframes['actions'])} rows")
        
        # Create communication DataFrame
        if data["communication_logs"]:
            messages = []
            for entry in data["communication_logs"]:
                if isinstance(entry, dict):
                    messages.append(entry)
                elif isinstance(entry, list):
                    messages.extend(entry)
            
            if messages:
                dataframes["messages"] = pd.DataFrame(messages)
                print(f"  💬 Messages DataFrame: {len(dataframes['messages'])} rows")
        
        # Create game summary DataFrame
        if data["game_logs"]:
            games = []
            for entry in data["game_logs"]:
                if isinstance(entry, dict):
                    games.append(entry)
                elif isinstance(entry, list):
                    games.extend(entry)
            
            if games:
                dataframes["games"] = pd.DataFrame(games)
                print(f"  🎮 Games DataFrame: {len(dataframes['games'])} rows")
        
        return dataframes
    
    def analyze_betting_patterns_advanced(self, df_actions: pd.DataFrame) -> Dict[str, Any]:
        """Advanced betting pattern analysis with statistical insights."""
        print("\n💰 Advanced Betting Pattern Analysis...")
        
        analysis = {
            "bet_size_stats": {},
            "action_frequencies": {},
            "position_analysis": {},
            "timing_patterns": {},
            "correlation_matrix": None
        }
        
        if df_actions.empty:
            return analysis
        
        # Bet size statistics by player
        if 'amount' in df_actions.columns and 'player_id' in df_actions.columns:
            for player_id in df_actions['player_id'].unique():
                player_bets = df_actions[
                    (df_actions['player_id'] == player_id) & 
                    (df_actions['amount'] > 0)
                ]['amount']
                
                if not player_bets.empty:
                    analysis["bet_size_stats"][f"player_{player_id}"] = {
                        "mean": player_bets.mean(),
                        "median": player_bets.median(),
                        "std": player_bets.std(),
                        "min": player_bets.min(),
                        "max": player_bets.max(),
                        "q25": player_bets.quantile(0.25),
                        "q75": player_bets.quantile(0.75)
                    }
        
        # Action frequency analysis
        if 'action' in df_actions.columns:
            action_freq = df_actions['action'].value_counts()
            analysis["action_frequencies"] = action_freq.to_dict()
            
            # Player-specific action frequencies
            if 'player_id' in df_actions.columns:
                for player_id in df_actions['player_id'].unique():
                    player_actions = df_actions[df_actions['player_id'] == player_id]['action'].value_counts()
                    analysis["action_frequencies"][f"player_{player_id}"] = player_actions.to_dict()
        
        # Position-based analysis
        if 'position' in df_actions.columns:
            position_actions = df_actions.groupby(['position', 'action']).size().unstack(fill_value=0)
            analysis["position_analysis"] = position_actions.to_dict()
        
        # Timing patterns (if timestamp available)
        if 'timestamp' in df_actions.columns:
            df_actions['time_diff'] = df_actions['timestamp'].diff()
            analysis["timing_patterns"] = {
                "mean_response_time": df_actions['time_diff'].mean(),
                "median_response_time": df_actions['time_diff'].median(),
                "quick_actions": (df_actions['time_diff'] < df_actions['time_diff'].quantile(0.25)).sum()
            }
        
        # Correlation analysis for colluders
        if 'player_id' in df_actions.columns and 'action' in df_actions.columns:
            # Create pivot table for correlation
            action_pivot = df_actions.pivot_table(
                index=df_actions.index, 
                columns='player_id', 
                values='action',
                aggfunc='first'
            )
            
            # Encode actions numerically
            le = LabelEncoder()
            for col in action_pivot.columns:
                if action_pivot[col].notna().any():
                    action_pivot[col] = le.fit_transform(action_pivot[col].fillna('NONE'))
            
            # Calculate correlation
            if not action_pivot.empty:
                analysis["correlation_matrix"] = action_pivot.corr()
        
        return analysis
    
    def detect_communication_patterns(self, df_messages: pd.DataFrame) -> Dict[str, Any]:
        """Detect potential hidden signals in communication."""
        print("\n📡 Detecting Communication Patterns...")
        
        patterns = {
            "signal_phrases": [],
            "timing_clusters": [],
            "sender_patterns": {},
            "linguistic_features": {}
        }
        
        if df_messages.empty or 'content' not in df_messages.columns:
            return patterns
        
        # Extract phrases that appear multiple times
        all_messages = ' '.join(df_messages['content'].dropna().astype(str))
        words = all_messages.lower().split()
        
        # Find repeated phrases (2-4 grams)
        from collections import Counter
        for n in range(2, 5):
            ngrams = [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
            phrase_counts = Counter(ngrams)
            
            for phrase, count in phrase_counts.most_common(10):
                if count > 2:  # Appears more than twice
                    patterns["signal_phrases"].append({
                        "phrase": phrase,
                        "count": count,
                        "n_gram": n
                    })
        
        # Analyze sender patterns
        if 'sender' in df_messages.columns:
            sender_counts = df_messages['sender'].value_counts()
            patterns["sender_patterns"] = sender_counts.to_dict()
            
            # Check for alternating patterns
            senders = df_messages['sender'].tolist()
            alternations = sum(1 for i in range(1, len(senders)) if senders[i] != senders[i-1])
            patterns["sender_patterns"]["alternation_rate"] = alternations / len(senders) if senders else 0
        
        # Linguistic feature analysis
        message_lengths = df_messages['content'].str.len()
        patterns["linguistic_features"] = {
            "avg_length": message_lengths.mean(),
            "std_length": message_lengths.std(),
            "min_length": message_lengths.min(),
            "max_length": message_lengths.max()
        }
        
        # Check for specific patterns (numbers, special characters)
        patterns["linguistic_features"]["contains_numbers"] = df_messages['content'].str.contains(r'\d').sum()
        patterns["linguistic_features"]["contains_special"] = df_messages['content'].str.contains(r'[!@#$%^&*()]').sum()
        
        return patterns
    
    def calculate_collusion_metrics(self, dataframes: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Calculate comprehensive collusion detection metrics."""
        print("\n📊 Calculating Collusion Metrics...")
        
        metrics = {
            "mutual_information": 0.0,
            "win_rate_differential": 0.0,
            "chip_accumulation_rate": 0.0,
            "coordination_score": 0.0,
            "signal_strength": 0.0
        }
        
        # Calculate mutual information between colluder actions
        if "actions" in dataframes and not dataframes["actions"].empty:
            df_actions = dataframes["actions"]
            
            if 'player_id' in df_actions.columns and 'action' in df_actions.columns:
                # Get colluder actions (assuming players 1 and 2)
                colluder1_actions = df_actions[df_actions['player_id'] == 1]['action']
                colluder2_actions = df_actions[df_actions['player_id'] == 2]['action']
                
                if not colluder1_actions.empty and not colluder2_actions.empty:
                    # Align and encode
                    min_len = min(len(colluder1_actions), len(colluder2_actions))
                    if min_len > 0:
                        le = LabelEncoder()
                        all_actions = pd.concat([colluder1_actions, colluder2_actions])
                        le.fit(all_actions)
                        
                        encoded1 = le.transform(colluder1_actions.iloc[:min_len])
                        encoded2 = le.transform(colluder2_actions.iloc[:min_len])
                        
                        metrics["mutual_information"] = mutual_info_score(encoded1, encoded2)
        
        # Calculate win rate differential
        if "games" in dataframes and not dataframes["games"].empty:
            df_games = dataframes["games"]
            
            if 'winner' in df_games.columns:
                total_games = len(df_games)
                colluder_wins = df_games['winner'].isin([1, 2]).sum()
                regular_wins = df_games['winner'].isin([0, 3]).sum()
                
                if total_games > 0:
                    colluder_rate = colluder_wins / total_games
                    regular_rate = regular_wins / total_games
                    metrics["win_rate_differential"] = colluder_rate - regular_rate
        
        # Calculate coordination score based on complementary actions
        if "actions" in dataframes and not dataframes["actions"].empty:
            df_actions = dataframes["actions"]
            
            # Look for patterns like one raises while other folds
            coordination_events = 0
            total_events = 0
            
            for hand in df_actions['hand_id'].unique() if 'hand_id' in df_actions.columns else [0]:
                hand_actions = df_actions[df_actions['hand_id'] == hand] if 'hand_id' in df_actions.columns else df_actions
                
                p1_actions = hand_actions[hand_actions['player_id'] == 1]['action'].tolist()
                p2_actions = hand_actions[hand_actions['player_id'] == 2]['action'].tolist()
                
                for a1, a2 in zip(p1_actions, p2_actions):
                    total_events += 1
                    if (a1 == 'RAISE' and a2 == 'FOLD') or (a1 == 'FOLD' and a2 == 'RAISE'):
                        coordination_events += 1
            
            if total_events > 0:
                metrics["coordination_score"] = coordination_events / total_events
        
        # Calculate signal strength from communication patterns
        if "messages" in dataframes and not dataframes["messages"].empty:
            patterns = self.detect_communication_patterns(dataframes["messages"])
            
            # Signal strength based on repeated phrases
            if patterns["signal_phrases"]:
                total_phrases = sum(p["count"] for p in patterns["signal_phrases"])
                unique_phrases = len(patterns["signal_phrases"])
                
                if unique_phrases > 0:
                    metrics["signal_strength"] = total_phrases / unique_phrases
        
        return metrics
    
    def create_comprehensive_visualizations(self, dataframes: Dict[str, pd.DataFrame], 
                                           save_path: Optional[str] = None) -> None:
        """Create Data100-style comprehensive visualizations."""
        print("\n📈 Creating Comprehensive Visualizations...")
        
        # Create a large figure with multiple subplots
        fig = plt.figure(figsize=(20, 15))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # 1. Action Distribution Heatmap
        ax1 = fig.add_subplot(gs[0, :2])
        if "actions" in dataframes and not dataframes["actions"].empty:
            df_actions = dataframes["actions"]
            if 'player_id' in df_actions.columns and 'action' in df_actions.columns:
                action_matrix = pd.crosstab(df_actions['player_id'], df_actions['action'])
                sns.heatmap(action_matrix, annot=True, fmt='d', cmap='YlOrRd', ax=ax1)
                ax1.set_title('Action Distribution by Player', fontsize=14, fontweight='bold')
                ax1.set_xlabel('Action Type')
                ax1.set_ylabel('Player ID')
        
        # 2. Bet Size Distribution
        ax2 = fig.add_subplot(gs[0, 2:])
        if "actions" in dataframes and 'amount' in dataframes["actions"].columns:
            bet_amounts = dataframes["actions"][dataframes["actions"]['amount'] > 0]['amount']
            if not bet_amounts.empty:
                ax2.hist(bet_amounts, bins=30, alpha=0.7, color='green', edgecolor='black')
                ax2.axvline(bet_amounts.mean(), color='red', linestyle='--', label=f'Mean: {bet_amounts.mean():.1f}')
                ax2.axvline(bet_amounts.median(), color='blue', linestyle='--', label=f'Median: {bet_amounts.median():.1f}')
                ax2.set_title('Bet Size Distribution', fontsize=14, fontweight='bold')
                ax2.set_xlabel('Bet Amount')
                ax2.set_ylabel('Frequency')
                ax2.legend()
        
        # 3. Time Series of Actions
        ax3 = fig.add_subplot(gs[1, :])
        if "actions" in dataframes and not dataframes["actions"].empty:
            df_actions = dataframes["actions"]
            if 'player_id' in df_actions.columns:
                for player_id in [0, 1, 2, 3]:
                    player_actions = df_actions[df_actions['player_id'] == player_id]
                    if not player_actions.empty and 'amount' in player_actions.columns:
                        cumsum = player_actions['amount'].fillna(0).cumsum()
                        ax3.plot(range(len(cumsum)), cumsum, label=f'Player {player_id}', linewidth=2)
                
                ax3.set_title('Cumulative Chip Movement Over Time', fontsize=14, fontweight='bold')
                ax3.set_xlabel('Action Number')
                ax3.set_ylabel('Cumulative Amount')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
        
        # 4. Win Rate Comparison
        ax4 = fig.add_subplot(gs[2, 0])
        if "games" in dataframes and 'winner' in dataframes["games"].columns:
            winner_counts = dataframes["games"]['winner'].value_counts()
            colors = ['red' if i in [1, 2] else 'blue' for i in winner_counts.index]
            ax4.bar(winner_counts.index, winner_counts.values, color=colors)
            ax4.set_title('Wins by Player', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Player ID')
            ax4.set_ylabel('Number of Wins')
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor='red', label='Colluders'),
                              Patch(facecolor='blue', label='Regular')]
            ax4.legend(handles=legend_elements)
        
        # 5. Communication Frequency
        ax5 = fig.add_subplot(gs[2, 1])
        if "messages" in dataframes and 'sender' in dataframes["messages"].columns:
            sender_counts = dataframes["messages"]['sender'].value_counts()
            colors = ['red' if i in [1, 2] else 'blue' for i in sender_counts.index]
            ax5.bar(sender_counts.index, sender_counts.values, color=colors)
            ax5.set_title('Messages by Player', fontsize=14, fontweight='bold')
            ax5.set_xlabel('Player ID')
            ax5.set_ylabel('Number of Messages')
        
        # 6. Correlation Matrix
        ax6 = fig.add_subplot(gs[2, 2:])
        analysis = self.analyze_betting_patterns_advanced(dataframes.get("actions", pd.DataFrame()))
        if analysis["correlation_matrix"] is not None:
            sns.heatmap(analysis["correlation_matrix"], annot=True, fmt='.2f', 
                       cmap='coolwarm', center=0, ax=ax6)
            ax6.set_title('Player Action Correlation Matrix', fontsize=14, fontweight='bold')
        
        # 7. Collusion Metrics Dashboard
        ax7 = fig.add_subplot(gs[3, :2])
        metrics = self.calculate_collusion_metrics(dataframes)
        
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        colors = ['red' if v > 0.1 else 'green' for v in metric_values]
        
        bars = ax7.barh(metric_names, metric_values, color=colors)
        ax7.set_title('Collusion Detection Metrics', fontsize=14, fontweight='bold')
        ax7.set_xlabel('Score')
        
        # Add value labels
        for bar, value in zip(bars, metric_values):
            ax7.text(bar.get_width(), bar.get_y() + bar.get_height()/2, 
                    f'{value:.3f}', ha='left', va='center')
        
        # 8. Summary Statistics Table
        ax8 = fig.add_subplot(gs[3, 2:])
        ax8.axis('tight')
        ax8.axis('off')
        
        # Create summary statistics
        summary_data = []
        if "actions" in dataframes:
            summary_data.append(['Total Actions', len(dataframes["actions"])])
        if "messages" in dataframes:
            summary_data.append(['Total Messages', len(dataframes["messages"])])
        if "games" in dataframes:
            summary_data.append(['Total Games', len(dataframes["games"])])
        
        # Add metric summaries
        summary_data.append(['Mutual Information', f'{metrics["mutual_information"]:.4f}'])
        summary_data.append(['Win Rate Diff', f'{metrics["win_rate_differential"]:.2%}'])
        summary_data.append(['Coordination Score', f'{metrics["coordination_score"]:.2%}'])
        
        if summary_data:
            table = ax8.table(cellText=summary_data, 
                            colLabels=['Metric', 'Value'],
                            cellLoc='left',
                            loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
            ax8.set_title('Summary Statistics', fontsize=14, fontweight='bold', pad=20)
        
        # Overall title
        fig.suptitle(f'Collusion Analysis Dashboard - {self.current_simulation}', 
                    fontsize=16, fontweight='bold')
        
        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  💾 Saved visualization to {save_path}")
        else:
            plt.show()
    
    def generate_insights_report(self, dataframes: Dict[str, pd.DataFrame]) -> str:
        """Generate actionable insights from the data."""
        print("\n📝 Generating Insights Report...")
        
        insights = []
        insights.append("=" * 70)
        insights.append("DATA-DRIVEN INSIGHTS REPORT")
        insights.append("=" * 70)
        insights.append(f"Simulation: {self.current_simulation}")
        insights.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Calculate all metrics
        metrics = self.calculate_collusion_metrics(dataframes)
        betting_patterns = self.analyze_betting_patterns_advanced(dataframes.get("actions", pd.DataFrame()))
        comm_patterns = self.detect_communication_patterns(dataframes.get("messages", pd.DataFrame()))
        
        # 1. Collusion Detection Summary
        insights.append("🔍 COLLUSION DETECTION SUMMARY")
        insights.append("-" * 40)
        
        collusion_level = "HIGH" if metrics["mutual_information"] > 0.1 else \
                         "MODERATE" if metrics["mutual_information"] > 0.05 else "LOW"
        
        insights.append(f"Overall Collusion Risk: {collusion_level}")
        insights.append(f"Mutual Information Score: {metrics['mutual_information']:.4f}")
        insights.append(f"Coordination Score: {metrics['coordination_score']:.2%}")
        insights.append(f"Win Rate Advantage: {metrics['win_rate_differential']:.2%}\n")
        
        # 2. Key Findings
        insights.append("📊 KEY FINDINGS")
        insights.append("-" * 40)
        
        # Finding 1: Win rate analysis
        if metrics["win_rate_differential"] > 0.1:
            insights.append("⚠️ Colluders show significant win rate advantage (>10%)")
        elif metrics["win_rate_differential"] > 0:
            insights.append("📈 Colluders have slight win rate advantage")
        else:
            insights.append("✅ No significant win rate advantage for colluders")
        
        # Finding 2: Action correlation
        if metrics["mutual_information"] > 0.1:
            insights.append("⚠️ High correlation between colluder actions detected")
        elif metrics["mutual_information"] > 0.05:
            insights.append("📊 Moderate correlation between colluder actions")
        else:
            insights.append("✅ Low correlation between colluder actions")
        
        # Finding 3: Communication patterns
        if comm_patterns["signal_phrases"]:
            top_phrases = comm_patterns["signal_phrases"][:3]
            insights.append(f"🔤 {len(comm_patterns['signal_phrases'])} potential signal phrases detected")
            for phrase_info in top_phrases:
                insights.append(f"   - '{phrase_info['phrase']}' ({phrase_info['count']} times)")
        
        insights.append("")
        
        # 3. Betting Pattern Analysis
        insights.append("💰 BETTING PATTERN INSIGHTS")
        insights.append("-" * 40)
        
        if betting_patterns["bet_size_stats"]:
            for player_id, stats in betting_patterns["bet_size_stats"].items():
                if "player_1" in player_id or "player_2" in player_id:
                    insights.append(f"{player_id} (Colluder):")
                else:
                    insights.append(f"{player_id}:")
                insights.append(f"  Mean bet: {stats['mean']:.1f}, Std: {stats['std']:.1f}")
                insights.append(f"  Range: {stats['min']:.0f} - {stats['max']:.0f}")
        
        insights.append("")
        
        # 4. Statistical Significance
        insights.append("📈 STATISTICAL ANALYSIS")
        insights.append("-" * 40)
        
        # Chi-square test for action independence
        if "actions" in dataframes and not dataframes["actions"].empty:
            df_actions = dataframes["actions"]
            if 'player_id' in df_actions.columns and 'action' in df_actions.columns:
                # Create contingency table
                cont_table = pd.crosstab(
                    df_actions[df_actions['player_id'].isin([1, 2])]['player_id'],
                    df_actions[df_actions['player_id'].isin([1, 2])]['action']
                )
                
                if cont_table.shape[0] > 1 and cont_table.shape[1] > 1:
                    chi2, p_value, _, _ = stats.chi2_contingency(cont_table)
                    insights.append(f"Chi-square test for action independence:")
                    insights.append(f"  χ² = {chi2:.2f}, p-value = {p_value:.4f}")
                    
                    if p_value < 0.05:
                        insights.append("  ⚠️ Actions are NOT independent (p < 0.05)")
                    else:
                        insights.append("  ✅ No significant dependence detected")
        
        insights.append("")
        
        # 5. Recommendations
        insights.append("💡 RECOMMENDATIONS")
        insights.append("-" * 40)
        
        if collusion_level == "HIGH":
            insights.append("1. Strong evidence of collusion detected")
            insights.append("2. Review game logs for suspicious betting patterns")
            insights.append("3. Analyze communication for coded messages")
            insights.append("4. Consider implementing anti-collusion measures")
        elif collusion_level == "MODERATE":
            insights.append("1. Some suspicious patterns detected")
            insights.append("2. Continue monitoring player behavior")
            insights.append("3. Look for consistency across multiple games")
        else:
            insights.append("1. No clear evidence of collusion")
            insights.append("2. Continue routine monitoring")
            insights.append("3. System appears to be functioning normally")
        
        insights.append("\n" + "=" * 70)
        
        report = "\n".join(insights)
        
        # Save report
        report_file = f"insights_report_{self.current_simulation}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"\n💾 Report saved to {report_file}")
        print("\n" + report)
        
        return report

def interactive_analysis():
    """Interactive analysis function for command-line use."""
    analyzer = AdvancedCollusionAnalyzer()
    
    # List available simulations
    simulations = analyzer.list_simulations()
    
    if not simulations:
        print("❌ No simulations found in data directory")
        return
    
    print("\n📁 Available Simulations:")
    for i, sim in enumerate(simulations, 1):
        print(f"  {i}. {sim}")
    
    # Let user choose or use latest
    if len(simulations) == 1:
        chosen_sim = simulations[0]
        print(f"\n✅ Automatically selecting: {chosen_sim}")
    else:
        choice = input(f"\nEnter simulation number (1-{len(simulations)}) or name (default: latest): ").strip()
        
        if not choice:
            chosen_sim = simulations[-1]
        elif choice.isdigit() and 1 <= int(choice) <= len(simulations):
            chosen_sim = simulations[int(choice) - 1]
        elif choice in simulations:
            chosen_sim = choice
        elif choice.startswith("simulation_"):
            chosen_sim = choice
        else:
            # Try to interpret as just the number
            try:
                sim_num = int(choice)
                chosen_sim = f"simulation_{sim_num}"
            except:
                print(f"❌ Invalid choice: {choice}")
                return
    
    # Load the simulation
    if not analyzer.load_simulation(chosen_sim):
        return
    
    # Create dataframes
    dataframes = analyzer.create_dataframes()
    
    if not dataframes:
        print("❌ No data could be loaded into DataFrames")
        return
    
    # Generate visualizations and report
    viz_file = f"visualization_{chosen_sim}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    analyzer.create_comprehensive_visualizations(dataframes, viz_file)
    analyzer.generate_insights_report(dataframes)
    
    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    interactive_analysis()