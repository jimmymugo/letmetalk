#!/usr/bin/env python3
"""
FPL Optimizer Demo
This script demonstrates the FPL Optimizer system working in the command line.
"""

from fetch_data import FPLDataFetcher
from optimizer import FPLOptimizer
import pandas as pd

def print_squad_table(squad, captain_id, vice_captain_id):
    """Print squad in a nice table format."""
    print("\n" + "="*80)
    print("🏆 OPTIMIZED FPL SQUAD")
    print("="*80)
    
    # Format the squad for display
    display_squad = squad.copy()
    display_squad['Role'] = ''
    display_squad.loc[display_squad['id'] == captain_id, 'Role'] = '⭐ Captain'
    display_squad.loc[display_squad['id'] == vice_captain_id, 'Role'] = '🅥 Vice-Captain'
    
    # Sort by predicted points (starting XI first, then bench)
    display_squad = display_squad.sort_values('predicted_points', ascending=False)
    
    # Print starting XI
    print("\n📋 STARTING XI:")
    print("-" * 80)
    starting_xi = display_squad.head(11)
    for idx, player in starting_xi.iterrows():
        role_icon = player['Role'] if player['Role'] else "  "
        print(f"{role_icon} {player['name']:<25} {player['team']:<20} {player['pos']:<12} £{player['cost']:>5.1f}m  {player['predicted_points']:>5.1f} pts")
    
    # Print bench
    print("\n🪑 BENCH:")
    print("-" * 80)
    bench = display_squad.tail(4)
    for idx, player in bench.iterrows():
        role_icon = player['Role'] if player['Role'] else "  "
        print(f"{role_icon} {player['name']:<25} {player['team']:<20} {player['pos']:<12} £{player['cost']:>5.1f}m  {player['predicted_points']:>5.1f} pts")
    
    print("-" * 80)

def print_captaincy_analysis(captaincy_results):
    """Print captaincy analysis."""
    print("\n" + "="*80)
    print("📈 CAPTAINCY ANALYSIS")
    print("="*80)
    
    # Sort by team total descending
    sorted_results = sorted(captaincy_results, key=lambda x: x['team_total'], reverse=True)
    
    print(f"{'Player':<25} {'Team':<20} {'Points':<8} {'Captain':<10} {'Team Total':<12}")
    print("-" * 80)
    
    for i, result in enumerate(sorted_results[:10]):  # Top 10
        captain_icon = "⭐" if i == 0 else "🅥" if i == 1 else "  "
        print(f"{captain_icon} {result['name']:<25} {result['team']:<20} {result['predicted_points']:<8.1f} {result['captain_points']:<10.1f} {result['team_total']:<12.1f}")

def print_summary(summary):
    """Print squad summary."""
    print("\n" + "="*80)
    print("📊 SQUAD SUMMARY")
    print("="*80)
    
    print(f"💰 Total Cost: £{summary['total_cost']:.1f}m")
    print(f"🎯 Total Predicted Points: {summary['total_predicted_points']:.1f}")
    print(f"⚽ Starting XI Points: {summary['starting_xi_points']:.1f}")
    print(f"🪑 Bench Points: {summary['bench_points']:.1f}")
    
    print(f"\n👥 Players by Position:")
    for pos, count in summary['players_by_position'].items():
        print(f"   {pos}: {count}")
    
    print(f"\n🏟️  Players by Team:")
    team_counts = summary['players_by_team']
    sorted_teams = sorted(team_counts.items(), key=lambda x: x[1], reverse=True)
    for team, count in sorted_teams[:10]:  # Top 10 teams
        print(f"   {team}: {count}")

def main():
    """Main demo function."""
    print("⚽ FPL Optimizer Demo")
    print("=" * 50)
    
    # Fetch data
    print("📡 Fetching FPL data...")
    fetcher = FPLDataFetcher()
    players = fetcher.fetch_and_clean()
    print(f"✅ Loaded {len(players)} players")
    
    # Optimize squad
    print("\n🧮 Optimizing squad...")
    optimizer = FPLOptimizer(players)
    squad_result = optimizer.optimize_squad()
    print(f"✅ Squad optimized! Total points: {squad_result['total_predicted_points']:.1f}")
    
    # Optimize captaincy
    print("\n👑 Optimizing captaincy...")
    captaincy_result = optimizer.optimize_captaincy()
    captain_id = captaincy_result['captain']['player_id']
    vice_captain_id = captaincy_result['vice_captain']['player_id']
    print(f"✅ Captain: {captaincy_result['captain']['name']}")
    print(f"✅ Vice-Captain: {captaincy_result['vice_captain']['name']}")
    
    # Optimize bench order
    print("\n🪑 Optimizing bench order...")
    bench_order = optimizer.optimize_bench_order()
    print(f"✅ Bench order optimized ({len(bench_order)} players)")
    
    # Get summary
    summary = optimizer.get_squad_summary()
    
    # Display results
    print_squad_table(squad_result['squad'], captain_id, vice_captain_id)
    print_captaincy_analysis(captaincy_result['captaincy_results'])
    print_summary(summary)
    
    print("\n" + "="*80)
    print("🎉 Demo completed successfully!")
    print("💡 Run 'streamlit run app.py' for the interactive web interface")
    print("="*80)

if __name__ == "__main__":
    main()