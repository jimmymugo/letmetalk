#!/usr/bin/env python3
"""
FPL Optimizer Demo Script
Demonstrates the full FPL Optimizer system with formatted output.
"""

import pandas as pd
from fetch_data import FPLDataFetcher
from optimizer import FPLOptimizer

def print_header():
    """Print a nice header."""
    print("=" * 80)
    print("⚽ FPL OPTIMIZER SYSTEM DEMO")
    print("=" * 80)
    print()

def print_squad_overview(result):
    """Print squad overview."""
    print("📊 SQUAD OVERVIEW")
    print("-" * 40)
    print(f"Total Points: {result['total_points']:.1f}")
    print(f"Squad Cost: £{result['squad_cost']:.1f}m")
    print(f"Budget Remaining: £{100.0 - result['squad_cost']:.1f}m")
    print(f"Captain: {result['captain']} ⭐")
    print(f"Vice-Captain: {result['vice_captain']} 🅥")
    print()

def print_starting_xi(result):
    """Print starting XI."""
    print("⚽ STARTING XI")
    print("-" * 40)
    starting_xi = result['starting_xi']
    
    # Format the table
    display_data = starting_xi[['name', 'pos', 'team', 'cost', 'predicted_points']].copy()
    display_data['Role'] = ''
    display_data.loc[display_data['name'] == result['captain'], 'Role'] = '⭐ Captain'
    display_data.loc[display_data['name'] == result['vice_captain'], 'Role'] = '🅥 Vice-Captain'
    
    # Format columns
    display_data['Cost (£m)'] = display_data['cost'].round(1)
    display_data['Points'] = display_data['predicted_points'].round(1)
    
    # Reorder and rename columns
    display_data = display_data[['name', 'pos', 'team', 'Cost (£m)', 'Points', 'Role']].rename(columns={
        'name': 'Player',
        'pos': 'Position',
        'team': 'Team'
    })
    
    # Print formatted table
    print(f"{'Player':<25} {'Position':<12} {'Team':<15} {'Cost':<8} {'Points':<8} {'Role':<12}")
    print("-" * 80)
    for _, row in display_data.iterrows():
        print(f"{row['Player']:<25} {row['Position']:<12} {row['Team']:<15} £{row['Cost (£m)']:<7} {row['Points']:<8} {row['Role']:<12}")
    print()

def print_bench(result):
    """Print bench."""
    print("🪑 BENCH (Autosub Order)")
    print("-" * 40)
    bench = result['bench']
    
    # Format the table
    display_data = bench[['name', 'pos', 'team', 'cost', 'predicted_points']].copy()
    display_data['Bench Position'] = range(1, len(display_data) + 1)
    display_data['Cost (£m)'] = display_data['cost'].round(1)
    display_data['Points'] = display_data['predicted_points'].round(1)
    
    # Reorder and rename columns
    display_data = display_data[['Bench Position', 'name', 'pos', 'team', 'Cost (£m)', 'Points']].rename(columns={
        'name': 'Player',
        'pos': 'Position',
        'team': 'Team'
    })
    
    # Print formatted table
    print(f"{'Pos':<4} {'Player':<25} {'Position':<12} {'Team':<15} {'Cost':<8} {'Points':<8}")
    print("-" * 80)
    for _, row in display_data.iterrows():
        print(f"{row['Bench Position']:<4} {row['Player']:<25} {row['Position']:<12} {row['Team']:<15} £{row['Cost (£m)']:<7} {row['Points']:<8}")
    print()

def print_captaincy_analysis(result):
    """Print captaincy analysis."""
    print("📈 CAPTAINCY ANALYSIS")
    print("-" * 40)
    captaincy_results = result['captaincy_results']
    
    # Get top 5 captaincy options
    sorted_captaincy = sorted(captaincy_results.items(), key=lambda x: x[1], reverse=True)[:5]
    
    print("Top 5 Captaincy Options:")
    print(f"{'Rank':<4} {'Player':<25} {'Team Total':<12} {'Points Diff':<12}")
    print("-" * 55)
    
    best_points = sorted_captaincy[0][1]
    for i, (player, points) in enumerate(sorted_captaincy, 1):
        points_diff = points - best_points
        marker = "⭐" if i == 1 else "🅥" if i == 2 else ""
        print(f"{i:<4} {player:<25} {points:<12.1f} {points_diff:+<12.1f} {marker}")
    print()

def print_position_breakdown(result):
    """Print position breakdown."""
    print("📊 POSITION BREAKDOWN")
    print("-" * 40)
    squad = result['squad']
    pos_counts = squad['pos'].value_counts()
    
    for position, count in pos_counts.items():
        print(f"{position:<12}: {count} players")
    print()

def print_team_breakdown(result):
    """Print team breakdown."""
    print("🏟️ TEAM BREAKDOWN")
    print("-" * 40)
    squad = result['squad']
    team_counts = squad['team'].value_counts()
    
    for team, count in team_counts.items():
        print(f"{team:<15}: {count} players")
    print()

def main():
    """Run the demo."""
    print_header()
    
    print("🔄 Fetching FPL data...")
    fetcher = FPLDataFetcher()
    players = fetcher.fetch_and_clean()
    print(f"✅ Loaded {len(players)} players")
    print()
    
    print("🧮 Optimizing squad...")
    optimizer = FPLOptimizer(players)
    result = optimizer.optimize_complete_squad()
    print("✅ Optimization complete!")
    print()
    
    # Print all sections
    print_squad_overview(result)
    print_starting_xi(result)
    print_bench(result)
    print_captaincy_analysis(result)
    print_position_breakdown(result)
    print_team_breakdown(result)
    
    print("=" * 80)
    print("🎉 DEMO COMPLETE!")
    print("=" * 80)
    print()
    print("💡 To use the web interface, run: streamlit run app.py")
    print("💡 To run tests, run: python test_system.py")

if __name__ == "__main__":
    main()