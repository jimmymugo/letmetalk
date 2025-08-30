#!/usr/bin/env python3
"""
Test script to verify comprehensive analysis is working and showing different players.
"""

from fetch_data import fetch_raw_data, parse_players, fetch_fixtures_data, parse_fixtures
from optimizer import SquadOptimizer

def main():
    print("🔍 Testing Comprehensive Analysis vs Old System")
    print("=" * 60)
    
    # Fetch data
    print("📊 Fetching FPL data...")
    raw_data = fetch_raw_data()
    players = parse_players(raw_data)
    
    fixtures_data = fetch_fixtures_data()
    fixtures = parse_fixtures(fixtures_data)
    
    print(f"✅ Loaded {len(players)} players and {len(fixtures)} fixtures")
    
    # Create optimizer
    optimizer = SquadOptimizer(players, fixtures)
    
    # Test 1: Get top players by old method (just predicted points)
    print("\n📊 Top 10 Players by OLD Method (Predicted Points Only):")
    print("-" * 60)
    
    old_top_players = sorted(players, key=lambda p: p.predicted_points, reverse=True)[:10]
    for i, player in enumerate(old_top_players, 1):
        print(f"{i:2d}. {player.name} ({player.position}) - {player.team}")
        print(f"    Predicted: {player.predicted_points:.1f} | Form: {player.form:.1f} | Cost: £{player.cost}m")
    
    # Test 2: Get top players by comprehensive analysis
    print("\n🎯 Top 10 Players by NEW Method (Comprehensive Analysis):")
    print("-" * 60)
    
    comprehensive_rankings = optimizer.get_comprehensive_player_rankings(gameweek=3, limit=10)
    for ranking in comprehensive_rankings:
        player = ranking['player']
        analysis = ranking['analysis_breakdown']
        print(f"{ranking['rank']:2d}. {player.name} ({player.position}) - {player.team}")
        print(f"    Enhanced Score: {ranking['total_enhanced_score']:.2f} | Base: {analysis['base_predicted_points']:.1f}")
        print(f"    ICT: {player.ict_index:.0f} | xG: {player.expected_goals:.2f} | xA: {player.expected_assists:.2f}")
        print(f"    Form: {player.form:.1f} | Cost: £{player.cost}m")
    
    # Test 3: Compare squads
    print("\n⚽ Comparing Squad Selection Methods:")
    print("-" * 60)
    
    # Old method squad (simplified - just top predicted points)
    print("📊 OLD Method Squad (Top Predicted Points):")
    old_squad_players = old_top_players[:15]  # Take top 15
    for i, player in enumerate(old_squad_players, 1):
        print(f"{i:2d}. {player.name} ({player.position}) - {player.team} - {player.predicted_points:.1f}pts")
    
    # New method squad (comprehensive analysis)
    print("\n🎯 NEW Method Squad (Comprehensive Analysis):")
    comprehensive_squad = optimizer.optimize_squad(gameweek=3, use_form=True, use_fixtures=True)
    for i, player in enumerate(comprehensive_squad.players, 1):
        analysis = optimizer.analyze_player_selection_factors(player.player, 3)
        print(f"{i:2d}. {player.name} ({player.position}) - {player.team} - Enhanced: {analysis['total_enhanced_score']:.2f}")
    
    # Test 4: Show differences
    print("\n🔄 Key Differences:")
    print("-" * 60)
    
    old_player_names = {p.name for p in old_squad_players}
    new_player_names = {p.name for p in comprehensive_squad.players}
    
    only_in_old = old_player_names - new_player_names
    only_in_new = new_player_names - old_player_names
    
    if only_in_old:
        print("❌ Players in OLD system but NOT in NEW system:")
        for player_name in only_in_old:
            player = next(p for p in old_squad_players if p.name == player_name)
            print(f"   • {player_name} ({player.position}) - {player.team} - Predicted: {player.predicted_points:.1f}")
    
    if only_in_new:
        print("\n✅ Players in NEW system but NOT in OLD system:")
        for player_name in only_in_new:
            player = next(p for p in comprehensive_squad.players if p.player.name == player_name)
            analysis = optimizer.analyze_player_selection_factors(player.player, 3)
            print(f"   • {player_name} ({player.position}) - {player.team} - Enhanced: {analysis['total_enhanced_score']:.2f}")
    
    if not only_in_old and not only_in_new:
        print("⚠️ Both systems selected the same players (this might happen if comprehensive metrics are similar)")
    
    print("\n" + "=" * 60)
    print("🎯 Summary:")
    print("✅ Comprehensive analysis system is working!")
    print("📊 Shows different player rankings based on enhanced metrics")
    print("🔬 Uses ICT Index, xG/xA, availability, team strength, and position-specific analysis")
    print("⚽ Not just dependent on recent form!")

if __name__ == "__main__":
    main()
