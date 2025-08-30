#!/usr/bin/env python3
"""
Test script to demonstrate comprehensive player analysis
that uses all available metrics instead of just recent form.
"""

from fetch_data import fetch_raw_data, parse_players, fetch_fixtures_data, parse_fixtures
from optimizer import SquadOptimizer

def main():
    print("🔍 Testing Comprehensive Player Analysis System")
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
    
    # Test comprehensive player rankings
    print("\n🏆 Top 10 Players by Comprehensive Analysis (Gameweek 3)")
    print("-" * 60)
    
    rankings = optimizer.get_comprehensive_player_rankings(gameweek=3, limit=10)
    
    for ranking in rankings:
        player = ranking['player']
        analysis = ranking['analysis_breakdown']
        
        print(f"\n{ranking['rank']:2d}. {player.name} ({player.position}) - {player.team}")
        print(f"    💰 Cost: £{player.cost}m | 📊 Enhanced Score: {ranking['total_enhanced_score']:.2f}")
        print(f"    🎯 Base Points: {analysis['base_predicted_points']:.1f} | 🔥 Form: {player.form:.1f}")
        print(f"    📈 ICT Index: {player.ict_index:.0f} | ⚽ xG: {player.expected_goals:.2f} | 🎯 xA: {player.expected_assists:.2f}")
        print(f"    ⏱️ Minutes: {player.minutes_played} | 📋 Status: {player.status}")
        
        if player.chance_of_playing_next_round is not None:
            print(f"    🎲 Chance of Playing: {player.chance_of_playing_next_round}%")
        
        # Show key factors
        factors = []
        if analysis['ict_index_score'] > 0:
            factors.append(f"ICT Bonus: +{analysis['ict_index_score']:.1f}")
        if analysis['expected_goals_assists'] > 0:
            factors.append(f"xG/xA: +{analysis['expected_goals_assists']:.1f}")
        if analysis['position_specific_score'] > 0:
            factors.append(f"Position: +{analysis['position_specific_score']:.1f}")
        if analysis['form_score'] > 0:
            factors.append(f"Form: +{analysis['form_score']:.1f}")
        if analysis['availability_score'] < 0:
            factors.append(f"Availability: {analysis['availability_score']:.1f}")
        
        if factors:
            print(f"    ✅ Key Factors: {', '.join(factors)}")
    
    # Test squad optimization
    print("\n" + "=" * 60)
    print("⚽ Optimizing Squad with Comprehensive Analysis")
    print("=" * 60)
    
    squad = optimizer.optimize_squad(gameweek=3, use_form=True, use_fixtures=True)
    
    print(f"\n🏆 Optimal Squad:")
    print(f"💰 Total Cost: £{squad.total_cost}m")
    print(f"📊 Total Predicted Points: {squad.total_predicted_points:.1f}")
    print(f"⭐ Captain: {squad.captain.name} ({squad.captain.team})")
    print(f"🅥 Vice-Captain: {squad.vice_captain.name} ({squad.vice_captain.team})")
    
    print(f"\n⚽ Starting XI:")
    for player in squad.starting_xi:
        captain_icon = "⭐" if player.is_captain else "🅥" if player.is_vice_captain else ""
        analysis = optimizer.analyze_player_selection_factors(player.player, 3)
        print(f"  {player.name} ({player.position}) - {player.team} - £{player.cost}m - {player.predicted_points:.1f}pts {captain_icon}")
        print(f"    ICT: {player.ict_index:.0f} | xG: {player.expected_goals:.2f} | xA: {player.expected_assists:.2f} | Enhanced: {analysis['total_enhanced_score']:.2f}")
    
    print(f"\n🪑 Bench:")
    for player in squad.bench:
        analysis = optimizer.analyze_player_selection_factors(player.player, 3)
        print(f"  {player.bench_position}. {player.name} ({player.position}) - {player.team} - £{player.cost}m - {player.predicted_points:.1f}pts")
        print(f"    ICT: {player.ict_index:.0f} | xG: {player.expected_goals:.2f} | xA: {player.expected_assists:.2f} | Enhanced: {analysis['total_enhanced_score']:.2f}")
    
    print(f"\n📋 Team Breakdown:")
    for team, count in squad.get_team_breakdown().items():
        print(f"  {team}: {count} players")
    
    print("\n" + "=" * 60)
    print("🎯 Key Improvements in This System:")
    print("=" * 60)
    print("✅ Uses ICT Index (Influence, Creativity, Threat) for player ability")
    print("✅ Incorporates Expected Goals (xG) and Expected Assists (xA)")
    print("✅ Analyzes availability and rotation risk")
    print("✅ Considers team strength and fixture difficulty")
    print("✅ Position-specific analysis (GK saves, DEF attacking threat, etc.)")
    print("✅ Bonus points potential analysis")
    print("✅ Minutes played and consistency analysis")
    print("✅ Not just dependent on recent form!")
    
    print("\n🚀 This system identifies players with high potential based on")
    print("   comprehensive metrics, not just those who performed well")
    print("   in the last 2 gameweeks.")

if __name__ == "__main__":
    main()
