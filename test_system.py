#!/usr/bin/env python3
"""
Test script for the FPL Optimizer system.
This script tests all components to ensure they work correctly.
"""

import sys
import traceback
from fetch_data import FPLDataFetcher
from optimizer import FPLOptimizer

def test_data_fetching():
    """Test the data fetching functionality."""
    print("🧪 Testing Data Fetching...")
    try:
        fetcher = FPLDataFetcher()
        players = fetcher.fetch_and_clean()
        
        # Basic checks
        assert len(players) > 0, "No players fetched"
        assert 'name' in players.columns, "Name column missing"
        assert 'pos' in players.columns, "Position column missing"
        assert 'team' in players.columns, "Team column missing"
        assert 'cost' in players.columns, "Cost column missing"
        assert 'predicted_points' in players.columns, "Predicted points column missing"
        
        # Data type checks
        assert players['cost'].dtype == 'float64', "Cost should be float"
        assert players['predicted_points'].dtype in ['float64', 'int64'], "Predicted points should be numeric"
        
        # Value checks
        assert players['cost'].min() > 0, "Cost should be positive"
        assert players['predicted_points'].min() >= 0, "Predicted points should be non-negative"
        
        # Position distribution
        positions = players['pos'].value_counts()
        assert 'Goalkeeper' in positions, "No goalkeepers found"
        assert 'Defender' in positions, "No defenders found"
        assert 'Midfielder' in positions, "No midfielders found"
        assert 'Forward' in positions, "No forwards found"
        
        print(f"✅ Data fetching successful! {len(players)} players loaded")
        print(f"   Positions: {dict(positions)}")
        print(f"   Cost range: £{players['cost'].min():.1f}m - £{players['cost'].max():.1f}m")
        print(f"   Points range: {players['predicted_points'].min():.1f} - {players['predicted_points'].max():.1f}")
        
        return players
        
    except Exception as e:
        print(f"❌ Data fetching failed: {e}")
        traceback.print_exc()
        return None

def test_optimization(players):
    """Test the optimization functionality."""
    print("\n🧪 Testing Optimization...")
    try:
        optimizer = FPLOptimizer(players)
        result = optimizer.optimize_complete_squad()
        
        # Basic checks
        assert 'squad' in result, "Squad missing from result"
        assert 'starting_xi' in result, "Starting XI missing from result"
        assert 'bench' in result, "Bench missing from result"
        assert 'captain' in result, "Captain missing from result"
        assert 'vice_captain' in result, "Vice-captain missing from result"
        assert 'total_points' in result, "Total points missing from result"
        assert 'squad_cost' in result, "Squad cost missing from result"
        
        squad = result['squad']
        starting_xi = result['starting_xi']
        bench = result['bench']
        
        # Squad size check
        assert len(squad) == 15, f"Squad should have 15 players, got {len(squad)}"
        assert len(starting_xi) == 11, f"Starting XI should have 11 players, got {len(starting_xi)}"
        assert len(bench) == 4, f"Bench should have 4 players, got {len(bench)}"
        
        # Budget check
        assert result['squad_cost'] <= 100.0, f"Squad cost {result['squad_cost']} exceeds £100m budget"
        
        # Position constraints check
        pos_counts = squad['pos'].value_counts()
        assert pos_counts.get('Goalkeeper', 0) == 2, f"Should have 2 goalkeepers, got {pos_counts.get('Goalkeeper', 0)}"
        assert pos_counts.get('Defender', 0) == 5, f"Should have 5 defenders, got {pos_counts.get('Defender', 0)}"
        assert pos_counts.get('Midfielder', 0) == 5, f"Should have 5 midfielders, got {pos_counts.get('Midfielder', 0)}"
        assert pos_counts.get('Forward', 0) == 3, f"Should have 3 forwards, got {pos_counts.get('Forward', 0)}"
        
        # Team constraints check
        team_counts = squad['team'].value_counts()
        assert team_counts.max() <= 3, f"Maximum 3 players per team exceeded: {team_counts.max()}"
        
        # Captain and vice-captain check
        assert result['captain'] in squad['name'].values, "Captain not in squad"
        assert result['vice_captain'] in squad['name'].values, "Vice-captain not in squad"
        assert result['captain'] != result['vice_captain'], "Captain and vice-captain should be different"
        
        print(f"✅ Optimization successful!")
        print(f"   Total Points: {result['total_points']:.1f}")
        print(f"   Squad Cost: £{result['squad_cost']:.1f}m")
        print(f"   Budget Remaining: £{100.0 - result['squad_cost']:.1f}m")
        print(f"   Captain: {result['captain']}")
        print(f"   Vice-Captain: {result['vice_captain']}")
        
        return result
        
    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        traceback.print_exc()
        return None

def test_captaincy_simulation(result):
    """Test the captaincy simulation functionality."""
    print("\n🧪 Testing Captaincy Simulation...")
    try:
        captaincy_results = result['captaincy_results']
        
        # Basic checks
        assert len(captaincy_results) == 15, f"Should have captaincy results for all 15 players, got {len(captaincy_results)}"
        
        # Check that all squad players are in captaincy results
        squad_players = set(result['squad']['name'].values)
        captaincy_players = set(captaincy_results.keys())
        assert squad_players == captaincy_players, "Captaincy results don't match squad players"
        
        # Check that captain has highest points
        sorted_captaincy = sorted(captaincy_results.items(), key=lambda x: x[1], reverse=True)
        best_captain = sorted_captaincy[0][0]
        assert best_captain == result['captain'], f"Best captain mismatch: {best_captain} vs {result['captain']}"
        
        print(f"✅ Captaincy simulation successful!")
        print(f"   Best captain points: {captaincy_results[result['captain']]:.1f}")
        print(f"   Vice-captain points: {captaincy_results[result['vice_captain']]:.1f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Captaincy simulation failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🚀 Starting FPL Optimizer System Tests\n")
    
    # Test data fetching
    players = test_data_fetching()
    if players is None:
        print("\n❌ System test failed at data fetching stage")
        sys.exit(1)
    
    # Test optimization
    result = test_optimization(players)
    if result is None:
        print("\n❌ System test failed at optimization stage")
        sys.exit(1)
    
    # Test captaincy simulation
    captaincy_success = test_captaincy_simulation(result)
    if not captaincy_success:
        print("\n❌ System test failed at captaincy simulation stage")
        sys.exit(1)
    
    print("\n🎉 All tests passed! The FPL Optimizer system is working correctly.")
    print("\n📊 Final Results Summary:")
    print(f"   Squad Size: {len(result['squad'])} players")
    print(f"   Starting XI: {len(result['starting_xi'])} players")
    print(f"   Bench: {len(result['bench'])} players")
    print(f"   Total Points: {result['total_points']:.1f}")
    print(f"   Squad Cost: £{result['squad_cost']:.1f}m")
    print(f"   Captain: {result['captain']}")
    print(f"   Vice-Captain: {result['vice_captain']}")
    
    print("\n✅ System is ready to use!")
    print("   Run 'streamlit run app.py' to start the web interface")

if __name__ == "__main__":
    main()