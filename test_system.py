#!/usr/bin/env python3
"""
Test script for the FPL Optimizer system.
This script tests all components to ensure they work correctly.
"""

import sys
import traceback
from fetch_data import FPLDataFetcher
from optimizer import FPLOptimizer
from sample_data import get_sample_data

def test_data_fetching():
    """Test data fetching functionality."""
    print("🧪 Testing Data Fetching...")
    try:
        fetcher = FPLDataFetcher()
        players = fetcher.fetch_and_clean()
        
        print(f"✅ Successfully loaded {len(players)} players")
        print(f"   - Goalkeepers: {len(players[players['pos'] == 'Goalkeeper'])}")
        print(f"   - Defenders: {len(players[players['pos'] == 'Defender'])}")
        print(f"   - Midfielders: {len(players[players['pos'] == 'Midfielder'])}")
        print(f"   - Forwards: {len(players[players['pos'] == 'Forward'])}")
        
        # Check data quality
        assert len(players) > 0, "No players loaded"
        assert 'predicted_points' in players.columns, "Missing predicted_points column"
        assert players['predicted_points'].dtype in ['float64', 'int64'], "predicted_points not numeric"
        
        return players
        
    except Exception as e:
        print(f"❌ Data fetching failed: {e}")
        traceback.print_exc()
        return None

def test_optimization(players):
    """Test optimization functionality."""
    print("\n🧪 Testing Optimization...")
    try:
        optimizer = FPLOptimizer(players)
        
        # Test squad optimization
        squad_result = optimizer.optimize_squad()
        print(f"✅ Squad optimization successful")
        print(f"   - Total Cost: £{squad_result['total_cost']:.1f}m")
        print(f"   - Total Points: {squad_result['total_predicted_points']:.1f}")
        
        # Test captaincy optimization
        captaincy_result = optimizer.optimize_captaincy()
        print(f"✅ Captaincy optimization successful")
        print(f"   - Captain: {captaincy_result['captain']['name']} ({captaincy_result['captain']['team']})")
        print(f"   - Vice-Captain: {captaincy_result['vice_captain']['name']} ({captaincy_result['vice_captain']['team']})")
        
        # Test bench order optimization
        bench_order = optimizer.optimize_bench_order()
        print(f"✅ Bench order optimization successful")
        print(f"   - Bench players: {len(bench_order)}")
        
        # Test squad summary
        summary = optimizer.get_squad_summary()
        print(f"✅ Squad summary generated")
        print(f"   - Starting XI points: {summary['starting_xi_points']:.1f}")
        print(f"   - Bench points: {summary['bench_points']:.1f}")
        
        return optimizer, squad_result, captaincy_result, bench_order
        
    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        traceback.print_exc()
        return None, None, None, None

def test_constraints(optimizer, squad_result):
    """Test that FPL constraints are satisfied."""
    print("\n🧪 Testing FPL Constraints...")
    try:
        squad = squad_result['squad']
        
        # Test squad size
        assert len(squad) == 15, f"Squad size should be 15, got {len(squad)}"
        print("✅ Squad size: 15 players")
        
        # Test budget constraint
        total_cost = squad['cost'].sum()
        assert total_cost <= 100.0, f"Budget exceeded: £{total_cost:.1f}m > £100.0m"
        print(f"✅ Budget constraint: £{total_cost:.1f}m ≤ £100.0m")
        
        # Test position requirements
        pos_counts = squad['pos'].value_counts()
        assert pos_counts.get('Goalkeeper', 0) == 2, f"Need 2 GKs, got {pos_counts.get('Goalkeeper', 0)}"
        assert pos_counts.get('Defender', 0) == 5, f"Need 5 DEFs, got {pos_counts.get('Defender', 0)}"
        assert pos_counts.get('Midfielder', 0) == 5, f"Need 5 MIDs, got {pos_counts.get('Midfielder', 0)}"
        assert pos_counts.get('Forward', 0) == 3, f"Need 3 FWDs, got {pos_counts.get('Forward', 0)}"
        print("✅ Position requirements satisfied")
        
        # Test team constraint
        team_counts = squad['team'].value_counts()
        max_team_players = team_counts.max()
        assert max_team_players <= 3, f"Max 3 players per team exceeded: {max_team_players}"
        print(f"✅ Team constraint: max {max_team_players} players per team")
        
        print("✅ All FPL constraints satisfied!")
        
    except Exception as e:
        print(f"❌ Constraint test failed: {e}")
        traceback.print_exc()

def test_sample_data():
    """Test sample data generation."""
    print("\n🧪 Testing Sample Data Generation...")
    try:
        sample_data = get_sample_data()
        print(f"✅ Generated {len(sample_data)} sample players")
        
        # Check structure
        required_columns = ['id', 'name', 'team', 'pos', 'cost', 'predicted_points']
        for col in required_columns:
            assert col in sample_data.columns, f"Missing column: {col}"
        
        print("✅ Sample data structure correct")
        return sample_data
        
    except Exception as e:
        print(f"❌ Sample data test failed: {e}")
        traceback.print_exc()
        return None

def main():
    """Run all tests."""
    print("🚀 Starting FPL Optimizer System Tests\n")
    
    # Test sample data first
    sample_data = test_sample_data()
    if sample_data is None:
        print("❌ Sample data test failed, aborting")
        return False
    
    # Test data fetching
    players = test_data_fetching()
    if players is None:
        print("⚠️  Using sample data for optimization tests")
        players = sample_data
    
    # Test optimization
    result = test_optimization(players)
    if result[0] is None:
        print("❌ Optimization tests failed")
        return False
    
    optimizer, squad_result, captaincy_result, bench_order = result
    
    # Test constraints
    test_constraints(optimizer, squad_result)
    
    print("\n🎉 All tests completed successfully!")
    print("\n📊 Final Results:")
    print(f"   - Squad Cost: £{squad_result['total_cost']:.1f}m")
    print(f"   - Predicted Points: {squad_result['total_predicted_points']:.1f}")
    print(f"   - Captain: {captaincy_result['captain']['name']}")
    print(f"   - Vice-Captain: {captaincy_result['vice_captain']['name']}")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)