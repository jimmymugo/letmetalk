import pulp
import pandas as pd
from typing import Dict, List, Tuple, Optional
import numpy as np

class FPLOptimizer:
    """FPL Squad Optimizer using Linear Programming."""
    
    def __init__(self, players_data: pd.DataFrame):
        self.players = players_data.copy()
        self.budget = 100.0  # £100m budget
        self.max_players_per_team = 3
        self.squad_size = 15
        
        # Position constraints
        self.position_limits = {
            'Goalkeeper': 2,
            'Defender': 5,
            'Midfielder': 5,
            'Forward': 3
        }
        
        # Starting XI constraints
        self.starting_xi_limits = {
            'Goalkeeper': 1,
            'Defender': 3,
            'Midfielder': 5,
            'Forward': 2
        }
        
    def optimize_squad(self) -> Tuple[pd.DataFrame, float]:
        """Optimize squad selection to maximize predicted points."""
        
        # Create optimization problem
        prob = pulp.LpProblem("FPL_Squad_Optimization", pulp.LpMaximize)
        
        # Decision variables: 1 if player is selected, 0 otherwise
        player_vars = pulp.LpVariable.dicts("Player",
                                          range(len(self.players)),
                                          cat='Binary')
        
        # Objective: Maximize total predicted points
        prob += pulp.lpSum([player_vars[i] * self.players.iloc[i]['predicted_points'] 
                           for i in range(len(self.players))])
        
        # Constraint 1: Budget constraint
        prob += pulp.lpSum([player_vars[i] * self.players.iloc[i]['cost'] 
                           for i in range(len(self.players))]) <= self.budget
        
        # Constraint 2: Squad size constraint
        prob += pulp.lpSum([player_vars[i] for i in range(len(self.players))]) == self.squad_size
        
        # Constraint 3: Position constraints
        for position, limit in self.position_limits.items():
            position_indices = self.players[self.players['pos'] == position].index.tolist()
            prob += pulp.lpSum([player_vars[i] for i in range(len(self.players)) 
                               if self.players.iloc[i]['pos'] == position]) == limit
        
        # Constraint 4: Max players per team
        for team in self.players['team'].unique():
            prob += pulp.lpSum([player_vars[i] for i in range(len(self.players)) 
                               if self.players.iloc[i]['team'] == team]) <= self.max_players_per_team
        
        # Solve the problem
        prob.solve()
        
        if prob.status != pulp.LpStatusOptimal:
            raise Exception("Optimization failed to find optimal solution")
        
        # Extract selected players
        selected_players = []
        total_cost = 0
        total_points = 0
        
        for i in range(len(self.players)):
            if player_vars[i].value() == 1:
                player = self.players.iloc[i].copy()
                selected_players.append(player)
                total_cost += player['cost']
                total_points += player['predicted_points']
        
        selected_squad = pd.DataFrame(selected_players)
        selected_squad = selected_squad.sort_values(['pos', 'predicted_points'], 
                                                   ascending=[True, False])
        
        return selected_squad, total_points
    
    def optimize_starting_xi(self, squad: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Optimize starting XI from the selected squad."""
        
        # Create optimization problem for starting XI
        prob = pulp.LpProblem("FPL_Starting_XI_Optimization", pulp.LpMaximize)
        
        # Decision variables: 1 if player starts, 0 if on bench
        start_vars = pulp.LpVariable.dicts("Start",
                                         range(len(squad)),
                                         cat='Binary')
        
        # Objective: Maximize total predicted points of starting XI
        prob += pulp.lpSum([start_vars[i] * squad.iloc[i]['predicted_points'] 
                           for i in range(len(squad))])
        
        # Constraint 1: Starting XI size
        prob += pulp.lpSum([start_vars[i] for i in range(len(squad))]) == 11
        
        # Constraint 2: Position constraints for starting XI
        for position, limit in self.starting_xi_limits.items():
            prob += pulp.lpSum([start_vars[i] for i in range(len(squad)) 
                               if squad.iloc[i]['pos'] == position]) >= limit
        
        # Constraint 3: At least one goalkeeper must start
        prob += pulp.lpSum([start_vars[i] for i in range(len(squad)) 
                           if squad.iloc[i]['pos'] == 'Goalkeeper']) >= 1
        
        # Solve the problem
        prob.solve()
        
        if prob.status != pulp.LpStatusOptimal:
            raise Exception("Starting XI optimization failed")
        
        # Extract starting XI and bench
        starting_xi = []
        bench = []
        
        for i in range(len(squad)):
            player = squad.iloc[i].copy()
            if start_vars[i].value() == 1:
                starting_xi.append(player)
            else:
                bench.append(player)
        
        starting_xi_df = pd.DataFrame(starting_xi)
        bench_df = pd.DataFrame(bench)
        
        # Sort bench by predicted points (ascending for autosubs)
        bench_df = bench_df.sort_values('predicted_points', ascending=True)
        
        return starting_xi_df, bench_df
    
    def captaincy_simulation(self, squad: pd.DataFrame) -> Tuple[str, str, Dict[str, float]]:
        """Simulate captaincy for each player and find best captain/vice-captain."""
        
        captaincy_results = {}
        
        for _, player in squad.iterrows():
            # Calculate team total with this player as captain (2x points)
            captain_points = player['predicted_points'] * 2
            other_players_points = squad[squad['name'] != player['name']]['predicted_points'].sum()
            total_points = captain_points + other_players_points
            
            captaincy_results[player['name']] = total_points
        
        # Sort by total points descending
        sorted_results = sorted(captaincy_results.items(), key=lambda x: x[1], reverse=True)
        
        best_captain = sorted_results[0][0]
        best_vice_captain = sorted_results[1][0]
        
        return best_captain, best_vice_captain, captaincy_results
    
    def optimize_complete_squad(self) -> Dict[str, any]:
        """Complete optimization pipeline."""
        
        print("Optimizing squad selection...")
        squad, squad_points = self.optimize_squad()
        
        print("Optimizing starting XI...")
        starting_xi, bench = self.optimize_starting_xi(squad)
        
        print("Simulating captaincy...")
        captain, vice_captain, captaincy_results = self.captaincy_simulation(squad)
        
        # Calculate final team total with best captain
        captain_player = squad[squad['name'] == captain].iloc[0]
        final_points = (captain_player['predicted_points'] * 2 + 
                       squad[squad['name'] != captain]['predicted_points'].sum())
        
        return {
            'squad': squad,
            'starting_xi': starting_xi,
            'bench': bench,
            'captain': captain,
            'vice_captain': vice_captain,
            'captaincy_results': captaincy_results,
            'total_points': final_points,
            'squad_cost': squad['cost'].sum()
        }

if __name__ == "__main__":
    # Test the optimizer
    from fetch_data import FPLDataFetcher
    
    fetcher = FPLDataFetcher()
    players = fetcher.fetch_and_clean()
    
    optimizer = FPLOptimizer(players)
    result = optimizer.optimize_complete_squad()
    
    print(f"\nOptimal Squad (Total Points: {result['total_points']:.2f}, Cost: £{result['squad_cost']:.1f}m)")
    print("\nStarting XI:")
    print(result['starting_xi'][['name', 'pos', 'team', 'cost', 'predicted_points']])
    print(f"\nCaptain: {result['captain']} ⭐")
    print(f"Vice-Captain: {result['vice_captain']} 🅥")
    print("\nBench:")
    print(result['bench'][['name', 'pos', 'team', 'cost', 'predicted_points']])