import pulp
import pandas as pd
from typing import Dict, List, Tuple, Any
import numpy as np

class FPLOptimizer:
    """Optimizes FPL squad selection using linear programming."""
    
    def __init__(self, players_data: pd.DataFrame):
        self.players_data = players_data
        self.squad = None
        self.captain = None
        self.vice_captain = None
        self.bench_order = None
        
        # FPL constraints
        self.max_budget = 100.0
        self.squad_size = 15
        self.max_players_per_team = 3
        
        # Position requirements
        self.position_requirements = {
            'Goalkeeper': 2,
            'Defender': 5,
            'Midfielder': 5,
            'Forward': 3
        }
        
    def optimize_squad(self) -> Dict[str, Any]:
        """Optimize squad selection to maximize predicted points."""
        
        # Create optimization problem
        prob = pulp.LpProblem("FPL_Squad_Optimization", pulp.LpMaximize)
        
        # Decision variables: 1 if player is selected, 0 otherwise
        player_vars = pulp.LpVariable.dicts("player",
                                           self.players_data.index,
                                           cat='Binary')
        
        # Objective: Maximize total predicted points
        prob += pulp.lpSum([player_vars[i] * self.players_data.loc[i, 'predicted_points'] 
                           for i in self.players_data.index])
        
        # Constraint 1: Squad size = 15
        prob += pulp.lpSum([player_vars[i] for i in self.players_data.index]) == self.squad_size
        
        # Constraint 2: Budget constraint
        prob += pulp.lpSum([player_vars[i] * self.players_data.loc[i, 'cost'] 
                           for i in self.players_data.index]) <= self.max_budget
        
        # Constraint 3: Position requirements
        for position, count in self.position_requirements.items():
            position_players = self.players_data[self.players_data['pos'] == position].index
            prob += pulp.lpSum([player_vars[i] for i in position_players]) == count
        
        # Constraint 4: Max 3 players per team
        teams = self.players_data['team'].unique()
        for team in teams:
            team_players = self.players_data[self.players_data['team'] == team].index
            prob += pulp.lpSum([player_vars[i] for i in team_players]) <= self.max_players_per_team
        
        # Solve the problem
        prob.solve()
        
        if prob.status != pulp.LpStatusOptimal:
            raise Exception("Optimization failed to find optimal solution")
        
        # Extract selected players
        selected_players = []
        for i in self.players_data.index:
            if player_vars[i].value() == 1:
                selected_players.append(i)
        
        self.squad = self.players_data.loc[selected_players].copy()
        
        return {
            'squad': self.squad,
            'total_cost': self.squad['cost'].sum(),
            'total_predicted_points': self.squad['predicted_points'].sum(),
            'objective_value': pulp.value(prob.objective)
        }
    
    def optimize_captaincy(self) -> Dict[str, Any]:
        """Find optimal captain and vice-captain."""
        if self.squad is None:
            raise Exception("Squad not optimized yet. Call optimize_squad() first.")
        
        captaincy_results = []
        
        # Test each player as captain
        for idx, player in self.squad.iterrows():
            # Calculate team total with this player as captain
            captain_points = player['predicted_points'] * 2
            other_players_points = self.squad[self.squad.index != idx]['predicted_points'].sum()
            team_total = captain_points + other_players_points
            
            captaincy_results.append({
                'player_id': player['id'],
                'name': player['name'],
                'team': player['team'],
                'pos': player['pos'],
                'predicted_points': player['predicted_points'],
                'captain_points': captain_points,
                'team_total': team_total
            })
        
        # Sort by team total descending
        captaincy_results.sort(key=lambda x: x['team_total'], reverse=True)
        
        # Select captain (best overall)
        self.captain = captaincy_results[0]
        
        # Select vice-captain (next best, ideally from different team)
        vice_captain_candidates = [p for p in captaincy_results[1:] 
                                 if p['team'] != self.captain['team']]
        
        if vice_captain_candidates:
            self.vice_captain = vice_captain_candidates[0]
        else:
            # If no different team, take second best overall
            self.vice_captain = captaincy_results[1]
        
        return {
            'captain': self.captain,
            'vice_captain': self.vice_captain,
            'captaincy_results': captaincy_results
        }
    
    def optimize_bench_order(self) -> List[Dict[str, Any]]:
        """Optimize bench order for maximum points."""
        if self.squad is None:
            raise Exception("Squad not optimized yet. Call optimize_squad() first.")
        
        # Separate starting XI and bench
        starting_xi = self.squad.head(11).copy()
        bench = self.squad.tail(4).copy()
        
        # Bench GK is always last (if there is one)
        bench_gks = bench[bench['pos'] == 'Goalkeeper']
        if len(bench_gks) > 0:
            bench_gk = bench_gks.iloc[0]
            outfield_bench = bench[bench['pos'] != 'Goalkeeper'].copy()
        else:
            # If no GK on bench, just use all outfield players
            bench_gk = None
            outfield_bench = bench.copy()
        
        # Sort outfield bench by predicted points ascending (worst first for autosubs)
        outfield_bench = outfield_bench.sort_values('predicted_points', ascending=True)
        
        # Create bench order
        bench_order = []
        
        # Add outfield bench players in order
        for idx, player in outfield_bench.iterrows():
            bench_order.append({
                'position': f"Bench {len(bench_order) + 1}",
                'name': player['name'],
                'team': player['team'],
                'pos': player['pos'],
                'cost': player['cost'],
                'predicted_points': player['predicted_points']
            })
        
        # Add GK last (if there is one)
        if bench_gk is not None:
            bench_order.append({
                'position': f"Bench {len(bench_order) + 1}",
                'name': bench_gk['name'],
                'team': bench_gk['team'],
                'pos': bench_gk['pos'],
                'cost': bench_gk['cost'],
                'predicted_points': bench_gk['predicted_points']
            })
        
        self.bench_order = bench_order
        return bench_order
    
    def get_starting_xi(self) -> pd.DataFrame:
        """Get the starting XI (top 11 players by predicted points)."""
        if self.squad is None:
            raise Exception("Squad not optimized yet. Call optimize_squad() first.")
        
        return self.squad.head(11).copy()
    
    def get_bench(self) -> pd.DataFrame:
        """Get the bench (bottom 4 players)."""
        if self.squad is None:
            raise Exception("Squad not optimized yet. Call optimize_squad() first.")
        
        return self.squad.tail(4).copy()
    
    def simulate_autosubs(self, starting_xi_minutes: Dict[int, int] = None) -> Dict[str, Any]:
        """Simulate autosubs if starting XI players have 0 minutes."""
        if self.squad is None:
            raise Exception("Squad not optimized yet. Call optimize_squad() first.")
        
        starting_xi = self.get_starting_xi()
        bench = self.get_bench()
        
        # Default: all starting XI players have minutes
        if starting_xi_minutes is None:
            starting_xi_minutes = {player['id']: 90 for _, player in starting_xi.iterrows()}
        
        # Find players with 0 minutes
        zero_minute_players = []
        for _, player in starting_xi.iterrows():
            if starting_xi_minutes.get(player['id'], 90) == 0:
                zero_minute_players.append(player)
        
        if not zero_minute_players:
            return {
                'autosubs_applied': False,
                'final_xi': starting_xi,
                'substitutions': []
            }
        
        # Apply autosubs
        final_xi = starting_xi.copy()
        substitutions = []
        
        # Sort bench by predicted points (best first for substitution)
        bench_sorted = bench.sort_values('predicted_points', ascending=False)
        
        for zero_minute_player in zero_minute_players:
            # Find best available bench player of same position
            same_pos_bench = bench_sorted[bench_sorted['pos'] == zero_minute_player['pos']]
            
            if len(same_pos_bench) > 0:
                substitute = same_pos_bench.iloc[0]
                
                # Remove substitute from bench
                bench_sorted = bench_sorted[bench_sorted.index != substitute.name]
                
                # Replace in final XI
                final_xi = final_xi[final_xi.index != zero_minute_player.name]
                final_xi = pd.concat([final_xi, pd.DataFrame([substitute])], ignore_index=True)
                
                substitutions.append({
                    'out': zero_minute_player['name'],
                    'in': substitute['name'],
                    'position': zero_minute_player['pos']
                })
        
        return {
            'autosubs_applied': True,
            'final_xi': final_xi,
            'substitutions': substitutions
        }
    
    def get_squad_summary(self) -> Dict[str, Any]:
        """Get comprehensive squad summary."""
        if self.squad is None:
            raise Exception("Squad not optimized yet. Call optimize_squad() first.")
        
        starting_xi = self.get_starting_xi()
        bench = self.get_bench()
        
        return {
            'total_cost': self.squad['cost'].sum(),
            'total_predicted_points': self.squad['predicted_points'].sum(),
            'starting_xi_points': starting_xi['predicted_points'].sum(),
            'bench_points': bench['predicted_points'].sum(),
            'players_by_position': self.squad['pos'].value_counts().to_dict(),
            'players_by_team': self.squad['team'].value_counts().to_dict(),
            'captain': self.captain,
            'vice_captain': self.vice_captain,
            'bench_order': self.bench_order
        }

if __name__ == "__main__":
    # Test the optimizer
    from fetch_data import FPLDataFetcher
    
    # Fetch data
    fetcher = FPLDataFetcher()
    players = fetcher.fetch_and_clean()
    
    # Optimize squad
    optimizer = FPLOptimizer(players)
    result = optimizer.optimize_squad()
    
    print("Optimization Results:")
    print(f"Total Cost: £{result['total_cost']:.1f}m")
    print(f"Total Predicted Points: {result['total_predicted_points']:.1f}")
    print(f"Objective Value: {result['objective_value']:.1f}")
    
    # Optimize captaincy
    captaincy = optimizer.optimize_captaincy()
    print(f"\nCaptain: {captaincy['captain']['name']} ({captaincy['captain']['team']})")
    print(f"Vice-Captain: {captaincy['vice_captain']['name']} ({captaincy['vice_captain']['team']})")
    
    # Optimize bench order
    bench_order = optimizer.optimize_bench_order()
    print("\nBench Order:")
    for player in bench_order:
        print(f"{player['position']}: {player['name']} ({player['pos']}) - {player['predicted_points']:.1f} pts")