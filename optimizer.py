from __future__ import annotations

import pulp
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from fetch_data import Player, PlayerPerformance, Fixture
from predictive_models import FPLPredictiveModel, AdvancedFPLOptimizer

# FPL Squad Constraints
MAX_BUDGET = 100.0  # million £
SQUAD_SIZE = 15
MAX_PLAYERS_PER_TEAM = 3

# Position requirements
POSITION_REQUIREMENTS = {
    "GK": 2,
    "DEF": 5,
    "MID": 5,
    "FWD": 3,
}

# Starting XI positions (first 11 players)
STARTING_XI_SIZE = 11


@dataclass
class SquadPlayer:
    """Player within a squad with additional squad-specific attributes."""

    player: Player
    is_captain: bool = False
    is_vice_captain: bool = False
    bench_position: Optional[int] = None  # 1-4, None if in starting XI

    @property
    def name(self) -> str:
        return self.player.name

    @property
    def position(self) -> str:
        return self.player.position

    @property
    def team(self) -> str:
        return self.player.team

    @property
    def cost(self) -> float:
        return self.player.cost

    @property
    def predicted_points(self) -> float:
        return self.player.predicted_points

    @property
    def form(self) -> float:
        return self.player.form
    
    @property
    def minutes_played(self) -> int:
        return self.player.minutes_played
    
    @property
    def goals_scored(self) -> int:
        return self.player.goals_scored
    
    @property
    def assists(self) -> int:
        return self.player.assists
    
    @property
    def clean_sheets(self) -> int:
        return self.player.clean_sheets
    
    @property
    def bonus(self) -> int:
        return self.player.bonus
    
    @property
    def total_points(self) -> int:
        return self.player.total_points
    
    @property
    def id(self) -> int:
        return self.player.id


@dataclass
class OptimizedSquad:
    """Complete optimized squad with captain, vice-captain, and bench order."""

    players: List[SquadPlayer]
    total_cost: float
    total_predicted_points: float
    captain: SquadPlayer
    vice_captain: SquadPlayer

    @property
    def starting_xi(self) -> List[SquadPlayer]:
        """Return the 11 players in the starting XI (non-bench players)."""
        return [p for p in self.players if p.bench_position is None]

    @property
    def bench(self) -> List[SquadPlayer]:
        """Return the 4 bench players ordered by bench position."""
        bench_players = [p for p in self.players if p.bench_position is not None]
        return sorted(bench_players, key=lambda p: p.bench_position)

    def get_team_breakdown(self) -> Dict[str, int]:
        """Get count of players per team."""
        team_counts = {}
        for player in self.players:
            team_counts[player.team] = team_counts.get(player.team, 0) + 1
        return team_counts


@dataclass
class SquadPerformance:
    """Squad performance analysis for a specific gameweek."""
    gameweek: int
    squad: OptimizedSquad
    predicted_total: float
    actual_total: float
    difference: float
    mae: float
    rmse: float
    accuracy: float
    player_performances: List[PlayerPerformance]
    overperformers: List[PlayerPerformance]
    underperformers: List[PlayerPerformance]


@dataclass
class PredictiveSquad:
    """Predictive squad with enhanced analytics."""
    squad: OptimizedSquad
    gameweek: int
    total_predicted_points: float
    captain_points: float
    vice_captain_points: float
    form_weighted_score: float
    fixture_difficulty_score: float
    confidence_rating: float


class SquadOptimizer:
    """Main optimizer class for building optimal FPL squads."""

    def __init__(self, players: List[Player], fixtures: Optional[List[Fixture]] = None):
        self.players = players
        self.fixtures = fixtures or []
        self.players_by_position = self._group_players_by_position()
        self.predictive_model = FPLPredictiveModel()
        self.advanced_optimizer = AdvancedFPLOptimizer(self.predictive_model)
        self.ml_enhanced = False
        # Update ML status based on loaded models
        self._update_ml_status()
    
    def _update_ml_status(self):
        """Update ML enhanced status based on loaded models."""
        self.ml_enhanced = self.predictive_model.is_trained and len(self.predictive_model.models) > 0

    def _group_players_by_position(self) -> Dict[str, List[Player]]:
        """Group players by their position."""
        grouped = {"GK": [], "DEF": [], "MID": [], "FWD": []}
        for player in self.players:
            if player.position in grouped:
                grouped[player.position].append(player)
        return grouped

    def optimize_squad(self, gameweek: Optional[int] = None, use_form: bool = True, use_fixtures: bool = True) -> OptimizedSquad:
        """Build the optimal 15-player squad using linear programming with comprehensive player ability analysis."""
        prob = pulp.LpProblem("FPL_Squad_Optimization", pulp.LpMaximize)

        # Decision variables: 1 if player is selected, 0 otherwise
        player_vars = {}
        for player in self.players:
            player_vars[player.id] = pulp.LpVariable(
                f"player_{player.id}", cat=pulp.LpBinary
            )

        # Calculate comprehensive player ability scores
        enhanced_points = {}
        for player in self.players:
            base_points = player.predicted_points
            
            # === PLAYER ABILITY ANALYSIS ===
            # ICT Index analysis (Influence, Creativity, Threat)
            ict_bonus = 0.0
            if player.ict_index > 100:
                ict_bonus = 0.4  # High ICT players are more likely to perform
            elif player.ict_index > 50:
                ict_bonus = 0.2
            elif player.ict_index < 20:
                ict_bonus = -0.3  # Low ICT penalty
            
            # Expected goals/assists analysis
            xg_xa_bonus = 0.0
            if player.expected_goals > 0.1:
                xg_xa_bonus += player.expected_goals * 0.3  # Goals are worth more
            if player.expected_assists > 0.1:
                xg_xa_bonus += player.expected_assists * 0.2  # Assists bonus
            
            # === AVAILABILITY ANALYSIS ===
            availability_penalty = 0.0
            if player.status != "a":
                availability_penalty = -2.0  # Significant penalty for unavailable players
            elif player.chance_of_playing_next_round is not None:
                if player.chance_of_playing_next_round < 50:
                    availability_penalty = -1.5  # High rotation risk
                elif player.chance_of_playing_next_round < 75:
                    availability_penalty = -0.5  # Moderate rotation risk
            
            # === TEAM STRENGTH ANALYSIS ===
            team_strength_bonus = 0.0
            if self.fixtures and gameweek:
                team_difficulty = self._get_fixture_difficulty(player.team, gameweek)
                if team_difficulty <= 2:  # Easy fixture
                    team_strength_bonus = 0.3
                elif team_difficulty >= 4:  # Hard fixture
                    team_strength_bonus = -0.3
            
            # === POSITION-SPECIFIC ANALYSIS ===
            position_bonus = 0.0
            if player.position == "GK":
                # Goalkeepers: consider saves and clean sheet potential
                if player.saves > 20:
                    position_bonus += 0.2  # Good save rate
                if player.clean_sheets > 0:
                    position_bonus += 0.1  # Clean sheet potential
            elif player.position == "DEF":
                # Defenders: consider attacking threat and clean sheets
                if player.goals_scored > 0 or player.assists > 0:
                    position_bonus += 0.3  # Attacking defender bonus
                if player.clean_sheets > 0:
                    position_bonus += 0.2  # Clean sheet potential
            elif player.position == "MID":
                # Midfielders: consider creativity and goal threat
                if player.creativity > 50:
                    position_bonus += 0.2  # Creative midfielder
                if player.threat > 50:
                    position_bonus += 0.2  # Goal threat
            elif player.position == "FWD":
                # Forwards: consider goal threat and expected goals
                if player.threat > 100:
                    position_bonus += 0.3  # High goal threat
                if player.expected_goals > 0.2:
                    position_bonus += 0.2  # Expected goals bonus
            
            # === FORM ANALYSIS (if enabled) ===
            form_bonus = 0.0
            if use_form and player.form > 0:
                if player.form > 7.0:
                    form_bonus = 0.3  # Excellent form
                elif player.form > 5.0:
                    form_bonus = 0.1  # Good form
                elif player.form < 3.0:
                    form_bonus = -0.2  # Poor form
            
            # === MINUTES PLAYED ANALYSIS ===
            minutes_bonus = 0.0
            if player.minutes_played < 90:
                minutes_bonus = -0.5  # Very low minutes penalty
            elif player.minutes_played < 270:
                minutes_bonus = -0.2  # Low minutes penalty
            elif player.minutes_played > 720:  # 8+ full games
                minutes_bonus = 0.1  # Consistent starter bonus
            
            # === BONUS POINTS POTENTIAL ===
            bonus_potential = 0.0
            if player.bps > 100:
                bonus_potential = 0.2  # High BPS potential
            elif player.bps > 50:
                bonus_potential = 0.1  # Moderate BPS potential
            
            # Calculate comprehensive enhanced points
            enhanced_points[player.id] = (
                base_points + 
                ict_bonus + 
                xg_xa_bonus + 
                availability_penalty + 
                team_strength_bonus + 
                position_bonus + 
                form_bonus + 
                minutes_bonus + 
                bonus_potential
            )

        # Objective: Maximize total comprehensive player ability score
        prob += pulp.lpSum(
            enhanced_points[player.id] * player_vars[player.id] for player in self.players
        )

        # Constraint 1: Budget constraint
        prob += (
            pulp.lpSum(player.cost * player_vars[player.id] for player in self.players)
            <= MAX_BUDGET
        )

        # Constraint 2: Squad size constraint
        prob += pulp.lpSum(player_vars[player.id] for player in self.players) == SQUAD_SIZE

        # Constraint 3: Position requirements (make more flexible)
        for pos, count in POSITION_REQUIREMENTS.items():
            pos_players = self.players_by_position[pos]
            if pos_players:  # Only add constraint if we have players for this position
                prob += (
                    pulp.lpSum(player_vars[player.id] for player in pos_players) == count
                )

        # Constraint 4: Max players per team
        teams = set(player.team for player in self.players)
        for team in teams:
            team_players = [p for p in self.players if p.team == team]
            if team_players:  # Only add constraint if we have players for this team
                prob += (
                    pulp.lpSum(player_vars[player.id] for player in team_players)
                    <= MAX_PLAYERS_PER_TEAM
                )

        # Solve the optimization problem
        prob.solve()

        if prob.status != pulp.LpStatusOptimal:
            # Try with relaxed constraints if infeasible
            return self._optimize_with_relaxed_constraints(gameweek, use_form, use_fixtures)

        # Extract selected players
        selected_players = []
        for player in self.players:
            if player_vars[player.id].value() == 1:
                selected_players.append(player)

        # Convert to SquadPlayer objects
        squad_players = [SquadPlayer(player=p) for p in selected_players]

        # Calculate total cost and points
        total_cost = sum(p.cost for p in selected_players)
        total_points = sum(enhanced_points[p.id] for p in selected_players)

        # Determine captain and vice-captain
        captain, vice_captain = self._select_captain_and_vice_captain(squad_players)

        # Set bench order
        self._set_bench_order(squad_players, captain, vice_captain)

        return OptimizedSquad(
            players=squad_players,
            total_cost=total_cost,
            total_predicted_points=total_points,
            captain=captain,
            vice_captain=vice_captain,
        )

    def _get_fixture_difficulty(self, team: str, gameweek: int) -> int:
        """Get fixture difficulty for a team in a specific gameweek."""
        for fixture in self.fixtures:
            if fixture.gameweek == gameweek:
                if fixture.home_team == team:
                    return fixture.home_difficulty
                elif fixture.away_team == team:
                    return fixture.away_difficulty
        return 3  # Default medium difficulty

    def _optimize_with_relaxed_constraints(self, gameweek: Optional[int] = None, use_form: bool = True, use_fixtures: bool = True) -> OptimizedSquad:
        """Fallback optimization with relaxed constraints."""
        # Calculate enhanced scores
        enhanced_scores = []
        for player in self.players:
            base_score = player.predicted_points
            
            # Form adjustment
            form_bonus = 0.0
            if use_form and player.form > 0:
                form_bonus = (player.form - 5.0) * 0.1
            
            # Fixture adjustment
            fixture_bonus = 0.0
            if use_fixtures and gameweek and self.fixtures:
                difficulty = self._get_fixture_difficulty(player.team, gameweek)
                if difficulty <= 2:
                    fixture_bonus = 0.5
                elif difficulty >= 4:
                    fixture_bonus = -0.5
            
            enhanced_score = base_score + form_bonus + fixture_bonus
            enhanced_scores.append((player, enhanced_score))
        
        # Sort by enhanced score
        enhanced_scores.sort(key=lambda x: x[1], reverse=True)
        
        selected_players = []
        total_cost = 0
        position_counts = {"GK": 0, "DEF": 0, "MID": 0, "FWD": 0}
        team_counts = {}
        
        # Greedy selection
        for player, score in enhanced_scores:
            if len(selected_players) >= SQUAD_SIZE:
                break
                
            # Check position constraint
            if position_counts[player.position] >= POSITION_REQUIREMENTS[player.position]:
                continue
                
            # Check team constraint
            if team_counts.get(player.team, 0) >= MAX_PLAYERS_PER_TEAM:
                continue
                
            # Check budget constraint
            if total_cost + player.cost > MAX_BUDGET:
                continue
                
            selected_players.append(player)
            total_cost += player.cost
            position_counts[player.position] += 1
            team_counts[player.team] = team_counts.get(player.team, 0) + 1
        
        # Fill remaining slots if needed
        while len(selected_players) < SQUAD_SIZE:
            for player, score in enhanced_scores:
                if player not in selected_players:
                    selected_players.append(player)
                    break
        
        # Convert to SquadPlayer objects
        squad_players = [SquadPlayer(player=p) for p in selected_players]
        total_points = sum(enhanced_scores[i][1] for i, p in enumerate(enhanced_scores) if p[0] in selected_players)
        
        # Determine captain and vice-captain
        captain, vice_captain = self._select_captain_and_vice_captain(squad_players)
        
        # Set bench order
        self._set_bench_order(squad_players, captain, vice_captain)
        
        return OptimizedSquad(
            players=squad_players,
            total_cost=total_cost,
            total_predicted_points=total_points,
            captain=captain,
            vice_captain=vice_captain,
        )

    def _select_captain_and_vice_captain(
        self, squad_players: List[SquadPlayer]
    ) -> Tuple[SquadPlayer, SquadPlayer]:
        """Select the best captain and vice-captain through simulation."""
        best_captain = None
        best_vice_captain = None
        best_total_points = 0

        # Try each player as captain
        for captain_candidate in squad_players:
            captain_points = captain_candidate.predicted_points * 2

            # Find best vice-captain (different team preferred)
            best_vc_points = 0
            best_vc_candidate = None

            for vc_candidate in squad_players:
                if vc_candidate == captain_candidate:
                    continue

                vc_points = vc_candidate.predicted_points
                total_with_vc = captain_points + vc_points

                # Prefer vice-captain from different team
                if vc_candidate.team != captain_candidate.team:
                    total_with_vc += 0.1  # Small bonus for different team

                if total_with_vc > best_vc_points:
                    best_vc_points = total_with_vc
                    best_vc_candidate = vc_candidate

            # Calculate total team points with this captain
            team_total = (
                sum(p.predicted_points for p in squad_players)
                + captain_candidate.predicted_points  # Captain bonus
                + best_vc_candidate.predicted_points  # Vice-captain points
            )

            if team_total > best_total_points:
                best_total_points = team_total
                best_captain = captain_candidate
                best_vice_captain = best_vc_candidate

        # Set captain and vice-captain flags
        best_captain.is_captain = True
        best_vice_captain.is_vice_captain = True

        return best_captain, best_vice_captain

    def _set_bench_order(
        self,
        squad_players: List[SquadPlayer],
        captain: SquadPlayer,
        vice_captain: SquadPlayer,
    ) -> None:
        """Set bench order according to FPL rules."""
        # First, ensure all players start with no bench position (starting XI)
        for player in squad_players:
            player.bench_position = None
        
        # Sort all players by predicted points (descending for starting XI)
        sorted_players = sorted(squad_players, key=lambda p: p.predicted_points, reverse=True)
        
        # Take the first 11 players as starting XI (no bench position)
        starting_xi_players = sorted_players[:11]
        
        # Take the last 4 players as bench players
        bench_players = sorted_players[11:]
        
        # Assign bench positions (1-4) to bench players only
        for i, player in enumerate(bench_players, 1):
            player.bench_position = i
        
        # Ensure captain and vice-captain are in starting XI
        if captain not in starting_xi_players:
            # Find a player to replace in starting XI
            for player in starting_xi_players:
                if player.bench_position is None and player != vice_captain:
                    player.bench_position = 1
                    captain.bench_position = None
                    break
        
        if vice_captain not in starting_xi_players:
            # Find a player to replace in starting XI
            for player in starting_xi_players:
                if player.bench_position is None and player != captain:
                    player.bench_position = 2
                    vice_captain.bench_position = None
                    break

    def get_captaincy_simulation_data(self, squad: OptimizedSquad) -> Dict[str, float]:
        """Get data for captaincy simulation chart."""
        simulation_data = {}
        
        for player in squad.players:
            # Calculate team total if this player is captain
            captain_bonus = player.predicted_points  # Captain gets 2x, so +1x bonus
            team_total = squad.total_predicted_points + captain_bonus
            simulation_data[player.name] = team_total
            
        return simulation_data

    def get_top_predicted_players(self, limit: int = 10) -> List[Player]:
        """Get top predicted players for a gameweek."""
        return sorted(self.players, key=lambda p: p.predicted_points, reverse=True)[:limit]

    def analyze_squad_performance(
        self, 
        squad: OptimizedSquad, 
        player_performances: List[PlayerPerformance]
    ) -> SquadPerformance:
        """Analyze squad performance against actual results."""
        
        # Create lookup for squad players
        squad_player_ids = {p.player.id for p in squad.players}
        
        # Filter performances to only squad players
        squad_performances = [p for p in player_performances if p.player_id in squad_player_ids]
        
        # Calculate predicted vs actual
        predicted_total = squad.total_predicted_points
        actual_total = sum(p.actual_points for p in squad_performances)
        difference = actual_total - predicted_total
        
        # Calculate metrics
        differences = [p.actual_points - p.predicted_points for p in squad_performances]
        mae = np.mean(np.abs(differences))
        rmse = np.sqrt(np.mean(np.array(differences) ** 2))
        
        # Calculate accuracy (percentage of predictions within 1 point)
        accurate_predictions = sum(1 for diff in differences if abs(diff) <= 1)
        accuracy = (accurate_predictions / len(differences)) * 100 if differences else 0
        
        # Find over/underperformers
        overperformers = [p for p in squad_performances if p.actual_points > p.predicted_points + 1]
        underperformers = [p for p in squad_performances if p.actual_points < p.predicted_points - 1]
        
        return SquadPerformance(
            gameweek=0,  # Will be set by caller
            squad=squad,
            predicted_total=predicted_total,
            actual_total=actual_total,
            difference=difference,
            mae=mae,
            rmse=rmse,
            accuracy=accuracy,
            player_performances=squad_performances,
            overperformers=overperformers,
            underperformers=underperformers
        )

    def create_predictive_squad(self, gameweek: int) -> PredictiveSquad:
        """Create a predictive squad with enhanced analytics."""
        squad = self.optimize_squad(gameweek, use_form=True, use_fixtures=True)
        
        # Calculate enhanced metrics
        form_weighted_score = sum(p.form * p.predicted_points for p in squad.players) / len(squad.players)
        
        fixture_difficulty_score = 0
        if self.fixtures:
            for player in squad.players:
                difficulty = self._get_fixture_difficulty(player.team, gameweek)
                fixture_difficulty_score += (5 - difficulty) * 0.1  # Bonus for easier fixtures
        
        # Calculate confidence rating (0-100)
        confidence_factors = []
        confidence_factors.append(min(100, squad.total_predicted_points / 2))  # Higher points = higher confidence
        confidence_factors.append(min(100, form_weighted_score * 10))  # Good form = higher confidence
        confidence_factors.append(min(100, fixture_difficulty_score * 20))  # Easy fixtures = higher confidence
        
        confidence_rating = np.mean(confidence_factors)
        
        return PredictiveSquad(
            squad=squad,
            gameweek=gameweek,
            total_predicted_points=squad.total_predicted_points,
            captain_points=squad.captain.predicted_points * 2,
            vice_captain_points=squad.vice_captain.predicted_points,
            form_weighted_score=form_weighted_score,
            fixture_difficulty_score=fixture_difficulty_score,
            confidence_rating=confidence_rating
        )
    
    def train_ml_models(self, historical_performances: List[PlayerPerformance]) -> Dict[str, any]:
        """Train machine learning models for enhanced predictions."""
        if not historical_performances:
            return {}
        
        print("Training ML models for enhanced predictions...")
        performance = self.predictive_model.train_models(self.players, historical_performances)
        
        if performance:
            self.ml_enhanced = True
            print("ML models trained successfully!")
        
        return performance
    
    def optimize_with_ml(self, gameweek: Optional[int] = None, historical_performances: List[PlayerPerformance] = None) -> OptimizedSquad:
        """Optimize squad using ML-enhanced predictions."""
        if not self.ml_enhanced and historical_performances:
            self.train_ml_models(historical_performances)
        
        if self.ml_enhanced:
            try:
                # Enhance player predictions using ML models
                enhanced_players = self.enhance_player_predictions(self.players, historical_performances)
                
                # Temporarily replace players with ML-enhanced versions
                original_players = self.players
                self.players = enhanced_players
                self.players_by_position = self._group_players_by_position()
                
                # Optimize with enhanced predictions
                squad = self.optimize_squad(gameweek, use_form=True, use_fixtures=True)
                
                # Restore original players
                self.players = original_players
                self.players_by_position = self._group_players_by_position()
                
                return squad
            except Exception as e:
                print(f"Error in ML optimization: {e}")
                # Fallback to regular optimization
                return self.optimize_squad(gameweek, use_form=True, use_fixtures=True)
        else:
            # Fallback to regular optimization
            return self.optimize_squad(gameweek, use_form=True, use_fixtures=True)
    
    def get_ml_model_performance(self):
        """Get performance metrics for all ML models."""
        # Update ML status before returning performance
        self._update_ml_status()
        return self.predictive_model.get_model_summary()
    
    def get_prediction_confidence(self, historical_performances: List[PlayerPerformance] = None) -> Dict[int, float]:
        """Get prediction confidence for each player using ML models."""
        if self.ml_enhanced:
            return self.advanced_optimizer.get_prediction_confidence(self.players, historical_performances)
        else:
            return {player.id: 0.5 for player in self.players}  # Default confidence
    
    def retrain_models(self, new_performances: List[PlayerPerformance]) -> Dict[str, any]:
        """Retrain ML models with new performance data."""
        if not new_performances:
            return {}
        
        print("Retraining ML models with new data...")
        performance = self.predictive_model.retrain_models(self.players, new_performances)
        
        if performance:
            self.ml_enhanced = True
            print("ML models retrained successfully!")
        
        return performance
    
    def save_ml_models(self, filepath: str = "fpl_models"):
        """Save trained ML models to disk."""
        self.predictive_model.save_models(filepath)
    
    def load_ml_models(self, filepath: str = "fpl_models") -> bool:
        """Load trained ML models from disk."""
        success = self.predictive_model.load_models(filepath)
        if success:
            self.ml_enhanced = True
        return success
    
    def enhance_player_predictions(self, players: List[Player], historical_performances: List[PlayerPerformance] = None) -> List[Player]:
        """Enhance player predictions using trained ML models."""
        if not self.ml_enhanced:
            return players
        
        try:
            # Convert historical performances to the format expected by predictive model
            historical_data = []
            if historical_performances:
                for perf in historical_performances:
                    historical_data.append({
                        'player_id': perf.player_id,
                        'actual_points': perf.actual_points,
                        'minutes_played': perf.minutes_played,
                        'goals_scored': perf.goals_scored,
                        'assists': perf.assists,
                        'clean_sheets': perf.clean_sheets,
                        'bonus': perf.bonus
                    })
            
            # Get ML predictions
            ml_predictions = self.predictive_model.predict_points(players, historical_data)
            
            # Update player predictions
            enhanced_players = []
            for player in players:
                if player.id in ml_predictions:
                    # Create a copy of the player with enhanced prediction
                    enhanced_player = Player(
                        id=player.id,
                        name=player.name,
                        team=player.team,
                        position=player.position,
                        cost=player.cost,
                        form=player.form,
                        minutes_played=player.minutes_played,
                        goals_scored=player.goals_scored,
                        assists=player.assists,
                        clean_sheets=player.clean_sheets,
                        bonus=player.bonus,
                        total_points=player.total_points,
                        predicted_points=ml_predictions[player.id],  # Use ML prediction
                        fixture_difficulty=player.fixture_difficulty
                    )
                    enhanced_players.append(enhanced_player)
                else:
                    enhanced_players.append(player)
            
            return enhanced_players
        except Exception as e:
            print(f"Error enhancing player predictions: {e}")
            return players

    def analyze_player_selection_factors(self, player: Player, gameweek: Optional[int] = None) -> Dict[str, float]:
        """Analyze all factors that contribute to a player's selection potential."""
        analysis = {
            'base_predicted_points': player.predicted_points,
            'ict_index_score': 0.0,
            'expected_goals_assists': 0.0,
            'availability_score': 0.0,
            'team_strength_score': 0.0,
            'position_specific_score': 0.0,
            'form_score': 0.0,
            'minutes_played_score': 0.0,
            'bonus_potential_score': 0.0,
            'total_enhanced_score': 0.0
        }
        
        # ICT Index analysis
        if player.ict_index > 100:
            analysis['ict_index_score'] = 0.4
        elif player.ict_index > 50:
            analysis['ict_index_score'] = 0.2
        elif player.ict_index < 20:
            analysis['ict_index_score'] = -0.3
        
        # Expected goals/assists analysis
        if player.expected_goals > 0.1:
            analysis['expected_goals_assists'] += player.expected_goals * 0.3
        if player.expected_assists > 0.1:
            analysis['expected_goals_assists'] += player.expected_assists * 0.2
        
        # Availability analysis
        if player.status != "a":
            analysis['availability_score'] = -2.0
        elif player.chance_of_playing_next_round is not None:
            if player.chance_of_playing_next_round < 50:
                analysis['availability_score'] = -1.5
            elif player.chance_of_playing_next_round < 75:
                analysis['availability_score'] = -0.5
        
        # Team strength analysis
        if self.fixtures and gameweek:
            team_difficulty = self._get_fixture_difficulty(player.team, gameweek)
            if team_difficulty <= 2:
                analysis['team_strength_score'] = 0.3
            elif team_difficulty >= 4:
                analysis['team_strength_score'] = -0.3
        
        # Position-specific analysis
        if player.position == "GK":
            if player.saves > 20:
                analysis['position_specific_score'] += 0.2
            if player.clean_sheets > 0:
                analysis['position_specific_score'] += 0.1
        elif player.position == "DEF":
            if player.goals_scored > 0 or player.assists > 0:
                analysis['position_specific_score'] += 0.3
            if player.clean_sheets > 0:
                analysis['position_specific_score'] += 0.2
        elif player.position == "MID":
            if player.creativity > 50:
                analysis['position_specific_score'] += 0.2
            if player.threat > 50:
                analysis['position_specific_score'] += 0.2
        elif player.position == "FWD":
            if player.threat > 100:
                analysis['position_specific_score'] += 0.3
            if player.expected_goals > 0.2:
                analysis['position_specific_score'] += 0.2
        
        # Form analysis
        if player.form > 7.0:
            analysis['form_score'] = 0.3
        elif player.form > 5.0:
            analysis['form_score'] = 0.1
        elif player.form < 3.0:
            analysis['form_score'] = -0.2
        
        # Minutes played analysis
        if player.minutes_played < 90:
            analysis['minutes_played_score'] = -0.5
        elif player.minutes_played < 270:
            analysis['minutes_played_score'] = -0.2
        elif player.minutes_played > 720:
            analysis['minutes_played_score'] = 0.1
        
        # Bonus potential analysis
        if player.bps > 100:
            analysis['bonus_potential_score'] = 0.2
        elif player.bps > 50:
            analysis['bonus_potential_score'] = 0.1
        
        # Calculate total enhanced score
        analysis['total_enhanced_score'] = (
            analysis['base_predicted_points'] +
            analysis['ict_index_score'] +
            analysis['expected_goals_assists'] +
            analysis['availability_score'] +
            analysis['team_strength_score'] +
            analysis['position_specific_score'] +
            analysis['form_score'] +
            analysis['minutes_played_score'] +
            analysis['bonus_potential_score']
        )
        
        return analysis
    
    def get_comprehensive_player_rankings(self, gameweek: Optional[int] = None, limit: int = 20) -> List[Dict]:
        """Get comprehensive player rankings based on all metrics, not just recent form."""
        player_rankings = []
        
        for player in self.players:
            analysis = self.analyze_player_selection_factors(player, gameweek)
            
            ranking_data = {
                'player': player,
                'rank': 0,  # Will be set after sorting
                'name': player.name,
                'team': player.team,
                'position': player.position,
                'cost': player.cost,
                'total_enhanced_score': analysis['total_enhanced_score'],
                'base_predicted_points': analysis['base_predicted_points'],
                'ict_index': player.ict_index,
                'expected_goals': player.expected_goals,
                'expected_assists': player.expected_assists,
                'form': player.form,
                'minutes_played': player.minutes_played,
                'status': player.status,
                'chance_of_playing': player.chance_of_playing_next_round,
                'analysis_breakdown': analysis
            }
            
            player_rankings.append(ranking_data)
        
        # Sort by total enhanced score
        player_rankings.sort(key=lambda x: x['total_enhanced_score'], reverse=True)
        
        # Add rank
        for i, ranking in enumerate(player_rankings[:limit], 1):
            ranking['rank'] = i
        
        return player_rankings[:limit]


if __name__ == "__main__":
    # Test the optimizer
    from fetch_data import fetch_raw_data, parse_players, fetch_fixtures_data, parse_fixtures
    
    raw_data = fetch_raw_data()
    players = parse_players(raw_data)
    
    fixtures_data = fetch_fixtures_data()
    fixtures = parse_fixtures(fixtures_data)
    
    optimizer = SquadOptimizer(players, fixtures)
    squad = optimizer.optimize_squad()
    
    print(f"Optimal Squad:")
    print(f"Total Cost: £{squad.total_cost}m")
    print(f"Total Predicted Points: {squad.total_predicted_points:.1f}")
    print(f"Captain: {squad.captain.name} ({squad.captain.team})")
    print(f"Vice-Captain: {squad.vice_captain.name} ({squad.vice_captain.team})")
    
    print("\nStarting XI:")
    for player in squad.starting_xi:
        captain_icon = "⭐" if player.is_captain else "🅥" if player.is_vice_captain else ""
        print(f"  {player.name} ({player.position}) - {player.team} - £{player.cost}m - {player.predicted_points:.1f}pts {captain_icon}")
    
    print("\nBench:")
    for player in squad.bench:
        print(f"  {player.bench_position}. {player.name} ({player.position}) - {player.team} - £{player.cost}m - {player.predicted_points:.1f}pts")
    
    print(f"\nTeam Breakdown:")
    for team, count in squad.get_team_breakdown().items():
        print(f"  {team}: {count} players")
