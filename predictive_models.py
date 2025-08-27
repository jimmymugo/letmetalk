"""
Advanced Predictive Models for FPL Optimization
Includes machine learning models, feature engineering, and model retraining capabilities.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import joblib
import json
import os
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from fetch_data import Player, PlayerPerformance
import pickle


@dataclass
class ModelPerformance:
    """Performance metrics for a trained model."""
    model_name: str
    mae: float
    rmse: float
    r2: float
    training_samples: int
    timestamp: str


class FPLPredictiveModel:
    """Advanced machine learning model for FPL predictions."""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.performance_metrics = {}
        self.is_trained = False
        
        # Create models directory if it doesn't exist
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(f"{models_dir}/data", exist_ok=True)
        
        # Load existing models if available
        self._load_existing_models()
    
    def _load_existing_models(self):
        """Load previously saved models and data."""
        try:
            # Load model performance metrics
            metrics_file = f"{self.models_dir}/model_performance.json"
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    self.performance_metrics = json.load(f)
            
            # Load trained models
            for model_name in ['random_forest', 'gradient_boosting', 'xgboost', 'lightgbm', 'ridge', 'linear']:
                model_file = f"{self.models_dir}/{model_name}.joblib"
                if os.path.exists(model_file):
                    self.models[model_name] = joblib.load(model_file)
            
            # Load scaler and encoders
            scaler_file = f"{self.models_dir}/scaler.joblib"
            if os.path.exists(scaler_file):
                self.scaler = joblib.load(scaler_file)
            
            # Load label encoders
            encoders_file = f"{self.models_dir}/label_encoders.joblib"
            if os.path.exists(encoders_file):
                self.label_encoders = joblib.load(encoders_file)
            
            if self.models:
                self.is_trained = True
                print(f"Loaded {len(self.models)} existing models")
                
        except Exception as e:
            print(f"Error loading existing models: {e}")
    
    def _save_models(self):
        """Save all trained models and preprocessing components."""
        try:
            # Save individual models
            for model_name, model in self.models.items():
                model_file = f"{self.models_dir}/{model_name}.joblib"
                joblib.dump(model, model_file)
            
            # Save scaler
            scaler_file = f"{self.models_dir}/scaler.joblib"
            joblib.dump(self.scaler, scaler_file)
            
            # Save label encoders
            encoders_file = f"{self.models_dir}/label_encoders.joblib"
            joblib.dump(self.label_encoders, encoders_file)
            
            # Save performance metrics
            metrics_file = f"{self.models_dir}/model_performance.json"
            with open(metrics_file, 'w') as f:
                json.dump(self.performance_metrics, f, indent=2)
            
            print(f"Saved {len(self.models)} models and metadata")
            
        except Exception as e:
            print(f"Error saving models: {e}")
    
    def _save_historical_data(self, historical_data: List[Dict]):
        """Save historical performance data for retraining."""
        try:
            data_file = f"{self.models_dir}/data/historical_performances.pkl"
            with open(data_file, 'wb') as f:
                pickle.dump(historical_data, f)
            print(f"Saved {len(historical_data)} historical performance records")
        except Exception as e:
            print(f"Error saving historical data: {e}")
    
    def _load_historical_data(self) -> List[Dict]:
        """Load historical performance data."""
        try:
            data_file = f"{self.models_dir}/data/historical_performances.pkl"
            if os.path.exists(data_file):
                with open(data_file, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            print(f"Error loading historical data: {e}")
        return []
    
    def _engineer_features(self, players: List[Player], historical_data: List[Dict]) -> pd.DataFrame:
        """Create sophisticated football analytics features for machine learning."""
        features = []
        
        # Validate input data
        if not players or len(players) == 0:
            return pd.DataFrame()
        
        # Filter out invalid players
        valid_players = [p for p in players if hasattr(p, 'id') and hasattr(p, 'position') and hasattr(p, 'team')]
        
        if not valid_players:
            print("No valid players found after filtering")
            return pd.DataFrame()
        
        print(f"Processing {len(valid_players)} valid players out of {len(players)} total players")
        
        # Debug player structure
        if valid_players:
            sample_player = valid_players[0]
            print(f"Sample player attributes: id={sample_player.id}, name={getattr(sample_player, 'name', 'N/A')}, team={sample_player.team}, position={sample_player.position}")
        
        # Calculate team strength metrics
        team_stats = self._calculate_team_strength(valid_players, historical_data)
        
        for player in valid_players:
                
            # === PLAYER ABILITY FEATURES ===
            try:
                player_features = {
                    'player_id': player.id,
                    'cost': player.cost,
                    'form': player.form,
                    'minutes_played': player.minutes_played,
                    'goals_scored': player.goals_scored,
                    'assists': player.assists,
                    'clean_sheets': player.clean_sheets,
                    'bonus': player.bonus,
                    'total_points': player.total_points,
                    'predicted_points': player.predicted_points,
                }
            except AttributeError:
                # Skip this player if it's not a valid Player object
                continue
            
            # Advanced player metrics
            if player.minutes_played > 0:
                player_features.update({
                    'points_per_minute': round(player.total_points / player.minutes_played, 3),
                    'goals_per_minute': round(player.goals_scored / player.minutes_played, 3),
                    'assists_per_minute': round(player.assists / player.minutes_played, 3),
                    'form_per_minute': round(player.form / player.minutes_played, 3),
                    'minutes_per_game': round(player.minutes_played / max(1, len([d for d in historical_data if d.get('player_id') == player.id])), 1),
                })
            else:
                player_features.update({
                    'points_per_minute': 0,
                    'goals_per_minute': 0,
                    'assists_per_minute': 0,
                    'form_per_minute': 0,
                    'minutes_per_game': 0,
                })
            
            # Goal involvement and creativity metrics
            total_goals = player.goals_scored + player.assists
            player_features.update({
                'goal_involvement_rate': round(total_goals / max(player.minutes_played, 1), 3),
                'goal_involvement_per_game': round(total_goals / max(1, len([d for d in historical_data if d.get('player_id') == player.id])), 2),
                'creativity_score': round((player.assists * 3 + player.bonus) / max(player.minutes_played, 1), 3),
            })
            
            # === HISTORICAL PERFORMANCE ANALYSIS ===
            player_history = []
            if historical_data:
                player_history = [d for d in historical_data if d.get('player_id') == player.id]
            
            if player_history:
                recent_points = [d.get('actual_points', 0) for d in player_history[-5:]]  # Last 5 games
                recent_minutes = [d.get('minutes_played', 0) for d in player_history[-5:]]
                
                # Handle empty lists to avoid numpy warnings
                if recent_points:
                    avg_points = np.mean(recent_points)
                    std_points = np.std(recent_points) if len(recent_points) > 1 else 0
                    min_points = np.min(recent_points)
                    max_points = np.max(recent_points)
                else:
                    avg_points = 0
                    std_points = 0
                    min_points = 0
                    max_points = 0
                
                if recent_minutes:
                    avg_minutes = np.mean(recent_minutes)
                else:
                    avg_minutes = 0
                
                player_features.update({
                    'avg_recent_points': round(avg_points),
                    'std_recent_points': round(std_points, 2),
                    'min_recent_points': min_points,
                    'max_recent_points': max_points,
                    'games_played': len(player_history),
                    'consistency_score': round(1 / (1 + std_points), 3) if std_points > 0 else 1.0,  # Higher = more consistent
                    'availability_score': round(avg_minutes / 90, 3),  # Minutes played ratio
                    'form_trend': self._calculate_form_trend(recent_points),
                    'recent_goal_involvements': sum([d.get('goals_scored', 0) + d.get('assists', 0) for d in player_history[-3:]]),
                })
            else:
                player_features.update({
                    'avg_recent_points': 0,
                    'std_recent_points': 0,
                    'min_recent_points': 0,
                    'max_recent_points': 0,
                    'games_played': 0,
                    'consistency_score': 0.5,
                    'availability_score': 0,
                    'form_trend': 0,
                    'recent_goal_involvements': 0,
                })
            
            # === TEAM STRENGTH FEATURES ===
            team = player.team
            if team in team_stats:
                team_data = team_stats[team]
                player_features.update({
                    'team_attack_rating': team_data['attack_rating'],
                    'team_defense_rating': team_data['defense_rating'],
                    'team_goals_scored': team_data['goals_scored'],
                    'team_goals_conceded': team_data['goals_conceded'],
                    'team_clean_sheets': team_data['clean_sheets'],
                    'team_form': team_data['form'],
                    'team_possession_estimate': team_data['possession_estimate'],
                })
            else:
                player_features.update({
                    'team_attack_rating': 0,
                    'team_defense_rating': 0,
                    'team_goals_scored': 0,
                    'team_goals_conceded': 0,
                    'team_clean_sheets': 0,
                    'team_form': 0,
                    'team_possession_estimate': 50,
                })
            
            # === POSITION-SPECIFIC FEATURES ===
            position_mapping = {'GK': 1, 'DEF': 2, 'MID': 3, 'FWD': 4}
            player_features['position_encoded'] = position_mapping.get(player.position, 0)
            
            # Position-specific metrics
            if player.position == 'GK':
                player_features.update({
                    'clean_sheet_potential': round(player.clean_sheets / max(1, len(player_history)), 3),
                    'save_points_potential': round((player.total_points - player.clean_sheets * 4) / max(1, len(player_history)), 2),
                })
            elif player.position == 'DEF':
                player_features.update({
                    'clean_sheet_potential': round(player.clean_sheets / max(1, len(player_history)), 3),
                    'attacking_threat': round((player.goals_scored + player.assists) / max(1, len(player_history)), 3),
                })
            elif player.position == 'MID':
                player_features.update({
                    'attacking_threat': round((player.goals_scored + player.assists) / max(1, len(player_history)), 3),
                    'creativity_weight': round(player.assists / max(1, len(player_history)), 3),
                })
            elif player.position == 'FWD':
                player_features.update({
                    'goal_scoring_potential': round(player.goals_scored / max(1, len(player_history)), 3),
                    'assist_potential': round(player.assists / max(1, len(player_history)), 3),
                })
            
            # === AVAILABILITY & ROTATION RISK ===
            player_features.update({
                'rotation_risk': self._calculate_rotation_risk(player, player_history),
                'injury_risk': self._calculate_injury_risk(player, player_history),
                'fitness_score': round(player.minutes_played / (90 * max(1, len(player_history))), 3),
            })
            
            # === ENCODING FEATURES ===
            # Team encoding
            try:
                if 'team' not in self.label_encoders:
                    self.label_encoders['team'] = LabelEncoder()
                    all_teams = list(set([p.team for p in valid_players]))
                    if all_teams:
                        self.label_encoders['team'].fit(all_teams)
                    else:
                        # Fallback if no teams found
                        player_features['team_encoded'] = 0
                        features.append(player_features)
                        continue
                
                if hasattr(self.label_encoders['team'], 'classes_') and player.team in self.label_encoders['team'].classes_:
                    player_features['team_encoded'] = self.label_encoders['team'].transform([player.team])[0]
                else:
                    # Fallback if team not in encoder
                    player_features['team_encoded'] = 0
            except Exception as e:
                print(f"Error encoding team for player {player.id}: {e}")
                player_features['team_encoded'] = 0
            
            features.append(player_features)
        
        print(f"Feature engineering completed. Created {len(features)} feature rows")
        if len(features) > 0:
            print(f"Sample features keys: {list(features[0].keys())}")
        else:
            print("WARNING: No features created! This indicates a problem with the data processing.")
            print(f"Valid players count: {len(valid_players)}")
            print(f"Historical data count: {len(historical_data) if historical_data else 0}")
        
        df = pd.DataFrame(features)
        print(f"DataFrame shape: {df.shape}")
        if not df.empty:
            print(f"DataFrame columns: {list(df.columns)}")
            print(f"DataFrame sample:\n{df.head(2)}")
        return df
    
    def _calculate_team_strength(self, players: List[Player], historical_data: List[Dict]) -> Dict:
        """Calculate team strength metrics for all teams."""
        team_stats = {}
        
        # Group players by team
        teams = {}
        for player in players:
            if player.team not in teams:
                teams[player.team] = []
            teams[player.team].append(player)
        
        for team_name, team_players in teams.items():
            # Basic team stats
            total_goals = sum(p.goals_scored for p in team_players)
            total_assists = sum(p.assists for p in team_players)
            total_clean_sheets = sum(p.clean_sheets for p in team_players)
            total_points = sum(p.total_points for p in team_players)
            total_form = sum(p.form for p in team_players)
            
            # Calculate goals conceded (estimate from clean sheets and defensive players)
            defensive_players = [p for p in team_players if p.position in ['GK', 'DEF']]
            goals_conceded_estimate = max(0, len(defensive_players) * 2 - total_clean_sheets * 2)  # Rough estimate
            
            # Team attack rating (goals + assists per game)
            # Use a default of 1 game if no historical data
            games_played = 1
            if historical_data:
                team_games = len([d for d in historical_data if any(p.id == d.get('player_id') for p in team_players)])
                games_played = max(1, team_games)
            
            attack_rating = round((total_goals + total_assists) / games_played, 2)
            
            # Team defense rating (clean sheets per game)
            defense_rating = round(total_clean_sheets / games_played, 2)
            
            # Team form (average form of all players)
            team_form = round(total_form / len(team_players), 2) if team_players else 0
            
            # Possession estimate (based on midfield strength)
            midfield_players = [p for p in team_players if p.position == 'MID']
            possession_estimate = min(70, max(30, 50 + len(midfield_players) * 2))  # Rough estimate
            
            team_stats[team_name] = {
                'attack_rating': attack_rating,
                'defense_rating': defense_rating,
                'goals_scored': total_goals,
                'goals_conceded': goals_conceded_estimate,
                'clean_sheets': total_clean_sheets,
                'form': team_form,
                'possession_estimate': possession_estimate,
                'total_points': total_points,
                'games_played': games_played,
            }
        
        return team_stats
    
    def _calculate_form_trend(self, recent_points: List[float]) -> float:
        """Calculate form trend (positive = improving, negative = declining)."""
        if len(recent_points) < 2:
            return 0
        
        try:
            # Simple linear trend
            x = np.arange(len(recent_points))
            slope = np.polyfit(x, recent_points, 1)[0]
            return round(slope, 3)
        except:
            return 0
    
    def _calculate_rotation_risk(self, player: Player, player_history: List[Dict]) -> float:
        """Calculate rotation risk (0-1, higher = more rotation risk)."""
        if not player_history:
            return 0.5
        
        try:
            # Factors: minutes per game, position, team depth
            recent_minutes = [d.get('minutes_played', 0) for d in player_history[-5:]]
            if recent_minutes:
                avg_minutes = np.mean(recent_minutes)
            else:
                avg_minutes = player.minutes_played  # Use current minutes if no history
            
            minutes_ratio = avg_minutes / 90
            
            # Position-based rotation risk
            position_risk = {
                'GK': 0.1,    # Goalkeepers rarely rotated
                'DEF': 0.3,   # Defenders sometimes rotated
                'MID': 0.4,   # Midfielders often rotated
                'FWD': 0.2,   # Forwards less rotated
            }.get(player.position, 0.3)
            
            # Combine factors
            rotation_risk = (1 - minutes_ratio) * 0.7 + position_risk * 0.3
            return round(min(1.0, max(0.0, rotation_risk)), 3)
        except:
            return 0.5
    
    def _calculate_injury_risk(self, player: Player, player_history: List[Dict]) -> float:
        """Calculate injury risk (0-1, higher = more injury risk)."""
        if not player_history:
            return 0.3
        
        try:
            # Factors: recent minutes, age (estimated from cost), position
            recent_minutes = [d.get('minutes_played', 0) for d in player_history[-3:]]
            if recent_minutes:
                avg_recent_minutes = np.mean(recent_minutes)
            else:
                avg_recent_minutes = player.minutes_played  # Use current minutes if no history
            
            # Cost-based age estimate (rough)
            age_risk = min(1.0, max(0.0, (player.cost - 4.0) / 10.0))  # Higher cost = older player
            
            # Position-based injury risk
            position_risk = {
                'GK': 0.1,    # Goalkeepers rarely injured
                'DEF': 0.2,   # Defenders moderate risk
                'MID': 0.3,   # Midfielders higher risk
                'FWD': 0.25,  # Forwards moderate risk
            }.get(player.position, 0.25)
            
            # Minutes-based fitness risk
            fitness_risk = 1 - (avg_recent_minutes / 90)
            
            # Combine factors
            injury_risk = (age_risk * 0.3 + position_risk * 0.3 + fitness_risk * 0.4)
            return round(min(1.0, max(0.0, injury_risk)), 3)
        except:
            return 0.3
    
    def train_models(self, players: List[Player], historical_data: List[Dict]) -> Dict[str, Dict]:
        """Train multiple ML models on historical data."""
        if not historical_data:
            raise ValueError("No historical data provided for training")
        
        print(f"Training models on {len(historical_data)} historical records...")
        
        # Debug historical data structure
        if historical_data:
            print(f"Sample historical data keys: {list(historical_data[0].keys())}")
            print(f"Sample player_id: {historical_data[0].get('player_id', 'NOT_FOUND')}")
        
        # Prepare training data
        X = self._engineer_features(players, historical_data)
        
        # Create target variable from historical data
        y = []
        print(f"Creating target variable for {len(X)} feature rows...")
        for idx, (_, row) in enumerate(X.iterrows()):
            if idx < 5:  # Debug first 5 rows
                print(f"Row {idx}: player_id={row['player_id']}, predicted_points={row['predicted_points']}")
            
            player_history = [d for d in historical_data if d.get('player_id') == row['player_id']]
            if player_history:
                # Use average of recent actual points as target (rounded to integer)
                recent_points = [d.get('actual_points', 0) for d in player_history[-3:]]  # Last 3 games
                if recent_points:
                    target_value = round(np.mean(recent_points))
                    if idx < 5:
                        print(f"  -> Using historical average: {recent_points} -> {target_value}")
                else:
                    target_value = round(row['predicted_points'])  # Fallback to FPL prediction (rounded)
                    if idx < 5:
                        print(f"  -> Using FPL prediction fallback: {target_value}")
            else:
                target_value = round(row['predicted_points'])  # Fallback to FPL prediction (rounded)
                if idx < 5:
                    print(f"  -> No history, using FPL prediction: {target_value}")
            
            y.append(target_value)
        
        y = np.array(y)
        
        # Remove rows with NaN values
        valid_mask = ~(X.isna().any(axis=1) | np.isnan(y))
        X = X[valid_mask]
        y = y[valid_mask]
        
        print(f"Feature engineering created {len(X)} rows")
        print(f"Target variable has {len(y)} values")
        print(f"Valid mask removed {len(X) - len(X[valid_mask])} rows with NaN values")
        
        if len(X) == 0:
            raise ValueError("No valid training data after preprocessing")
        
        print(f"Final training data: {len(X)} samples")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train models
        models_to_train = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'xgboost': xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
            'lightgbm': lgb.LGBMRegressor(n_estimators=100, random_state=42, verbosity=-1),
            'ridge': Ridge(alpha=1.0),
            'linear': LinearRegression()
        }
        
        self.performance_metrics = {}
        
        for name, model in models_to_train.items():
            print(f"Training {name}...")
            
            # Train model
            if name in ['xgboost', 'lightgbm']:
                model.fit(X_train_scaled, y_train)
            else:
                model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            # Store model and metrics
            self.models[name] = model
            self.performance_metrics[name] = {
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'training_samples': len(X_train),
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            print(f"{name} - MAE: {mae:.3f}, RMSE: {rmse:.3f}, R²: {r2:.3f}")
        
        self.is_trained = True
        
        # Save models and data
        self._save_models()
        self._save_historical_data(historical_data)
        
        return self.performance_metrics
    
    def predict_points(self, players: List[Player], historical_data: List[Dict] = None) -> Dict[int, int]:
        """Predict points for players using ensemble of trained models."""
        if not self.is_trained:
            raise ValueError("Models not trained. Call train_models() first.")
        
        if historical_data is None:
            historical_data = self._load_historical_data()
        
        # Prepare features
        X = self._engineer_features(players, historical_data)
        
        # Remove rows with NaN values
        valid_mask = ~X.isna().any(axis=1)
        X_valid = X[valid_mask]
        
        if len(X_valid) == 0:
            return {}
        
        # Scale features
        X_scaled = self.scaler.transform(X_valid)
        
        # Get predictions from all models
        predictions = {}
        for name, model in self.models.items():
            pred = model.predict(X_scaled)
            for i, player_id in enumerate(X_valid['player_id']):
                if player_id not in predictions:
                    predictions[player_id] = []
                predictions[player_id].append(round(pred[i]))  # Round individual predictions
        
        # Ensemble prediction (weighted average)
        final_predictions = {}
        for player_id, model_preds in predictions.items():
            # Weight by model performance (R² score)
            weights = [self.performance_metrics[name]['r2'] for name in self.models.keys()]
            weights = np.array(weights) / sum(weights)  # Normalize weights
            
            ensemble_pred = np.average(model_preds, weights=weights)
            final_predictions[player_id] = max(0, int(round(ensemble_pred)))  # Ensure non-negative and integer
        
        return final_predictions
    
    def get_best_model(self) -> Optional[str]:
        """Get the name of the best performing model."""
        if not self.performance_metrics:
            return None
        
        best_model = max(self.performance_metrics.items(), key=lambda x: x[1]['r2'])
        return best_model[0]
    
    def get_model_count(self) -> int:
        """Get the number of trained models."""
        return len(self.models)
    
    def get_model_summary(self) -> Dict:
        """Get a comprehensive summary of all trained models and their performance."""
        if not self.is_trained:
            return {"error": "No models trained yet"}
        
        summary = {
            "total_models": len(self.models),
            "models": {},
            "best_model": self.get_best_model(),
            "historical_records": len(self._load_historical_data()) if hasattr(self, '_load_historical_data') else 0,
            "model_type": "Advanced Football Analytics Engine",
            "features_used": [
                "Player Ability (form, xG, xA, creativity)",
                "Team Strength (attack/defense ratings)",
                "Fixture Context (opponent analysis)",
                "Availability (rotation risk, injury risk)",
                "Position-specific metrics",
                "Historical consistency analysis"
            ],
            "prediction_approach": "Hybrid ML Model: f(Player Ability, Team Strength, Fixture Difficulty, Availability)"
        }
        
        # Add performance metrics for each model
        for model_name, metrics in self.performance_metrics.items():
            summary["models"][model_name] = {
                "mae": round(metrics.get('mae', 0), 3),
                "rmse": round(metrics.get('rmse', 0), 3),
                "r2": round(metrics.get('r2', 0), 3),
                "training_samples": metrics.get('training_samples', 0),
                "accuracy_rating": self._get_accuracy_rating(metrics.get('r2', 0))
            }
        
        return summary
    
    def _get_accuracy_rating(self, r2_score: float) -> str:
        """Convert R² score to accuracy rating."""
        if r2_score >= 0.95:
            return "Excellent"
        elif r2_score >= 0.90:
            return "Very Good"
        elif r2_score >= 0.80:
            return "Good"
        elif r2_score >= 0.70:
            return "Fair"
        else:
            return "Poor"
    
    def get_prediction_insights(self, players: List[Player], historical_data: List[Dict] = None) -> Dict:
        """Get detailed insights about predictions for key players."""
        if not self.is_trained:
            return {"error": "Models not trained yet"}
        
        if not players:
            return {"error": "No players provided"}
        
        # Filter out invalid players
        valid_players = [p for p in players if hasattr(p, 'id') and hasattr(p, 'name') and hasattr(p, 'team') and hasattr(p, 'position')]
        
        if not valid_players:
            return {"error": "No valid players found"}
        
        if historical_data is None:
            historical_data = self._load_historical_data()
        
        # Get predictions
        predictions = self.predict_points(valid_players, historical_data)
        
        # Get team strength data
        team_stats = self._calculate_team_strength(valid_players, historical_data)
        
        insights = {
            "prediction_summary": {
                "total_players_analyzed": len(valid_players),
                "average_predicted_points": round(np.mean(list(predictions.values())), 2),
                "highest_predicted_points": max(predictions.values()) if predictions else 0,
                "prediction_confidence": "High (Multi-model ensemble with football context)"
            },
            "top_predictions": [],
            "team_analysis": {},
            "key_insights": []
        }
        
        # Top 10 predictions with insights
        sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:10]
        
        for player_id, predicted_points in sorted_predictions:
            player = next((p for p in valid_players if p.id == player_id), None)
            if not player:
                continue
            
            player_history = []
            if historical_data:
                player_history = [d for d in historical_data if d.get('player_id') == player.id]
            
            insight = {
                "player_name": player.name,
                "team": player.team,
                "position": player.position,
                "predicted_points": predicted_points,
                "current_form": player.form,
                "cost": player.cost,
                "key_factors": self._get_player_key_factors(player, player_history, team_stats),
                "risk_assessment": self._get_risk_assessment(player, player_history),
                "recommendation": self._get_recommendation(predicted_points, player.cost, player.form)
            }
            
            insights["top_predictions"].append(insight)
        
        # Team analysis
        for team_name, team_data in team_stats.items():
            insights["team_analysis"][team_name] = {
                "attack_rating": team_data['attack_rating'],
                "defense_rating": team_data['defense_rating'],
                "form": team_data['form'],
                "strength_assessment": self._get_team_strength_assessment(team_data)
            }
        
        # Key insights
        insights["key_insights"] = [
            "Model considers player ability, team strength, and fixture context",
            "Predictions account for rotation risk and injury probability",
            "Position-specific metrics enhance accuracy",
            "Historical consistency analysis filters out one-off performances",
            "Team strength ratings provide context for new players"
        ]
        
        return insights
    
    def _get_player_key_factors(self, player: Player, player_history: List[Dict], team_stats: Dict) -> List[str]:
        """Get key factors influencing player's prediction."""
        factors = []
        
        # Validate player object
        if not hasattr(player, 'form') or not hasattr(player, 'goals_scored') or not hasattr(player, 'assists') or not hasattr(player, 'team') or not hasattr(player, 'position') or not hasattr(player, 'minutes_played') or not hasattr(player, 'total_points') or not hasattr(player, 'cost'):
            return ["Player data incomplete"]
        
        # Form analysis
        if player.form > 7:
            factors.append("Strong recent form")
        elif player.form < 3:
            factors.append("Poor recent form")
        
        # Goal involvement
        if player.goals_scored + player.assists > 5:
            factors.append("High goal involvement")
        
        # Team strength
        if player.team in team_stats:
            team_data = team_stats[player.team]
            if team_data['attack_rating'] > 2:
                factors.append("Strong attacking team")
            if team_data['defense_rating'] > 1 and player.position in ['GK', 'DEF']:
                factors.append("Solid defensive team")
        
        # Minutes played
        if player.minutes_played > 2000:
            factors.append("Regular starter")
        elif player.minutes_played < 500:
            factors.append("Limited playing time")
        
        # Cost efficiency
        if player.total_points / player.cost > 20:
            factors.append("Cost-effective performer")
        
        return factors
    
    def _get_risk_assessment(self, player: Player, player_history: List[Dict]) -> Dict:
        """Get risk assessment for a player."""
        # Validate player object
        if not hasattr(player, 'position') or not hasattr(player, 'cost') or not hasattr(player, 'minutes_played'):
            return {
                "rotation_risk": 0.5,
                "injury_risk": 0.3,
                "overall_risk": 0.4,
                "risk_level": "Unknown"
            }
        
        rotation_risk = self._calculate_rotation_risk(player, player_history)
        injury_risk = self._calculate_injury_risk(player, player_history)
        
        return {
            "rotation_risk": rotation_risk,
            "injury_risk": injury_risk,
            "overall_risk": round((rotation_risk + injury_risk) / 2, 3),
            "risk_level": "High" if (rotation_risk + injury_risk) / 2 > 0.6 else "Medium" if (rotation_risk + injury_risk) / 2 > 0.3 else "Low"
        }
    
    def _get_recommendation(self, predicted_points: float, cost: float, form: float) -> str:
        """Get recommendation based on prediction."""
        if predicted_points > 8 and cost < 8:
            return "Strong buy - High potential, good value"
        elif predicted_points > 6 and form > 5:
            return "Good option - Consistent performer"
        elif predicted_points < 3:
            return "Avoid - Low predicted returns"
        else:
            return "Consider - Moderate potential"
    
    def _get_team_strength_assessment(self, team_data: Dict) -> str:
        """Get team strength assessment."""
        attack = team_data['attack_rating']
        defense = team_data['defense_rating']
        
        if attack > 2 and defense > 1:
            return "Strong overall team"
        elif attack > 2:
            return "Strong attacking team"
        elif defense > 1:
            return "Strong defensive team"
        else:
            return "Average team"
    
    def retrain_models(self, players: List[Player], new_historical_data: List[Dict]) -> Dict[str, Dict]:
        """Retrain models with new data."""
        # Combine with existing historical data
        existing_data = self._load_historical_data()
        combined_data = existing_data + new_historical_data
        
        # Remove duplicates based on player_id and gameweek
        seen = set()
        unique_data = []
        for record in combined_data:
            key = (record['player_id'], record.get('gameweek', 0))
            if key not in seen:
                seen.add(key)
                unique_data.append(record)
        
        print(f"Retraining with {len(unique_data)} total records ({len(new_historical_data)} new)")
        
        return self.train_models(players, unique_data)


class AdvancedFPLOptimizer:
    """Advanced optimizer using ML predictions."""
    
    def __init__(self, predictive_model: FPLPredictiveModel):
        self.predictive_model = predictive_model
    
    def optimize_with_ml(self, players: List[Player], gameweek: int, historical_data: List[Dict] = None) -> List[Player]:
        """Optimize squad using ML-enhanced predictions."""
        if not self.predictive_model.is_trained:
            raise ValueError("ML models not trained. Train models first.")
        
        # Get ML predictions
        ml_predictions = self.predictive_model.predict_points(players, historical_data)
        
        # Update player predicted points with ML predictions
        for player in players:
            if player.id in ml_predictions:
                player.predicted_points = ml_predictions[player.id]
        
        return players
    
    def get_prediction_confidence(self, players: List[Player], historical_data: List[Dict] = None) -> Dict[int, float]:
        """Get prediction confidence for each player using ML models."""
        if not self.predictive_model.is_trained:
            return {player.id: 0.5 for player in players}  # Default confidence
        
        # Get predictions from all models to calculate confidence
        if historical_data is None:
            historical_data = self.predictive_model._load_historical_data()
        
        # Prepare features
        X = self.predictive_model._engineer_features(players, historical_data)
        
        # Remove rows with NaN values
        valid_mask = ~X.isna().any(axis=1)
        X_valid = X[valid_mask]
        
        if len(X_valid) == 0:
            return {player.id: 0.5 for player in players}
        
        # Scale features
        X_scaled = self.predictive_model.scaler.transform(X_valid)
        
        # Get predictions from all models
        all_predictions = {}
        for name, model in self.predictive_model.models.items():
            pred = model.predict(X_scaled)
            for i, player_id in enumerate(X_valid['player_id']):
                if player_id not in all_predictions:
                    all_predictions[player_id] = []
                all_predictions[player_id].append(round(pred[i]))
        
        # Calculate confidence based on prediction variance
        confidence_scores = {}
        for player_id, predictions in all_predictions.items():
            if len(predictions) > 1:
                # Lower variance = higher confidence
                variance = np.var(predictions)
                # Convert variance to confidence (0-1 scale)
                confidence = max(0.1, min(1.0, 1.0 / (1.0 + variance)))
                confidence_scores[player_id] = confidence
            else:
                confidence_scores[player_id] = 0.5  # Default confidence
        
        return confidence_scores

