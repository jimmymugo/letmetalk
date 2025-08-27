import pandas as pd
import numpy as np
from typing import Dict, Any

def generate_sample_fpl_data() -> pd.DataFrame:
    """Generate sample FPL data for testing when API is not accessible."""
    
    # Sample teams
    teams = ['Arsenal', 'Aston Villa', 'Bournemouth', 'Brentford', 'Brighton', 
             'Burnley', 'Chelsea', 'Crystal Palace', 'Everton', 'Fulham',
             'Liverpool', 'Luton', 'Manchester City', 'Manchester United', 
             'Newcastle', 'Nottingham Forest', 'Sheffield United', 'Tottenham', 
             'West Ham', 'Wolves']
    
    # Sample player names (first names and last names)
    first_names = ['James', 'John', 'Robert', 'Michael', 'William', 'David', 'Richard', 
                   'Joseph', 'Thomas', 'Christopher', 'Charles', 'Daniel', 'Matthew', 
                   'Anthony', 'Mark', 'Donald', 'Steven', 'Paul', 'Andrew', 'Joshua',
                   'Kevin', 'Brian', 'George', 'Timothy', 'Ronald', 'Jason', 'Edward',
                   'Jeffrey', 'Ryan', 'Jacob', 'Gary', 'Nicholas', 'Eric', 'Jonathan',
                   'Stephen', 'Larry', 'Justin', 'Scott', 'Brandon', 'Benjamin']
    
    last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller',
                  'Davis', 'Rodriguez', 'Martinez', 'Hernandez', 'Lopez', 'Gonzalez',
                  'Wilson', 'Anderson', 'Thomas', 'Taylor', 'Moore', 'Jackson', 'Martin',
                  'Lee', 'Perez', 'Thompson', 'White', 'Harris', 'Sanchez', 'Clark',
                  'Ramirez', 'Lewis', 'Robinson', 'Walker', 'Young', 'Allen', 'King',
                  'Wright', 'Scott', 'Torres', 'Nguyen', 'Hill', 'Flores']
    
    # Generate players
    players = []
    player_id = 1
    
    # Generate goalkeepers (2 per team)
    for team in teams:
        for i in range(2):
            first_name = np.random.choice(first_names)
            last_name = np.random.choice(last_names)
            name = f"{first_name} {last_name}"
            cost = round(np.random.uniform(4.0, 6.5), 1)
            predicted_points = round(np.random.uniform(2.0, 8.0), 1)
            
            players.append({
                'id': player_id,
                'name': name,
                'team': team,
                'pos': 'Goalkeeper',
                'cost': cost,
                'predicted_points': predicted_points
            })
            player_id += 1
    
    # Generate defenders (4 per team)
    for team in teams:
        for i in range(4):
            first_name = np.random.choice(first_names)
            last_name = np.random.choice(last_names)
            name = f"{first_name} {last_name}"
            cost = round(np.random.uniform(4.0, 8.0), 1)
            predicted_points = round(np.random.uniform(2.0, 12.0), 1)
            
            players.append({
                'id': player_id,
                'name': name,
                'team': team,
                'pos': 'Defender',
                'cost': cost,
                'predicted_points': predicted_points
            })
            player_id += 1
    
    # Generate midfielders (4 per team)
    for team in teams:
        for i in range(4):
            first_name = np.random.choice(first_names)
            last_name = np.random.choice(last_names)
            name = f"{first_name} {last_name}"
            cost = round(np.random.uniform(4.5, 12.0), 1)
            predicted_points = round(np.random.uniform(2.0, 15.0), 1)
            
            players.append({
                'id': player_id,
                'name': name,
                'team': team,
                'pos': 'Midfielder',
                'cost': cost,
                'predicted_points': predicted_points
            })
            player_id += 1
    
    # Generate forwards (2 per team)
    for team in teams:
        for i in range(2):
            first_name = np.random.choice(first_names)
            last_name = np.random.choice(last_names)
            name = f"{first_name} {last_name}"
            cost = round(np.random.uniform(5.0, 14.0), 1)
            predicted_points = round(np.random.uniform(2.0, 18.0), 1)
            
            players.append({
                'id': player_id,
                'name': name,
                'team': team,
                'pos': 'Forward',
                'cost': cost,
                'predicted_points': predicted_points
            })
            player_id += 1
    
    # Create DataFrame
    df = pd.DataFrame(players)
    
    # Sort by predicted points descending
    df = df.sort_values('predicted_points', ascending=False)
    
    return df

def get_sample_data() -> pd.DataFrame:
    """Get sample FPL data."""
    return generate_sample_fpl_data()

if __name__ == "__main__":
    # Test the sample data generator
    sample_data = generate_sample_fpl_data()
    print(f"Generated {len(sample_data)} players")
    print("\nSample data:")
    print(sample_data.head())
    
    print(f"\nPlayers by position:")
    for pos in ['Goalkeeper', 'Defender', 'Midfielder', 'Forward']:
        pos_players = sample_data[sample_data['pos'] == pos]
        print(f"{pos}: {len(pos_players)} players")
    
    print(f"\nPlayers by team:")
    team_counts = sample_data['team'].value_counts()
    print(team_counts.head())