import requests
import pandas as pd
from typing import Dict, List, Any

class FPLDataFetcher:
    """Fetches and cleans FPL data from the official API."""
    
    def __init__(self):
        self.api_url = "https://fantasy.premierleague.com/api/bootstrap-static/"
        self.players_data = None
        self.teams_data = None
        
    def fetch_data(self) -> Dict[str, Any]:
        """Fetch data from FPL API."""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        try:
            response = requests.get(self.api_url, headers=headers, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise Exception(f"Failed to fetch FPL data: {e}")
    
    def clean_players(self, raw_data: Dict[str, Any]) -> pd.DataFrame:
        """Clean and filter player data."""
        players = raw_data.get('elements', [])
        
        # Convert to DataFrame
        df = pd.DataFrame(players)
        
        # Select required fields and rename for clarity
        required_fields = {
            'id': 'id',
            'first_name': 'first_name',
            'second_name': 'second_name',
            'team': 'team_id',
            'element_type': 'position_id',
            'now_cost': 'cost',
            'ep_next': 'predicted_points',
            'status': 'status',
            'minutes': 'minutes'
        }
        
        df = df[list(required_fields.keys())].copy()
        df = df.rename(columns=required_fields)
        
        # Filter for available players with at least 90 minutes
        df = df[
            (df['status'] == 'a') & 
            (df['minutes'] >= 90)
        ].copy()
        
        # Convert predicted_points to numeric and filter out negative values
        df['predicted_points'] = pd.to_numeric(df['predicted_points'], errors='coerce')
        df = df[df['predicted_points'] > 0].copy()
        
        # Convert cost from tenths of millions to millions
        df['cost'] = df['cost'] / 10.0
        
        # Convert predicted_points to numeric (already done above, just ensure it's float)
        df['predicted_points'] = df['predicted_points'].astype(float)
        
        # Create full name
        df['name'] = df['first_name'] + ' ' + df['second_name']
        
        # Add team and position names
        teams_dict = {team['id']: team['name'] for team in raw_data.get('teams', [])}
        positions_dict = {pos['id']: pos['singular_name'] for pos in raw_data.get('element_types', [])}
        
        df['team'] = df['team_id'].map(teams_dict)
        df['pos'] = df['position_id'].map(positions_dict)
        
        # Final column selection
        final_columns = ['id', 'name', 'team', 'pos', 'cost', 'predicted_points']
        df = df[final_columns].copy()
        
        # Sort by predicted points descending
        df = df.sort_values('predicted_points', ascending=False)
        
        return df
    
    def get_teams_data(self, raw_data: Dict[str, Any]) -> pd.DataFrame:
        """Extract teams data."""
        teams = raw_data.get('teams', [])
        return pd.DataFrame(teams)
    
    def get_positions_data(self, raw_data: Dict[str, Any]) -> pd.DataFrame:
        """Extract positions data."""
        positions = raw_data.get('element_types', [])
        return pd.DataFrame(positions)
    
    def fetch_and_clean(self) -> pd.DataFrame:
        """Main method to fetch and clean all data."""
        print("Fetching FPL data...")
        raw_data = self.fetch_data()
        
        print("Cleaning player data...")
        self.players_data = self.clean_players(raw_data)
        self.teams_data = self.get_teams_data(raw_data)
        
        print(f"Successfully processed {len(self.players_data)} players")
        return self.players_data
    
    def get_players_by_position(self, position: str) -> pd.DataFrame:
        """Get players filtered by position."""
        if self.players_data is None:
            raise Exception("Data not fetched yet. Call fetch_and_clean() first.")
        
        return self.players_data[self.players_data['pos'] == position].copy()

if __name__ == "__main__":
    # Test the data fetcher
    fetcher = FPLDataFetcher()
    players = fetcher.fetch_and_clean()
    print(f"Total players: {len(players)}")
    print(f"Positions: {players['pos'].value_counts().to_dict()}")
    print(f"Teams: {players['team'].value_counts().to_dict()}")
    print("\nTop 10 players by predicted points:")
    print(players.head(10)[['name', 'pos', 'team', 'cost', 'predicted_points']])