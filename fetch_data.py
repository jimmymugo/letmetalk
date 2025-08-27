import requests
import pandas as pd
from typing import Dict, List, Any
from sample_data import get_sample_data

class FPLDataFetcher:
    """Fetches and cleans FPL data from the official API."""
    
    def __init__(self):
        self.api_url = "https://fantasy.premierleague.com/api/bootstrap-static/"
        self.players_data = None
        self.teams_data = None
        
    def fetch_data(self) -> Dict[str, Any]:
        """Fetch all data from FPL API."""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'application/json',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
            response = requests.get(self.api_url, headers=headers, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise Exception(f"Failed to fetch FPL data: {e}")
    
    def clean_players(self, raw_data: Dict[str, Any]) -> pd.DataFrame:
        """Clean and filter players data."""
        players = raw_data.get('elements', [])
        
        # Convert to DataFrame
        df = pd.DataFrame(players)
        
        # Filter for available players with at least 90 minutes
        df = df[
            (df['status'] == 'a') & 
            (df['minutes'] >= 90)
        ].copy()
        
        # Select required fields and rename for clarity
        required_fields = {
            'id': 'id',
            'first_name': 'first_name',
            'second_name': 'second_name',
            'team': 'team_id',
            'element_type': 'position_id',
            'now_cost': 'cost',
            'ep_next': 'predicted_points'
        }
        
        df = df[list(required_fields.keys())].copy()
        df = df.rename(columns=required_fields)
        
        # Create full name
        df['name'] = df['first_name'] + ' ' + df['second_name']
        
        # Convert cost from tenths of millions to millions
        df['cost'] = df['cost'] / 10.0
        
        # Get team names
        teams = {team['id']: team['name'] for team in raw_data.get('teams', [])}
        df['team'] = df['team_id'].map(teams)
        
        # Get position names
        positions = {pos['id']: pos['singular_name'] for pos in raw_data.get('element_types', [])}
        df['pos'] = df['position_id'].map(positions)
        
        # Final columns in desired order
        final_columns = ['id', 'name', 'team', 'pos', 'cost', 'predicted_points']
        df = df[final_columns].copy()
        
        # Convert predicted_points to numeric, handling any non-numeric values
        df['predicted_points'] = pd.to_numeric(df['predicted_points'], errors='coerce')
        
        # Remove any rows with NaN predicted points
        df = df.dropna(subset=['predicted_points'])
        
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
        try:
            print("Fetching FPL data from API...")
            raw_data = self.fetch_data()
            
            print("Cleaning players data...")
            self.players_data = self.clean_players(raw_data)
            self.teams_data = self.get_teams_data(raw_data)
            
            print(f"Successfully processed {len(self.players_data)} players from API")
            return self.players_data
            
        except Exception as e:
            print(f"API fetch failed: {e}")
            print("Using sample data for demonstration...")
            
            # Use sample data as fallback
            self.players_data = get_sample_data()
            print(f"Successfully loaded {len(self.players_data)} sample players")
            return self.players_data
    
    def get_players_by_position(self, position: str) -> pd.DataFrame:
        """Get players filtered by position."""
        if self.players_data is None:
            raise Exception("Data not fetched yet. Call fetch_and_clean() first.")
        
        return self.players_data[self.players_data['pos'] == position].copy()
    
    def get_players_by_team(self, team: str) -> pd.DataFrame:
        """Get players filtered by team."""
        if self.players_data is None:
            raise Exception("Data not fetched yet. Call fetch_and_clean() first.")
        
        return self.players_data[self.players_data['team'] == team].copy()

if __name__ == "__main__":
    # Test the data fetcher
    fetcher = FPLDataFetcher()
    players = fetcher.fetch_and_clean()
    print("\nSample of cleaned data:")
    print(players.head())
    
    print(f"\nPlayers by position:")
    for pos in ['Goalkeeper', 'Defender', 'Midfielder', 'Forward']:
        pos_players = fetcher.get_players_by_position(pos)
        print(f"{pos}: {len(pos_players)} players")