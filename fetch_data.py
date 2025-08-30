from __future__ import annotations

import requests
from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime

API_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
FIXTURES_URL = "https://fantasy.premierleague.com/api/fixtures/"


@dataclass
class Player:
    """Lightweight representation of an FPL player after cleaning."""

    id: int
    name: str
    team: str
    position: str
    cost: float  # in million £
    predicted_points: float
    form: float = 0.0  # Average points in last 3-5 matches
    minutes_played: int = 0
    goals_scored: int = 0
    assists: int = 0
    clean_sheets: int = 0
    bonus: int = 0
    total_points: int = 0
    
    # Enhanced player ability metrics
    influence: float = 0.0  # ICT Index component
    creativity: float = 0.0  # ICT Index component  
    threat: float = 0.0  # ICT Index component
    ict_index: float = 0.0  # Combined ICT Index
    
    # Expected goals and assists (if available from API)
    expected_goals: float = 0.0
    expected_assists: float = 0.0
    
    # Advanced metrics
    bps: int = 0  # Bonus points system
    saves: int = 0  # For goalkeepers
    goals_conceded: int = 0  # For defensive players
    penalties_saved: int = 0  # For goalkeepers
    penalties_missed: int = 0
    yellow_cards: int = 0
    red_cards: int = 0
    
    # Availability metrics
    status: str = "a"  # a=available, u=unavailable, i=injured, n=not available
    chance_of_playing_next_round: Optional[int] = None
    news: str = ""  # Injury/news information
    
    # Historical consistency metrics
    consistency_score: float = 0.0  # Based on historical performance variance
    rotation_risk: float = 0.0  # Risk of being rotated
    injury_risk: float = 0.0  # Risk of injury based on history

    def __post_init__(self):
        # Safety: ensure cost is a float with 1 decimal place and predicted_points is float
        self.cost = round(float(self.cost), 1)
        self.predicted_points = float(self.predicted_points)


@dataclass
class GameweekEvent:
    """Gameweek event information."""
    id: int
    name: str
    deadline_time: str
    average_entry_score: Optional[float] = None
    finished: bool = False
    data_checked: bool = False


@dataclass
class PlayerPerformance:
    """Player performance data for a specific gameweek."""
    player_id: int
    name: str
    team: str
    position: str
    predicted_points: float
    actual_points: float
    minutes_played: int
    goals_scored: int = 0
    assists: int = 0
    clean_sheets: int = 0
    goals_conceded: int = 0
    own_goals: int = 0
    penalties_saved: int = 0
    penalties_missed: int = 0
    yellow_cards: int = 0
    red_cards: int = 0
    saves: int = 0
    bonus: int = 0
    bps: int = 0
    influence: float = 0.0
    creativity: float = 0.0
    threat: float = 0.0
    ict_index: float = 0.0


@dataclass
class Fixture:
    """Fixture information for predictive analytics."""
    id: int
    gameweek: int
    home_team: str
    away_team: str
    home_difficulty: int
    away_difficulty: int
    home_score: Optional[int] = None
    away_score: Optional[int] = None
    finished: bool = False


@dataclass
class GameweekInsights:
    """Comprehensive insights for a gameweek."""
    gameweek: int
    squad_performance: SquadPerformance
    top_performers: List[PlayerPerformance]
    position_insights: Dict[str, Dict]
    overall_accuracy: float
    key_insights: List[str]


def fetch_raw_data(url: str = API_URL) -> Dict:
    """Fetch raw JSON data from the official FPL endpoint."""

    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.json()


def fetch_fixtures_data() -> Dict:
    """Fetch fixtures data from FPL API."""
    response = requests.get(FIXTURES_URL, timeout=30)
    response.raise_for_status()
    return response.json()


def fetch_live_event_data(gameweek: int) -> Dict:
    """Fetch live event data for a specific gameweek."""
    url = f"https://fantasy.premierleague.com/api/event/{gameweek}/live/"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data
    except Exception as e:
        print(f"Error fetching live data for GW{gameweek}: {e}")
        return {}


def parse_players(raw_json: Dict) -> List[Player]:
    """Clean and transform raw FPL data into a list of Player instances.

    Filters:
    - Only players whose status == "a" (available)
    - Must have played at least 90 minutes in the season so far
    """

    # Mapping helpers for team names and position names
    teams = {team["id"]: team["name"] for team in raw_json["teams"]}
    positions = {
        pos["id"]: pos["singular_name_short"] for pos in raw_json["element_types"]
    }

    players: List[Player] = []
    

    
    for element in raw_json["elements"]:
        # Include all available players and high-profile players regardless of minutes
        # Status values: "a" = available, "u" = unavailable, "i" = injured, "n" = not available
        # We'll include "a" (available) and also high-value players who might return
        if element["status"] not in ["a"]:
            # For non-available players, only exclude if they have very low value and no minutes
            # This keeps high-profile players who might return from injury/suspension
            if element.get("minutes", 0) < 10 and element.get("now_cost", 0) < 50:  # Very low minutes AND low value
                continue

        predicted = element.get("ep_next")
        base_predicted_points = float(predicted) if predicted not in (None, "") else 0.0

        # Calculate form (average of last 3 matches)
        form = 0.0
        if element.get("form"):
            try:
                form = float(element["form"])
            except (ValueError, TypeError):
                form = 0.0

        # Extract comprehensive player ability metrics
        influence = float(element.get("influence", 0))
        creativity = float(element.get("creativity", 0))
        threat = float(element.get("threat", 0))
        ict_index = float(element.get("ict_index", 0))
        
        # Expected goals and assists (if available)
        expected_goals = float(element.get("expected_goals", 0))
        expected_assists = float(element.get("expected_assists", 0))
        
        # Advanced metrics
        bps = element.get("bps", 0)
        saves = element.get("saves", 0)
        goals_conceded = element.get("goals_conceded", 0)
        penalties_saved = element.get("penalties_saved", 0)
        penalties_missed = element.get("penalties_missed", 0)
        yellow_cards = element.get("yellow_cards", 0)
        red_cards = element.get("red_cards", 0)
        
        # Availability metrics
        status = element.get("status", "a")
        chance_of_playing_next_round = element.get("chance_of_playing_next_round")
        news = element.get("news", "")

        # Make predictions more realistic based on comprehensive metrics
        minutes_played = element.get("minutes", 0)
        
        # Enhanced prediction using comprehensive metrics
        base_predicted_points = float(predicted) if predicted not in (None, "") else 0.0
        
        # ICT Index adjustment (higher ICT = better performance potential)
        ict_adjustment = 0.0
        if ict_index > 100:
            ict_adjustment = 0.3  # High ICT players tend to perform better
        elif ict_index > 50:
            ict_adjustment = 0.1  # Moderate ICT boost
        elif ict_index < 20:
            ict_adjustment = -0.2  # Low ICT penalty
        
        # Expected goals/assists adjustment
        xg_xa_adjustment = 0.0
        if expected_goals > 0.1 or expected_assists > 0.1:
            xg_xa_adjustment = (expected_goals * 4 + expected_assists * 3) * 0.1  # Convert to points potential
        
        # Form adjustment (players in good form tend to perform better)
        form_adjustment = 0.0
        if form > 7.0:
            form_adjustment = 0.5  # Boost for excellent form
        elif form > 5.0:
            form_adjustment = 0.2  # Small boost for good form
        elif form < 3.0:
            form_adjustment = -0.3  # Penalty for poor form
        
        # Availability adjustment
        availability_adjustment = 0.0
        if status != "a":
            availability_adjustment = -1.0  # Significant penalty for unavailable players
        elif chance_of_playing_next_round is not None and chance_of_playing_next_round < 75:
            availability_adjustment = -0.5  # Penalty for rotation risk
        
        # Adjust based on minutes played (more minutes = more realistic prediction)
        minutes_adjustment = 0.0
        if minutes_played < 90:
            minutes_adjustment = -0.5  # Penalty for very low minutes
        elif minutes_played < 270:
            minutes_adjustment = -0.2  # Small penalty for low minutes
        
        # Position-specific adjustments
        position_adjustment = 0.0
        position = positions.get(element["element_type"], "UNK")
        if position == "GK":
            # Goalkeepers are more predictable, less adjustment
            position_adjustment = 0.0
        elif position == "DEF":
            # Defenders can be inconsistent, slight penalty
            position_adjustment = -0.1
        elif position == "MID":
            # Midfielders are moderately predictable
            position_adjustment = 0.0
        elif position == "FWD":
            # Forwards can be very inconsistent, slight penalty
            position_adjustment = -0.2
        
        # Apply all adjustments to make predictions more realistic
        predicted_points = max(0.0, base_predicted_points + ict_adjustment + xg_xa_adjustment + 
                              form_adjustment + availability_adjustment + minutes_adjustment + position_adjustment)
        
        # Round to 1 decimal place for realism
        predicted_points = round(predicted_points, 1)

        player = Player(
            id=element["id"],
            name=f"{element['first_name']} {element['second_name']}",
            team=teams.get(element["team"], "Unknown"),
            position=positions.get(element["element_type"], "UNK"),
            cost=element["now_cost"] / 10.0,  # convert to million £
            predicted_points=predicted_points,
            form=form,
            minutes_played=element.get("minutes", 0),
            goals_scored=element.get("goals_scored", 0),
            assists=element.get("assists", 0),
            clean_sheets=element.get("clean_sheets", 0),
            bonus=element.get("bonus", 0),
            total_points=element.get("total_points", 0),
            # Enhanced metrics
            influence=influence,
            creativity=creativity,
            threat=threat,
            ict_index=ict_index,
            expected_goals=expected_goals,
            expected_assists=expected_assists,
            bps=bps,
            saves=saves,
            goals_conceded=goals_conceded,
            penalties_saved=penalties_saved,
            penalties_missed=penalties_missed,
            yellow_cards=yellow_cards,
            red_cards=red_cards,
            status=status,
            chance_of_playing_next_round=chance_of_playing_next_round,
            news=news
        )
        players.append(player)

    return players


def parse_fixtures(raw_json: List) -> List[Fixture]:
    """Parse fixtures data for predictive analytics."""
    fixtures = []
    
    # The fixtures API returns a list directly, not a dict
    try:
        for fixture_data in raw_json:
            fixture = Fixture(
                id=fixture_data["id"],
                gameweek=fixture_data["event"],
                home_team=fixture_data.get("team_h_name", "Unknown"),
                away_team=fixture_data.get("team_a_name", "Unknown"),
                home_difficulty=fixture_data.get("team_h_difficulty", 3),
                away_difficulty=fixture_data.get("team_a_difficulty", 3),
                home_score=fixture_data.get("team_h_score"),
                away_score=fixture_data.get("team_a_score"),
                finished=fixture_data.get("finished", False)
            )
            fixtures.append(fixture)
    except Exception as e:
        print(f"Error parsing fixtures: {e}")
        # Return empty list if parsing fails
        return []
    
    return fixtures


def parse_gameweek_events(raw_json: Dict) -> List[GameweekEvent]:
    """Parse gameweek events from raw data."""
    events = []
    for event in raw_json.get("events", []):
        gameweek = GameweekEvent(
            id=event["id"],
            name=event["name"],
            deadline_time=event["deadline_time"],
            average_entry_score=event.get("average_entry_score"),
            finished=event.get("finished", False),
            data_checked=event.get("data_checked", False)
        )
        events.append(gameweek)
    return events


def parse_player_performances(live_data: Dict, players: List[Player]) -> List[PlayerPerformance]:
    """Parse player performances from live event data."""
    performances = []
    
    # Create player lookup
    player_lookup = {p.id: p for p in players}
    
    # The live data structure has 'elements' as a list, not a dict
    elements_data = live_data.get("elements", [])
    
    for element_data in elements_data:
        player_id = element_data.get("id")
        player = player_lookup.get(player_id)
        
        if not player:
            continue
            
        stats = element_data.get("stats", {})
        
        # Get actual points from total_points in stats
        actual_points = stats.get("total_points", 0)
            
        performance = PlayerPerformance(
            player_id=player_id,
            name=player.name,
            team=player.team,
            position=player.position,
            predicted_points=player.predicted_points,
            actual_points=actual_points,
            minutes_played=stats.get("minutes", 0),
            goals_scored=stats.get("goals_scored", 0),
            assists=stats.get("assists", 0),
            clean_sheets=stats.get("clean_sheets", 0),
            goals_conceded=stats.get("goals_conceded", 0),
            own_goals=stats.get("own_goals", 0),
            penalties_saved=stats.get("penalties_saved", 0),
            penalties_missed=stats.get("penalties_missed", 0),
            yellow_cards=stats.get("yellow_cards", 0),
            red_cards=stats.get("red_cards", 0),
            saves=stats.get("saves", 0),
            bonus=stats.get("bonus", 0),
            bps=stats.get("bps", 0),
            influence=stats.get("influence", 0.0),
            creativity=stats.get("creativity", 0.0),
            threat=stats.get("threat", 0.0),
            ict_index=stats.get("ict_index", 0.0)
        )
        performances.append(performance)
    
    return performances


def get_available_gameweeks(raw_json: Dict) -> List[int]:
    """Get list of available gameweeks."""
    events = raw_json.get("events", [])
    return [event["id"] for event in events if event.get("finished", False)]


def get_next_gameweek(raw_json: Dict) -> Optional[int]:
    """Get the next gameweek number."""
    events = raw_json.get("events", [])
    for event in events:
        if not event.get("finished", False):
            return event["id"]
    return None


def get_current_gameweek(raw_json: Dict) -> int:
    """Get the current gameweek number."""
    events = raw_json.get("events", [])
    for event in events:
        if event.get("is_current", False):
            return event["id"]
    # Fallback: return the most recent finished gameweek
    finished_events = [event for event in events if event.get("finished", False)]
    if finished_events:
        return max(event["id"] for event in finished_events)
    return 1  # Default fallback


def find_player_by_name(players: List[Player], query: str) -> Optional[Player]:
    """Find a player by searching across all name variations."""
    query = query.lower()
    
    for player in players:
        # Get the raw player data to access FPL API fields
        # We need to reconstruct the name variations from our Player object
        name_parts = player.name.split()
        if len(name_parts) >= 2:
            first_name = name_parts[0]
            second_name = name_parts[-1]  # Last part as second name
            
            # Create all possible name variations
            full_name = f"{first_name} {second_name}".lower()
            short_dot = f"{first_name[0]}.{second_name}".lower()
            short_space = f"{first_name[0]} {second_name}".lower()
            web_name = second_name.lower()  # Assume web_name is the last name
            
            # Check if query matches any variation
            if (query in full_name or 
                query in short_dot or 
                query in short_space or 
                query in web_name or
                query in player.name.lower()):
                return player
    
    return None


def calculate_fixture_difficulty(fixtures: List[Fixture], team: str, gameweek: int) -> int:
    """Calculate fixture difficulty for a team in a specific gameweek."""
    for fixture in fixtures:
        if fixture.gameweek == gameweek:
            if fixture.home_team == team:
                return fixture.home_difficulty
            elif fixture.away_team == team:
                return fixture.away_difficulty
    return 3  # Default medium difficulty


def get_position_insights(performances: List[PlayerPerformance]) -> Dict[str, Dict]:
    """Generate insights by position."""
    position_data = {}
    
    for pos in ["GK", "DEF", "MID", "FWD"]:
        pos_performances = [p for p in performances if p.position == pos]
        if pos_performances:
            avg_predicted = sum(p.predicted_points for p in pos_performances) / len(pos_performances)
            avg_actual = sum(p.actual_points for p in pos_performances) / len(pos_performances)
            avg_difference = avg_actual - avg_predicted
            
            position_data[pos] = {
                "count": len(pos_performances),
                "avg_predicted": avg_predicted,
                "avg_actual": avg_actual,
                "avg_difference": avg_difference,
                "overperformers": [p for p in pos_performances if p.actual_points > p.predicted_points + 1],
                "underperformers": [p for p in pos_performances if p.actual_points < p.predicted_points - 1]
            }
    
    return position_data


if __name__ == "__main__":
    raw = fetch_raw_data()
    cleaned_players = parse_players(raw)
    events = parse_gameweek_events(raw)
    available_gws = get_available_gameweeks(raw)
    next_gw = get_next_gameweek(raw)
    
    print(f"Fetched {len(cleaned_players)} eligible players.")
    print(f"Available gameweeks: {available_gws}")
    print(f"Next gameweek: {next_gw}")
    # Preview first 5 players
    for p in cleaned_players[:5]:
        print(p)
