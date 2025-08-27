from __future__ import annotations

import requests
from dataclasses import dataclass
from typing import List, Dict

API_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"


@dataclass
class Player:
    """Lightweight representation of an FPL player after cleaning."""

    id: int
    name: str
    team: str
    position: str
    cost: float  # in million £
    predicted_points: float

    def __post_init__(self):
        # Safety: ensure cost is a float with 1 decimal place and predicted_points is float
        self.cost = round(float(self.cost), 1)
        self.predicted_points = float(self.predicted_points)


def fetch_raw_data(url: str = API_URL) -> Dict:
    """Fetch raw JSON data from the official FPL endpoint."""

    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.json()


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
        if element["status"] != "a" or element["minutes"] < 90:
            continue

        predicted = element.get("ep_next")
        predicted_points = float(predicted) if predicted not in (None, "") else 0.0

        player = Player(
            id=element["id"],
            name=f"{element['first_name']} {element['second_name']}",
            team=teams.get(element["team"], "Unknown"),
            position=positions.get(element["element_type"], "UNK"),
            cost=element["now_cost"] / 10.0,  # convert to million £
            predicted_points=predicted_points,
        )
        players.append(player)

    return players


if __name__ == "__main__":
    raw = fetch_raw_data()
    cleaned_players = parse_players(raw)
    print(f"Fetched {len(cleaned_players)} eligible players.")
    # Preview first 5 players
    for p in cleaned_players[:5]:
        print(p)