import requests
import pandas as pd
from typing import Dict, Any, List


FPL_BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"


def _safe_float(value: Any) -> float:
	"""Convert value to float safely, defaulting to 0.0 on failure."""
	try:
		if value is None:
			return 0.0
		if isinstance(value, (int, float)):
			return float(value)
		value_str = str(value).strip()
		if value_str == "":
			return 0.0
		return float(value_str)
	except Exception:
		return 0.0


def fetch_bootstrap_data() -> Dict[str, Any]:
	"""Fetch the FPL bootstrap-static JSON payload."""
	resp = requests.get(FPL_BOOTSTRAP_URL, timeout=30)
	resp.raise_for_status()
	return resp.json()


def build_players_dataframe(
		min_minutes: int = 90,
		only_available: bool = True,
        keep_columns_only: bool = False,
) -> pd.DataFrame:
	"""
	Fetch and clean players from FPL API.

	Filtering:
	- Keep players with status == 'a' if only_available is True
	- Keep players with minutes >= min_minutes

	Returned columns include at least:
	- id, name, team, pos, cost, predicted_points

	We also include helper fields used internally:
	- team_id, minutes, chance_play (chance_of_playing_next_round, default 100)

	Set keep_columns_only=True to drop helper fields.
	"""
	data = fetch_bootstrap_data()
	teams = {t["id"]: t["name"] for t in data.get("teams", [])}
	position_map = {p["id"]: p["singular_name_short"] for p in data.get("element_types", [])}

	rows: List[Dict[str, Any]] = []
	for e in data.get("elements", []):
		if only_available and e.get("status") != "a":
			continue
		if int(e.get("minutes", 0)) < int(min_minutes):
			continue

		player_id = e.get("id")
		name = e.get("web_name") or f"{e.get('first_name', '')} {e.get('second_name', '')}".strip()
		team_id = e.get("team")
		team_name = teams.get(team_id, str(team_id))
		pos = position_map.get(e.get("element_type"), "UNK")
		cost = float(e.get("now_cost", 0)) / 10.0
		predicted_points = _safe_float(e.get("ep_next"))
		chance_play = e.get("chance_of_playing_next_round")
		chance_play = 100 if chance_play is None else int(chance_play)

		rows.append(
			{
				"id": player_id,
				"name": name,
				"team": team_name,
				"team_id": team_id,
				"pos": pos,
				"cost": cost,
				"predicted_points": predicted_points,
				"minutes": int(e.get("minutes", 0)),
				"chance_play": chance_play,
			}
		)

	df = pd.DataFrame(rows)
	if keep_columns_only:
		return df[["id", "name", "team", "pos", "cost", "predicted_points"]].copy()
	return df


__all__ = [
	"FPL_BOOTSTRAP_URL",
	"fetch_bootstrap_data",
	"build_players_dataframe",
]

