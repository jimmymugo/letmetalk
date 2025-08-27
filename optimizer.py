from typing import Dict, Tuple, List
import pandas as pd
import pulp


DEFAULT_SQUAD_REQUIREMENTS: Dict[str, int] = {
	"GKP": 2,
	"DEF": 5,
	"MID": 5,
	"FWD": 3,
}


def optimize_squad(
	players: pd.DataFrame,
	budget_millions: float = 100.0,
	max_per_team: int = 3,
	squad_requirements: Dict[str, int] = None,
) -> Tuple[pd.DataFrame, float]:
	"""
	Optimize selection of a 15-player squad under FPL constraints to maximize predicted points.

	Returns (squad_df, objective_value).
	"""
	if squad_requirements is None:
		squad_requirements = DEFAULT_SQUAD_REQUIREMENTS

	players = players.copy().reset_index(drop=True)
	player_indices: List[int] = list(players.index)

	model = pulp.LpProblem("FPL_Squad_Selection", pulp.LpMaximize)
	x = {i: pulp.LpVariable(f"x_{i}", lowBound=0, upBound=1, cat=pulp.LpBinary) for i in player_indices}

	# Objective: maximize total predicted points of selected players
	model += pulp.lpSum(players.loc[i, "predicted_points"] * x[i] for i in player_indices)

	# Exactly 15 players
	model += pulp.lpSum(x[i] for i in player_indices) == 15, "total_players"

	# Position constraints
	for pos, required in squad_requirements.items():
		idx = [i for i in player_indices if players.loc[i, "pos"] == pos]
		model += pulp.lpSum(x[i] for i in idx) == required, f"pos_{pos}"

	# Max per team
	for team_name, group in players.groupby("team"):
		idx = list(group.index)
		model += pulp.lpSum(x[i] for i in idx) <= max_per_team, f"team_{team_name}"

	# Budget
	model += pulp.lpSum(players.loc[i, "cost"] * x[i] for i in player_indices) <= budget_millions, "budget"

	# Solve
	status = model.solve(pulp.PULP_CBC_CMD(msg=False))
	if pulp.LpStatus[status] != "Optimal":
		raise RuntimeError(f"Optimization failed: {pulp.LpStatus[status]}")

	selected_indices = [i for i in player_indices if x[i].value() >= 0.99]
	squad_df = players.loc[selected_indices].copy()
	squad_df.sort_values(["pos", "predicted_points"], ascending=[True, False], inplace=True)

	objective_value = pulp.value(model.objective)
	return squad_df.reset_index(drop=True), float(objective_value)


def pick_starting_xi(squad_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
	"""Pick a starting XI and bench from a 15-player squad using simple greedy logic.

	Rules:
	- 1 starting GKP (highest predicted_points), the other GKP is bench last
	- Outfield must satisfy minimums: DEF>=3, MID>=2, FWD>=1
	- Fill remaining slots by highest predicted_points regardless of outfield position
	- Bench outfield ordered by predicted_points ascending, then GKP last
	"""
	assert len(squad_df) == 15, "Squad must have 15 players"

	# Select starting goalkeeper
	gks = squad_df[squad_df["pos"] == "GKP"].sort_values("predicted_points", ascending=False)
	start_gk = gks.iloc[[0]]
	bench_gk = gks.iloc[[1]]

	# Outfield selection
	defs = squad_df[squad_df["pos"] == "DEF"].sort_values("predicted_points", ascending=False)
	mids = squad_df[squad_df["pos"] == "MID"].sort_values("predicted_points", ascending=False)
	fwds = squad_df[squad_df["pos"] == "FWD"].sort_values("predicted_points", ascending=False)

	selected_outfield = pd.concat([
		defs.head(3),
		mids.head(2),
		fwds.head(1),
	])

	remaining_needed = 10 - len(selected_outfield)
	remaining_outfield_pool = pd.concat([
		defs.iloc[3:],
		mids.iloc[2:],
		fwds.iloc[1:],
	])
	remaining_outfield = remaining_outfield_pool.sort_values("predicted_points", ascending=False).head(remaining_needed)

	starting_xi = pd.concat([start_gk, selected_outfield, remaining_outfield])
	starting_xi = starting_xi.sort_values(["pos", "predicted_points"], ascending=[True, False]).reset_index(drop=True)

	# Bench is the rest
	bench = squad_df[~squad_df.index.isin(starting_xi.index)]
	bench_outfield = bench[bench["pos"] != "GKP"].sort_values("predicted_points", ascending=True)
	bench_ordered = pd.concat([bench_outfield, bench_gk])
	bench_ordered = bench_ordered.reset_index(drop=True)

	return starting_xi, bench_ordered


def simulate_captaincy(starting_xi: pd.DataFrame) -> Tuple[str, str, pd.DataFrame]:
	"""
	Simulate team total points for each starting XI player as captain.
	Returns (captain_name, vice_captain_name, results_df)
	results_df columns: name, team, pos, player_points, team_total
	"""
	base_sum = float(starting_xi["predicted_points"].sum())
	results = []
	for _, row in starting_xi.iterrows():
		player_points = float(row["predicted_points"])
		total = base_sum + player_points  # doubling captain adds +player_points
		results.append({
			"name": row["name"],
			"team": row["team"],
			"pos": row["pos"],
			"player_points": player_points,
			"team_total": total,
		})

	results_df = pd.DataFrame(results).sort_values("team_total", ascending=False).reset_index(drop=True)
	captain_name = str(results_df.iloc[0]["name"]) if not results_df.empty else ""

	# Vice-captain: next best, ideally from different team
	vice_name = ""
	if not results_df.empty:
		captain_team = results_df.iloc[0]["team"]
		diff_team = results_df[results_df["team"] != captain_team]
		if not diff_team.empty:
			vice_name = str(diff_team.iloc[0]["name"])
		elif len(results_df) > 1:
			vice_name = str(results_df.iloc[1]["name"])

	return captain_name, vice_name, results_df


def apply_autosubs_if_needed(starting_xi: pd.DataFrame, bench: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
	"""
	Optional autosubs: if a starting XI player has chance_play == 0, bring bench on in order.
	This is a naive simulation and does not re-check formation constraints.
	"""
	xi = starting_xi.copy()
	bn = bench.copy()

	# Determine which starting players have 0 chance to play
	non_playing_mask = xi.get("chance_play", 100) == 0
	non_playing_indices = list(xi[non_playing_mask].index)

	for idx in non_playing_indices:
		if bn.empty:
			break
		# Take first bench outfield if possible; if only GK left, take it
		replacement = bn.iloc[0:1]
		bn = bn.iloc[1:]
		xi.loc[idx] = replacement.iloc[0]

	return xi.reset_index(drop=True), bn.reset_index(drop=True)


__all__ = [
	"optimize_squad",
	"pick_starting_xi",
	"simulate_captaincy",
	"apply_autosubs_if_needed",
]

