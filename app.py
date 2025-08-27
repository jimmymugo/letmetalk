import streamlit as st
import pandas as pd
import plotly.express as px

from fetch_data import build_players_dataframe
from optimizer import optimize_squad, pick_starting_xi, simulate_captaincy, apply_autosubs_if_needed


st.set_page_config(page_title="FPL Optimizer", layout="wide")


@st.cache_data(ttl=3600)
def get_players(min_minutes: int, only_available: bool) -> pd.DataFrame:
	return build_players_dataframe(min_minutes=min_minutes, only_available=only_available, keep_columns_only=False)


def display_squad_table(squad: pd.DataFrame, captain: str, vice: str):
	display = squad.copy()
	def mark_name(name: str) -> str:
		if name == captain:
			return f"{name} ⭐"
		if name == vice:
			return f"{name} 🅥"
		return name
	
	display["Name"] = display["name"].map(mark_name)
	display = display[["Name", "pos", "team", "cost", "predicted_points"]]
	display = display.rename(columns={
		"pos": "Position",
		"team": "Team",
		"cost": "Cost (£m)",
		"predicted_points": "Predicted Points",
	})
	st.dataframe(display, use_container_width=True, hide_index=True)


def main():
	st.title("FPL Optimizer (MVP)")

	with st.sidebar:
		st.header("Settings")
		budget = st.number_input("Budget (£m)", min_value=90.0, max_value=120.0, value=100.0, step=0.5)
		min_minutes = st.number_input("Minimum minutes (season)", min_value=0, max_value=3420, value=90, step=30)
		only_available = st.checkbox("Only available players (status 'a')", value=True)
		use_autosubs = st.checkbox("Simulate autosubs (chance_play==0)", value=False)
		page = st.radio("Page", ["Best Squad", "Captaincy Simulation"], horizontal=False)

	players = get_players(min_minutes=min_minutes, only_available=only_available)

	try:
		squad, obj = optimize_squad(players, budget_millions=budget)
	except Exception as e:
		st.error(f"Optimization error: {e}")
		return

	# Starting XI and bench
	starting_xi, bench = pick_starting_xi(squad)
	if use_autosubs:
		starting_xi, bench = apply_autosubs_if_needed(starting_xi, bench)

	captain, vice, cap_results = simulate_captaincy(starting_xi)

	if page == "Best Squad":
		st.subheader("Optimized Squad")
		display_squad_table(squad, captain, vice)

		st.subheader("Starting XI")
		st.dataframe(
			starting_xi[["name", "pos", "team", "cost", "predicted_points"]]
			.rename(columns={
				"name": "Name",
				"pos": "Position",
				"team": "Team",
				"cost": "Cost (£m)",
				"predicted_points": "Predicted Points",
			}),
			use_container_width=True,
			hide_index=True,
		)

		st.subheader("Bench Order")
		bench_display = bench[["name", "pos", "team", "predicted_points"]].rename(columns={
			"name": "Name",
			"pos": "Position",
			"team": "Team",
			"predicted_points": "Predicted Points",
		})
		st.dataframe(bench_display, use_container_width=True, hide_index=True)

		st.info(f"Captain: {captain} | Vice-Captain: {vice}")

	else:
		st.subheader("Captaincy Simulation (Starting XI)")
		fig = px.bar(cap_results, x="name", y="team_total", color="pos", title="Team total points by captain choice")
		fig.update_layout(xaxis_title="Captain", yaxis_title="Team Total Predicted Points")
		st.plotly_chart(fig, use_container_width=True)

		st.dataframe(
			cap_results.rename(columns={
				"name": "Player",
				"team": "Team",
				"pos": "Position",
				"player_points": "Player Points",
				"team_total": "Team Total",
			}),
			use_container_width=True,
			hide_index=True,
		)


if __name__ == "__main__":
	main()

