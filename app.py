import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional
import time
from datetime import datetime
import numpy as np
import pandas as pd

from fetch_data import (
    fetch_raw_data, parse_players, fetch_live_event_data, 
    parse_player_performances, parse_gameweek_events, get_available_gameweeks,
    fetch_fixtures_data, parse_fixtures, get_next_gameweek, get_current_gameweek, get_position_insights,
    find_player_by_name, Player, GameweekEvent
)
from optimizer import SquadOptimizer, OptimizedSquad, SquadPerformance, PredictiveSquad
from predictive_models import FPLPredictiveModel


def main():
    st.set_page_config(
        page_title="FPL AI Optimizer",
        page_icon="⚽",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Header
    st.title("⚽ FPL AI Optimizer")
    st.markdown("---")

    # Sidebar navigation
    st.sidebar.title("Navigation")
    
    # Initialize session state
    if "page" not in st.session_state:
        st.session_state.page = "Best Squad"
    if "gw_tab" not in st.session_state:
        st.session_state.gw_tab = "Gameweek Insights"
    
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["Best Squad", "Captaincy Analysis", "Bench & Autosubs", "Player Comparisons", "Gameweek Explorer", "ML Models", "Settings"],
        index=["Best Squad", "Captaincy Analysis", "Bench & Autosubs", "Player Comparisons", "Gameweek Explorer", "ML Models", "Settings"].index(st.session_state.page)
    )
    
    # Update session state
    st.session_state.page = page

    # Load data with caching
    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def load_fpl_data():
        with st.spinner("Fetching FPL data..."):
            raw_data = fetch_raw_data()
            players = parse_players(raw_data)
            events = parse_gameweek_events(raw_data)
            available_gws = get_available_gameweeks(raw_data)
            next_gw = get_next_gameweek(raw_data)
            
            # Fetch fixtures data
            try:
                fixtures_data = fetch_fixtures_data()
                fixtures = parse_fixtures(fixtures_data)
            except Exception as e:
                st.warning(f"Could not fetch fixtures data: {e}")
                fixtures = []
            
            return players, raw_data, events, available_gws, next_gw, fixtures

    try:
        players, raw_data, events, available_gws, next_gw, fixtures = load_fpl_data()
        
        if not players:
            st.error("No players found. Please check your internet connection.")
            return



        st.sidebar.success(f"✅ Loaded {len(players)} eligible players")
        
        # Get current gameweek info
        current_gw = get_current_gameweek(raw_data)
        
        # Gameweek Selector in Sidebar
        st.sidebar.markdown("---")
        st.sidebar.markdown("**📅 Gameweek Selector**")
        
        # Create list of available gameweeks
        available_gameweeks = []
        for event in events:
            if event.finished:
                available_gameweeks.append(event.id)
        
        # Sort gameweeks
        available_gameweeks.sort()
        
        # Add current and next gameweek if not already included
        if current_gw not in available_gameweeks:
            available_gameweeks.append(current_gw)
        if next_gw not in available_gameweeks:
            available_gameweeks.append(next_gw)
        
        available_gameweeks.sort()
        
        # Gameweek dropdown
        selected_gw = st.sidebar.selectbox(
            "Select Gameweek:",
            available_gameweeks,
            index=len(available_gameweeks) - 1,  # Default to latest
            format_func=lambda x: f"GW {x} {'(Current)' if x == current_gw else '(Next)' if x == next_gw else ''}"
        )
        
        # Quick navigation buttons
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            if st.button(f"📊 GW {selected_gw} Insights", help=f"View insights for Gameweek {selected_gw}"):
                st.session_state.page = "Gameweek Explorer"
                st.session_state.gw_tab = "Gameweek Insights"
                st.session_state.selected_gw = selected_gw
                st.rerun()
        
        with col2:
            if st.button(f"⚽ GW {selected_gw} Squad", help=f"View squad for Gameweek {selected_gw}"):
                st.session_state.page = "Gameweek Explorer"
                st.session_state.gw_tab = "Next Gameweek Squad"
                st.rerun()

        # Create optimizer with fixtures
        optimizer = SquadOptimizer(players, fixtures)

        # Page routing
        if page == "Best Squad":
            show_best_squad_page(optimizer, current_gw)
        elif page == "Captaincy Analysis":
            show_captaincy_analysis_page(optimizer, raw_data)
        elif page == "Bench & Autosubs":
            show_bench_autosubs_page(optimizer)
        elif page == "Player Comparisons":
            show_player_comparisons_page(players)
        elif page == "Gameweek Explorer":
            show_gameweek_explorer_page(optimizer, players, events, available_gws, next_gw)
        elif page == "ML Models":
            show_ml_models_page(optimizer, players, events, available_gws)
        elif page == "Settings":
            show_settings_page()

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("Please check your internet connection and try again.")


def show_best_squad_page(optimizer: SquadOptimizer, current_gw: str):
    """Display the optimal squad page."""
    st.header(f"🏆 Best Squad - Gameweek {current_gw}")
    
    # Gameweek Status
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"📅 **Current Gameweek**: {current_gw}")
    with col2:
        st.success(f"✅ **Data Status**: Live")
    with col3:
        st.warning(f"🔄 **Last Updated**: {datetime.now().strftime('%H:%M:%S')}")
    
    # Get recent performance data
    try:
        with st.spinner("Fetching recent performance data..."):
            # Try current gameweek first, fall back to previous if not available
            recent_gw = current_gw
            live_data = fetch_live_event_data(recent_gw)
            
            # Check if current gameweek has sufficient data
            if live_data and "elements" in live_data:
                recent_performances = parse_player_performances(live_data, optimizer.players)
                # If we have good data for current gameweek, use it
                if recent_performances and len(recent_performances) > 100:  # Reasonable amount of data
                    pass  # Use current gameweek
                else:
                    # Fall back to previous gameweek
                    recent_gw = max(1, current_gw - 1)
                    live_data = fetch_live_event_data(recent_gw)
                    recent_performances = parse_player_performances(live_data, optimizer.players) if live_data else []
            else:
                # Fall back to previous gameweek
                recent_gw = max(1, current_gw - 1)
                live_data = fetch_live_event_data(recent_gw)
                recent_performances = parse_player_performances(live_data, optimizer.players) if live_data else []
    except:
        recent_performances = []

    # Optimize squad
    with st.spinner("Optimizing squad..."):
        squad = optimizer.optimize_squad()

    # 2.1 Squad Summary (Top Panel)
    st.subheader("📊 Squad Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("💰 Team Value", f"£{squad.total_cost:.1f}m")
    
    with col2:
        st.metric("⭐ Predicted Points", f"{squad.total_predicted_points:.1f}")
    
    with col3:
        st.metric("👑 Captain", f"{squad.captain.name}")
    
    with col4:
        st.metric("🅥 Vice-Captain", f"{squad.vice_captain.name}")
    
    # Recent Performance Summary
    if recent_performances:
        if recent_gw == current_gw:
            st.subheader(f"📈 Recent Performance (Gameweek {recent_gw} - Current)")
        else:
            st.subheader(f"📈 Recent Performance (Gameweek {recent_gw} - Most Recent Available)")
        
        # Calculate squad performance in the gameweek
        squad_players = [p.player.id for p in squad.players]
        recent_squad_performance = [p for p in recent_performances if p.player_id in squad_players]
        
        if recent_squad_performance:
            # Calculate actual FPL points for starting XI with captain bonus and bench substitutions
            starting_xi_actual = 0
            captain_actual = 0
            vice_captain_actual = 0
            bench_substitutions = []
            
            # Get starting XI and bench players
            starting_xi_players = [p for p in squad.players if p.bench_position is None]
            bench_players = squad.bench  # Already sorted by bench position
            
            # Track which players actually played (got minutes)
            players_used = []
            
            # First, check if captain played
            captain_played = False
            for squad_player in starting_xi_players:
                if squad_player.is_captain:
                    player_perf = next((p for p in recent_squad_performance if p.player_id == squad_player.player.id), None)
                    if player_perf and player_perf.minutes_played > 0:
                        captain_played = True
                    break
            
            # Now calculate points for starting XI
            for squad_player in starting_xi_players:
                player_perf = next((p for p in recent_squad_performance if p.player_id == squad_player.player.id), None)
                
                if player_perf and player_perf.minutes_played > 0:
                    # Player played, add their points
                    base_points = player_perf.actual_points
                    players_used.append(squad_player.player.id)
                    
                    # Apply captain/vice-captain bonus
                    if squad_player.is_captain:
                        captain_actual = base_points * 2  # Captain gets double points
                        starting_xi_actual += captain_actual
                    elif squad_player.is_vice_captain:
                        # Vice-captain only gets double points if captain didn't play
                        if not captain_played:
                            vice_captain_actual = base_points * 2
                            starting_xi_actual += vice_captain_actual
                        else:
                            vice_captain_actual = base_points  # Normal points if captain played
                            starting_xi_actual += base_points
                    else:
                        starting_xi_actual += base_points
                else:
                    # Player didn't play, need bench substitution
                    bench_substitutions.append(squad_player.name)
            
            # Handle bench substitutions (FPL rules: bench players come in order if starters don't play)
            bench_index = 0
            for squad_player in bench_players:
                if bench_index >= len(bench_substitutions):
                    break  # No more substitutions needed
                
                player_perf = next((p for p in recent_squad_performance if p.player_id == squad_player.player.id), None)
                
                if player_perf and player_perf.minutes_played > 0:
                    # Bench player played, substitute them in
                    base_points = player_perf.actual_points
                    players_used.append(squad_player.player.id)
                    starting_xi_actual += base_points
                    bench_substitutions[bench_index] = f"{bench_substitutions[bench_index]} → {squad_player.name} ({base_points} pts)"
                    bench_index += 1
                else:
                    # Bench player also didn't play, try next one
                    continue
            
            # Calculate predicted points for starting XI
            starting_xi_predicted = sum(p.predicted_points for p in starting_xi_players)
            accuracy = (starting_xi_actual / starting_xi_predicted * 100) if starting_xi_predicted > 0 else 0
            
            # Get highest individual player score for reference
            highest_player_score = max(p.actual_points for p in recent_performances) if recent_performances else 0
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📊 Starting XI Points", f"{starting_xi_actual:.1f}", 
                         help="Total points scored by the 11 starting players (including captain bonus and bench substitutions)")
            
            with col2:
                st.metric("🎯 Predicted Points", f"{starting_xi_predicted:.1f}")
            
            with col3:
                st.metric("📈 Accuracy", f"{accuracy:.1f}%")
            
            with col4:
                diff = starting_xi_actual - starting_xi_predicted
                st.metric("📊 Difference", f"{diff:+.1f}")
            
            # Show captain and vice-captain performance
            if captain_actual > 0 or vice_captain_actual > 0:
                captain_info = f"Captain: {captain_actual:.1f} pts" if captain_actual > 0 else "Captain: 0 pts"
                if not captain_played and vice_captain_actual > 0:
                    vice_info = f"Vice-Captain: {vice_captain_actual:.1f} pts (doubled - captain didn't play)"
                elif vice_captain_actual > 0:
                    vice_info = f"Vice-Captain: {vice_captain_actual:.1f} pts (normal - captain played)"
                else:
                    vice_info = "Vice-Captain: 0 pts"
                st.info(f"ℹ️ **Captain Performance**: {captain_info} | {vice_info}")
            
            # Show bench substitutions if any
            if bench_substitutions:
                st.info(f"🔄 **Bench Substitutions**: {', '.join(bench_substitutions)}")
            
            # Add note about highest individual score
            st.info(f"ℹ️ **Note**: Highest individual player score in Gameweek {recent_gw} was {highest_player_score} points")
            
            # Show detailed comparison of predicted vs actual points for all squad players
            st.subheader("📊 Squad Performance Comparison")
            
            # Create comparison data for all squad players
            comparison_data = []
            for squad_player in squad.players:
                player_perf = next((p for p in recent_squad_performance if p.player_id == squad_player.player.id), None)
                
                if player_perf:
                    # Player has performance data
                    actual_points = player_perf.actual_points
                    predicted_points = squad_player.predicted_points
                    difference = actual_points - predicted_points
                    minutes_played = player_perf.minutes_played
                    
                    # Determine if player was in starting XI or bench
                    if squad_player.bench_position is None:
                        position_status = "Starting XI"
                        if squad_player.is_captain:
                            position_status += " ⭐ (C)"
                        elif squad_player.is_vice_captain:
                            position_status += " 🅥 (VC)"
                    else:
                        position_status = f"Bench ({squad_player.bench_position})"
                    
                    comparison_data.append({
                        "Player": squad_player.name,
                        "Position": squad_player.position,
                        "Team": squad_player.team,
                        "Status": position_status,
                        "Predicted": f"{predicted_points:.1f}",
                        "Actual": f"{actual_points:.1f}",
                        "Difference": f"{difference:+.1f}",
                        "Minutes": f"{minutes_played}",
                        "Performance": "✅ Played" if minutes_played > 0 else "❌ No Minutes"
                    })
                else:
                    # Player has no performance data (didn't play)
                    predicted_points = squad_player.predicted_points
                    
                    if squad_player.bench_position is None:
                        position_status = "Starting XI"
                        if squad_player.is_captain:
                            position_status += " ⭐ (C)"
                        elif squad_player.is_vice_captain:
                            position_status += " 🅥 (VC)"
                    else:
                        position_status = f"Bench ({squad_player.bench_position})"
                    
                    comparison_data.append({
                        "Player": squad_player.name,
                        "Position": squad_player.position,
                        "Team": squad_player.team,
                        "Status": position_status,
                        "Predicted": f"{predicted_points:.1f}",
                        "Actual": "0.0",
                        "Difference": f"{-predicted_points:+.1f}",
                        "Minutes": "0",
                        "Performance": "❌ No Minutes"
                    })
            
            # Sort by actual points (descending) then by predicted points (descending)
            comparison_data.sort(key=lambda x: (float(x["Actual"]), float(x["Predicted"])), reverse=True)
            
            # Display the comparison table
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, width='stretch', hide_index=True)
            
            # Summary statistics
            total_predicted = sum(float(row["Predicted"]) for row in comparison_data)
            total_actual = sum(float(row["Actual"]) for row in comparison_data)
            total_difference = total_actual - total_predicted
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Total Predicted", f"{total_predicted:.1f}")
            with col2:
                st.metric("📈 Total Actual", f"{total_actual:.1f}")
            with col3:
                st.metric("📊 Total Difference", f"{total_difference:+.1f}")
    
    # Top Performers from the Gameweek
    if recent_performances:
        if recent_gw == current_gw:
            st.subheader(f"🏆 Top Performers (Gameweek {recent_gw} - Current)")
        else:
            st.subheader(f"🏆 Top Performers (Gameweek {recent_gw} - Most Recent Available)")
        
        # Get top 5 performers
        top_performers = sorted(recent_performances, key=lambda p: p.actual_points, reverse=True)[:5]
        
        top_data = []
        for i, player in enumerate(top_performers, 1):
            top_data.append({
                "Rank": i,
                "Player": player.name,
                "Team": player.team,
                "Position": player.position,
                "Points": player.actual_points,
                "Predicted": f"{player.predicted_points:.1f}",
                "Difference": f"{player.actual_points - player.predicted_points:+.1f}"
            })
        
        df = pd.DataFrame(top_data)
        st.dataframe(df, width='stretch', hide_index=True)
    
    # Key Insights and Alerts
    st.subheader("🔍 Key Insights & Alerts")
    
    # Check for injured players in the squad
    injured_players = [p for p in squad.players if p.minutes_played < 45 and p.total_points > 0]
    if injured_players:
        st.warning(f"⚠️ **Injury Alert**: {len(injured_players)} players in your squad have limited recent minutes:")
        for player in injured_players:
            st.write(f"   • {player.name} ({player.team}) - {player.minutes_played} minutes played")
    
    # Check for high-value players
    high_value_players = [p for p in squad.players if p.cost > 10.0]
    if high_value_players:
        st.info(f"💰 **Premium Players**: {len(high_value_players)} players cost over £10m:")
        for player in high_value_players:
            st.write(f"   • {player.name} ({player.team}) - £{player.cost:.1f}m")
    
    # Check for differential picks (low ownership)
    differential_players = [p for p in squad.players if p.cost < 6.0 and p.predicted_points > 4.0]
    if differential_players:
        st.success(f"🎯 **Differential Picks**: {len(differential_players)} budget players with high potential:")
        for player in differential_players:
            st.write(f"   • {player.name} ({player.team}) - £{player.cost:.1f}m, {player.predicted_points:.1f} pts")

    st.markdown("---")

    # 2.2 Starting XI Display
    st.subheader("⚽ Starting XI")
    
    # Pitch View and Table View tabs
    tab1, tab2 = st.tabs(["🎯 Pitch View", "📋 Table View"])
    
    with tab1:
        show_pitch_view(squad)
    
    with tab2:
        show_starting_xi_table(squad)

    # 2.3 Bench Order
    st.subheader("🪑 Bench Order")
    show_bench_order(squad)

    # Team Breakdown
    st.subheader("🏟️ Team Breakdown")
    show_team_breakdown(squad)


def show_pitch_view(squad: OptimizedSquad):
    """Display squad in a pitch-like formation."""
    # Debug: Check if squad has players
    if not squad.players:
        st.error("No players in squad!")
        return
    
    # Debug: Check starting XI
    starting_xi = squad.starting_xi
    if not starting_xi:
        st.error("No starting XI players found!")
        st.write(f"Total players: {len(squad.players)}")
        st.write(f"Players with bench_position None: {len([p for p in squad.players if p.bench_position is None])}")
        
        # Show all players and their bench positions
        st.write("**All Players:**")
        for i, player in enumerate(squad.players):
            st.write(f"{i+1}. {player.name} ({player.position}) - Bench: {player.bench_position}")
        return

    # Group players by position
    gk_players = [p for p in starting_xi if p.position == "GK"]
    def_players = [p for p in starting_xi if p.position == "DEF"]
    mid_players = [p for p in starting_xi if p.position == "MID"]
    fwd_players = [p for p in starting_xi if p.position == "FWD"]
    
    # Debug: Show player counts and starting XI players
    st.write(f"**Starting XI Breakdown:** GK: {len(gk_players)}, DEF: {len(def_players)}, MID: {len(mid_players)}, FWD: {len(fwd_players)}")
    st.write(f"**Total Starting XI:** {len(starting_xi)} players")
    
    # Show all starting XI players
    st.write("**Starting XI Players:**")
    for i, player in enumerate(starting_xi):
        captain_icon = "⭐" if player.is_captain else "🅥" if player.is_vice_captain else ""
        st.write(f"{i+1}. {player.name} {captain_icon} ({player.position}) - {player.team} - £{player.cost}m - {player.predicted_points:.1f}pts")

    # Display formation using Streamlit components
    st.markdown("### 🏟️ Pitch Formation")
    
    # Goalkeeper row
    if gk_players:
        st.markdown("**Goalkeeper:**")
        cols = st.columns(len(gk_players))
        for i, player in enumerate(gk_players):
            with cols[i]:
                captain_icon = "⭐" if player.is_captain else "🅥" if player.is_vice_captain else ""
                st.info(f"""
                **{player.name}** {captain_icon}
                
                {player.position} | {player.team}
                £{player.cost}m | {player.predicted_points:.1f}pts
                """)
    
    # Defenders row
    if def_players:
        st.markdown("**Defenders:**")
        cols = st.columns(len(def_players))
        for i, player in enumerate(def_players):
            with cols[i]:
                captain_icon = "⭐" if player.is_captain else "🅥" if player.is_vice_captain else ""
                st.info(f"""
                **{player.name}** {captain_icon}
                
                {player.position} | {player.team}
                £{player.cost}m | {player.predicted_points:.1f}pts
                """)
    
    # Midfielders row
    if mid_players:
        st.markdown("**Midfielders:**")
        cols = st.columns(len(mid_players))
        for i, player in enumerate(mid_players):
            with cols[i]:
                captain_icon = "⭐" if player.is_captain else "🅥" if player.is_vice_captain else ""
                st.info(f"""
                **{player.name}** {captain_icon}
                
                {player.position} | {player.team}
                £{player.cost}m | {player.predicted_points:.1f}pts
                """)
    
    # Forwards row
    if fwd_players:
        st.markdown("**Forwards:**")
        cols = st.columns(len(fwd_players))
        for i, player in enumerate(fwd_players):
            with cols[i]:
                captain_icon = "⭐" if player.is_captain else "🅥" if player.is_vice_captain else ""
                st.info(f"""
                **{player.name}** {captain_icon}
                
                {player.position} | {player.team}
                £{player.cost}m | {player.predicted_points:.1f}pts
                """)


def show_starting_xi_table(squad: OptimizedSquad):
    """Display starting XI in a sortable table."""
    starting_xi_data = []
    for player in squad.starting_xi:
        captain_icon = "⭐" if player.is_captain else "🅥" if player.is_vice_captain else ""
        starting_xi_data.append({
            "Name": f"{player.name} {captain_icon}",
            "Position": player.position,
            "Team": player.team,
            "Cost (£m)": player.cost,
            "Predicted Points": player.predicted_points
        })
    
    if starting_xi_data:
        st.dataframe(
            starting_xi_data,
            width='stretch',
            hide_index=True
        )


def show_bench_order(squad: OptimizedSquad):
    """Display bench order."""
    bench_data = []
    for player in squad.bench:
        bench_data.append({
            "Position": f"{player.bench_position}.",
            "Name": player.name,
            "Position": player.position,
            "Team": player.team,
            "Cost (£m)": player.cost,
            "Predicted Points": f"{player.predicted_points:.1f}"
        })
    
    if bench_data:
        st.dataframe(
            bench_data,
            width='stretch',
            hide_index=True
        )

    # Show bench summary
    bench_summary = " | ".join([f"{p.name} ({p.bench_position})" for p in squad.bench])
    st.info(f"**Bench Order:** {bench_summary}")


def show_team_breakdown(squad: OptimizedSquad):
    """Display team breakdown."""
    team_breakdown = squad.get_team_breakdown()
    
    # Create bar chart
    fig = px.bar(
        x=list(team_breakdown.keys()),
        y=list(team_breakdown.values()),
        title="Players per Team",
        labels={"x": "Team", "y": "Number of Players"}
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, width='stretch')

    # Show as table
    team_data = [{"Team": team, "Players": count} for team, count in team_breakdown.items()]
    st.dataframe(team_data, width='stretch', hide_index=True)


def show_captaincy_analysis_page(optimizer: SquadOptimizer, raw_data: Dict):
    """Display the captaincy analysis page."""
    st.header("🎯 Captaincy Analysis")
    
    # Get current gameweek
    current_gw = get_current_gameweek(raw_data)
    next_gw = current_gw + 1
    
    st.info(f"**Top 5 players with highest potential for Gameweek {next_gw}**")

    # Get all players and sort by predicted points
    all_players = optimizer.players
    top_captaincy_players = sorted(all_players, key=lambda p: p.predicted_points, reverse=True)[:5]
    
    # Create captaincy analysis table
    captaincy_data = []
    for i, player in enumerate(top_captaincy_players, 1):
        # Calculate captain points (double)
        captain_points = player.predicted_points * 2
        
        # Get fixture difficulty if available
        fixture_difficulty = "N/A"
        if optimizer.fixtures:
            for fixture in optimizer.fixtures:
                if fixture.gameweek == next_gw:
                    if fixture.home_team == player.team:
                        fixture_difficulty = f"{fixture.home_difficulty}/5"
                        break
                    elif fixture.away_team == player.team:
                        fixture_difficulty = f"{fixture.away_difficulty}/5"
                        break
        
        captaincy_data.append({
            "Rank": f"#{i}",
            "Player": player.name,
            "Position": player.position,
            "Team": player.team,
            "Cost": f"£{player.cost}m",
            "Predicted Points": f"{player.predicted_points:.1f}",
            "Captain Points": f"{captain_points:.1f}",
            "Form": f"{player.form:.1f}",
            "Fixture Difficulty": fixture_difficulty,
            "Minutes Played": player.minutes_played
        })
    
    # Display captaincy analysis table
    st.dataframe(
        captaincy_data,
        width='stretch',
        hide_index=True,
        column_config={
            "Rank": st.column_config.TextColumn("Rank", width="small"),
            "Player": st.column_config.TextColumn("Player", width="medium"),
            "Position": st.column_config.TextColumn("Pos", width="small"),
            "Team": st.column_config.TextColumn("Team", width="small"),
            "Cost": st.column_config.TextColumn("Cost", width="small"),
            "Predicted Points": st.column_config.TextColumn("Predicted", width="small"),
            "Captain Points": st.column_config.TextColumn("Captain", width="small"),
            "Form": st.column_config.TextColumn("Form", width="small"),
            "Fixture Difficulty": st.column_config.TextColumn("Fixture", width="small"),
            "Minutes Played": st.column_config.NumberColumn("Minutes", width="small")
        }
    )
    
    # Add insights about the top captaincy choices
    st.subheader("💡 Captaincy Insights")
    
    if captaincy_data:
        top_player = captaincy_data[0]
        st.success(f"**🎯 Top Pick:** {top_player['Player']} ({top_player['Team']}) - {top_player['Captain Points']} captain points")
        
        # Show fixture analysis
        if top_player['Fixture Difficulty'] != "N/A":
            difficulty = int(top_player['Fixture Difficulty'].split('/')[0])
            if difficulty <= 2:
                st.info(f"✅ **Favorable Fixture** - Difficulty {difficulty}/5")
            elif difficulty >= 4:
                st.warning(f"⚠️ **Tough Fixture** - Difficulty {difficulty}/5")
            else:
                st.info(f"📊 **Average Fixture** - Difficulty {difficulty}/5")
        
        # Show form analysis
        form = float(top_player['Form'])
        if form >= 7.0:
            st.success(f"🔥 **Excellent Form** - {form:.1f} average")
        elif form >= 5.0:
            st.info(f"📈 **Good Form** - {form:.1f} average")
        else:
            st.warning(f"📉 **Poor Form** - {form:.1f} average")
        
        # Show minutes analysis
        minutes = top_player['Minutes Played']
        if minutes >= 270:  # 3 full games
            st.success(f"⏱️ **Regular Starter** - {minutes} minutes played")
        elif minutes >= 90:  # At least 1 full game
            st.info(f"🔄 **Rotation Risk** - {minutes} minutes played")
        else:
            st.warning(f"❌ **Limited Minutes** - {minutes} minutes played")
    
    # 3.2 Visualization
    st.subheader("📊 Captaincy Points Visualization")
    
    # Create bar chart for top 5 players
    names = [player['Player'] for player in captaincy_data]
    captain_points = [float(player['Captain Points']) for player in captaincy_data]
    predicted_points = [float(player['Predicted Points']) for player in captaincy_data]
    
    fig = go.Figure()
    
    # Add bars for captain points (doubled)
    fig.add_trace(go.Bar(
        name='Captain Points (2x)',
        x=names,
        y=captain_points,
        marker_color='#FFD700',  # Gold
        text=[f"{p:.1f}" for p in captain_points],
        textposition='auto',
    ))
    
    # Add bars for predicted points (normal)
    fig.add_trace(go.Bar(
        name='Predicted Points (1x)',
        x=names,
        y=predicted_points,
        marker_color='#1f77b4',  # Blue
        text=[f"{p:.1f}" for p in predicted_points],
        textposition='auto',
    ))
    
    fig.update_layout(
        title=f"Top 5 Captaincy Options for Gameweek {next_gw}",
        xaxis_title="Player",
        yaxis_title="Points",
        barmode='group',
        xaxis_tickangle=-45,
        height=500
    )
    
    st.plotly_chart(fig, width='stretch')
    
    # 3.3 Squad Captaincy Simulation (Original functionality)
    st.subheader("🏆 Squad Captaincy Simulation")
    st.markdown("Team predicted points if each squad player is captain:")
    
    # Get the optimal squad
    with st.spinner("Loading squad for captaincy simulation..."):
        squad = optimizer.optimize_squad()
    
    # Get captaincy simulation data
    simulation_data = optimizer.get_captaincy_simulation_data(squad)
    
    # Sort by points (descending)
    sorted_data = sorted(simulation_data.items(), key=lambda x: x[1], reverse=True)
    
    # Create simulation table
    simulation_table_data = []
    base_points = squad.total_predicted_points
    
    for name, points in sorted_data:
        captain_icon = "⭐" if name == squad.captain.name else "🅥" if name == squad.vice_captain.name else ""
        difference = points - base_points
        simulation_table_data.append({
            "Player (C)": f"{name} {captain_icon}",
            "Team": next((p.team for p in squad.players if p.name == name), ""),
            "Predicted Team Points": f"{points:.1f}",
            "Difference vs Best": f"{difference:+.1f}"
        })
    
    st.dataframe(
        simulation_table_data,
        width='stretch',
        hide_index=True
    )
    
    # Squad captaincy visualization
    st.subheader("📈 Squad Captaincy Impact")
    
    squad_names = [item[0] for item in sorted_data]
    squad_points = [item[1] for item in sorted_data]
    
    # Color coding: captain in gold, vice-captain in silver, others in blue
    colors = []
    for name in squad_names:
        if name == squad.captain.name:
            colors.append("#FFD700")  # Gold
        elif name == squad.vice_captain.name:
            colors.append("#C0C0C0")  # Silver
        else:
            colors.append("#1f77b4")  # Blue
    
    fig2 = go.Figure(data=[
        go.Bar(
            x=squad_names,
            y=squad_points,
            marker_color=colors,
            text=[f"{p:.1f}" for p in squad_points],
            textposition='auto',
        )
    ])
    
    fig2.update_layout(
        title="Captaincy Impact on Team Points",
        xaxis_title="Player",
        yaxis_title="Total Team Predicted Points",
        xaxis_tickangle=-45,
        height=500
    )
    
    st.plotly_chart(fig2, width='stretch')
    
    # 3.4 Recommendation
    st.subheader("🎯 Recommendation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success(f"**Best Captain:** {squad.captain.name} ⭐")
        st.metric("Captain Points", f"{squad.captain.predicted_points * 2:.1f}")
    
    with col2:
        st.info(f"**Best Vice-Captain:** {squad.vice_captain.name} 🅥")
        st.metric("Vice-Captain Points", f"{squad.vice_captain.predicted_points:.1f}")


def show_bench_autosubs_page(optimizer: SquadOptimizer):
    """Display bench and autosubs analysis."""
    st.header("🪑 Bench & Autosubs")

    with st.spinner("Loading squad for bench analysis..."):
        squad = optimizer.optimize_squad()

    # 4.1 Bench Order Optimization
    st.subheader("4.1 Bench Order Optimization")
    
    bench_data = []
    for player in squad.bench:
        bench_data.append({
            "Position": f"{player.bench_position}.",
            "Name": player.name,
            "Position": player.position,
            "Team": player.team,
            "Cost (£m)": player.cost,
            "Predicted Points": f"{player.predicted_points:.1f}"
        })
    
    if bench_data:
        st.dataframe(
            bench_data,
            width='stretch',
            hide_index=True
        )

    # Show bench summary
    bench_summary = " | ".join([f"{p.name} ({p.bench_position})" for p in squad.bench])
    st.info(f"**Current Bench Order:** {bench_summary}")

    # 4.2 Autosub Simulation (Optional)
    st.subheader("4.2 Autosub Simulation")
    st.info("💡 This feature simulates what happens if a starting player doesn't play (0 minutes).")
    
    # Simple autosub simulation
    starting_xi = squad.starting_xi
    bench_players = squad.bench
    
    # Find the first outfield bench player (not GK)
    first_bench_outfield = next((p for p in bench_players if p.position != "GK"), None)
    
    if first_bench_outfield:
        # Simulate if the lowest predicted points starter doesn't play
        lowest_starter = min(starting_xi, key=lambda p: p.predicted_points)
        
        st.markdown(f"""
        **Scenario:** If {lowest_starter.name} doesn't play (0 minutes)
        
        **Autosub:** {first_bench_outfield.name} would come in
        
        **Points Impact:** 
        - Without autosub: {squad.total_predicted_points:.1f} points
        - With autosub: {squad.total_predicted_points - lowest_starter.predicted_points + first_bench_outfield.predicted_points:.1f} points
        - Difference: {first_bench_outfield.predicted_points - lowest_starter.predicted_points:+.1f} points
        """)


def show_player_comparisons_page(players: List[Player]):
    """Display player comparison tool."""
    st.header("🔍 Player Comparisons")

    st.subheader("Select 2-3 players to compare:")
    
    # Player selection
    player_names = [p.name for p in players]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        player1 = st.selectbox("Player 1:", ["Select player..."] + player_names)
    
    with col2:
        player2 = st.selectbox("Player 2:", ["Select player..."] + player_names)
    
    with col3:
        player3 = st.selectbox("Player 3 (optional):", ["Select player..."] + player_names)

    # Get selected players
    selected_players = []
    for name in [player1, player2, player3]:
        if name != "Select player...":
            player = next((p for p in players if p.name == name), None)
            if player:
                selected_players.append(player)

    if len(selected_players) >= 2:
        st.subheader("Comparison Results")
        
        # Create comparison table
        comparison_data = []
        for player in selected_players:
            comparison_data.append({
                "Name": player.name,
                "Team": player.team,
                "Position": player.position,
                "Cost (£m)": player.cost,
                "Predicted Points": f"{player.predicted_points:.1f}",
                "Value (pts/£m)": f"{player.predicted_points / player.cost:.2f}"
            })
        
        st.dataframe(
            comparison_data,
            width='stretch',
            hide_index=True
        )

        # Value comparison chart
        st.subheader("Value Comparison (Points per £m)")
        
        fig = px.bar(
            x=[p.name for p in selected_players],
            y=[p.predicted_points / p.cost for p in selected_players],
            title="Value for Money Comparison",
            labels={"x": "Player", "y": "Points per £m"}
        )
        st.plotly_chart(fig, width='stretch')


def show_gameweek_explorer_page(optimizer: SquadOptimizer, players: List[Player], events: List[GameweekEvent], available_gws: List[int], next_gw: Optional[int]):
    """Display the enhanced gameweek explorer page."""
    st.header("📊 Gameweek Explorer")
    
    # Handle session state for tab selection
    if "gw_tab" not in st.session_state:
        st.session_state.gw_tab = "Gameweek Insights"
    
    # Create tabs for different sections
    tab1, tab2 = st.tabs(["🟢 Gameweek Insights (Past & Current)", "🔮 Next Gameweek Squad (Predictive)"])
    
    with tab1:
        show_gameweek_insights_tab(optimizer, players, events, available_gws)
    
    with tab2:
        show_next_gameweek_tab(optimizer, next_gw, players)


def show_gameweek_insights_tab(optimizer: SquadOptimizer, players: List[Player], events: List[GameweekEvent], available_gws: List[int]):
    """Show gameweek insights for past and current gameweeks."""
    st.subheader("🟢 Gameweek Insights (Past & Current)")
    
    # 1. Gameweek Selector
    st.markdown("**1. Gameweek Selector**")
    
    if not available_gws:
        st.warning("No completed gameweeks available for analysis.")
        return
    
    # Initialize selected gameweek from session state if available
    if "selected_gw" in st.session_state and st.session_state.selected_gw in available_gws:
        default_index = available_gws.index(st.session_state.selected_gw)
    else:
        default_index = 0
    
    selected_gw = st.selectbox(
        "Select Gameweek:",
        available_gws,
        index=default_index,
        format_func=lambda x: f"Gameweek {x}"
    )
    
    # Update session state
    st.session_state.selected_gw = selected_gw
    
    # Get event info
    event_info = next((e for e in events if e.id == selected_gw), None)
    if event_info:
        st.info(f"**{event_info.name}** - Average Score: {event_info.average_entry_score or 'N/A'}")
    
    # 2. Top Players Table (Actual)
    st.markdown("**2. Top Players Table (Actual)**")
    
    try:
        with st.spinner(f"Fetching actual results for Gameweek {selected_gw}..."):
            live_data = fetch_live_event_data(selected_gw)
            
            # Debug: Check if we have data for this gameweek
            if not live_data:
                st.error(f"No live data available for Gameweek {selected_gw}. This gameweek may not have completed yet.")
                return
                
            player_performances = parse_player_performances(live_data, players)
            

        
        if player_performances:
            # Sort by actual points and get top 15 performers
            top_performers = sorted(player_performances, key=lambda p: p.actual_points, reverse=True)[:15]
            

            
            top_players_data = []
            for i, player in enumerate(top_performers, 1):
                difference = player.actual_points - player.predicted_points
                top_players_data.append({
                    "Rank": i,
                    "Player": player.name,
                    "Team": player.team,
                    "Position": player.position,
                    "Actual Points": player.actual_points,
                    "Predicted Points": f"{player.predicted_points:.1f}",
                    "Difference": f"{difference:+.1f}"
                })
            
            # Create dataframe and sort by actual points (descending)
            import pandas as pd
            df = pd.DataFrame(top_players_data)
            df = df.sort_values('Actual Points', ascending=False)
            
            # Display the table with better formatting
            st.dataframe(
                df,
                width='stretch',
                hide_index=True,
                column_config={
                    "Rank": st.column_config.NumberColumn("Rank", format="%d"),
                    "Player": st.column_config.TextColumn("Player", width="medium"),
                    "Team": st.column_config.TextColumn("Team", width="small"),
                    "Position": st.column_config.TextColumn("Pos", width="small"),
                    "Actual Points": st.column_config.NumberColumn("Actual Points", format="%d"),
                    "Predicted Points": st.column_config.TextColumn("Predicted", width="small"),
                    "Difference": st.column_config.TextColumn("Diff", width="small")
                }
            )
            
            # Add summary statistics
            st.markdown("**📊 Summary Statistics**")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_actual = df['Actual Points'].mean()
                st.metric("Avg Actual Points", f"{avg_actual:.1f}")
            
            with col2:
                avg_predicted = df['Predicted Points'].astype(float).mean()
                st.metric("Avg Predicted Points", f"{avg_predicted:.1f}")
            
            with col3:
                avg_diff = df['Difference'].astype(float).mean()
                st.metric("Avg Difference", f"{avg_diff:+.1f}")
            
            with col4:
                overperformers = len(df[df['Difference'].astype(float) > 0])
                st.metric("Overperformers", f"{overperformers}/15")
            
            # 3. Squad Accuracy Analysis
            st.markdown("**3. Squad Accuracy (for optimizer's best 15 that GW)**")
            
            # Optimize squad for this gameweek
            with st.spinner(f"Optimizing squad for Gameweek {selected_gw}..."):
                squad = optimizer.optimize_squad(selected_gw)
            
            # Analyze squad performance
            performance = optimizer.analyze_squad_performance(squad, player_performances)
            performance.gameweek = selected_gw
            
            # Show accuracy metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Predicted Total", f"{performance.predicted_total:.1f}")
            with col2:
                st.metric("Actual Total", f"{performance.actual_total:.1f}")
            with col3:
                st.metric("Difference", f"{performance.difference:+.1f}")
            with col4:
                st.metric("Accuracy %", f"{performance.accuracy:.1f}%")
            
            # Show detailed metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("MAE", f"{performance.mae:.2f}")
            with col2:
                st.metric("RMSE", f"{performance.rmse:.2f}")
            with col3:
                st.metric("Players Analyzed", len(performance.player_performances))
            
            # 4. Charts
            st.markdown("**4. Charts**")
            
            # Predicted vs Actual per player
            if performance.player_performances:
                player_names = [p.name for p in performance.player_performances]
                predicted_points = [p.predicted_points for p in performance.player_performances]
                actual_points = [p.actual_points for p in performance.player_performances]
                
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    name='Predicted',
                    x=player_names,
                    y=predicted_points,
                    marker_color='lightblue'
                ))
                
                fig.add_trace(go.Bar(
                    name='Actual',
                    x=player_names,
                    y=actual_points,
                    marker_color='orange'
                ))
                
                fig.update_layout(
                    title=f"Predicted vs Actual Points - Gameweek {selected_gw}",
                    xaxis_title="Player",
                    yaxis_title="Points",
                    barmode='group',
                    xaxis_tickangle=-45,
                    height=500
                )
                
                st.plotly_chart(fig, width='stretch')
            
            # 5. Insights Section
            st.markdown("**5. Insights Section**")
            
            # Position insights
            position_insights = get_position_insights(performance.player_performances)
            
            for pos, data in position_insights.items():
                if data["count"] > 0:
                    avg_diff = data["avg_difference"]
                    if avg_diff > 0:
                        st.success(f"✅ {pos} delivered above average (+{avg_diff:.1f} pts vs predicted)")
                    elif avg_diff < 0:
                        st.error(f"⚠️ {pos} underperformed ({avg_diff:.1f} pts vs predicted)")
                    else:
                        st.info(f"📊 {pos} performed as expected")
            
            # Over/underperformers
            if performance.overperformers:
                st.success("**Overperformers:**")
                for player in performance.overperformers[:5]:
                    diff = player.actual_points - player.predicted_points
                    st.write(f"✅ {player.name} exceeded prediction by +{diff:.1f}")
            
            if performance.underperformers:
                st.error("**Underperformers:**")
                for player in performance.underperformers[:5]:
                    diff = player.actual_points - player.predicted_points
                    st.write(f"⚠️ {player.name} underperformed by {diff:.1f} vs expected")
            
            # Detailed insights
            st.markdown("**🔍 Detailed Insights**")
            
            # Position analysis
            position_stats = {}
            for player in performance.player_performances:
                pos = player.position
                if pos not in position_stats:
                    position_stats[pos] = {'actual': [], 'predicted': [], 'count': 0}
                position_stats[pos]['actual'].append(player.actual_points)
                position_stats[pos]['predicted'].append(player.predicted_points)
                position_stats[pos]['count'] += 1
            
            if position_stats:
                st.markdown("**Position Performance Analysis:**")
                pos_cols = st.columns(len(position_stats))
                for i, (pos, stats) in enumerate(position_stats.items()):
                    with pos_cols[i]:
                        avg_actual = sum(stats['actual']) / len(stats['actual'])
                        avg_pred = sum(stats['predicted']) / len(stats['predicted'])
                        diff = avg_actual - avg_pred
                        
                        if diff > 0:
                            st.success(f"**{pos}:** +{diff:.1f} pts")
                        else:
                            st.error(f"**{pos}:** {diff:.1f} pts")
                        st.caption(f"Avg: {avg_actual:.1f} vs {avg_pred:.1f}")
            
            # Top overperformers and underperformers
            sorted_players = sorted(performance.player_performances, 
                                  key=lambda p: p.actual_points - p.predicted_points, reverse=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**🏆 Top Overperformers:**")
                for i, player in enumerate(sorted_players[:3], 1):
                    diff = player.actual_points - player.predicted_points
                    if diff > 0:
                        st.write(f"{i}. **{player.name}** ({player.team}): +{diff:.1f} pts")
            
            with col2:
                st.markdown("**📉 Top Underperformers:**")
                for i, player in enumerate(sorted_players[-3:], 1):
                    diff = player.actual_points - player.predicted_points
                    if diff < 0:
                        st.write(f"{i}. **{player.name}** ({player.team}): {diff:.1f} pts")
            
            # Overall assessment
            if performance.difference > 0:
                st.success(f"🎉 **Great performance!** The squad outperformed predictions by {performance.difference:.1f} points.")
            elif performance.difference < 0:
                st.warning(f"⚠️ **Below expectations.** The squad underperformed by {abs(performance.difference):.1f} points.")
            else:
                st.info("📊 **On target.** The squad performed exactly as predicted.")
                
        else:
            st.warning("No performance data available for this gameweek.")
            
    except Exception as e:
        st.error(f"Error fetching gameweek data: {str(e)}")
        st.info("This gameweek may not have completed data yet.")


def show_next_gameweek_tab(optimizer: SquadOptimizer, next_gw: Optional[int], players: List[Player]):
    """Show comprehensive next gameweek predictive squad analysis."""
    
    # ============================================================================
    # HEADER SECTION
    # ============================================================================
    st.markdown("""
    <div style="text-align: center; padding: 20px; background: linear-gradient(90deg, #1f77b4, #ff7f0e); border-radius: 10px; margin-bottom: 20px;">
        <h1 style="color: white; margin: 0;">🔮 Next Gameweek Squad (Predictive Analytics)</h1>
        <p style="color: white; margin: 5px 0 0 0; font-size: 18px;">Advanced AI-powered squad optimization for Gameweek {}</p>
    </div>
    """.format(next_gw), unsafe_allow_html=True)
    
    if not next_gw:
        st.error("❌ **No next gameweek available.** Please check FPL API status.")
        return
    
    # ============================================================================
    # GAMEWEEK OVERVIEW
    # ============================================================================
    st.markdown("### 📅 Gameweek Overview")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"**🎯 Target Gameweek:** {next_gw}")
    with col2:
        st.info(f"**📊 Total Players Available:** {len(players)}")
    with col3:
        st.info(f"**🤖 ML Models Ready:** {'✅ Yes' if optimizer.predictive_model.is_trained else '❌ No'}")
    
    # ============================================================================
    # OPTIMIZATION CONTROLS
    # ============================================================================
    st.markdown("### ⚙️ Optimization Controls")
    
    with st.expander("🔧 Advanced Settings", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            use_form = st.checkbox("📈 Use Form (last 3-5 matches)", value=True, help="Incorporate recent form into predictions")
        with col2:
            use_fixtures = st.checkbox("🏟️ Use Fixture Difficulty", value=True, help="Consider opponent strength and home/away")
        with col3:
            min_minutes = st.slider("⏱️ Min Minutes Threshold", 0, 2000, 90, 10, 
                                  help="Minimum minutes played to consider player (reduces rotation risk)")
        with col4:
            models_trained = optimizer.predictive_model.is_trained and len(optimizer.predictive_model.models) > 0
            use_ml = st.checkbox("🤖 Use ML Predictions", value=models_trained, disabled=not models_trained,
                               help="Enable machine learning enhanced predictions")
            if not models_trained:
                st.caption("⚠️ Train models first in ML Models page")
    
    # ============================================================================
    # SQUAD GENERATION
    # ============================================================================
    st.markdown("### 🚀 Squad Generation")
    
    with st.spinner(f"🎯 Creating optimized squad for Gameweek {next_gw}..."):
        try:
            # Check if we have historical data and ML models
            has_historical_data = 'historical_performances' in st.session_state and st.session_state.historical_performances
            models_trained = optimizer.predictive_model.is_trained and len(optimizer.predictive_model.models) > 0
            
            if use_ml and models_trained and has_historical_data:
                # Use ML-enhanced optimization
                st.success("🤖 **ML-Enhanced Optimization Active**")
                squad = optimizer.optimize_with_ml(next_gw, st.session_state.historical_performances)
                predictive_squad = optimizer.create_predictive_squad(next_gw)
            else:
                # Use comprehensive analysis optimization (enhanced)
                if use_ml and not models_trained:
                    st.warning("⚠️ **ML models not trained.** Using comprehensive analysis optimization.")
                elif use_ml and not has_historical_data:
                    st.warning("⚠️ **No historical data available.** Using comprehensive analysis optimization.")
                    st.info("💡 **Tip:** Collect historical data in the ML Models page to enable ML predictions.")
                else:
                    st.success("📊 **Using Comprehensive Analysis Optimization**")
                
                # Use the enhanced optimize_squad method with comprehensive analysis
                squad = optimizer.optimize_squad(gameweek=next_gw, use_form=use_form, use_fixtures=use_fixtures)
                predictive_squad = optimizer.create_predictive_squad(next_gw)
        except Exception as e:
            st.error(f"❌ **Error creating predictive squad:** {str(e)}")
            st.info("🔄 **Falling back to comprehensive analysis optimization...**")
            squad = optimizer.optimize_squad(gameweek=next_gw, use_form=use_form, use_fixtures=use_fixtures)
            predictive_squad = optimizer.create_predictive_squad(next_gw)
    
    # ============================================================================
    # COMPREHENSIVE ANALYSIS STATUS
    # ============================================================================
    st.markdown("### 🔬 Comprehensive Analysis Status")
    
    # Show that comprehensive analysis is being used
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.success("✅ **Comprehensive Analysis Active**")
        st.info("Using ICT Index, xG/xA, availability, team strength, and position-specific metrics")
    with col2:
        st.info("📊 **Enhanced Player Selection**")
        st.info("Players selected based on comprehensive ability metrics, not just recent form")
    with col3:
        st.info("🎯 **Advanced Captaincy**")
        st.info("Captain selection considers all available metrics and fixture context")
    with col4:
        st.info("⚽ **Position-Specific Analysis**")
        st.info("Different metrics for GK, DEF, MID, FWD positions")
    
    # ============================================================================
    # SQUAD SUMMARY METRICS
    # ============================================================================
    st.markdown("### 📊 Squad Summary")
    
    # Calculate enhanced squad metrics
    squad_enhanced_score = sum(optimizer.analyze_player_selection_factors(p.player, next_gw)['total_enhanced_score'] for p in squad.players)
    squad_ict_total = sum(p.player.ict_index for p in squad.players)
    squad_xg_total = sum(p.player.expected_goals for p in squad.players)
    squad_xa_total = sum(p.player.expected_assists for p in squad.players)
    
    # Create a more visually appealing metrics display
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            label="🎯 Enhanced Squad Score",
            value=f"{squad_enhanced_score:.1f}",
            delta=None,
            help="Total enhanced score using comprehensive analysis"
        )
    with col2:
        st.metric(
            label="💰 Team Value",
            value=f"£{squad.total_cost:.1f}m",
            delta=None,
            help="Total cost of the selected squad"
        )
    with col3:
        st.metric(
            label="📈 ICT Index Total",
            value=f"{squad_ict_total:.0f}",
            delta=None,
            help="Total ICT Index of all squad players"
        )
    with col4:
        st.metric(
            label="⚽ Expected Goals/Assists",
            value=f"{squad_xg_total:.1f}/{squad_xa_total:.1f}",
            delta=None,
            help="Total expected goals and assists for the squad"
        )
    
    # ============================================================================
    # ML MODEL INFORMATION
    # ============================================================================
    if use_ml and optimizer.ml_enhanced:
        st.markdown("### 🤖 Machine Learning Model Information")
        
        try:
            model_summary = optimizer.get_ml_model_performance()
            if model_summary and 'models' in model_summary:
                # Create a more detailed ML model display
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.info(f"**📊 Models Trained:** {model_summary.get('total_models', 0)}")
                with col2:
                    best_model = model_summary.get('best_model', 'N/A')
                    st.success(f"**🏆 Best Model:** {best_model}")
                with col3:
                    accuracy = model_summary.get('accuracy_rating', 'N/A')
                    st.warning(f"**📈 Accuracy:** {accuracy}")
                
                # Show detailed model information
                if 'prediction_approach' in model_summary:
                    st.info(f"**🔬 Prediction Approach:** {model_summary['prediction_approach']}")
                if 'features_used' in model_summary:
                    st.info(f"**🎯 Features Used:** {model_summary['features_used']}")
        except Exception as e:
            st.warning(f"⚠️ **Could not load ML model information:** {str(e)}")
    
    # ============================================================================
    # CAPTAIN & VICE-CAPTAIN SELECTION
    # ============================================================================
    st.markdown("### ⭐ Captain & Vice-Captain Selection")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(90deg, #28a745, #20c997); padding: 15px; border-radius: 10px; text-align: center;">
            <h3 style="color: white; margin: 0;">⭐ Captain</h3>
            <h2 style="color: white; margin: 10px 0;">{}</h2>
            <p style="color: white; margin: 0;">Predicted Points: <strong>{:.1f}</strong></p>
        </div>
        """.format(squad.captain.name, predictive_squad.captain_points), unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(90deg, #17a2b8, #6f42c1); padding: 15px; border-radius: 10px; text-align: center;">
            <h3 style="color: white; margin: 0;">🅥 Vice-Captain</h3>
            <h2 style="color: white; margin: 10px 0;">{}</h2>
            <p style="color: white; margin: 0;">Predicted Points: <strong>{:.1f}</strong></p>
        </div>
        """.format(squad.vice_captain.name, predictive_squad.vice_captain_points), unsafe_allow_html=True)
    
    # ============================================================================
    # COMPREHENSIVE PLAYER ANALYSIS
    # ============================================================================
    st.markdown("### 🎯 Comprehensive Player Analysis")
    
    # Use the enhanced comprehensive player rankings instead of just predicted points
    st.markdown("**📊 Top Players by Comprehensive Analysis (All Metrics)**")
    
    # Get comprehensive player rankings using our enhanced system
    comprehensive_rankings = optimizer.get_comprehensive_player_rankings(gameweek=next_gw, limit=15)
    
    # Create comprehensive analysis table
    comprehensive_data = []
    for ranking in comprehensive_rankings:
        player = ranking['player']
        analysis = ranking['analysis_breakdown']
        
        # Get fixture information
        fixture_difficulty = "N/A"
        opponent = "N/A"
        home_away = "N/A"
        if optimizer.fixtures:
            for fixture in optimizer.fixtures:
                if fixture.gameweek == next_gw:
                    if fixture.home_team == player.team:
                        fixture_difficulty = f"{fixture.home_difficulty}/5"
                        opponent = fixture.away_team
                        home_away = "Home"
                        break
                    elif fixture.away_team == player.team:
                        fixture_difficulty = f"{fixture.away_difficulty}/5"
                        opponent = fixture.home_team
                        home_away = "Away"
                        break
        
        comprehensive_data.append({
            "Rank": f"#{ranking['rank']}",
            "Player": player.name,
            "Position": player.position,
            "Team": player.team,
            "Cost": f"£{player.cost}m",
            "Enhanced Score": f"{ranking['total_enhanced_score']:.2f}",
            "Base Points": f"{analysis['base_predicted_points']:.1f}",
            "ICT Index": f"{player.ict_index:.0f}",
            "Expected Goals": f"{player.expected_goals:.2f}",
            "Expected Assists": f"{player.expected_assists:.2f}",
            "Form": f"{player.form:.1f}",
            "Minutes": player.minutes_played,
            "Status": player.status,
            "Opponent": opponent,
            "Fixture": fixture_difficulty,
            "Availability": f"{player.chance_of_playing_next_round}%" if player.chance_of_playing_next_round else "N/A"
        })
    
    # Display comprehensive analysis table
    st.dataframe(
        comprehensive_data,
        width='stretch',
        hide_index=True,
        column_config={
            "Rank": st.column_config.TextColumn("Rank", width="small"),
            "Player": st.column_config.TextColumn("Player", width="medium"),
            "Position": st.column_config.TextColumn("Pos", width="small"),
            "Team": st.column_config.TextColumn("Team", width="small"),
            "Cost": st.column_config.TextColumn("Cost", width="small"),
            "Enhanced Score": st.column_config.TextColumn("Enhanced", width="small"),
            "Base Points": st.column_config.TextColumn("Base", width="small"),
            "ICT Index": st.column_config.TextColumn("ICT", width="small"),
            "Expected Goals": st.column_config.TextColumn("xG", width="small"),
            "Expected Assists": st.column_config.TextColumn("xA", width="small"),
            "Form": st.column_config.TextColumn("Form", width="small"),
            "Minutes": st.column_config.NumberColumn("Minutes", width="small"),
            "Status": st.column_config.TextColumn("Status", width="small"),
            "Opponent": st.column_config.TextColumn("Opponent", width="small"),
            "Fixture": st.column_config.TextColumn("Fixture", width="small"),
            "Availability": st.column_config.TextColumn("Availability", width="small")
        }
    )
    
    # ============================================================================
    # COMPREHENSIVE CAPTAINCY ANALYSIS
    # ============================================================================
    st.markdown("### 🎯 Comprehensive Captaincy Analysis")
    
    # Use comprehensive rankings for captaincy analysis
    top_captaincy_players = [ranking['player'] for ranking in comprehensive_rankings[:10]]
    
    # Enhanced captaincy data with comprehensive analysis
    captaincy_data = []
    for i, player in enumerate(top_captaincy_players, 1):
        captain_points = player.predicted_points * 2
        
        # Get comprehensive analysis for this player
        player_analysis = optimizer.analyze_player_selection_factors(player, next_gw)
        
        # Get fixture difficulty and opponent
        fixture_difficulty = "N/A"
        opponent = "N/A"
        home_away = "N/A"
        if optimizer.fixtures:
            for fixture in optimizer.fixtures:
                if fixture.gameweek == next_gw:
                    if fixture.home_team == player.team:
                        fixture_difficulty = f"{fixture.home_difficulty}/5"
                        opponent = fixture.away_team
                        home_away = "Home"
                        break
                    elif fixture.away_team == player.team:
                        fixture_difficulty = f"{fixture.away_difficulty}/5"
                        opponent = fixture.home_team
                        home_away = "Away"
                        break
        
        # Calculate enhanced captaincy score using comprehensive analysis
        enhanced_captaincy_score = player_analysis['total_enhanced_score'] * 2  # Captain gets double points
        
        captaincy_data.append({
            "Rank": f"#{i}",
            "Player": player.name,
            "Position": player.position,
            "Team": player.team,
            "Opponent": opponent,
            "H/A": home_away,
            "Cost": f"£{player.cost}m",
            "Enhanced Score": f"{player_analysis['total_enhanced_score']:.2f}",
            "Captain Points": f"{captain_points:.1f}",
            "Enhanced Captain Score": f"{enhanced_captaincy_score:.2f}",
            "ICT Index": f"{player.ict_index:.0f}",
            "Expected Goals": f"{player.expected_goals:.2f}",
            "Expected Assists": f"{player.expected_assists:.2f}",
            "Form": f"{player.form:.1f}",
            "Fixture Difficulty": fixture_difficulty,
            "Minutes Played": player.minutes_played,
            "Availability": f"{player.chance_of_playing_next_round}%" if player.chance_of_playing_next_round else "N/A"
        })
    
    # Display enhanced captaincy analysis table
    st.markdown("**📋 Top 10 Captaincy Candidates (Comprehensive Analysis)**")
    st.dataframe(
        captaincy_data,
        width='stretch',
        hide_index=True,
        column_config={
            "Rank": st.column_config.TextColumn("Rank", width="small"),
            "Player": st.column_config.TextColumn("Player", width="medium"),
            "Position": st.column_config.TextColumn("Pos", width="small"),
            "Team": st.column_config.TextColumn("Team", width="small"),
            "Opponent": st.column_config.TextColumn("Opponent", width="small"),
            "H/A": st.column_config.TextColumn("H/A", width="small"),
            "Cost": st.column_config.TextColumn("Cost", width="small"),
            "Enhanced Score": st.column_config.TextColumn("Enhanced", width="small"),
            "Captain Points": st.column_config.TextColumn("Captain", width="small"),
            "Enhanced Captain Score": st.column_config.TextColumn("Enhanced C", width="small"),
            "ICT Index": st.column_config.TextColumn("ICT", width="small"),
            "Expected Goals": st.column_config.TextColumn("xG", width="small"),
            "Expected Assists": st.column_config.TextColumn("xA", width="small"),
            "Form": st.column_config.TextColumn("Form", width="small"),
            "Fixture Difficulty": st.column_config.TextColumn("Fixture", width="small"),
            "Minutes Played": st.column_config.NumberColumn("Minutes", width="small"),
            "Availability": st.column_config.TextColumn("Availability", width="small")
        }
    )
    
    # ============================================================================
    # DETAILED CAPTAINCY INSIGHTS
    # ============================================================================
    st.markdown("### 💡 Detailed Captaincy Insights")
    
    if captaincy_data:
        # Top 3 recommendations with detailed analysis
        for i in range(min(3, len(captaincy_data))):
            player = captaincy_data[i]
            
            # Create expandable section for each top recommendation
            with st.expander(f"🎯 #{i+1} Recommendation: {player['Player']} ({player['Team']}) - {player['Captain Points']} captain points", expanded=(i==0)):
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**📊 Comprehensive Player Stats**")
                    st.write(f"**Position:** {player['Position']}")
                    st.write(f"**Cost:** {player['Cost']}")
                    st.write(f"**Enhanced Score:** {player['Enhanced Score']}")
                    st.write(f"**ICT Index:** {player['ICT Index']}")
                    st.write(f"**Expected Goals:** {player['Expected Goals']}")
                    st.write(f"**Expected Assists:** {player['Expected Assists']}")
                    st.write(f"**Form:** {player['Form']}")
                    st.write(f"**Minutes:** {player['Minutes Played']}")
                    st.write(f"**Availability:** {player['Availability']}")
                
                with col2:
                    st.markdown("**🏟️ Fixture Analysis**")
                    if player['Opponent'] != "N/A":
                        st.write(f"**Opponent:** {player['Opponent']}")
                        st.write(f"**Home/Away:** {player['H/A']}")
                        st.write(f"**Difficulty:** {player['Fixture Difficulty']}")
                        
                        # Fixture difficulty analysis
                        if player['Fixture Difficulty'] != "N/A":
                            difficulty = int(player['Fixture Difficulty'].split('/')[0])
                            if difficulty <= 2:
                                st.success(f"✅ **Favorable Fixture** - Difficulty {difficulty}/5")
                            elif difficulty >= 4:
                                st.warning(f"⚠️ **Tough Fixture** - Difficulty {difficulty}/5")
                            else:
                                st.info(f"📊 **Average Fixture** - Difficulty {difficulty}/5")
                    else:
                        st.warning("⚠️ Fixture information not available")
                
                with col3:
                    st.markdown("**📈 Comprehensive Performance Analysis**")
                    
                    # Enhanced score analysis
                    enhanced_score = float(player['Enhanced Score'])
                    if enhanced_score >= 8.0:
                        st.success(f"🌟 **Excellent Enhanced Score** - {enhanced_score:.2f}")
                    elif enhanced_score >= 6.0:
                        st.info(f"📈 **Good Enhanced Score** - {enhanced_score:.2f}")
                    else:
                        st.warning(f"📉 **Low Enhanced Score** - {enhanced_score:.2f}")
                    
                    # ICT Index analysis
                    ict_index = float(player['ICT Index'])
                    if ict_index >= 100:
                        st.success(f"🔥 **High ICT Index** - {ict_index:.0f}")
                    elif ict_index >= 50:
                        st.info(f"📊 **Moderate ICT Index** - {ict_index:.0f}")
                    else:
                        st.warning(f"📉 **Low ICT Index** - {ict_index:.0f}")
                    
                    # Expected goals/assists analysis
                    xg = float(player['Expected Goals'])
                    xa = float(player['Expected Assists'])
                    if xg > 0.2 or xa > 0.2:
                        st.success(f"⚽ **High xG/xA** - xG: {xg:.2f}, xA: {xa:.2f}")
                    elif xg > 0.1 or xa > 0.1:
                        st.info(f"📊 **Moderate xG/xA** - xG: {xg:.2f}, xA: {xa:.2f}")
                    else:
                        st.warning(f"📉 **Low xG/xA** - xG: {xg:.2f}, xA: {xa:.2f}")
                    
                    # Form analysis
                    form = float(player['Form'])
                    if form >= 7.0:
                        st.success(f"🔥 **Excellent Form** - {form:.1f} average")
                    elif form >= 5.0:
                        st.info(f"📈 **Good Form** - {form:.1f} average")
                    else:
                        st.warning(f"📉 **Poor Form** - {form:.1f} average")
                    
                    # Minutes analysis
                    minutes = player['Minutes Played']
                    if minutes >= 270:  # 3 full games
                        st.success(f"⏱️ **Regular Starter** - {minutes} minutes")
                    elif minutes >= 90:  # At least 1 full game
                        st.info(f"🔄 **Rotation Risk** - {minutes} minutes")
                    else:
                        st.warning(f"❌ **Limited Minutes** - {minutes} minutes")
                    
                    # Enhanced captaincy score analysis
                    enhanced_captain_score = float(player['Enhanced Captain Score'])
                    st.write(f"**🎯 Enhanced Captain Score:** {enhanced_captain_score:.2f}")
                    if enhanced_captain_score >= 16.0:
                        st.success("🌟 **Excellent Captaincy Choice**")
                    elif enhanced_captain_score >= 12.0:
                        st.info("👍 **Good Captaincy Choice**")
                    else:
                        st.warning("⚠️ **Risky Captaincy Choice**")
    
    # ============================================================================
    # SQUAD BREAKDOWN
    # ============================================================================
    st.markdown("### 📋 Squad Breakdown")
    
    # Show starting XI and bench
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**⚽ Starting XI (Enhanced Analysis)**")
        starting_xi = [p for p in squad.players if p.bench_position is None]
        for i, player in enumerate(starting_xi, 1):
            analysis = optimizer.analyze_player_selection_factors(player.player, next_gw)
            enhanced_score = analysis['total_enhanced_score']
            ict_index = player.player.ict_index
            xg = player.player.expected_goals
            xa = player.player.expected_assists
            
            st.write(f"{i}. **{player.name}** ({player.position}) - {player.team}")
            st.write(f"   💰 £{player.cost}m | 🎯 Enhanced: {enhanced_score:.2f} | 📊 ICT: {ict_index:.0f} | ⚽ xG: {xg:.2f} | 🎯 xA: {xa:.2f}")
    
    with col2:
        st.markdown("**🪑 Bench (Enhanced Analysis)**")
        bench = [p for p in squad.players if p.bench_position is not None]
        # Sort bench by bench position
        bench.sort(key=lambda p: p.bench_position)
        for i, player in enumerate(bench, 1):
            analysis = optimizer.analyze_player_selection_factors(player.player, next_gw)
            enhanced_score = analysis['total_enhanced_score']
            ict_index = player.player.ict_index
            xg = player.player.expected_goals
            xa = player.player.expected_assists
            
            st.write(f"{i}. **{player.name}** ({player.position}) - {player.team} (Bench {player.bench_position})")
            st.write(f"   💰 £{player.cost}m | 🎯 Enhanced: {enhanced_score:.2f} | 📊 ICT: {ict_index:.0f} | ⚽ xG: {xg:.2f} | 🎯 xA: {xa:.2f}")
    
    # ============================================================================
    # POSITION ANALYSIS
    # ============================================================================
    st.markdown("### 📊 Position Analysis")
    
    # Count players by position
    position_counts = {}
    for player in squad.players:
        pos = player.position
        if pos not in position_counts:
            position_counts[pos] = 0
        position_counts[pos] += 1
    
    col1, col2, col3, col4 = st.columns(4)
    positions = ['GK', 'DEF', 'MID', 'FWD']
    for i, pos in enumerate(positions):
        with [col1, col2, col3, col4][i]:
            count = position_counts.get(pos, 0)
            st.metric(f"{pos} Players", count)
    
    # ============================================================================
    # TEAM DISTRIBUTION
    # ============================================================================
    st.markdown("### 🏆 Team Distribution")
    
    # Count players by team
    team_counts = {}
    for player in squad.players:
        team = player.team
        if team not in team_counts:
            team_counts[team] = 0
        team_counts[team] += 1
    
    # Show teams with more than 1 player
    teams_with_multiple = {team: count for team, count in team_counts.items() if count > 1}
    if teams_with_multiple:
        st.markdown("**Teams with Multiple Players:**")
        for team, count in sorted(teams_with_multiple.items(), key=lambda x: x[1], reverse=True):
            st.write(f"• **{team}:** {count} players")
    else:
        st.info("✅ **Well-balanced squad** - No team has more than 1 player")
    
    # ============================================================================
    # RISK ASSESSMENT
    # ============================================================================
    st.markdown("### ⚠️ Risk Assessment")
    
    # Identify potential risks
    risks = []
    
    # Check for rotation risks
    rotation_risks = [p for p in squad.players if p.minutes_played < 270]
    if rotation_risks:
        risks.append(f"🔄 **Rotation Risk:** {len(rotation_risks)} players with limited minutes")
    
    # Check for form risks
    form_risks = [p for p in squad.players if p.form < 3.0]
    if form_risks:
        risks.append(f"📉 **Form Risk:** {len(form_risks)} players in poor form")
    
    # Check for fixture risks
    fixture_risks = []
    if optimizer.fixtures:
        for player in squad.players:
            for fixture in optimizer.fixtures:
                if fixture.gameweek == next_gw:
                    if fixture.home_team == player.team and fixture.home_difficulty >= 4:
                        fixture_risks.append(player.name)
                    elif fixture.away_team == player.team and fixture.away_difficulty >= 4:
                        fixture_risks.append(player.name)
                    break
    
    if fixture_risks:
        risks.append(f"🏟️ **Fixture Risk:** {len(set(fixture_risks))} players with tough fixtures")
    
    if risks:
        for risk in risks:
            st.warning(risk)
    else:
        st.success("✅ **Low Risk Squad** - No significant risks identified")
    
    # ============================================================================
    # WHY COMPREHENSIVE ANALYSIS IS BETTER
    # ============================================================================
    st.markdown("### 🎯 Why Comprehensive Analysis is Better")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**❌ Old System (Limited)**")
        st.write("• Only considered recent form (last 2 gameweeks)")
        st.write("• Basic predicted points only")
        st.write("• Simple fixture difficulty")
        st.write("• No consideration of underlying ability")
        st.write("• Missed high-potential players with poor recent form")
    
    with col2:
        st.markdown("**✅ New System (Comprehensive)**")
        st.write("• **ICT Index** - Influence, Creativity, Threat analysis")
        st.write("• **Expected Goals/Assists** - Underlying performance metrics")
        st.write("• **Availability** - Rotation risk and injury assessment")
        st.write("• **Team Strength** - Fixture context and opponent analysis")
        st.write("• **Position-Specific** - Different metrics for GK/DEF/MID/FWD")
        st.write("• **Historical Consistency** - Minutes played and reliability")
    
    # ============================================================================
    # COMPREHENSIVE SQUAD ANALYSIS
    # ============================================================================
    st.markdown("### 🔬 Comprehensive Squad Analysis")
    
    # Get comprehensive analysis for all squad players
    squad_analysis_data = []
    for player in squad.players:
        analysis = optimizer.analyze_player_selection_factors(player.player, next_gw)
        
        squad_analysis_data.append({
            "Player": player.name,
            "Position": player.position,
            "Team": player.team,
            "Enhanced Score": f"{analysis['total_enhanced_score']:.2f}",
            "Base Points": f"{analysis['base_predicted_points']:.1f}",
            "ICT Bonus": f"{analysis['ict_index_score']:.2f}",
            "xG/xA Bonus": f"{analysis['expected_goals_assists']:.2f}",
            "Availability": f"{analysis['availability_score']:.2f}",
            "Team Strength": f"{analysis['team_strength_score']:.2f}",
            "Position Bonus": f"{analysis['position_specific_score']:.2f}",
            "Form Bonus": f"{analysis['form_score']:.2f}",
            "Minutes Bonus": f"{analysis['minutes_played_score']:.2f}",
            "Bonus Potential": f"{analysis['bonus_potential_score']:.2f}",
            "ICT Index": f"{player.player.ict_index:.0f}",
            "Expected Goals": f"{player.player.expected_goals:.2f}",
            "Expected Assists": f"{player.player.expected_assists:.2f}",
            "Form": f"{player.player.form:.1f}",
            "Minutes": player.player.minutes_played,
            "Status": player.player.status
        })
    
    # Display comprehensive squad analysis
    st.markdown("**📊 Squad Players - Comprehensive Analysis Breakdown**")
    st.dataframe(
        squad_analysis_data,
        width='stretch',
        hide_index=True,
        column_config={
            "Player": st.column_config.TextColumn("Player", width="medium"),
            "Position": st.column_config.TextColumn("Pos", width="small"),
            "Team": st.column_config.TextColumn("Team", width="small"),
            "Enhanced Score": st.column_config.TextColumn("Enhanced", width="small"),
            "Base Points": st.column_config.TextColumn("Base", width="small"),
            "ICT Bonus": st.column_config.TextColumn("ICT", width="small"),
            "xG/xA Bonus": st.column_config.TextColumn("xG/xA", width="small"),
            "Availability": st.column_config.TextColumn("Avail", width="small"),
            "Team Strength": st.column_config.TextColumn("Team", width="small"),
            "Position Bonus": st.column_config.TextColumn("Pos", width="small"),
            "Form Bonus": st.column_config.TextColumn("Form", width="small"),
            "Minutes Bonus": st.column_config.TextColumn("Min", width="small"),
            "Bonus Potential": st.column_config.TextColumn("Bonus", width="small"),
            "ICT Index": st.column_config.TextColumn("ICT", width="small"),
            "Expected Goals": st.column_config.TextColumn("xG", width="small"),
            "Expected Assists": st.column_config.TextColumn("xA", width="small"),
            "Form": st.column_config.TextColumn("Form", width="small"),
            "Minutes": st.column_config.NumberColumn("Minutes", width="small"),
            "Status": st.column_config.TextColumn("Status", width="small")
        }
    )
    
    # ============================================================================
    # ADVANCED ANALYTICS
    # ============================================================================
    with st.expander("🔬 Advanced Analytics", expanded=False):
        st.markdown("### 📈 Advanced Metrics")
        
        # Show player count by position
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            gk_count = len([p for p in players if p.position == "GK"])
            st.metric("Total Goalkeepers", gk_count)
        with col2:
            def_count = len([p for p in players if p.position == "DEF"])
            st.metric("Total Defenders", def_count)
        with col3:
            mid_count = len([p for p in players if p.position == "MID"])
            st.metric("Total Midfielders", mid_count)
        with col4:
            fwd_count = len([p for p in players if p.position == "FWD"])
            st.metric("Total Forwards", fwd_count)
        
        # Show top 10 players by comprehensive analysis
        st.markdown("**🏆 Top 10 Players by Comprehensive Analysis:**")
        comprehensive_top = optimizer.get_comprehensive_player_rankings(gameweek=next_gw, limit=10)
        top_data = []
        for ranking in comprehensive_top:
            player = ranking['player']
            top_data.append({
                "Rank": ranking['rank'],
                "Player": player.name,
                "Team": player.team,
                "Position": player.position,
                "Cost": f"£{player.cost}m",
                "Enhanced Score": f"{ranking['total_enhanced_score']:.2f}",
                "Form": f"{player.form:.1f}",
                "Minutes": player.minutes_played
            })
        
        st.dataframe(
            top_data,
            width='stretch',
            hide_index=True,
            column_config={
                "Rank": st.column_config.NumberColumn("Rank", format="%d"),
                "Player": st.column_config.TextColumn("Player", width="medium"),
                "Team": st.column_config.TextColumn("Team", width="small"),
                "Position": st.column_config.TextColumn("Pos", width="small"),
                "Cost": st.column_config.TextColumn("Cost", width="small"),
                "Enhanced Score": st.column_config.TextColumn("Enhanced", width="small"),
                "Form": st.column_config.TextColumn("Form", width="small"),
                "Minutes": st.column_config.NumberColumn("Minutes", width="small")
            }
        )
    
    # ============================================================================
    # ACTION ITEMS
    # ============================================================================
    st.markdown("### 🎯 Action Items")
    
    st.markdown("""
    **📋 Next Steps:**
    1. **Review the squad** - Check if all selected players align with your strategy
    2. **Verify captain choice** - Ensure the recommended captain fits your risk tolerance
    3. **Check bench order** - Review the bench ordering for autosubstitutions
    4. **Monitor injuries** - Check for any late injury news before deadline
    5. **Consider transfers** - Use the insights to plan future transfers
    """)
    
    # ============================================================================
    # FOOTER
    # ============================================================================
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 10px; background-color: #f8f9fa; border-radius: 5px;">
        <p style="margin: 0; color: #6c757d;">
            🤖 <strong>AI-Powered FPL Optimization</strong> | 
            📊 <strong>Machine Learning Enhanced</strong> | 
            ⚽ <strong>Data-Driven Decisions</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)


def show_ml_models_page(optimizer: SquadOptimizer, players: List[Player], events: List[GameweekEvent], available_gws: List[int]):
    """Display ML models management page."""
    st.header("🤖 Machine Learning Models")
    st.markdown("Advanced predictive models for enhanced FPL optimization")
    
    # Model Status
    st.subheader("📊 Model Status")
    
    # Check if models are actually trained (not just the flag)
    models_trained = optimizer.predictive_model.is_trained and len(optimizer.predictive_model.models) > 0
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if models_trained:
            st.success("✅ ML Models Trained")
        else:
            st.warning("⚠️ ML Models Not Trained")
    
    with col2:
        if models_trained:
            best_model = optimizer.predictive_model.get_best_model()
            if best_model:
                st.info(f"🏆 Best Model: {best_model.replace('_', ' ').title()}")
            else:
                st.info("🏆 Best Model: N/A")
        else:
            st.info("🏆 Best Model: N/A")
    
    with col3:
        if models_trained:
            model_count = len(optimizer.predictive_model.models)
            st.metric("Models Available", model_count)
        else:
            st.metric("Models Available", 0)
    
    # Training Section
    st.subheader("🎯 Train Models")
    
    # Collect historical data for training
    st.markdown("**Step 1: Collect Historical Data**")
    
    col1, col2 = st.columns(2)
    with col1:
        start_gw = st.selectbox("Start Gameweek:", available_gws, index=0)
    with col2:
        end_gw = st.selectbox("End Gameweek:", available_gws, index=min(len(available_gws)-1, 2))
    
    if st.button("📊 Collect Historical Data", help="Fetch performance data from selected gameweeks"):
        with st.spinner("Collecting historical data..."):
            historical_data = []
            
            for gw in range(start_gw, end_gw + 1):
                try:
                    live_data = fetch_live_event_data(gw)
                    if live_data and "elements" in live_data:
                        player_performances = parse_player_performances(live_data, players)
                        for perf in player_performances:
                            historical_data.append({
                                'player_id': perf.player_id,
                                'name': perf.name,
                                'gameweek': gw,
                                'actual_points': perf.actual_points,
                                'minutes': perf.minutes_played,
                                'goals_scored': perf.goals_scored,
                                'assists': perf.assists,
                                'clean_sheets': perf.clean_sheets,
                                'bonus': perf.bonus
                            })
                except Exception as e:
                    st.warning(f"Could not fetch data for GW{gw}: {e}")
            
            if historical_data:
                st.session_state.historical_performances = historical_data
                st.success(f"✅ Collected {len(historical_data)} performance records from GW{start_gw}-{end_gw}")
                
                # Auto-save historical data persistently
                try:
                    optimizer.predictive_model._save_historical_data(historical_data)
                    st.info("💾 Historical data saved persistently")
                except Exception as e:
                    st.warning(f"Could not save historical data: {e}")
            else:
                st.error("❌ No historical data collected")
    
    # Train Models
    st.markdown("**Step 2: Train ML Models**")
    
    if 'historical_performances' in st.session_state and st.session_state.historical_performances:
        if st.button("🚀 Train ML Models", help="Train multiple ML models for enhanced predictions"):
            with st.spinner("Training ML models..."):
                try:
                    # Ensure players is a valid list
                    if not players or not isinstance(players, list):
                        st.error("No valid players data available for training")
                        return
                        
                    # Filter out any invalid player objects
                    valid_players = [p for p in players if hasattr(p, 'id') and hasattr(p, 'name') and hasattr(p, 'team') and hasattr(p, 'position')]
                    
                    if not valid_players:
                        st.error("No valid players found for training")
                        return
                        
                    performance_metrics = optimizer.predictive_model.train_models(
                        valid_players, st.session_state.historical_performances
                    )
                    
                    # Display results
                    st.success("✅ ML models trained successfully!")
                    
                    # Show performance metrics
                    st.subheader("📈 Model Performance")
                    metrics_df = pd.DataFrame([
                        {
                            'Model': name,
                            'MAE': f"{metrics['mae']:.3f}",
                            'RMSE': f"{metrics['rmse']:.3f}",
                            'R²': f"{metrics['r2']:.3f}",
                            'Training Samples': metrics['training_samples']
                        }
                        for name, metrics in performance_metrics.items()
                    ])
                    st.dataframe(metrics_df, width='stretch')
                    
                    # Enable ML-enhanced optimization
                    optimizer.ml_enhanced = True
                    
                except Exception as e:
                    st.error(f"❌ Error training models: {e}")
    else:
        st.info("ℹ️ Collect historical data first to train models")
    
    # Model Performance
    if models_trained:
        st.subheader("📈 Model Performance")
        
        # Show enhanced model summary
        model_summary = optimizer.get_ml_model_performance()
        if model_summary and "error" not in model_summary:
            # Display model type and approach
            if "model_type" in model_summary:
                st.info(f"🎯 **{model_summary['model_type']}**")
                st.write(f"**Prediction Approach:** {model_summary['prediction_approach']}")
                
                # Show features used
                if "features_used" in model_summary:
                    st.subheader("🔧 Features Used")
                    for feature in model_summary["features_used"]:
                        st.write(f"• {feature}")
            
            # Convert dictionary to DataFrame for display
            if "models" in model_summary:
                models_data = []
                for model_name, metrics in model_summary["models"].items():
                    models_data.append({
                        "Model": model_name.replace("_", " ").title(),
                        "MAE": metrics["mae"],
                        "RMSE": metrics["rmse"],
                        "R²": metrics["r2"],
                        "Accuracy": metrics.get("accuracy_rating", "N/A"),
                        "Training Samples": metrics["training_samples"]
                    })
                
                if models_data:
                    model_df = pd.DataFrame(models_data)
                    st.dataframe(model_df, width='stretch')
                    
                    # Performance visualization
                    fig = px.bar(
                        model_df, 
                        x='Model', 
                        y='R²', 
                        title="Model Performance (R² Score)",
                        color='R²',
                        color_continuous_scale='viridis'
                    )
                    st.plotly_chart(fig, width='stretch')
                    
                    # Show summary metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Models", model_summary["total_models"])
                    with col2:
                        st.metric("Best Model", model_summary["best_model"].replace("_", " ").title() if model_summary["best_model"] else "N/A")
                    with col3:
                        st.metric("Historical Records", model_summary["historical_records"])
        else:
            st.warning("No trained models available")
        
        # Show prediction insights
        st.subheader("🔍 Prediction Insights")
        
        try:
            # Ensure players is a valid list
            if not players or not isinstance(players, list):
                st.warning("No valid players data available for insights")
                return
                
            # Filter out any invalid player objects
            valid_players = [p for p in players if hasattr(p, 'id') and hasattr(p, 'name') and hasattr(p, 'team') and hasattr(p, 'position')]
            
            if not valid_players:
                st.warning("No valid players found for insights")
                return
                
            insights = optimizer.predictive_model.get_prediction_insights(valid_players)
            
            if "error" not in insights:
                # Prediction summary
                summary = insights["prediction_summary"]
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Players Analyzed", summary["total_players_analyzed"])
                with col2:
                    st.metric("Avg Predicted Points", f"{summary['average_predicted_points']:.1f}")
                with col3:
                    st.metric("Highest Prediction", summary["highest_predicted_points"])
                with col4:
                    st.metric("Confidence", summary["prediction_confidence"])
                
                # Top predictions with insights
                if insights["top_predictions"]:
                    st.subheader("🏆 Top 10 Predictions with Analysis")
                    
                    for i, prediction in enumerate(insights["top_predictions"][:5], 1):
                        with st.expander(f"{i}. {prediction['player_name']} ({prediction['team']}) - {prediction['predicted_points']} pts"):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write(f"**Position:** {prediction['position']}")
                                st.write(f"**Cost:** £{prediction['cost']}m")
                                st.write(f"**Form:** {prediction['current_form']}")
                                st.write(f"**Recommendation:** {prediction['recommendation']}")
                            
                            with col2:
                                risk = prediction['risk_assessment']
                                st.write(f"**Risk Level:** {risk['risk_level']}")
                                st.write(f"**Rotation Risk:** {risk['rotation_risk']:.1%}")
                                st.write(f"**Injury Risk:** {risk['injury_risk']:.1%}")
                            
                            # Key factors
                            if prediction['key_factors']:
                                st.write("**Key Factors:**")
                                for factor in prediction['key_factors']:
                                    st.write(f"• {factor}")
                
                # Team analysis
                if insights["team_analysis"]:
                    st.subheader("🏟️ Team Strength Analysis")
                    
                    team_data = []
                    for team_name, team_info in insights["team_analysis"].items():
                        team_data.append({
                            "Team": team_name,
                            "Attack Rating": team_info["attack_rating"],
                            "Defense Rating": team_info["defense_rating"],
                            "Form": team_info["form"],
                            "Assessment": team_info["strength_assessment"]
                        })
                    
                    team_df = pd.DataFrame(team_data)
                    st.dataframe(team_df, width='stretch')
                
                # Key insights
                if insights["key_insights"]:
                    st.subheader("💡 Key Insights")
                    for insight in insights["key_insights"]:
                        st.write(f"• {insight}")
            
        except Exception as e:
            st.error(f"Error generating insights: {e}")
    
    # Auto-load historical data on page load if available
    if 'historical_performances' not in st.session_state:
        try:
            historical_data = optimizer.predictive_model._load_historical_data()
            if historical_data:
                st.session_state.historical_performances = historical_data
                st.info(f"📂 Loaded {len(historical_data)} historical performance records from persistent storage")
        except Exception as e:
            pass
    
    # Retraining Section
    st.subheader("🔄 Retrain Models")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔄 Retrain with New Data", help="Retrain models with latest performance data"):
            if 'historical_performances' in st.session_state:
                with st.spinner("Retraining models..."):
                    try:
                        # Ensure players is a valid list
                        if not players or not isinstance(players, list):
                            st.error("No valid players data available for retraining")
                            return
                            
                        # Filter out any invalid player objects
                        valid_players = [p for p in players if hasattr(p, 'id') and hasattr(p, 'name') and hasattr(p, 'team') and hasattr(p, 'position')]
                        
                        if not valid_players:
                            st.error("No valid players found for retraining")
                            return
                            
                        performance_metrics = optimizer.predictive_model.retrain_models(
                            valid_players, st.session_state.historical_performances
                        )
                        st.success("✅ Models retrained successfully!")
                    except Exception as e:
                        st.error(f"❌ Error retraining: {e}")
            else:
                st.warning("⚠️ No historical data available for retraining")
    
    with col2:
        if st.button("📂 Load Saved Models", help="Load previously saved models"):
            try:
                optimizer.predictive_model._load_existing_models()
                if optimizer.predictive_model.is_trained:
                    st.success("✅ Models loaded successfully!")
                else:
                    st.warning("No saved models found")
            except Exception as e:
                st.error(f"❌ Error loading models: {e}")
    
    with col3:
        if st.button("💾 Save Models", help="Save trained models to disk"):
            if optimizer.predictive_model.is_trained:
                try:
                    optimizer.predictive_model._save_models()
                    st.success("✅ Models saved successfully!")
                except Exception as e:
                    st.error(f"❌ Error saving models: {e}")
            else:
                st.warning("No trained models to save")
    
    # ML-Enhanced Optimization
    st.subheader("🎯 ML-Enhanced Optimization")
    
    if models_trained:
        st.success("✅ ML models are ready for enhanced optimization!")
        
        # Show prediction confidence
        try:
            # Ensure players is a valid list
            if not players or not isinstance(players, list):
                st.warning("No valid players data available for confidence calculation")
                confidence_scores = {}
            else:
                # Filter out any invalid player objects
                valid_players = [p for p in players if hasattr(p, 'id') and hasattr(p, 'name') and hasattr(p, 'team') and hasattr(p, 'position')]
                
                if not valid_players:
                    st.warning("No valid players found for confidence calculation")
                    confidence_scores = {}
                else:
                    confidence_scores = optimizer.get_prediction_confidence(valid_players)
        except Exception as e:
            st.error(f"Error calculating prediction confidence: {e}")
            confidence_scores = {}
        
        # Top 10 most confident predictions
        top_confidence = sorted(confidence_scores.items(), key=lambda x: x[1], reverse=True)[:10]
        
        confidence_data = []
        for player_id, confidence in top_confidence:
            player = next((p for p in players if p.id == player_id), None)
            if player:
                confidence_data.append({
                    "Player": player.name,
                    "Team": player.team,
                    "Position": player.position,
                    "Confidence": f"{confidence:.2f}"
                })
        
        if confidence_data:
            st.markdown("**Top 10 Most Confident Predictions:**")
            st.dataframe(confidence_data, width='stretch')
    else:
        st.info("ℹ️ Train ML models first to enable enhanced optimization")
    
    # Feature Importance
    if models_trained and 'ml_performance' in st.session_state:
        st.subheader("🔍 Feature Importance")
        
        best_model_name = optimizer.predictive_model.get_best_model()
        if best_model_name and best_model_name in optimizer.predictive_model.model_performance:
            feature_importance = optimizer.predictive_model.model_performance[best_model_name].feature_importance
            
            # Sort features by importance
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
            
            feature_data = []
            for feature, importance in sorted_features:
                feature_data.append({
                    "Feature": feature,
                    "Importance": f"{importance:.4f}"
                })
            
            if feature_data:
                st.dataframe(feature_data, use_container_width=True)
                
                # Feature importance chart
                fig = px.bar(
                    x=[f[0] for f in sorted_features],
                    y=[f[1] for f in sorted_features],
                    title=f"Feature Importance - {best_model_name.title()}",
                    labels={"x": "Feature", "y": "Importance"}
                )
                st.plotly_chart(fig, use_container_width=True)


def show_settings_page():
    """Display settings page."""
    st.header("⚙️ Settings & Controls")
    
    st.subheader("Budget Control")
    budget = st.slider("Budget (£m)", min_value=80.0, max_value=120.0, value=100.0, step=0.5)
    st.info(f"Current budget: £{budget}m")
    
    st.subheader("Player Filters")
    min_minutes = st.slider("Minimum minutes played", min_value=0, max_value=2000, value=90, step=10)
    st.info(f"Only players with {min_minutes}+ minutes this season")
    
    st.subheader("Team Constraints")
    max_per_team = st.slider("Max players per team", min_value=2, max_value=5, value=3)
    st.info(f"Maximum {max_per_team} players from any single team")
    
    st.subheader("Risk Tolerance")
    include_doubtful = st.checkbox("Include doubtful players", value=False)
    if include_doubtful:
        st.warning("⚠️ Including players with injury doubts may affect reliability")
    else:
        st.success("✅ Only fully fit players included")


if __name__ == "__main__":
    main()
