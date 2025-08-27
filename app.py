import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from fetch_data import FPLDataFetcher
from optimizer import FPLOptimizer
import time

# Page configuration
st.set_page_config(
    page_title="FPL Optimizer",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .captain-badge {
        background-color: #ffd700;
        color: #000;
        padding: 0.2rem 0.5rem;
        border-radius: 0.3rem;
        font-weight: bold;
        font-size: 0.8rem;
    }
    .vice-captain-badge {
        background-color: #c0c0c0;
        color: #000;
        padding: 0.2rem 0.5rem;
        border-radius: 0.3rem;
        font-weight: bold;
        font-size: 0.8rem;
    }
    .bench-player {
        background-color: #f8f9fa;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.2rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_and_optimize():
    """Load FPL data and optimize squad."""
    try:
        with st.spinner("Fetching FPL data..."):
            fetcher = FPLDataFetcher()
            players = fetcher.fetch_and_clean()
        
        with st.spinner("Optimizing squad..."):
            optimizer = FPLOptimizer(players)
            squad_result = optimizer.optimize_squad()
            captaincy_result = optimizer.optimize_captaincy()
            bench_order = optimizer.optimize_bench_order()
            
        return optimizer, squad_result, captaincy_result, bench_order
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None, None

def display_squad_page(optimizer, squad_result, captaincy_result, bench_order):
    """Display the Best Squad page."""
    st.markdown('<h1 class="main-header">⚽ FPL Optimizer - Best Squad</h1>', unsafe_allow_html=True)
    
    # Squad summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Cost",
            value=f"£{squad_result['total_cost']:.1f}m",
            delta=None
        )
    
    with col2:
        st.metric(
            label="Total Predicted Points",
            value=f"{squad_result['total_predicted_points']:.1f}",
            delta=None
        )
    
    with col3:
        starting_xi = optimizer.get_starting_xi()
        st.metric(
            label="Starting XI Points",
            value=f"{starting_xi['predicted_points'].sum():.1f}",
            delta=None
        )
    
    with col4:
        bench = optimizer.get_bench()
        st.metric(
            label="Bench Points",
            value=f"{bench['predicted_points'].sum():.1f}",
            delta=None
        )
    
    st.markdown("---")
    
    # Squad display
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🏆 Optimized Squad")
        
        # Create squad dataframe with captain/vice-captain indicators
        squad_df = squad_result['squad'].copy()
        squad_df['Role'] = ''
        
        # Mark captain and vice-captain
        captain_id = captaincy_result['captain']['player_id']
        vice_captain_id = captaincy_result['vice_captain']['player_id']
        
        squad_df.loc[squad_df['id'] == captain_id, 'Role'] = '⭐ Captain'
        squad_df.loc[squad_df['id'] == vice_captain_id, 'Role'] = '🅥 Vice-Captain'
        
        # Format the display
        display_df = squad_df[['name', 'team', 'pos', 'cost', 'predicted_points', 'Role']].copy()
        display_df['cost'] = display_df['cost'].apply(lambda x: f"£{x:.1f}m")
        display_df['predicted_points'] = display_df['predicted_points'].apply(lambda x: f"{x:.1f}")
        
        # Highlight starting XI vs bench
        starting_xi_ids = set(optimizer.get_starting_xi()['id'])
        
        def highlight_starting_xi(row):
            if row['id'] in starting_xi_ids:
                return ['background-color: #e8f4fd'] * len(row)
            else:
                return ['background-color: #f8f9fa'] * len(row)
        
        # Apply highlighting
        styled_df = display_df.style.apply(highlight_starting_xi, axis=1)
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.subheader("🪑 Bench Order")
        
        for i, player in enumerate(bench_order):
            with st.container():
                st.markdown(f"""
                <div class="bench-player">
                    <strong>{player['position']}</strong><br>
                    {player['name']} ({player['pos']})<br>
                    <small>£{player['cost']:.1f}m • {player['predicted_points']:.1f} pts</small>
                </div>
                """, unsafe_allow_html=True)
    
    # Team and position breakdown
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Players by Team")
        team_counts = squad_result['squad']['team'].value_counts()
        fig_team = px.bar(
            x=team_counts.values,
            y=team_counts.index,
            orientation='h',
            title="Players per Team",
            labels={'x': 'Number of Players', 'y': 'Team'}
        )
        fig_team.update_layout(height=400)
        st.plotly_chart(fig_team, use_container_width=True)
    
    with col2:
        st.subheader("📊 Players by Position")
        pos_counts = squad_result['squad']['pos'].value_counts()
        fig_pos = px.pie(
            values=pos_counts.values,
            names=pos_counts.index,
            title="Players by Position"
        )
        st.plotly_chart(fig_pos, use_container_width=True)

def display_captaincy_page(optimizer, captaincy_result):
    """Display the Captaincy Simulation page."""
    st.markdown('<h1 class="main-header">⚽ FPL Optimizer - Captaincy Simulation</h1>', unsafe_allow_html=True)
    
    # Captain and Vice-Captain summary
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>⭐ Captain</h3>
            <h2>{}</h2>
            <p>{} • {} • {:.1f} pts</p>
        </div>
        """.format(
            captaincy_result['captain']['name'],
            captaincy_result['captain']['team'],
            captaincy_result['captain']['pos'],
            captaincy_result['captain']['predicted_points']
        ), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🅥 Vice-Captain</h3>
            <h2>{}</h2>
            <p>{} • {} • {:.1f} pts</p>
        </div>
        """.format(
            captaincy_result['vice_captain']['name'],
            captaincy_result['vice_captain']['team'],
            captaincy_result['vice_captain']['pos'],
            captaincy_result['vice_captain']['predicted_points']
        ), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Captaincy simulation chart
    st.subheader("📈 Captaincy Impact on Team Total")
    
    # Prepare data for chart
    captaincy_data = captaincy_result['captaincy_results']
    df_chart = pd.DataFrame(captaincy_data)
    
    # Sort by team total
    df_chart = df_chart.sort_values('team_total', ascending=True)
    
    # Create horizontal bar chart
    fig = px.bar(
        df_chart,
        x='team_total',
        y='name',
        orientation='h',
        color='team',
        title="Team Total Points if Each Player is Captain",
        labels={'team_total': 'Team Total Points', 'name': 'Player'},
        hover_data=['pos', 'predicted_points', 'captain_points']
    )
    
    # Highlight captain and vice-captain
    captain_name = captaincy_result['captain']['name']
    vice_captain_name = captaincy_result['vice_captain']['name']
    
    fig.update_layout(
        height=600,
        showlegend=True,
        xaxis_title="Team Total Points",
        yaxis_title="Player"
    )
    
    # Add annotations for captain and vice-captain
    for i, row in df_chart.iterrows():
        if row['name'] == captain_name:
            fig.add_annotation(
                x=row['team_total'] + 0.5,
                y=row['name'],
                text="⭐ Captain",
                showarrow=True,
                arrowhead=2,
                arrowcolor="gold",
                font=dict(color="gold", size=12, weight="bold")
            )
        elif row['name'] == vice_captain_name:
            fig.add_annotation(
                x=row['team_total'] + 0.5,
                y=row['name'],
                text="🅥 Vice-Captain",
                showarrow=True,
                arrowhead=2,
                arrowcolor="silver",
                font=dict(color="silver", size=12, weight="bold")
            )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed captaincy table
    st.subheader("📋 Detailed Captaincy Analysis")
    
    # Format table
    table_df = df_chart[['name', 'team', 'pos', 'predicted_points', 'captain_points', 'team_total']].copy()
    table_df['predicted_points'] = table_df['predicted_points'].apply(lambda x: f"{x:.1f}")
    table_df['captain_points'] = table_df['captain_points'].apply(lambda x: f"{x:.1f}")
    table_df['team_total'] = table_df['team_total'].apply(lambda x: f"{x:.1f}")
    
    # Sort by team total descending
    table_df = table_df.sort_values('team_total', ascending=False)
    
    st.dataframe(table_df, use_container_width=True, hide_index=True)

def main():
    """Main application function."""
    
    # Sidebar
    st.sidebar.title("⚽ FPL Optimizer")
    st.sidebar.markdown("---")
    
    # Page selection
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["Best Squad", "Captaincy Simulation"]
    )
    
    # Load data and optimize
    optimizer, squad_result, captaincy_result, bench_order = load_and_optimize()
    
    if optimizer is None:
        st.error("Failed to load data. Please check your internet connection and try again.")
        return
    
    # Display appropriate page
    if page == "Best Squad":
        display_squad_page(optimizer, squad_result, captaincy_result, bench_order)
    elif page == "Captaincy Simulation":
        display_captaincy_page(optimizer, captaincy_result)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **Data Source:** FPL Official API  
    **Last Updated:** Real-time  
    **Optimization:** Linear Programming (PuLP)
    """)
    
    # Refresh button
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

if __name__ == "__main__":
    main()