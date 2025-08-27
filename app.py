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
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .player-table {
        background-color: white;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .captain-badge {
        background-color: #ffd700;
        color: #000;
        padding: 0.2rem 0.5rem;
        border-radius: 0.3rem;
        font-weight: bold;
    }
    .vice-captain-badge {
        background-color: #c0c0c0;
        color: #000;
        padding: 0.2rem 0.5rem;
        border-radius: 0.3rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_data():
    """Load and cache FPL data."""
    fetcher = FPLDataFetcher()
    return fetcher.fetch_and_clean()

@st.cache_data(ttl=3600)  # Cache for 1 hour
def optimize_squad(players_data):
    """Optimize squad and cache results."""
    optimizer = FPLOptimizer(players_data)
    return optimizer.optimize_complete_squad()

def display_squad_page(result):
    """Display the optimal squad page."""
    st.markdown('<h1 class="main-header">⚽ FPL Optimizer - Best Squad</h1>', unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Points", f"{result['total_points']:.1f}")
    
    with col2:
        st.metric("Squad Cost", f"£{result['squad_cost']:.1f}m")
    
    with col3:
        st.metric("Budget Remaining", f"£{100.0 - result['squad_cost']:.1f}m")
    
    with col4:
        st.metric("Captain", result['captain'])
    
    st.markdown("---")
    
    # Squad overview
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Squad Overview")
        
        # Create squad display with captain/vice-captain indicators
        squad_display = result['squad'].copy()
        squad_display['Role'] = ''
        squad_display.loc[squad_display['name'] == result['captain'], 'Role'] = '⭐ Captain'
        squad_display.loc[squad_display['name'] == result['vice_captain'], 'Role'] = '🅥 Vice-Captain'
        
        # Format cost and points
        squad_display['Cost (£m)'] = squad_display['cost'].round(1)
        squad_display['Predicted Points'] = squad_display['predicted_points'].round(1)
        
        # Reorder columns
        display_cols = ['name', 'pos', 'team', 'Cost (£m)', 'Predicted Points', 'Role']
        squad_display = squad_display[display_cols].rename(columns={
            'name': 'Player',
            'pos': 'Position',
            'team': 'Team'
        })
        
        st.dataframe(
            squad_display,
            use_container_width=True,
            hide_index=True
        )
    
    with col2:
        st.subheader("📈 Position Distribution")
        
        # Position breakdown
        pos_counts = result['squad']['pos'].value_counts()
        fig = px.pie(
            values=pos_counts.values,
            names=pos_counts.index,
            title="Squad by Position"
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Team breakdown
        st.subheader("🏟️ Team Distribution")
        team_counts = result['squad']['team'].value_counts()
        fig2 = px.bar(
            x=team_counts.values,
            y=team_counts.index,
            orientation='h',
            title="Players per Team"
        )
        fig2.update_layout(height=300, xaxis_title="Players", yaxis_title="Team")
        st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown("---")
    
    # Starting XI and Bench
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⚽ Starting XI")
        
        starting_xi_display = result['starting_xi'].copy()
        starting_xi_display['Role'] = ''
        starting_xi_display.loc[starting_xi_display['name'] == result['captain'], 'Role'] = '⭐ Captain'
        starting_xi_display.loc[starting_xi_display['name'] == result['vice_captain'], 'Role'] = '🅥 Vice-Captain'
        
        starting_xi_display['Cost (£m)'] = starting_xi_display['cost'].round(1)
        starting_xi_display['Predicted Points'] = starting_xi_display['predicted_points'].round(1)
        
        display_cols = ['name', 'pos', 'team', 'Cost (£m)', 'Predicted Points', 'Role']
        starting_xi_display = starting_xi_display[display_cols].rename(columns={
            'name': 'Player',
            'pos': 'Position',
            'team': 'Team'
        })
        
        st.dataframe(
            starting_xi_display,
            use_container_width=True,
            hide_index=True
        )
    
    with col2:
        st.subheader("🪑 Bench (Autosub Order)")
        
        bench_display = result['bench'].copy()
        bench_display['Bench Position'] = range(1, len(bench_display) + 1)
        bench_display['Cost (£m)'] = bench_display['cost'].round(1)
        bench_display['Predicted Points'] = bench_display['predicted_points'].round(1)
        
        display_cols = ['Bench Position', 'name', 'pos', 'team', 'Cost (£m)', 'Predicted Points']
        bench_display = bench_display[display_cols].rename(columns={
            'name': 'Player',
            'pos': 'Position',
            'team': 'Team'
        })
        
        st.dataframe(
            bench_display,
            use_container_width=True,
            hide_index=True
        )

def display_captaincy_page(result):
    """Display the captaincy simulation page."""
    st.markdown('<h1 class="main-header">⚽ FPL Optimizer - Captaincy Simulation</h1>', unsafe_allow_html=True)
    
    # Captaincy results
    captaincy_data = result['captaincy_results']
    
    # Create DataFrame for plotting
    captaincy_df = pd.DataFrame([
        {'Player': player, 'Team Total Points': points}
        for player, points in captaincy_data.items()
    ])
    
    # Add player info
    squad_info = result['squad'].set_index('name')[['pos', 'team', 'predicted_points']]
    captaincy_df = captaincy_df.merge(squad_info, left_on='Player', right_index=True, how='left')
    
    # Sort by team total points
    captaincy_df = captaincy_df.sort_values('Team Total Points', ascending=False)
    
    # Highlight current captain and vice-captain
    captaincy_df['Is Captain'] = captaincy_df['Player'] == result['captain']
    captaincy_df['Is Vice Captain'] = captaincy_df['Player'] == result['vice_captain']
    
    st.subheader("📊 Captaincy Impact on Team Total")
    
    # Create bar chart
    fig = px.bar(
        captaincy_df.head(15),  # Show top 15
        x='Player',
        y='Team Total Points',
        color='pos',
        title="Team Total Points if Each Player is Captain",
        hover_data=['team', 'predicted_points']
    )
    
    # Add captain and vice-captain markers
    for i, row in captaincy_df.head(15).iterrows():
        if row['Is Captain']:
            fig.add_annotation(
                x=row['Player'],
                y=row['Team Total Points'] + 0.5,
                text="⭐ Captain",
                showarrow=False,
                font=dict(color="gold", size=12, weight="bold")
            )
        elif row['Is Vice Captain']:
            fig.add_annotation(
                x=row['Player'],
                y=row['Team Total Points'] + 0.5,
                text="🅥 Vice-Captain",
                showarrow=False,
                font=dict(color="silver", size=12, weight="bold")
            )
    
    fig.update_layout(
        height=600,
        xaxis_tickangle=-45,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Captaincy analysis
    st.subheader("📈 Captaincy Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Best Captain", result['captain'])
        best_captain_points = captaincy_data[result['captain']]
        st.metric("Team Total with Best Captain", f"{best_captain_points:.1f}")
    
    with col2:
        st.metric("Vice-Captain", result['vice_captain'])
        vice_captain_points = captaincy_data[result['vice_captain']]
        st.metric("Team Total with Vice-Captain", f"{vice_captain_points:.1f}")
    
    # Detailed captaincy table
    st.subheader("📋 Detailed Captaincy Results")
    
    captaincy_display = captaincy_df.copy()
    captaincy_display['Role'] = ''
    captaincy_display.loc[captaincy_display['Is Captain'], 'Role'] = '⭐ Captain'
    captaincy_display.loc[captaincy_display['Is Vice Captain'], 'Role'] = '🅥 Vice-Captain'
    
    captaincy_display['Team Total Points'] = captaincy_display['Team Total Points'].round(1)
    captaincy_display['Predicted Points'] = captaincy_display['predicted_points'].round(1)
    
    display_cols = ['Player', 'pos', 'team', 'Predicted Points', 'Team Total Points', 'Role']
    captaincy_display = captaincy_display[display_cols].rename(columns={
        'pos': 'Position',
        'team': 'Team'
    })
    
    st.dataframe(
        captaincy_display,
        use_container_width=True,
        hide_index=True
    )

def main():
    """Main Streamlit application."""
    
    # Sidebar
    st.sidebar.title("⚽ FPL Optimizer")
    st.sidebar.markdown("---")
    
    # Page selection
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["Best Squad", "Captaincy Simulation"]
    )
    
    # Load data with progress indicator
    with st.spinner("Fetching FPL data..."):
        try:
            players_data = load_data()
        except Exception as e:
            st.error(f"Failed to load FPL data: {e}")
            st.stop()
    
    # Optimize squad with progress indicator
    with st.spinner("Optimizing squad..."):
        try:
            result = optimize_squad(players_data)
        except Exception as e:
            st.error(f"Failed to optimize squad: {e}")
            st.stop()
    
    # Display appropriate page
    if page == "Best Squad":
        display_squad_page(result)
    elif page == "Captaincy Simulation":
        display_captaincy_page(result)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Data Source:** FPL Official API")
    st.sidebar.markdown("**Last Updated:** " + time.strftime("%Y-%m-%d %H:%M"))

if __name__ == "__main__":
    main()