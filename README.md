# ⚽ FPL Optimizer

A full-stack Fantasy Premier League (FPL) optimization system that uses linear programming to build optimal squads and simulate captaincy scenarios.

## 🎯 Features

### Core Functionality
- **Real-time Data Fetching**: Connects to the official FPL API to get live player data
- **Squad Optimization**: Uses PuLP linear programming to build optimal 15-player squads
- **Captaincy Simulation**: Tests every player as captain to find the best choice
- **Starting XI Optimization**: Automatically selects the best 11 players to start
- **Bench Ordering**: Orders bench players for optimal autosub scenarios

### FPL Rules Compliance
- ✅ £100m budget constraint
- ✅ Exactly 15 players: 2 GK, 5 DEF, 5 MID, 3 FWD
- ✅ Maximum 3 players per real team
- ✅ Valid starting XI formation (1 GK, 3+ DEF, 3+ MID, 1+ FWD)
- ✅ Captain gets 2x points, Vice-Captain as backup

### Frontend Features
- **Beautiful Streamlit Interface**: Modern, responsive web application
- **Two Main Pages**:
  - **Best Squad**: Complete squad overview with metrics and visualizations
  - **Captaincy Simulation**: Interactive charts showing captaincy impact
- **Real-time Data**: Cached data with automatic refresh
- **Interactive Visualizations**: Position distribution, team breakdown, captaincy analysis

## 🚀 Quick Start

### Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd fpl-optimizer
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Run the application**:
```bash
streamlit run app.py
```

4. **Open your browser** and navigate to `http://localhost:8501`

### Alternative: Run Individual Modules

Test the data fetching:
```bash
python fetch_data.py
```

Test the optimizer:
```bash
python optimizer.py
```

## 📁 Project Structure

```
fpl-optimizer/
├── app.py              # Streamlit frontend application
├── fetch_data.py       # FPL API data fetching and cleaning
├── optimizer.py        # Linear programming optimization engine
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## 🔧 Technical Details

### Data Processing
- **API Source**: Official FPL API (`https://fantasy.premierleague.com/api/bootstrap-static/`)
- **Data Filtering**: Only available players (`status == "a"`) with 90+ minutes played
- **Fields Used**: id, name, team, position, cost, predicted_points (`ep_next`)

### Optimization Algorithm
- **Method**: Linear Programming using PuLP
- **Objective**: Maximize total predicted points
- **Constraints**: Budget, squad size, position limits, team limits
- **Starting XI**: Secondary optimization for best 11 players

### Captaincy Analysis
- **Simulation**: Tests each squad player as captain (2x points)
- **Selection**: Automatically picks best captain and vice-captain
- **Visualization**: Interactive bar charts showing impact

## 📊 Key Metrics

The system provides comprehensive analysis including:
- **Total Predicted Points**: Squad's expected performance
- **Budget Utilization**: How much of £100m is spent
- **Position Distribution**: Visual breakdown of squad composition
- **Team Distribution**: Players per Premier League team
- **Captaincy Impact**: Points difference with different captains

## 🎮 Usage Guide

### Page 1: Best Squad
1. View the optimal 15-player squad
2. See starting XI and bench order
3. Check position and team distributions
4. Review key metrics (points, cost, budget remaining)

### Page 2: Captaincy Simulation
1. Analyze captaincy impact on team total
2. Compare all players as potential captains
3. View detailed captaincy results table
4. See automatic captain/vice-captain selection

## 🔄 Data Updates

- **Cache Duration**: 1 hour (configurable)
- **Automatic Refresh**: Data updates when cache expires
- **Real-time API**: Always uses latest FPL data

## 🛠️ Customization

### Modify Constraints
Edit `optimizer.py` to change:
- Budget limit (default: £100m)
- Position limits
- Team limits
- Starting XI formation

### Add Features
- Extend `fetch_data.py` for additional player stats
- Modify `optimizer.py` for new optimization objectives
- Enhance `app.py` with additional visualizations

## 📈 Performance

- **Optimization Time**: Typically 2-5 seconds
- **Data Fetching**: 1-3 seconds (depends on API response)
- **Memory Usage**: Minimal (cached data ~5MB)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **FPL Official API**: For providing the data
- **PuLP**: For linear programming optimization
- **Streamlit**: For the web interface
- **Plotly**: For interactive visualizations

## 🐛 Troubleshooting

### Common Issues

**API Connection Error**:
- Check internet connection
- Verify FPL API is accessible
- Try refreshing the page

**Optimization Fails**:
- Ensure sufficient players meet criteria
- Check position constraints are valid
- Verify budget constraints

**Display Issues**:
- Clear browser cache
- Update Streamlit: `pip install --upgrade streamlit`
- Check browser compatibility

### Support

For issues or questions:
1. Check the troubleshooting section
2. Review the code comments
3. Open an issue on GitHub

---

**Happy FPL Managing!** ⚽🎯