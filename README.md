# ⚽ FPL Optimizer

A full-stack Fantasy Premier League (FPL) optimization system that uses linear programming to find the optimal 15-player squad to maximize predicted points.

## 🚀 Features

### Core Functionality
- **Real-time Data Fetching**: Automatically fetches current FPL data from the official API
- **Squad Optimization**: Uses PuLP linear programming to find the optimal 15-player squad
- **Captaincy Simulation**: Tests each player as captain to find the best choice
- **Bench Order Optimization**: Automatically orders bench players for optimal autosubs
- **Autosub Simulation**: Simulates bench substitutions when starting XI players don't play

### FPL Rules Compliance
- ✅ Maximum budget: £100.0m
- ✅ Squad size: Exactly 15 players
- ✅ Position requirements: 2 GK, 5 DEF, 5 MID, 3 FWD
- ✅ Maximum 3 players per real team
- ✅ Captain gets 2x points
- ✅ Vice-captain as backup (ideally from different team)

### Frontend Features
- **Best Squad Page**: Displays optimized squad with captain/vice-captain indicators
- **Captaincy Simulation Page**: Interactive bar chart showing team totals for each captain choice
- **Real-time Updates**: Data refreshes automatically every hour
- **Responsive Design**: Beautiful, modern UI with Streamlit

## 📁 Project Structure

```
fpl-optimizer/
├── fetch_data.py      # Data fetching and cleaning module
├── optimizer.py       # Linear programming optimization engine
├── app.py            # Streamlit frontend application
├── requirements.txt   # Python dependencies
└── README.md         # This file
```

## 🛠️ Installation

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

## 🎯 Usage

### Running the Application
1. Start the Streamlit app: `streamlit run app.py`
2. Open your browser to the provided URL (usually `http://localhost:8501`)
3. Navigate between "Best Squad" and "Captaincy Simulation" pages using the sidebar

### Understanding the Output

#### Best Squad Page
- **Squad Table**: Shows all 15 players with their details
- **Starting XI**: Highlighted in blue (top 11 players by predicted points)
- **Bench**: Highlighted in gray (bottom 4 players)
- **Captain**: Marked with ⭐
- **Vice-Captain**: Marked with 🅥
- **Bench Order**: Shows optimal substitution order

#### Captaincy Simulation Page
- **Bar Chart**: Shows team total points if each player is captain
- **Captain Selection**: Automatically picks the player that maximizes team total
- **Vice-Captain**: Picks the next best option, ideally from a different team
- **Detailed Table**: Shows all captaincy scenarios with exact numbers

## 🔧 Technical Details

### Data Processing
- **API Source**: FPL Official API (`https://fantasy.premierleague.com/api/bootstrap-static/`)
- **Player Filtering**: Only includes available players (`status == "a"`) with at least 90 minutes played
- **Data Fields**: id, name, team, position, cost, predicted_points (`ep_next`)

### Optimization Algorithm
- **Method**: Linear Programming using PuLP
- **Objective**: Maximize total predicted points
- **Constraints**: Budget, squad size, position requirements, team limits
- **Solver**: Default PuLP solver (usually CBC)

### Performance
- **Caching**: Data cached for 1 hour to avoid excessive API calls
- **Optimization Time**: Typically completes in 1-3 seconds
- **Memory Usage**: Minimal, processes only required data

## 🎨 Customization

### Modifying Constraints
Edit `optimizer.py` to change:
- Budget limit (`max_budget`)
- Squad size (`squad_size`)
- Position requirements (`position_requirements`)
- Maximum players per team (`max_players_per_team`)

### Adding Features
- **Transfer Planning**: Modify to consider transfer costs
- **Form Integration**: Add recent form data to predictions
- **Fixture Difficulty**: Incorporate fixture difficulty ratings
- **Chip Strategy**: Add support for Triple Captain, Bench Boost, etc.

## 🐛 Troubleshooting

### Common Issues

1. **API Connection Error**:
   - Check internet connection
   - Verify FPL API is accessible
   - Try refreshing data using sidebar button

2. **Optimization Fails**:
   - Ensure sufficient players available in each position
   - Check if budget constraints are too restrictive
   - Verify data quality from API

3. **Streamlit Issues**:
   - Update Streamlit: `pip install --upgrade streamlit`
   - Clear cache: Use "🔄 Refresh Data" button
   - Check browser compatibility

### Debug Mode
Run individual modules for testing:
```bash
python fetch_data.py    # Test data fetching
python optimizer.py     # Test optimization
```

## 📊 Data Sources

- **FPL Official API**: Player data, teams, positions, predicted points
- **Real-time Updates**: Data refreshes automatically
- **Historical Data**: Available through FPL API endpoints

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This tool is for educational and entertainment purposes only. FPL predictions are inherently uncertain, and past performance does not guarantee future results. Always make your own informed decisions when managing your FPL team.

## 🆘 Support

For issues, questions, or feature requests:
1. Check the troubleshooting section above
2. Review existing issues on GitHub
3. Create a new issue with detailed information

---

**Happy FPL Managing!** ⚽🎯