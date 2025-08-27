# ⚽ FPL AI Optimizer

A comprehensive Fantasy Premier League (FPL) optimization tool that uses machine learning and advanced analytics to help you build the best possible squad for each gameweek.

## 🚀 Features

### 🤖 **Machine Learning Enhanced Predictions**
- **Advanced ML Models**: Random Forest, XGBoost, LightGBM, Gradient Boosting, Ridge, and Linear Regression
- **Feature Engineering**: Player abilities, team strength, fixture difficulty, and availability metrics
- **Realistic Predictions**: Form-based adjustments, minutes played analysis, and position-specific volatility
- **Model Persistence**: Trained models are saved and persist across app refreshes

### 📊 **Comprehensive Squad Optimization**
- **15-Player Squad Builder**: Optimizes under FPL rules (budget, position limits, team limits)
- **Captaincy Analysis**: Top 5 players with highest potential, considering fixture difficulty
- **Bench Ordering**: Automatic bench ordering with autosubstitution logic
- **Form Integration**: Uses last 3-5 matches for form-based optimization
- **Fixture Difficulty**: Incorporates opponent strength and home/away factors

### 📈 **Gameweek Analytics**
- **Past Performance Analysis**: Compare predicted vs actual performance
- **Top Performers Tracking**: Monitor best and worst performers
- **Position Analysis**: Breakdown by goalkeeper, defender, midfielder, forward
- **Over/Under Performers**: Identify players exceeding or falling short of expectations

### 🎯 **Captaincy & Vice-Captaincy**
- **Smart Captain Selection**: Fixture difficulty, form, and minutes played analysis
- **Vice-Captain Logic**: Proper FPL rules implementation (only doubles if captain doesn't play)
- **Captaincy Insights**: Detailed analysis of top captaincy candidates

### 🔍 **Player Comparisons**
- **Head-to-Head Analysis**: Compare any two players across multiple metrics
- **Position-Specific Metrics**: Relevant stats for each position
- **Cost-Benefit Analysis**: Value for money calculations

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/fpl-ai-optimizer.git
   cd fpl-ai-optimizer
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser**
   Navigate to `http://localhost:8501` (or the URL shown in the terminal)

## 📋 Requirements

The following packages are required (see `requirements.txt`):

- **Core**: streamlit, pandas, numpy
- **Optimization**: pulp (for linear programming)
- **Machine Learning**: scikit-learn, xgboost, lightgbm
- **Visualization**: plotly
- **Data Handling**: requests, joblib
- **Utilities**: typing, dataclasses

## 🎮 Usage

### 1. **Best Squad Page**
- View your optimized 15-player squad
- See captain and vice-captain selections
- Analyze squad breakdown by position and team
- View pitch visualization of starting XI

### 2. **Captaincy Analysis Page**
- Get top 5 captaincy candidates for next gameweek
- Detailed insights on fixture difficulty and form
- Captaincy simulation with points breakdown

### 3. **Bench & Autosubs Page**
- View bench ordering and autosubstitution logic
- Understand how bench players will be used

### 4. **Player Comparisons Page**
- Compare any two players head-to-head
- Analyze value for money and performance metrics

### 5. **Gameweek Explorer Page**
- **Gameweek Insights**: Analyze past and current gameweek performance
- **Next Gameweek Squad**: Predictive squad for upcoming gameweek with ML enhancements

### 6. **ML Models Page**
- Train and manage machine learning models
- View model performance metrics
- Collect historical data for better predictions
- Retrain models with new data

## 🔧 Technical Details

### **Architecture**
- **Frontend**: Streamlit web application
- **Backend**: Python with modular design
- **Optimization**: Linear Programming using PuLP
- **ML Pipeline**: Scikit-learn with ensemble methods
- **Data Source**: Official FPL API

### **Key Components**
- `app.py`: Main Streamlit application
- `fetch_data.py`: FPL API data fetching and parsing
- `optimizer.py`: Squad optimization logic
- `predictive_models.py`: Machine learning models and predictions

### **Data Sources**
- **FPL API**: Player stats, fixtures, live data
- **Historical Data**: Gameweek performance tracking
- **Feature Engineering**: Advanced metrics calculation

## 📊 Prediction Features

### **Player Ability Metrics**
- Form (last 3-5 matches average)
- Goals and assists per minute
- Goal involvement rate
- Creativity and threat scores
- Consistency metrics

### **Team Strength Analysis**
- Team attack and defense ratings
- Goals scored/conceded
- Clean sheet potential
- Team form and possession estimates

### **Fixture Context**
- Opponent defensive/attacking strength
- Home/away fixture difficulty
- Historical performance vs specific teams

### **Availability & Risk**
- Minutes played analysis
- Rotation risk assessment
- Injury risk factors
- Fitness scores

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **FPL API**: For providing comprehensive player and fixture data
- **Streamlit**: For the excellent web app framework
- **Scikit-learn**: For the robust machine learning tools
- **PuLP**: For the optimization capabilities

## 📞 Support

If you encounter any issues or have questions:
1. Check the [Issues](https://github.com/yourusername/fpl-ai-optimizer/issues) page
2. Create a new issue with detailed information
3. Include error messages and steps to reproduce

---

**⚽ Happy FPL Managing! May your captain always return! 🏆**