# ⚽ FPL AI Optimizer

A comprehensive Fantasy Premier League (FPL) optimization tool that uses **advanced analytics and machine learning** to help you build the best possible squad for each gameweek. **Now featuring comprehensive player analysis that goes beyond recent form!**

## 🚀 Features

### 🔬 **Comprehensive Player Analysis System** ⭐ **NEW!**
- **ICT Index Analysis**: Influence, Creativity, Threat metrics for player ability assessment
- **Expected Goals (xG) & Assists (xA)**: Underlying performance metrics, not just actual returns
- **Availability & Rotation Risk**: Advanced assessment of player availability and rotation risk
- **Team Strength Integration**: Fixture context and opponent analysis
- **Position-Specific Metrics**: Different analysis for GK, DEF, MID, FWD positions
- **Historical Consistency**: Minutes played analysis and reliability assessment
- **Bonus Points Potential**: BPS analysis for additional point potential

### 🤖 **Machine Learning Enhanced Predictions**
- **Advanced ML Models**: Random Forest, XGBoost, LightGBM, Gradient Boosting, Ridge, and Linear Regression
- **Feature Engineering**: Player abilities, team strength, fixture difficulty, and availability metrics
- **Realistic Predictions**: Form-based adjustments, minutes played analysis, and position-specific volatility
- **Model Persistence**: Trained models are saved and persist across app refreshes

### 📊 **Enhanced Squad Optimization**
- **15-Player Squad Builder**: Optimizes under FPL rules (budget, position limits, team limits)
- **Comprehensive Player Selection**: Uses all available metrics, not just recent form
- **Advanced Captaincy Analysis**: Top players with highest comprehensive scores
- **Bench Ordering**: Automatic bench ordering with autosubstitution logic
- **Enhanced Form Integration**: Uses comprehensive form analysis with underlying metrics
- **Fixture Difficulty**: Incorporates opponent strength and home/away factors

### 📈 **Gameweek Analytics**
- **Past Performance Analysis**: Compare predicted vs actual performance
- **Top Performers Tracking**: Monitor best and worst performers
- **Position Analysis**: Breakdown by goalkeeper, defender, midfielder, forward
- **Over/Under Performers**: Identify players exceeding or falling short of expectations

### 🎯 **Advanced Captaincy & Vice-Captaincy**
- **Comprehensive Captain Selection**: ICT Index, xG/xA, availability, and fixture analysis
- **Enhanced Vice-Captain Logic**: Proper FPL rules implementation with comprehensive metrics
- **Detailed Captaincy Insights**: Complete analysis of top captaincy candidates with all metrics

### 🔍 **Player Comparisons**
- **Head-to-Head Analysis**: Compare any two players across comprehensive metrics
- **Position-Specific Metrics**: Relevant stats for each position
- **Cost-Benefit Analysis**: Value for money calculations with enhanced metrics

## 🆕 **What's New: Comprehensive Analysis System**

### **❌ Old System (Limited)**
- Only considered recent form (last 2 gameweeks)
- Basic predicted points only
- Simple fixture difficulty
- No consideration of underlying ability
- Missed high-potential players with poor recent form

### **✅ New System (Comprehensive)**
- **ICT Index** - Influence, Creativity, Threat analysis
- **Expected Goals/Assists** - Underlying performance metrics
- **Availability** - Rotation risk and injury assessment
- **Team Strength** - Fixture context and opponent analysis
- **Position-Specific** - Different metrics for GK/DEF/MID/FWD
- **Historical Consistency** - Minutes played and reliability

### **🎯 Key Benefits**
- **Identifies High-Potential Players**: Finds players with good underlying metrics even if they haven't scored recently
- **Reduces Recency Bias**: Not just picking players who scored in the last 2 weeks
- **Considers Context**: Takes into account fixtures, team strength, and availability
- **Position-Specific**: Different analysis for goalkeepers, defenders, midfielders, and forwards
- **Risk Assessment**: Identifies rotation risks and injury concerns

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
- **Next Gameweek Squad**: **Enhanced predictive squad with comprehensive analysis** - Shows different players based on ICT Index, xG/xA, availability, and team strength, not just recent form

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
- `app.py`: Main Streamlit application with comprehensive analysis UI
- `fetch_data.py`: Enhanced FPL API data fetching with ICT Index, xG/xA, and availability metrics
- `optimizer.py`: Advanced squad optimization with comprehensive player analysis
- `predictive_models.py`: Machine learning models and predictions
- `test_comprehensive_analysis.py`: Test scripts for comprehensive analysis verification
- `test_comprehensive_squad.py`: Comparison tools for old vs new analysis systems

### **Data Sources**
- **FPL API**: Player stats, fixtures, live data
- **Historical Data**: Gameweek performance tracking
- **Feature Engineering**: Advanced metrics calculation

## 📊 Enhanced Prediction Features

### **Comprehensive Player Ability Metrics** ⭐ **NEW!**
- **ICT Index**: Influence, Creativity, Threat analysis for overall player impact
- **Expected Goals (xG)**: Underlying goal-scoring potential, not just actual goals
- **Expected Assists (xA)**: Underlying assist potential, not just actual assists
- **Enhanced Form Analysis**: Comprehensive form assessment with underlying metrics
- **Availability Metrics**: Rotation risk, injury risk, and playing time analysis
- **Position-Specific Analysis**: Different metrics for GK, DEF, MID, FWD positions
- **Historical Consistency**: Minutes played analysis and reliability assessment
- **Bonus Points Potential**: BPS analysis for additional point potential

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

## 🔬 **Comprehensive Analysis System Details**

### **How It Works**
The comprehensive analysis system evaluates players using multiple dimensions:

1. **ICT Index Analysis**: 
   - **Influence**: How much the player affects the game
   - **Creativity**: Chance creation and assist potential
   - **Threat**: Goal-scoring threat and attacking potential

2. **Expected Goals/Assists (xG/xA)**:
   - Uses underlying performance metrics
   - More reliable than actual goals/assists for prediction
   - Considers shot quality, pass quality, and positioning

3. **Availability Assessment**:
   - Rotation risk based on minutes played
   - Injury risk assessment
   - Team selection probability

4. **Position-Specific Analysis**:
   - **Goalkeepers**: Saves, clean sheet potential, save points
   - **Defenders**: Attacking threat, clean sheets, bonus potential
   - **Midfielders**: Creativity, goal threat, assist potential
   - **Forwards**: Goal-scoring potential, assist potential, bonus points

5. **Team Strength Integration**:
   - Fixture difficulty analysis
   - Opponent defensive/attacking strength
   - Home/away advantage consideration

### **Why This Matters**
Traditional FPL analysis often focuses on recent form, which can be misleading. The comprehensive system identifies players with:
- High underlying performance metrics (good xG/xA)
- Strong ICT Index scores (consistent influence)
- Good availability (low rotation risk)
- Favorable fixtures
- Position-specific strengths

This leads to better squad selection and improved FPL performance!

---

**⚽ Happy FPL Managing! May your captain always return! 🏆**

**🎯 Now with comprehensive analysis that goes beyond recent form!**