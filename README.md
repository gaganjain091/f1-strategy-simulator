# 🏎️ F1 Strategy Simulator

**[Live Demo](https://f1-strategy-simulator-26.streamlit.app/)** | Real-time race simulation with AI-powered pit stop recommendations

---

## Overview

F1 Strategy Simulator is an interactive web application that models Formula 1 race dynamics in real-time. Drive as Max Verstappen, manage tyres and fuel, execute pit stops strategically, and compete against the full 2024 grid. The app uses **machine learning models** trained on historical pit stop and overtaking data to recommend optimal strategies tailored to each of the **24 real F1 circuits**.

### Key Capabilities
- **Live Race Simulation** — lap-by-lap race progression with 10 AI-driven competitor cars
- **Strategy Engine** — rule-based recommendations (pit timing, tyre compounds, gap management)
- **ML Predictions** — scikit-learn models predict pit stop probability and position gains
- **24 Real Circuits** — track-specific data including tyre wear, pit lane loss, SC probability, overtake difficulty
- **Live Timing Display** — real-time leaderboard, gap progression, tyre strategy
- **Interactive Charts** — Plotly visualizations of tyre performance, position, fuel load, overtake probability

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| **Frontend** | [Streamlit](https://streamlit.io/) (Python web framework) |
| **Visualizations** | [Plotly](https://plotly.com/) (interactive charts) |
| **ML Models** | [scikit-learn](https://scikit-learn.org/) (pit & overtaking prediction) |
| **Data Processing** | [Pandas](https://pandas.pydata.org/), NumPy |
| **Data Source** | [Kaggle F1 Dataset](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020) (pit stops, overtakes, race history) |
| **Language** | Python 3.14+ |
| **Deployment** | Streamlit Cloud |

---

## Features

### 🏁 Core Gameplay
- **Player-Controlled Race** — Start/pause/reset race; pit on-demand or auto-pit on critical wear
- **Dynamic Weather** — Random rain/damp conditions trigger tyre changes (Dry/Wet/Intermediate)
- **Safety Cars** — Track-specific SC probability; free pit window when deployed
- **Position Gains/Losses** — Overtaking/defending logic based on gap and tyre delta

### 🤖 AI & Strategy
- **Pit Stop Predictor** — ML model suggests pit probability based on lap, fuel, tyre age, gap
- **Position Gain Forecast** — Predict how many positions you'll gain/lose if you pit now
- **Rule-Based Engine** — Strategy recommendations (undercut, overcut, stay out, pit soon)
- **10 AI Drivers** — Realistic competitor pit timing and tyre strategies

### 📊 Data & Visualization
- **Live Timing Table** — Position, driver, gap to leader, tyre age/compound, pit count
- **Multi-Tab Layout**
  - 🏁 **Live Timing** — real-time leaderboard
  - 📊 **Charts** — tyre perf, position, gap trends, overtake probability
  - 📋 **Lap History** — detailed lap-by-lap data export
- **Track-Specific Info** — Description, best strategy, pit loss time, SC chance

### 🏆 24 Real F1 Circuits
Bahrain, Saudi Arabia, Australia, Japan, China, Miami, Monaco, Spain, Austria, Silverstone, Hungary, Spa, Monza, Singapore, Mexico, Austin, Brazil, Abu Dhabi, Las Vegas, Qatar, Suzuka, Baku, Hungary, Canada—each with unique:
- Lap count
- Overtake difficulty (Easy/Medium/Hard)
- Tyre wear multiplier
- Pit lane loss (time penalty in seconds)
- Safety car probability

---

## Quick Start

### Prerequisites
- Python 3.10+
- pip or conda

### Local Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/gaganjain091/f1-strategy-simulator.git
   cd f1-strategy-simulator
   ```

2. **Create a virtual environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app**
   ```bash
   streamlit run app.py
   ```

5. **Open in browser**
   ```
   Local URL: http://localhost:8501
   ```

---

## Project Structure

```
f1-strategy-simulator/
├── app.py                 # Main Streamlit app (race logic, UI, rendering)
├── tracks.py              # 24 F1 circuits with track-specific data
├── train_model.py         # ML model training pipeline
├── pit_model.pkl          # Trained pit stop probability model
├── pos_model.pkl          # Trained position gain model
├── requirements.txt       # Python dependencies
├── *.csv                  # Kaggle F1 dataset (pit stops, drivers, races, etc.)
└── README.md              # This file
```

---

## How It Works

### Race Loop
1. **Init** → Load track, initialize player car (P5, Soft tyres, 100% fuel)
2. **Tick** → Each lap: age tyres, consume fuel, update AI positions, check collisions
3. **Strategy** → Evaluate pit timing vs. gap/tyre wear; run ML models
4. **Pit Logic** → Auto-pit if weather changes or tyres critical; user can pit on-demand
5. **Overtaking** → Random chance to pass ahead/defend based on gap and tyre delta
6. **Repeat** → Continue until lap 57 (or track-specific lap count)

### Strategy Recommendation Engine
The rule-based strategy recommends action based on:
- **Tyre wear %** (0–100%)
- **Gap to leader** (seconds ahead)
- **Gap behind** (gap from follower)
- **Weather condition** (Dry/Damp/Heavy)
- **Safety car deployed?**
- **Fuel level**
- **Overtake probability** (track difficulty + tyre delta)
- **Laps remaining**

Example rules:
- If tyre wear ≥92% → 🔴 PIT NOW (critical)
- If gap ahead <2s, gap behind >5s, wear ≥58% → 🟡 UNDERCUT (opportune)
- If only 8 laps left, wear <88% → 🟢 STAY OUT (finish safely)

### ML Models
Two scikit-learn models trained on historical Kaggle F1 data:

**Pit Stop Predictor**
- Input: lap, total_laps, laps_left, lap_pct, tyre_age, tyre_pct, fuel, grid_position
- Output: pit probability (0–100%)
- Use: Suggest pit stop likelihood in current race state

**Position Gain Forecast**
- Input: grid_position, pit_lap_pct, pit_status, total_laps
- Output: expected position change (positive = gain, negative = loss)
- Use: Show impact of pitting now vs. staying out

---

## Usage Tips

### Gameplay
- **Manage Fuel** — Fuel depletes 1.4% per lap; plan pit stops accordingly
- **Monitor Tyres** — Watch tyre wear % in metrics; pit before >90%
- **Read the Gap** — Gap ahead <1s? You might undercut. Gap behind tight? Hold position.
- **Weather Pivot** — When rain appears, pit immediately for Wet tyres
- **Trust the ML** — If ML pit probability >60%, the model sees a strong pit window

### Tuning (Paused Mode)
- Adjust **tyre compound** mid-race (if paused)
- Change **position** and **gaps** to explore scenarios
- Test different **track difficulties**

---

## Roadmap

### Phase 2 (Frontend)
- [ ] React/TypeScript frontend replacement for better UX
- [ ] Dark mode toggle
- [ ] Mobile-responsive design

### Phase 3 (Analytics)
- [ ] Historical race comparison — load a real 2024 race and compare your strategy to actual winner
- [ ] Pit stop heatmaps — when/where did drivers pit in real races?
- [ ] PDF export — generate strategy report (gaps, recommendations, lap history)

### Phase 4 (ML Enhancements)
- [ ] XGBoost models for better overtaking prediction
- [ ] Real-time model retraining on each session
- [ ] Driver personality profiles (aggressive vs. defensive)

---

## Data Source

Race data sourced from [Kaggle F1 Dataset](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020):
- `pit_stops.csv` — Historical pit stop timings, lap numbers, tyre compounds
- `drivers.csv`, `races.csv` — Driver and race metadata
- `results.csv` — Final race positions and points
- `qualifying.csv` — Grid positions

---

## Performance

- **Race Tick Speed** — ~300ms per lap (normal mode), configurable (Slow/Normal/Fast)
- **Chart Rendering** — Real-time Plotly updates, ~100 laps stored in history
- **ML Inference** — <50ms per prediction on standard hardware
- **Deployment** — Streamlit Cloud; cold start ~3-5s, warm loads <1s

---

## Contributing

Contributions welcome! Areas to improve:
- Additional circuit data accuracy
- ML model refinements (more features, better hyperparameters)
- New strategy recommendation rules
- UI/UX polish
- Bug reports and feature requests

---

## License

MIT License — see LICENSE file for details.

---

## Author

**Gagan Jain** — Product Manager & Data Enthusiast  
[GitHub](https://github.com/gaganjain091) | [LinkedIn](#)

---

## Acknowledgments

- **Kaggle** for the comprehensive F1 dataset
- **Streamlit** for the simple, powerful web framework
- **Plotly** for beautiful interactive visualizations
- **scikit-learn** for accessible ML tools
- Formula 1 fans and data enthusiasts who inspired this project

---

**Enjoy your race! 🏁**