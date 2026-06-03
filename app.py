import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="F1 Strategy Simulator", page_icon="🏎️", layout="wide")

# Load data
@st.cache_data
def load_data():
    results = pd.read_csv("results.csv")
    drivers = pd.read_csv("drivers.csv")
    races = pd.read_csv("races.csv")
    races = races[["raceId", "year", "name"]]
    drivers = drivers[["driverId", "forename", "surname"]]
    drivers["driver_name"] = drivers["forename"] + " " + drivers["surname"]
    merged = results.merge(drivers, on="driverId").merge(races, on="raceId")
    return merged

df = load_data()

# Header
st.title("🏎️ F1 Race Strategy Simulator")
st.markdown("Simulate pit stop strategy based on real F1 historical data.")

# Sidebar inputs
st.sidebar.header("Race Conditions")
current_lap = st.sidebar.slider("Current Lap", 1, 70, 30)
current_position = st.sidebar.slider("Current Position", 1, 20, 5)
tyre_age = st.sidebar.slider("Tyre Age (laps)", 1, 40, 15)
tyre_type = st.sidebar.selectbox("Tyre Compound", ["Soft", "Medium", "Hard"])
weather = st.sidebar.selectbox("Weather", ["Dry", "Wet", "Intermediate"])
gap_ahead = st.sidebar.slider("Gap to Car Ahead (seconds)", 0, 30, 5)
gap_behind = st.sidebar.slider("Gap to Car Behind (seconds)", 0, 30, 8)

# Strategy logic
def recommend_strategy(tyre_age, tyre_type, current_lap, current_position, weather, gap_ahead, gap_behind):
    # Tyre life thresholds
    thresholds = {"Soft": 20, "Medium": 30, "Hard": 40}
    limit = thresholds[tyre_type]
    tyre_pct = (tyre_age / limit) * 100

    if weather == "Wet":
        return "⛈️ PIT NOW", "Switch to Wet/Intermediate tyres immediately.", "red"
    if tyre_pct >= 85:
        return "🔴 PIT NOW", f"Your {tyre_type} tyres are critically worn ({tyre_age} laps). Box this lap.", "red"
    if tyre_pct >= 65 and gap_behind > 5:
        return "🟡 PIT SOON", f"Tyres at {tyre_pct:.0f}% life. Safe gap behind ({gap_behind}s). Pit in 2-3 laps.", "orange"
    if tyre_pct >= 65 and gap_behind <= 5:
        return "🟠 HOLD POSITION", f"Tyres worn but gap behind too tight ({gap_behind}s). Wait for a gap.", "orange"
    if current_position <= 3 and tyre_pct < 50:
        return "🟢 STAY OUT", f"You're P{current_position} with fresh enough tyres. Push and stay out.", "green"
    return "🟢 STAY OUT", f"Tyres at {tyre_pct:.0f}% life. No need to pit yet.", "green"

decision, reason, color = recommend_strategy(tyre_age, tyre_type, current_lap, current_position, weather, gap_ahead, gap_behind)

# Main display
col1, col2, col3 = st.columns(3)
col1.metric("Current Lap", f"{current_lap}/70")
col2.metric("Position", f"P{current_position}")
col3.metric("Tyre Age", f"{tyre_age} laps")

st.markdown("---")

# Strategy recommendation
st.subheader("Strategy Recommendation")
if color == "red":
    st.error(f"**{decision}** — {reason}")
elif color == "orange":
    st.warning(f"**{decision}** — {reason}")
else:
    st.success(f"**{decision}** — {reason}")

st.markdown("---")

# Tyre degradation chart
st.subheader("Tyre Degradation Model")
laps = list(range(1, 41))
deg = {
    "Soft":   [max(0, 100 - (l ** 1.6)) for l in laps],
    "Medium": [max(0, 100 - (l ** 1.4)) for l in laps],
    "Hard":   [max(0, 100 - (l ** 1.2)) for l in laps],
}
deg_df = pd.DataFrame(deg, index=laps).reset_index()
deg_df.columns = ["Lap", "Soft", "Medium", "Hard"]
deg_melted = deg_df.melt(id_vars="Lap", var_name="Compound", value_name="Performance")
fig1 = px.line(deg_melted, x="Lap", y="Performance", color="Compound",
               color_discrete_map={"Soft": "#E8593C", "Medium": "#F0C040", "Hard": "#AAAAAA"},
               title="Tyre Performance vs Lap Age")
fig1.add_vline(x=tyre_age, line_dash="dash", line_color="white", annotation_text="Current")
fig1.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig1, use_container_width=True)

st.markdown("---")

# Historical pit stop data
st.subheader("Historical Data Explorer")
year_filter = st.selectbox("Select Season", sorted(df["year"].unique(), reverse=True))
filtered = df[df["year"] == year_filter].copy()
filtered["driver_name"] = filtered["forename"] + " " + filtered["surname"]
top_drivers = filtered.groupby("driver_name")["points"].sum().reset_index()
top_drivers = top_drivers.sort_values("points", ascending=False).head(10)
fig2 = px.bar(top_drivers, x="driver_name", y="points",
              title=f"{year_filter} Season — Top 10 Drivers by Points",
              color="points", color_continuous_scale="Reds")
fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")
st.caption("Built with real F1 Kaggle data · Powered by Streamlit")