import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import time

st.set_page_config(page_title="F1 Strategy Simulator", page_icon="🏎️", layout="wide")

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .stApp { background-color: #0f0f0f; color: #ffffff; }
  .metric-box { background: #1a1a1a; border: 1px solid #333; border-radius: 8px; padding: 16px; text-align: center; }
  .metric-label { font-size: 12px; color: #888; text-transform: uppercase; letter-spacing: 1px; }
  .metric-value { font-size: 28px; font-weight: 700; color: #e8002d; }
  .rec-box { padding: 16px; border-radius: 8px; font-size: 16px; font-weight: 600; margin: 12px 0; }
  .rec-pit { background: #3d0000; border: 1px solid #e8002d; color: #ff4444; }
  .rec-soon { background: #3d2000; border: 1px solid #ff8800; color: #ffaa44; }
  .rec-stay { background: #003d00; border: 1px solid #00cc44; color: #44ff88; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
  .stButton > button { background-color: #1a1a1a !important; color: #ffffff !important; border: 1px solid #555 !important; font-weight: 600 !important; padding: 8px 16px !important; }
  .stButton > button:hover { background-color: #e8002d !important; border-color: #e8002d !important; color: #ffffff !important; }
  .stSelectbox label, .stSlider label { color: #cccccc !important; font-size: 13px !important; }
  div[data-testid="stMetricValue"] { color: #e8002d !important; }
  div[data-testid="stMetricLabel"] { color: #888888 !important; }
  .stToggle label { color: #ffffff !important; font-weight: 600 !important; }
  p { color: #cccccc !important; }
  label { color: #cccccc !important; }
  .stToggle p { color: #ffffff !important; }
</style>
""", unsafe_allow_html=True)

# ── Load Data ────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    results = pd.read_csv("results.csv")
    drivers = pd.read_csv("drivers.csv")
    races   = pd.read_csv("races.csv")[["raceId","year","name"]]
    drivers["driver_name"] = drivers["forename"] + " " + drivers["surname"]
    return results.merge(drivers[["driverId","driver_name"]], on="driverId").merge(races, on="raceId")

df = load_data()

# ── Session State ────────────────────────────────────────────────────────────
defaults = dict(
    lap=1, running=False, pit_this_lap=False,
    tyre_age=0, tyre_type="Medium", position=5,
    gap_ahead=3.2, gap_behind=5.1, fuel=100.0,
    weather="Dry", safety_car=False,
    history=[]
)
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Tyre Model ───────────────────────────────────────────────────────────────
TYRE_LIFE  = {"Soft": 20, "Medium": 30, "Hard": 40, "Wet": 25, "Inter": 30}
TYRE_PACE  = {"Soft": 0,  "Medium": 0.4, "Hard": 0.9, "Wet": 1.5, "Inter": 0.8}

def tyre_performance(tyre, age):
    limit = TYRE_LIFE[tyre]
    pct = age / limit
    if pct < 0.5:  return 100 - (pct * 20)
    if pct < 0.8:  return 90  - (pct * 40)
    return max(0,  80  - (pct * 80))

def lap_time_delta(tyre, age, fuel):
    base = TYRE_PACE[tyre]
    deg  = (age / TYRE_LIFE[tyre]) * 2.5
    fuel_effect = (fuel / 100) * 0.35
    return round(base + deg + fuel_effect, 3)

# ── Strategy Engine ──────────────────────────────────────────────────────────
def strategy_recommendation(lap, tyre, age, pos, gap_ahead, gap_behind, weather, safety_car, fuel):
    limit   = TYRE_LIFE[tyre]
    pct     = (age / limit) * 100
    perf    = tyre_performance(tyre, age)
    delta   = lap_time_delta(tyre, age, fuel)
    laps_left = 70 - lap

    # Weather change
    if weather in ["Wet", "Inter"] and tyre not in ["Wet", "Inter"]:
        return "🔴 PIT NOW", "Wrong tyre for conditions. Box immediately.", "pit", pct, perf, delta

    # Safety car window
    if safety_car:
        return "🔴 PIT NOW (SC)", "Safety car deployed — free pit stop window. Box now.", "pit", pct, perf, delta

    # Critically worn
    if pct >= 90:
        return "🔴 PIT NOW", f"{tyre} tyres critically worn ({age} laps). Box this lap.", "pit", pct, perf, delta

    # Undercut opportunity
    if pct >= 60 and gap_ahead < 2.5 and gap_behind > 4:
        return "🟡 UNDERCUT", f"Gap ahead only {gap_ahead}s. Pit now to undercut P{pos-1}.", "soon", pct, perf, delta

    # Overcut opportunity
    if pct >= 60 and gap_ahead > 4 and gap_behind < 2:
        return "🟡 OVERCUT", f"Car behind {gap_behind}s back. Stay out and overcut.", "soon", pct, perf, delta

    # Pit soon
    if pct >= 65 and gap_behind > 5:
        return "🟡 PIT SOON", f"Tyres at {pct:.0f}%. Safe window — pit in 2-3 laps.", "soon", pct, perf, delta

    # Traffic on exit
    if pct >= 65 and gap_behind <= 3:
        return "🟠 HOLD", f"Tyres worn but gap behind tight ({gap_behind}s). Wait.", "soon", pct, perf, delta

    # End of race — no point pitting
    if laps_left < 8 and pct < 85:
        return "🟢 STAY OUT", f"Only {laps_left} laps left. Nurse the tyres home.", "stay", pct, perf, delta

    # All good
    return "🟢 STAY OUT", f"Tyres at {pct:.0f}% — no action needed.", "stay", pct, perf, delta

# ── Simulate One Lap ─────────────────────────────────────────────────────────
def simulate_lap():
    s = st.session_state
    if s.lap >= 70:
        s.running = False
        return

    s.tyre_age += 1
    s.fuel     = max(0, s.fuel - (100 / 70))

    # Random-ish gap drift
    import random
    s.gap_ahead  = max(0.1, s.gap_ahead  + random.uniform(-0.3, 0.3))
    s.gap_behind = max(0.1, s.gap_behind + random.uniform(-0.3, 0.3))

    decision, reason, rec_type, pct, perf, delta = strategy_recommendation(
        s.lap, s.tyre_type, s.tyre_age, s.position,
        s.gap_ahead, s.gap_behind, s.weather, s.safety_car, s.fuel
    )

    s.history.append({
        "lap": s.lap, "position": s.position,
        "tyre_type": s.tyre_type, "tyre_age": s.tyre_age,
        "tyre_pct": round(pct, 1), "tyre_perf": round(perf, 1),
        "gap_ahead": round(s.gap_ahead, 2), "gap_behind": round(s.gap_behind, 2),
        "fuel": round(s.fuel, 1), "delta": delta,
        "recommendation": decision, "safety_car": s.safety_car
    })

    if s.pit_this_lap:
        s.tyre_age   = 0
        s.position   = min(20, s.position + 2)
        s.pit_this_lap = False

    s.lap += 1

# ── UI ───────────────────────────────────────────────────────────────────────
st.markdown("## 🏎️ F1 Race Strategy Simulator")
st.markdown("---")

# Controls row
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    if st.button("▶ Play" if not st.session_state.running else "⏸ Pause", use_container_width=True):
        st.session_state.running = not st.session_state.running
with col2:
    if st.button("⏹ Reset", use_container_width=True):
        for k, v in defaults.items():
            st.session_state[k] = v
        st.rerun()
with col3:
    if st.button("🔧 Pit This Lap", use_container_width=True):
        st.session_state.pit_this_lap = True
with col4:
    sc = st.toggle("🟡 Safety Car", value=st.session_state.safety_car)
    st.session_state.safety_car = sc
with col5:
    weather = st.selectbox("Weather", ["Dry","Wet","Inter"], index=["Dry","Wet","Inter"].index(st.session_state.weather))
    st.session_state.weather = weather

st.markdown("---")

# Manual overrides (visible when paused)
if not st.session_state.running:
    st.markdown("#### ⚙️ Adjust Conditions")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.session_state.tyre_type = st.selectbox("Tyre Compound", ["Soft","Medium","Hard","Wet","Inter"],
            index=["Soft","Medium","Hard","Wet","Inter"].index(st.session_state.tyre_type))
    with c2:
        st.session_state.position = st.slider("Position", 1, 20, st.session_state.position)
    with c3:
        st.session_state.gap_ahead = st.slider("Gap Ahead (s)", 0.0, 30.0, float(st.session_state.gap_ahead), 0.1)
    with c4:
        st.session_state.gap_behind = st.slider("Gap Behind (s)", 0.0, 30.0, float(st.session_state.gap_behind), 0.1)
    st.markdown("---")

# Current metrics
s = st.session_state
decision, reason, rec_type, pct, perf, delta = strategy_recommendation(
    s.lap, s.tyre_type, s.tyre_age, s.position,
    s.gap_ahead, s.gap_behind, s.weather, s.safety_car, s.fuel
)

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Lap", f"{s.lap}/70")
m2.metric("Position", f"P{s.position}")
m3.metric("Tyre", f"{s.tyre_type} — {s.tyre_age} laps")
m4.metric("Tyre Life", f"{pct:.0f}%")
m5.metric("Gap Ahead", f"{s.gap_ahead:.1f}s")
m6.metric("Gap Behind", f"{s.gap_behind:.1f}s")

# Recommendation
css_class = {"pit": "rec-pit", "soon": "rec-soon", "stay": "rec-stay"}[rec_type]
st.markdown(f'<div class="rec-box {css_class}">{decision} — {reason}</div>', unsafe_allow_html=True)

# Lap time delta
st.caption(f"Lap time delta vs optimal: +{delta}s | Fuel load: {s.fuel:.1f}%")

st.markdown("---")

# Charts
if s.history:
    hist_df = pd.DataFrame(s.history)

    c1, c2 = st.columns(2)

    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist_df["lap"], y=hist_df["tyre_perf"],
            mode="lines+markers", name="Tyre Performance",
            line=dict(color="#e8002d", width=2)))
        fig.update_layout(title="Tyre Performance", paper_bgcolor="#1a1a1a",
            plot_bgcolor="#1a1a1a", font_color="#ffffff", height=280,
            margin=dict(l=40,r=20,t=40,b=30))
        fig.update_xaxes(gridcolor="#333")
        fig.update_yaxes(gridcolor="#333", range=[0,105])
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=hist_df["lap"], y=hist_df["gap_ahead"],
            mode="lines", name="Gap Ahead", line=dict(color="#00aaff", width=2)))
        fig2.add_trace(go.Scatter(x=hist_df["lap"], y=hist_df["gap_behind"],
            mode="lines", name="Gap Behind", line=dict(color="#ffaa00", width=2, dash="dash")))
        fig2.update_layout(title="Gaps (seconds)", paper_bgcolor="#1a1a1a",
            plot_bgcolor="#1a1a1a", font_color="#ffffff", height=280,
            margin=dict(l=40,r=20,t=40,b=30))
        fig2.update_xaxes(gridcolor="#333")
        fig2.update_yaxes(gridcolor="#333")
        st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=hist_df["lap"], y=hist_df["delta"],
            mode="lines", name="Lap Delta",
            line=dict(color="#ff6600", width=2), fill="tozeroy",
            fillcolor="rgba(255,102,0,0.15)"))
        fig3.update_layout(title="Lap Time Delta vs Optimal (+s)", paper_bgcolor="#1a1a1a",
            plot_bgcolor="#1a1a1a", font_color="#ffffff", height=280,
            margin=dict(l=40,r=20,t=40,b=30))
        fig3.update_xaxes(gridcolor="#333")
        fig3.update_yaxes(gridcolor="#333")
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=hist_df["lap"], y=hist_df["fuel"],
            mode="lines", name="Fuel Load",
            line=dict(color="#44ff88", width=2), fill="tozeroy",
            fillcolor="rgba(68,255,136,0.1)"))
        fig4.update_layout(title="Fuel Load (%)", paper_bgcolor="#1a1a1a",
            plot_bgcolor="#1a1a1a", font_color="#ffffff", height=280,
            margin=dict(l=40,r=20,t=40,b=30))
        fig4.update_xaxes(gridcolor="#333")
        fig4.update_yaxes(gridcolor="#333", range=[0,105])
        st.plotly_chart(fig4, use_container_width=True)

    # Lap history table
    st.markdown("#### Lap History")
    display_cols = ["lap","position","tyre_type","tyre_age","tyre_pct","gap_ahead","gap_behind","fuel","delta","recommendation"]
    st.dataframe(hist_df[display_cols].tail(20), use_container_width=True, hide_index=True)

# ── Auto-advance when running ────────────────────────────────────────────────
if st.session_state.running and st.session_state.lap < 70:
    simulate_lap()
    time.sleep(0.8)
    st.rerun()