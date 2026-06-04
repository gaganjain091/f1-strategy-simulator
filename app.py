import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import time
import random
from tracks import TRACKS, TRACK_NAMES

import pickle
import numpy as np

# Load ML models
@st.cache_resource
def load_models():
    with open("pit_model.pkl","rb") as f:
        pit_model = pickle.load(f)
    with open("pos_model.pkl","rb") as f:
        pos_model = pickle.load(f)
    return pit_model, pos_model

pit_model, pos_model = load_models()

def ml_pit_probability(lap, total_laps, tyre_age, tyre_pct, fuel, grid_pos):
    laps_left = total_laps - lap
    lap_pct   = lap / total_laps
    features  = [[lap, total_laps, laps_left, lap_pct, tyre_age, tyre_pct, fuel, grid_pos]]
    prob      = pit_model.predict_proba(features)[0]
    pit_prob  = prob[1] if len(prob) > 1 else 0.0
    return round(pit_prob * 100, 1)

def ml_position_gain(grid, pit_lap_pct, total_laps):
    features = [[grid, pit_lap_pct, 1, total_laps]]
    gain     = pos_model.predict(features)[0]
    return round(gain, 1)

st.set_page_config(page_title="F1 Strategy Simulator", page_icon="🏎️", layout="wide")

st.markdown("""
<style>
  .stApp { background-color: #0f0f0f; color: #ffffff; }
  .stButton > button { background-color: #1a1a1a !important; color: #ffffff !important; border: 1px solid #555 !important; font-weight: 600 !important; }
  .stButton > button:hover { background-color: #e8002d !important; border-color: #e8002d !important; }
  .stSelectbox label, .stSlider label { color: #cccccc !important; }
  .stToggle label, .stToggle p { color: #ffffff !important; }
  .stTabs [data-baseweb="tab"] { color: #888 !important; background: #1a1a1a !important; border-radius: 6px 6px 0 0; padding: 8px 20px; }
  .stTabs [aria-selected="true"] { color: #ffffff !important; background: #e8002d !important; }
  div[data-testid="stMetricValue"] { color: #e8002d !important; }
  div[data-testid="stMetricLabel"] { color: #888888 !important; }
  label { color: #cccccc !important; }
  .rec-box { padding: 14px 18px; border-radius: 8px; font-size: 15px; font-weight: 600; margin: 10px 0; }
  .rec-pit  { background:#3d0000; border:1px solid #e8002d; color:#ff4444; }
  .rec-soon { background:#3d2000; border:1px solid #ff8800; color:#ffaa44; }
  .rec-stay { background:#003d00; border:1px solid #00cc44; color:#44ff88; }
  .timing-table { width:100%; border-collapse:collapse; font-size:13px; }
  .timing-table th { background:#1a1a1a; color:#888; padding:8px; text-align:left; border-bottom:1px solid #333; }
  .timing-table td { padding:7px 8px; border-bottom:1px solid #222; color:#ddd; }
  .timing-table tr.player td { background:#1a1a2e; border-left: 3px solid #e8002d; }
  .tyre-S { color:#e8002d; font-weight:700; }
  .tyre-M { color:#f0c040; font-weight:700; }
  .tyre-H { color:#aaaaaa; font-weight:700; }
  .tyre-W { color:#4488ff; font-weight:700; }
  .tyre-I { color:#44cc88; font-weight:700; }
  .sc-banner { background:#3d3d00; border:1px solid #ffcc00; color:#ffcc00; padding:10px 16px; border-radius:6px; font-weight:700; margin:8px 0; }
  .pos-gain { color:#44ff88; font-weight:700; }
  .pos-loss { color:#ff4444; font-weight:700; }
</style>
""", unsafe_allow_html=True)

# ── Constants ────────────────────────────────────────────────────────────────
TYRE_LIFE  = {"S":18,"M":28,"H":40,"W":25,"I":30}
TYRE_PACE  = {"S":0,"M":0.5,"H":1.1,"W":2.0,"I":1.0}
TYRE_NAMES = {"S":"Soft","M":"Medium","H":"Hard","W":"Wet","I":"Inter"}
TYRE_CLASS = {"S":"tyre-S","M":"tyre-M","H":"tyre-H","W":"tyre-W","I":"tyre-I"}
PIT_LOSS   = 23.0
TOTAL_LAPS = 57
TRACK_OVERTAKE = {"Easy":0.75,"Medium":0.45,"Hard":0.18}

DRIVERS = [
    ("VER","Verstappen",95),("HAM","Hamilton",93),("LEC","Leclerc",91),
    ("NOR","Norris",90),("ALO","Alonso",92),("SAI","Sainz",88),
    ("RUS","Russell",87),("PER","Perez",86),("STR","Stroll",80),("OCO","Ocon",82)
]

# ── Helpers ──────────────────────────────────────────────────────────────────
def tyre_perf(tyre, age):
    limit = TYRE_LIFE[tyre]
    pct = age / limit
    if pct < 0.5:  return 100 - pct * 15
    if pct < 0.75: return 92  - pct * 35
    return max(0,  75  - pct * 75)

def lap_delta(tyre, age, fuel):
    base    = TYRE_PACE[tyre]
    deg     = (age / TYRE_LIFE[tyre]) ** 1.8 * 3.0
    fuel_ef = (fuel / 100) * 0.32
    return round(base + deg + fuel_ef, 3)

def choose_tyre(weather, laps_left, position):
    if weather == "Heavy": return "W"
    if weather == "Damp":  return "I"
    if laps_left < 15:     return "S"
    if laps_left < 28:     return "M"
    return "H" if position <= 3 else "M"

def weather_progression(lap, events):
    for start, end, w in events:
        if start <= lap <= end:
            return w
    return "Dry"

def generate_weather_events():
    s = st.session_state
    total_laps = s.get("total_laps", 57)
    events = []
    if random.random() < 0.55:
        lap = random.randint(10, 30)
        dur = random.randint(5, 14)
        w   = random.choice(["Damp","Heavy","Damp"])
        events.append((lap, lap + dur, w))
        if random.random() < 0.35:
            lap2 = lap + dur + random.randint(6, 15)
            dur2 = random.randint(3, 8)
            events.append((lap2, min(lap2+dur2, total_laps), "Damp"))
    return events

def overtake_prob(gap, track_diff, tyre_delta):
    base = TRACK_OVERTAKE[track_diff]
    if gap < 0.5:   base += 0.3
    elif gap < 1.0: base += 0.15
    if tyre_delta > 0.5: base += 0.25
    return min(0.95, max(0.02, base))

# ── Strategy Engine ──────────────────────────────────────────────────────────
def strategy_rec(lap, tyre, age, pos, gap_ahead, gap_behind, weather, sc, fuel, track_diff, laps_left):
    limit   = TYRE_LIFE[tyre]
    pct     = (age / limit) * 100
    perf    = tyre_perf(tyre, age)
    delta   = lap_delta(tyre, age, fuel)
    ot_prob = overtake_prob(gap_ahead, track_diff, delta)

    if weather == "Heavy" and tyre != "W":
        return "🔴 PIT NOW","Rain — switch to Wet tyres immediately.","pit",pct,perf,delta,ot_prob
    if weather == "Damp" and tyre not in ["I","W"]:
        return "🔴 PIT NOW","Damp — switch to Intermediates.","pit",pct,perf,delta,ot_prob
    if sc:
        if pct > 40:
            return "🔴 PIT NOW (SC)",f"Free pit window. Tyre life {pct:.0f}%.","pit",pct,perf,delta,ot_prob
        return "🟢 STAY OUT (SC)","Tyres fresh — gain track position.","stay",pct,perf,delta,ot_prob
    if pct >= 92:
        return "🔴 PIT NOW",f"{TYRE_NAMES[tyre]} critically worn. Box this lap.","pit",pct,perf,delta,ot_prob
    if pct >= 58 and gap_ahead < 2.0 and gap_behind > 5 and laps_left > 10:
        return f"🟡 UNDERCUT ({ot_prob*100:.0f}%)",f"Gap ahead {gap_ahead:.1f}s — pit to undercut. OT prob {ot_prob*100:.0f}%.","soon",pct,perf,delta,ot_prob
    if pct >= 55 and gap_ahead > 5 and gap_behind < 1.5 and laps_left > 12:
        return "🟡 OVERCUT",f"Car behind closing ({gap_behind:.1f}s). Overcut after they pit.","soon",pct,perf,delta,ot_prob
    if pct >= 65 and gap_behind <= 2.5:
        return "🟠 HOLD",f"Tyres worn but tight gap behind ({gap_behind:.1f}s).","soon",pct,perf,delta,ot_prob
    if pct >= 65 and gap_behind > 5:
        return "🟡 PIT SOON",f"Tyres at {pct:.0f}%. Good window — pit in 1-2 laps.","soon",pct,perf,delta,ot_prob
    if laps_left <= 8 and pct < 88:
        return "🟢 STAY OUT",f"Only {laps_left} laps left. Nurse it home.","stay",pct,perf,delta,ot_prob
    return "🟢 STAY OUT",f"Tyres at {pct:.0f}% — no action needed.","stay",pct,perf,delta,ot_prob

# ── Car Init ─────────────────────────────────────────────────────────────────
def init_cars():
    tyres = ["S","S","M","M","M","H","H","S","M","H"]
    cars  = []
    for i,(code,name,skill) in enumerate(DRIVERS):
        cars.append({
            "code":code,"name":name,"skill":skill,
            "pos":i+1,"gap_to_leader":i*1.3,
            "tyre":tyres[i],"tyre_age":0,
            "fuel":100.0,"pits":0,"last_pit":0,
            "is_player": code=="VER"
        })
    return cars

def simulate_cars(cars, lap, weather, sc, track_diff):
    s = st.session_state
    total_laps = s.get("total_laps", 57)
    track = s.get("track", {})
    laps_left = total_laps - lap
    active = [c for c in cars if not c.get("dnf")]

    for car in active:
        car["tyre_age"] += track.get("tyre_wear_multiplier", 1.0)
        car["fuel"] = max(0, car["fuel"] - (100/total_laps))
        tyre = car["tyre"]
        age  = car["tyre_age"]
        pct  = (age / TYRE_LIFE[tyre]) * 100

        # AI pit logic
        should_pit = False
        if weather == "Heavy" and tyre != "W": should_pit = True
        elif weather == "Damp" and tyre not in ["I","W"]: should_pit = True
        elif sc and pct > 45 and laps_left > 8: should_pit = True
        elif pct >= 88: should_pit = True
        elif pct >= 62 and laps_left > 10 and random.random() < 0.25: should_pit = True

        if should_pit and not car["is_player"]:
            car["tyre"]     = choose_tyre(weather, laps_left, car["pos"])
            car["tyre_age"] = 0
            car["pits"]    += 1
            car["last_pit"] = lap
            pit_loss = track.get("pit_lane_loss", 23.0)
            car["gap_to_leader"] += pit_loss * random.uniform(0.88, 1.08)

        # Lap time based on skill + tyre delta
        delta = lap_delta(car["tyre"], car["tyre_age"], car["fuel"])
        skill_bonus = (car["skill"] - 87) * 0.05
        drift = (delta - 1.2 - skill_bonus) * 0.12
        if sc: drift = 0
        car["gap_to_leader"] = max(0, car["gap_to_leader"] + drift + random.uniform(-0.08, 0.08))

    # Re-sort
    active.sort(key=lambda x: x["gap_to_leader"])
    for i,c in enumerate(active): c["pos"] = i+1

    # Compute gaps
    leader_gap = active[0]["gap_to_leader"] if active else 0
    for i,c in enumerate(active):
        c["gap_to_leader_display"] = c["gap_to_leader"] - leader_gap
        c["gap_ahead"]  = (c["gap_to_leader"] - active[i-1]["gap_to_leader"]) if i > 0 else 0
        c["gap_behind"] = (active[i+1]["gap_to_leader"] - c["gap_to_leader"]) if i < len(active)-1 else 99

    return cars

# ── Session State ─────────────────────────────────────────────────────────────
def reset_state(track_name="Bahrain"):
    track = TRACKS.get(track_name, TRACKS["Bahrain"])
    st.session_state.update(dict(
        lap=1, running=False,
        tyre="M", tyre_age=0, position=5,
        gap_ahead=3.2, gap_behind=5.1,
        fuel=100.0, sc=False, weather="Dry",
        track_name=track_name, track=track,
        total_laps=track["laps"], track_diff=track["overtake_difficulty"],
        pit_now=False,
        history=[], pits_taken=0,
        cars=init_cars(),
        wx_events=generate_weather_events(),
        prev_pos=5, positions_gained=0
    ))

if "lap" not in st.session_state:
    reset_state()

# ── Tick ──────────────────────────────────────────────────────────────────────
def tick():
    s = st.session_state
    total_laps = s.get("total_laps", 57)
    track = s.get("track", {})
    if s.lap >= total_laps:
        s.running = False
        return

    laps_left = total_laps - s.lap
    s.weather = weather_progression(s.lap, s.wx_events)
    s.fuel    = max(0, s.fuel - (100/total_laps))
    s.tyre_age += track.get("tyre_wear_multiplier", 1.0)

    # SC trigger
    sc_prob = track.get("sc_probability", 0.04)
    if not s.sc and random.random() < sc_prob: s.sc = True
    elif s.sc and random.random() < 0.30:   s.sc = False

    # Simulate field first
    s.cars = simulate_cars(s.cars, s.lap, s.weather, s.sc, s.track_diff)

    # Sync player car gaps from field
    player = next((c for c in s.cars if c["is_player"]), None)
    if player:
        s.gap_ahead  = max(0.1, player.get("gap_ahead", s.gap_ahead))
        s.gap_behind = max(0.1, player.get("gap_behind", s.gap_behind))

    my_delta = lap_delta(s.tyre, s.tyre_age, s.fuel)

    # Pit this lap?
    tyre_pct  = (s.tyre_age / TYRE_LIFE[s.tyre]) * 100
    auto_pit  = False
    if s.weather == "Heavy" and s.tyre != "W": auto_pit = True
    elif s.weather == "Damp" and s.tyre not in ["I","W"]: auto_pit = True
    elif tyre_pct >= 90: auto_pit = True

    if s.pit_now or auto_pit:
        new_tyre     = choose_tyre(s.weather, laps_left, s.position)
        s.tyre       = new_tyre
        s.tyre_age   = 0
        s.pits_taken += 1
        s.gap_behind = random.uniform(18, 26)
        s.gap_ahead  = random.uniform(2.0, 5.0)
        s.pit_now    = False
        # Drop positions on pit stop
        s.position   = min(20, s.position + random.randint(2, 4))
        if player:
            player["tyre"]     = new_tyre
            player["tyre_age"] = 0
            player["pits"]     = s.pits_taken
            pit_loss = track.get("pit_lane_loss", 23.0)
            player["gap_to_leader"] += pit_loss * random.uniform(0.9, 1.1)

    # Overtaking on track — player tries to pass car ahead
    ot_prob = overtake_prob(s.gap_ahead, s.track_diff, my_delta)
    if s.gap_ahead < 1.0 and random.random() < ot_prob:
        s.position   = max(1, s.position - 1)
        s.gap_ahead  = random.uniform(1.8, 3.5)
        if player:
            player["pos"] = s.position
            # Move player up in gap_to_leader
            active = sorted([c for c in s.cars], key=lambda x: x["pos"])
            for i, c in enumerate(active):
                if c["is_player"] and i > 0:
                    player["gap_to_leader"] = active[i-1]["gap_to_leader"] - random.uniform(0.2, 0.8)

    # Being overtaken
    if s.gap_behind < 0.5 and random.random() < TRACK_OVERTAKE[s.track_diff] * 0.4:
        s.position   = min(20, s.position + 1)
        s.gap_behind = random.uniform(1.5, 3.0)
        if player: player["pos"] = s.position

    # Track position gain/loss
    s.positions_gained += (s.prev_pos - s.position)
    s.prev_pos = s.position

    if player: player["pos"] = s.position

    dec,reason,rec_type,pct,perf,delta,ot_prob = strategy_rec(
        s.lap,s.tyre,s.tyre_age,s.position,
        s.gap_ahead,s.gap_behind,s.weather,s.sc,s.fuel,
        s.track_diff,laps_left
    )

    s.history.append({
        "lap":s.lap,"pos":s.position,
        "tyre":s.tyre,"tyre_age":s.tyre_age,
        "tyre_pct":round(pct,1),"tyre_perf":round(perf,1),
        "gap_ahead":round(s.gap_ahead,2),"gap_behind":round(s.gap_behind,2),
        "fuel":round(s.fuel,1),"delta":delta,
        "weather":s.weather,"sc":s.sc,
        "rec":dec,"ot_prob":round(ot_prob*100,1)
    })

    s.lap += 1

# ── UI ────────────────────────────────────────────────────────────────────────
s = st.session_state
total_laps = s.get("total_laps", 57)
laps_left = total_laps - s.lap

st.markdown("## 🏎️ F1 Race Strategy Simulator")
if s.sc:
    st.markdown('<div class="sc-banner">🟡 SAFETY CAR DEPLOYED</div>', unsafe_allow_html=True)

# Track info display
if s.get("track"):
    track = s["track"]
    track_name = s.get("track_name", "Bahrain")
    flag = track.get("flag", "🏁")
    st.markdown(f"**{flag} {track_name}** — {track.get('description', 'N/A')} | Best strat: {track.get('best_strategy', 'N/A')} | Pit loss: {track.get('pit_lane_loss', 23.0):.1f}s | SC chance: {track.get('sc_probability', 0.04)*100:.1f}%")

wx_icon = {"Dry":"☀️","Damp":"🌦️","Heavy":"⛈️"}.get(s.weather,"☀️")
st.caption(f"{wx_icon} Weather: **{s.weather}** | Track: **{s.track_diff}** | Laps left: **{laps_left}** | Positions gained: **{s.positions_gained:+d}**")

# Controls
c1,c2,c3,c4,c5,c6 = st.columns(6)
with c1:
    if st.button("▶ Play" if not s.running else "⏸ Pause", use_container_width=True):
        s.running = not s.running
with c2:
    if st.button("⏹ Reset", use_container_width=True):
        reset_state(); st.rerun()
with c3:
    if st.button("🔧 Pit Now", use_container_width=True):
        s.pit_now = True
with c4:
    s.sc = st.toggle("🟡 SC", value=s.sc)
with c5:
    selected_track = st.selectbox("Circuit", TRACK_NAMES,
        index=TRACK_NAMES.index(s.get("track_name", "Bahrain")))
    if selected_track != s.get("track_name", "Bahrain"):
        reset_state(selected_track)
        st.rerun()
with c6:
    spd = st.selectbox("Speed", ["Slow","Normal","Fast"], index=1)

sleep_map = {"Slow":2.0,"Normal":1.2,"Fast":0.3}

# Pause adjust
if not s.running:
    st.markdown("#### ⚙️ Adjust Conditions")
    a1,a2,a3,a4 = st.columns(4)
    with a1:
        s.tyre = st.selectbox("Compound",["S","M","H","W","I"],
            index=["S","M","H","W","I"].index(s.tyre),
            format_func=lambda x: TYRE_NAMES[x])
    with a2:
        s.position = st.slider("Position",1,20,s.position)
    with a3:
        s.gap_ahead = st.slider("Gap Ahead (s)",0.0,30.0,float(s.gap_ahead),0.1)
    with a4:
        s.gap_behind = st.slider("Gap Behind (s)",0.0,30.0,float(s.gap_behind),0.1)

st.markdown("---")

# Metrics
dec,reason,rec_type,pct,perf,delta,ot_prob = strategy_rec(
    s.lap,s.tyre,s.tyre_age,s.position,
    s.gap_ahead,s.gap_behind,s.weather,s.sc,s.fuel,
    s.track_diff,laps_left
)

m1,m2,m3,m4,m5,m6,m7 = st.columns(7)
m1.metric("Lap", f"{s.lap}/{total_laps}")
m2.metric("Position", f"P{s.position}")
m3.metric("Tyre", TYRE_NAMES[s.tyre])
m4.metric("Tyre Life", f"{pct:.0f}%")
m5.metric("Gap Ahead", f"{s.gap_ahead:.1f}s")
m6.metric("Gap Behind", f"{s.gap_behind:.1f}s")
m7.metric("OT Prob", f"{ot_prob*100:.0f}%")

css = {"pit":"rec-pit","soon":"rec-soon","stay":"rec-stay"}[rec_type]
st.markdown(f'<div class="rec-box {css}">{dec} — {reason}</div>', unsafe_allow_html=True)

ml_prob = ml_pit_probability(
    s.lap, total_laps, s.tyre_age,
    pct, s.fuel, s.position
)
ml_gain = ml_position_gain(s.position, s.lap / total_laps, total_laps)
gain_str = f"+{ml_gain:.1f}" if ml_gain >= 0 else f"{ml_gain:.1f}"

st.caption(f"Lap delta: +{delta}s | Fuel: {s.fuel:.1f}% | Pits: {s.pits_taken}")

col_ml1, col_ml2 = st.columns(2)
with col_ml1:
    if ml_prob > 60:
        st.error(f"🤖 ML Pit Probability: **{ml_prob}%** — Model strongly suggests pitting")
    elif ml_prob > 30:
        st.warning(f"🤖 ML Pit Probability: **{ml_prob}%** — Model sees a pit window opening")
    else:
        st.success(f"🤖 ML Pit Probability: **{ml_prob}%** — Model says stay out")
with col_ml2:
    st.info(f"📈 Predicted position gain if pitting now: **{gain_str} positions**")

st.markdown("---")

# ── Tabbed Layout ─────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🏁 Live Timing", "📊 Charts", "📋 Lap History"])

with tab1:
    active = sorted([c for c in s.cars], key=lambda x: x["pos"])
    leader_gap = active[0]["gap_to_leader"] if active else 0
    rows = ""
    for car in active:
        gap     = car["gap_to_leader"] - leader_gap
        gap_str = "LEADER" if gap < 0.001 else f"+{gap:.3f}s"
        tc      = TYRE_CLASS[car["tyre"]]
        tn      = TYRE_NAMES[car["tyre"]]
        tr_cls  = "player" if car["is_player"] else ""
        rows += f"""<tr class="{tr_cls}">
            <td><b>{car['pos']}</b></td>
            <td>{'⭐ ' if car['is_player'] else ''}<b>{car['code']}</b></td>
            <td>{car['name']}</td>
            <td>{gap_str}</td>
            <td><span class="{tc}">{tn} ({car['tyre_age']}L)</span></td>
            <td>{car['pits']}</td>
            <td>{car['last_pit'] if car['last_pit'] else '—'}</td>
        </tr>"""
    st.markdown(f"""
    <table class="timing-table">
      <thead><tr>
        <th>POS</th><th>CODE</th><th>DRIVER</th><th>GAP</th>
        <th>TYRE</th><th>STOPS</th><th>LAST PIT</th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table>""", unsafe_allow_html=True)

with tab2:
    if s.history:
        hist = pd.DataFrame(s.history)
        r1c1, r1c2 = st.columns(2)

        with r1c1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist["lap"],y=hist["tyre_perf"],
                mode="lines",line=dict(color="#e8002d",width=2),
                fill="tozeroy",fillcolor="rgba(232,0,45,0.1)",name="Tyre Perf"))
            fig.update_layout(title="Tyre Performance",paper_bgcolor="#1a1a1a",
                plot_bgcolor="#1a1a1a",font_color="#fff",height=260,
                margin=dict(l=40,r=20,t=36,b=30))
            fig.update_xaxes(gridcolor="#333")
            fig.update_yaxes(gridcolor="#333",range=[0,105])
            st.plotly_chart(fig,use_container_width=True)

        with r1c2:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=hist["lap"],y=hist["pos"],
                mode="lines+markers",line=dict(color="#ff6600",width=2),
                marker=dict(size=4),name="Position"))
            fig2.update_layout(title="Race Position",paper_bgcolor="#1a1a1a",
                plot_bgcolor="#1a1a1a",font_color="#fff",height=260,
                margin=dict(l=40,r=20,t=36,b=30))
            fig2.update_xaxes(gridcolor="#333")
            fig2.update_yaxes(gridcolor="#333",autorange="reversed")
            st.plotly_chart(fig2,use_container_width=True)

        r2c1, r2c2 = st.columns(2)

        with r2c1:
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=hist["lap"],y=hist["gap_ahead"],
                mode="lines",line=dict(color="#00aaff",width=2),name="Gap Ahead"))
            fig3.add_trace(go.Scatter(x=hist["lap"],y=hist["gap_behind"],
                mode="lines",line=dict(color="#ffaa00",width=2,dash="dash"),name="Gap Behind"))
            fig3.update_layout(title="Gaps (s)",paper_bgcolor="#1a1a1a",
                plot_bgcolor="#1a1a1a",font_color="#fff",height=260,
                margin=dict(l=40,r=20,t=36,b=30))
            fig3.update_xaxes(gridcolor="#333")
            fig3.update_yaxes(gridcolor="#333")
            st.plotly_chart(fig3,use_container_width=True)

        with r2c2:
            fig4 = go.Figure()
            fig4.add_trace(go.Scatter(x=hist["lap"],y=hist["ot_prob"],
                mode="lines",line=dict(color="#aa44ff",width=2),
                fill="tozeroy",fillcolor="rgba(170,68,255,0.1)",name="OT Prob %"))
            fig4.update_layout(title="Overtake Probability %",paper_bgcolor="#1a1a1a",
                plot_bgcolor="#1a1a1a",font_color="#fff",height=260,
                margin=dict(l=40,r=20,t=36,b=30))
            fig4.update_xaxes(gridcolor="#333")
            fig4.update_yaxes(gridcolor="#333",range=[0,100])
            st.plotly_chart(fig4,use_container_width=True)
    else:
        st.info("Start the race to see charts.")

with tab3:
    if s.history:
        hist = pd.DataFrame(s.history)
        st.dataframe(
            hist[["lap","pos","tyre","tyre_age","tyre_pct","gap_ahead","gap_behind","weather","sc","rec"]].tail(20),
            use_container_width=True,hide_index=True)
    else:
        st.info("Start the race to see lap history.")

# ── Auto advance ──────────────────────────────────────────────────────────────
if s.running and s.lap < total_laps:
    tick()
    time.sleep(sleep_map[spd])
    st.rerun()
elif s.lap >= total_laps:
    gained = s.positions_gained
    color  = "pos-gain" if gained >= 0 else "pos-loss"
    st.success(f"🏁 Race complete! Finished **P{s.position}** with **{s.pits_taken}** pit stop(s). Positions gained: {gained:+d}")