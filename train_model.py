import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, mean_absolute_error
import pickle

print("Loading data...")

results    = pd.read_csv("results.csv")
races      = pd.read_csv("races.csv")[["raceId","year","name","round"]]
drivers    = pd.read_csv("drivers.csv")[["driverId","forename","surname"]]
pit_stops  = pd.read_csv("pit_stops.csv")

drivers["driver_name"] = drivers["forename"] + " " + drivers["surname"]

# ── Merge ────────────────────────────────────────────────────────────────────
df = results.merge(races, on="raceId").merge(drivers, on="driverId")
df = df.merge(pit_stops[["raceId","driverId","lap","stop"]].rename(
    columns={"lap":"pit_lap","stop":"pit_number"}), on=["raceId","driverId"], how="left")

print(f"Total records: {len(df)}")

# ── Feature Engineering ───────────────────────────────────────────────────────
df["pit_lap"]    = pd.to_numeric(df["pit_lap"], errors="coerce")
df["points"]     = pd.to_numeric(df["points"], errors="coerce").fillna(0)
df["grid"]       = pd.to_numeric(df["grid"], errors="coerce").fillna(10)
df["laps"]       = pd.to_numeric(df["laps"], errors="coerce").fillna(50)
df["positionOrder"] = pd.to_numeric(df["positionOrder"], errors="coerce").fillna(10)

# Position gained
df["positions_gained"] = df["grid"] - df["positionOrder"]

# Pit lap as % of race distance
df["pit_lap_pct"] = df["pit_lap"] / df["laps"].replace(0, 57)

# Was this a points finish?
df["points_finish"] = (df["points"] > 0).astype(int)

# Round of season (early/mid/late)
df["season_stage"] = pd.cut(df["round"], bins=[0,7,14,24], labels=["early","mid","late"])

# Drop rows with no pit data for pit model
pit_df = df.dropna(subset=["pit_lap","pit_lap_pct"])
print(f"Records with pit data: {len(pit_df)}")

# ── MODEL 1: Pit Lap Predictor ────────────────────────────────────────────────
# Given race state, predict if this is a good lap to pit
# We'll generate lap-by-lap synthetic states from real pit data

print("\nBuilding pit predictor training data...")

rows = []
for _, race_group in pit_df.groupby(["raceId","driverId"]):
    total_laps = race_group["laps"].iloc[0]
    if total_laps < 20: continue
    pit_laps = race_group["pit_lap"].dropna().tolist()
    grid_pos = race_group["grid"].iloc[0]

    for lap in range(1, int(total_laps)+1):
        tyre_age     = lap % 28 + 1  # simplified tyre age
        fuel         = max(0, 100 - (lap / total_laps * 100))
        laps_left    = total_laps - lap
        tyre_pct     = min(100, (tyre_age / 28) * 100)
        should_pit   = 1 if lap in pit_laps else 0

        rows.append({
            "lap": lap,
            "total_laps": total_laps,
            "laps_left": laps_left,
            "lap_pct": lap / total_laps,
            "tyre_age": tyre_age,
            "tyre_pct": tyre_pct,
            "fuel": fuel,
            "grid_pos": grid_pos,
            "should_pit": should_pit
        })

pit_train = pd.DataFrame(rows)
print(f"Pit training samples: {len(pit_train)}")

features_pit = ["lap","total_laps","laps_left","lap_pct","tyre_age","tyre_pct","fuel","grid_pos"]
X_pit = pit_train[features_pit]
y_pit = pit_train["should_pit"]

X_train, X_test, y_train, y_test = train_test_split(X_pit, y_pit, test_size=0.2, random_state=42)

print("Training pit predictor (Random Forest)...")
pit_model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
pit_model.fit(X_train, y_train)

y_pred = pit_model.predict(X_test)
print("\nPit Predictor Results:")
print(classification_report(y_test, y_pred))

# Feature importance
importances = pd.Series(pit_model.feature_importances_, index=features_pit).sort_values(ascending=False)
print("\nFeature Importance:")
print(importances)

# ── MODEL 2: Position Gain Predictor ─────────────────────────────────────────
print("\nBuilding position gain predictor...")

pos_df = df.dropna(subset=["positions_gained","pit_lap_pct","grid"])
pos_df = pos_df[pos_df["laps"] > 20]

features_pos = ["grid","pit_lap_pct","points_finish","laps"]
pos_df = pos_df.dropna(subset=features_pos+["positions_gained"])

X_pos = pos_df[features_pos]
y_pos = pos_df["positions_gained"].clip(-10, 10)

X_train2, X_test2, y_train2, y_test2 = train_test_split(X_pos, y_pos, test_size=0.2, random_state=42)

print("Training position gain predictor...")
pos_model = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
pos_model.fit(X_train2, y_train2)

y_pred2 = pos_model.predict(X_test2)
mae = mean_absolute_error(y_test2, y_pred2)
print(f"Position Gain MAE: {mae:.2f} positions")

# ── Save Models ───────────────────────────────────────────────────────────────
print("\nSaving models...")
with open("pit_model.pkl","wb") as f:
    pickle.dump(pit_model, f)
with open("pos_model.pkl","wb") as f:
    pickle.dump(pos_model, f)
with open("pit_features.pkl","wb") as f:
    pickle.dump(features_pit, f)
with open("pos_features.pkl","wb") as f:
    pickle.dump(features_pos, f)

print("\n✅ Models saved: pit_model.pkl, pos_model.pkl")
print("Ready to integrate into app.py")