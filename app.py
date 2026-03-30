import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Soccer xG Model",
    page_icon="⚽",
    layout="wide"
)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load('models/lr_xg_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    return model, scaler

model, scaler = load_model()

# ── Helper functions ──────────────────────────────────────────────────────────
GOAL_X = 120
GOAL_Y = 40

def compute_distance(x, y):
    return np.sqrt((x - GOAL_X)**2 + (y - GOAL_Y)**2)

def compute_angle(x, y):
    angle = np.abs(np.arctan2(
        8 * (GOAL_X - x),
        (GOAL_X - x)**2 + (y - GOAL_Y)**2 - 16
    ))
    return np.degrees(angle)

import pandas as pd

def predict_xg(x, y, is_header, is_open_play, is_first_time,
               is_one_on_one, is_open_goal, under_pressure,
               is_aerial, is_deflected, is_volley):
    distance = compute_distance(x, y)
    angle = compute_angle(x, y)
    features = pd.DataFrame([[distance, angle, is_header, is_open_play,
                              is_first_time, is_one_on_one, is_open_goal,
                              under_pressure, is_aerial, is_deflected, is_volley]],
                            columns=['distance', 'angle_degrees', 'is_header',
                                     'is_open_play', 'is_first_time', 'is_one_on_one',
                                     'is_open_goal', 'under_pressure', 'is_aerial',
                                     'is_deflected', 'is_volley'])
    features_scaled = scaler.transform(features)
    xg = model.predict_proba(features_scaled)[0][1]
    return xg, distance, angle

def draw_pitch(shot_x=None, shot_y=None):
    fig, ax = plt.subplots(figsize=(10, 7))
    # Force green background as a patch
    green_rect = patches.Rectangle((0, 0), 120, 80,
                                    linewidth=0, facecolor='#2d6a4f', zorder=0)
    ax.add_patch(green_rect)
    fig.patch.set_facecolor('#0d1117')

    # Pitch outline
    ax.plot([0, 0, 120, 120, 0], [0, 80, 80, 0, 0], color='white', linewidth=2)
    ax.plot([60, 60], [0, 80], color='white', linewidth=1)
    circle = plt.Circle((60, 40), 10, color='white', fill=False, linewidth=1)
    ax.add_patch(circle)

    # Penalty areas
    ax.plot([102, 102, 120, 120, 102], [18, 62, 62, 18, 18], color='white', linewidth=1.5)
    ax.plot([0, 0, 18, 18, 0], [18, 62, 62, 18, 18], color='white', linewidth=1.5)

    # 6-yard boxes
    ax.plot([114, 114, 120, 120, 114], [30, 50, 50, 30, 30], color='white', linewidth=1)
    ax.plot([0, 0, 6, 6, 0], [30, 50, 50, 30, 30], color='white', linewidth=1)

    # Goals
    ax.plot([120, 120], [36, 44], color='white', linewidth=5)
    ax.plot([0, 0], [36, 44], color='white', linewidth=5)

    # Penalty spots
    ax.plot(108, 40, 'o', color='white', markersize=3)
    ax.plot(12, 40, 'o', color='white', markersize=3)

    # Shot marker
    if shot_x is not None and shot_y is not None:
        ax.plot(shot_x, shot_y, 'o', color='red', markersize=14,
                markeredgecolor='white', markeredgewidth=2, zorder=5)
        # Line to goal
        ax.annotate('', xy=(120, 40), xytext=(shot_x, shot_y),
                    arrowprops=dict(arrowstyle='->', color='yellow',
                                   lw=1.5, linestyle='dashed'))

    ax.set_xlim(-2, 122)
    ax.set_ylim(-2, 82)
    ax.axis('off')
    ax.set_aspect('equal')
    return fig

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("⚽ Soccer Expected Goals (xG) Model")
st.markdown("**FIFA World Cup 2022** — Logistic Regression model trained on StatsBomb open data")
st.markdown("---")

col1, col2 = st.columns([1.4, 1])

with col1:
    st.subheader("📍 Shot Location")
    st.markdown("Use the sliders to position the shot on the pitch.")

    shot_x = st.slider("X coordinate (0 = own goal, 120 = attacking goal)",
                        min_value=60.0, max_value=119.0,
                        value=105.0, step=0.5)
    shot_y = st.slider("Y coordinate (0 = left touchline, 80 = right touchline)",
                        min_value=0.0, max_value=80.0,
                        value=40.0, step=0.5)

    fig = draw_pitch(shot_x, shot_y)
    st.pyplot(fig, width='stretch')
    plt.close()

with col2:
    st.subheader("🎛️ Shot Characteristics")

    is_header     = st.checkbox("Header")
    is_open_play  = st.checkbox("Open Play", value=True)
    is_first_time = st.checkbox("First Time Shot")
    is_one_on_one = st.checkbox("One on One")
    is_open_goal  = st.checkbox("Open Goal")
    under_pressure = st.checkbox("Under Pressure")
    is_aerial     = st.checkbox("Aerial")
    is_deflected  = st.checkbox("Deflected")
    is_volley     = st.checkbox("Volley")

    st.markdown("---")

    xg, distance, angle = predict_xg(
        shot_x, shot_y,
        int(is_header), int(is_open_play), int(is_first_time),
        int(is_one_on_one), int(is_open_goal), int(under_pressure),
        int(is_aerial), int(is_deflected), int(is_volley)
    )

    # Color code the xG value
    if xg >= 0.5:
        color = "🟢"
    elif xg >= 0.2:
        color = "🟡"
    else:
        color = "🔴"

    st.markdown(f"### {color} Predicted xG: `{xg:.3f}`")
    st.progress(float(xg))

    st.markdown(f"""
    **Shot details:**
    - Distance to goal: `{distance:.1f} yards`
    - Shot angle: `{angle:.1f}°`
    - Conversion likelihood: `{xg*100:.1f}%`
    """)

    # Contextual interpretation
    if xg >= 0.7:
        st.success("High quality chance — a top striker scores this more often than not.")
    elif xg >= 0.4:
        st.info("Good chance — a quality opportunity that should trouble the goalkeeper.")
    elif xg >= 0.15:
        st.warning("Moderate chance — possible but the goalkeeper is favored.")
    else:
        st.error("Low quality chance — unlikely to score from here.")

    st.markdown("---")
    st.markdown("**Model:** Logistic Regression | **Data:** StatsBomb Open Data")
    st.markdown("**Competition:** FIFA World Cup 2022 | **ROC-AUC:** 0.835")