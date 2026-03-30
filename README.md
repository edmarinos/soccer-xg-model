# Soccer Expected Goals (xG) Model
### FIFA World Cup 2022 | StatsBomb Open Data | Logistic Regression

A machine learning model that predicts the probability of a shot resulting 
in a goal based on pre-shot characteristics. Built using StatsBomb open 
event data from the 2022 FIFA World Cup.

**[Try the live app →](https://soccer-xg-model.streamlit.app/)**

---

## Results

| Model | Log Loss | Brier Score | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.2898 | 0.0863 | 0.835 |
| XGBoost | 0.2988 | 0.0866 | 0.817 |
| StatsBomb xG (professional baseline) | 0.2934 | 0.0862 | 0.827 |

The Logistic Regression model marginally outperforms StatsBomb's professional 
xG model across Log Loss and ROC-AUC despite using only 11 engineered features, 
compared to StatsBomb's richer feature set which includes defensive freeze frame 
and goalkeeper positioning data.

---

## Features

All features were engineered from raw StatsBomb event data:

- **Distance to goal** — Euclidean distance from shot location to goal center
- **Shot angle** — Angular width of the goal from the shot position
- **Body part** — Header vs foot
- **Shot type** — Open play vs set piece
- **First time** — Whether the shot was struck first time
- **One on one** — Whether the shooter was through on goal
- **Open goal** — Whether the goalkeeper was absent
- **Under pressure** — Whether a defender was closing the shooter
- **Aerial** — Whether it was an aerial challenge
- **Deflected** — Whether the shot was deflected
- **Volley** — Whether the shot was a volley or half volley

---

## Notable Predictions

**Biggest Upset — Salem Al-Dawsari (Saudi Arabia vs Argentina, 52')**  
Our model assigned 3.8% xG to Al-Dawsari's famous equalizer — one of the 
most improbable goals of the tournament. The goal came from ~23 yards under 
pressure, exactly the profile our model correctly identifies as low probability.

**Best Miss — Romelu Lukaku (Belgium, 89')**  
Lukaku's ruled-out goal carried a 73.8% xG — the highest quality chance in 
the test set. The goal was disallowed for offside, validating that the shot 
quality was genuine.

**Biggest Model Disagreement — Jamal Musiala (Germany, 99')**  
Our model (63.2% xG) and StatsBomb (8.5% xG) disagreed dramatically. The 
gap likely reflects StatsBomb's use of defensive freeze frame data — a 
crowded penalty area context our geometric features cannot capture.

---

## Project Structure
```
soccer-xg-model/
├── app.py                  # Streamlit application
├── requirements.txt
├── lr_xg_model.pkl         # Trained Logistic Regression model
├── scaler.pkl              # Fitted StandardScaler
├── 01_eda.ipynb            # Full EDA, feature engineering, and modeling
├── calibration_curve.png
├── distance_angle.png
├── feature_importance.png
├── lr_coefficients.png
├── roc_curve.png
├── shot_characteristics.png
└── shot_map.png
```

---

## How to Run Locally
```bash
git clone https://github.com/edmarinos/soccer-xg-model.git
cd soccer-xg-model
pip install -r requirements.txt
python -m streamlit run app.py
```

---

## Tech Stack

Python, scikit-learn, XGBoost, StatsBombPy, Streamlit, pandas, NumPy, matplotlib

---

## Data Source

[StatsBomb Open Data](https://github.com/statsbomb/open-data) — freely available 
event-level match data. All data is used in accordance with StatsBomb's open 
data terms of use.
