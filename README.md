# 🚦 AI-Powered Parking Intelligence Platform
### Flipkart Gridlock Hackathon 2.0 — Round 2 Prototype
### Theme 1: Poor Visibility on Parking-Induced Congestion

**Author:** Chinmay Kumar Rath
ITER, SOA University | B.Tech CSE 2027 

---

## 📋 Problem Statement

Bengaluru, a city of over 14 million people, faces chronic traffic congestion — and illegal parking is one of its most persistent but least quantified contributors. On-street illegal parking near commercial areas, metro stations, and busy intersections physically reduces road capacity, forces lane merging, and creates bottlenecks that cascade into wider gridlock.

The core operational challenge faced by the Bengaluru Traffic Police is that enforcement is entirely **reactive and patrol-based**. Officers respond only after congestion has already formed. There is no:

- Spatial heatmap showing where parking violations cluster historically
- Predictive capability to anticipate where violations will peak before they occur
- Congestion impact assessment to quantify which hotspots cause the most road blockage
- Data-driven resource allocation framework for officers and tow trucks

This platform addresses all four gaps using AI forecasting built entirely on real Bengaluru Traffic Police enforcement data from the ASTraM unit.

**Problem Statement Direction (official):**
> *"How can AI-driven parking intelligence detect illegal parking hotspots and quantify their impact on traffic flow to enable targeted enforcement?"*

---

## 📊 Dataset Overview

- **Source:** Bengaluru Traffic Police ASTraM Unit (provided by HackerEarth)
- **Original size:** 298,450 parking violation records
- **Time span:** November 2023 – April 2024 (approximately 5 months)
- **Coverage:** 169 named junctions, 54 police stations, complete GPS coordinates for all records
- **Key columns:** `junction_name`, `vehicle_type`, `violation_type`, `police_station`, `latitude`, `longitude`, `created_datetime`, `validation_status`, `device_id`

---

## 🔍 Exploratory Data Analysis

### Missing Value Analysis

| Column | Missing % | Treatment |
|---|---|---|
| `description` | 100% | Dropped — completely empty |
| `closed_datetime` | 100% | Dropped — completely empty |
| `action_taken_timestamp` | 100% | Dropped — completely empty |
| `validation_status` | 41.97% | Filled as "unvalidated" — system rollout gap |
| `Temperature` | 3.23% | Filled with geo3-zone median |
| `Weather` | 1.03% | Filled with mode |
| `RoadType` | 0.78% | Filled with mode |

**Key decision on `validation_status`:** The 125,254 missing rows were concentrated in February–April 2024, consistent with a system rollout gap rather than invalid detections. Dropping them would have removed 42% of the dataset unnecessarily. These rows were retained as "unvalidated."

### Data Quality Filtering

Records explicitly marked as `rejected` (49,754) or `duplicate` (320) were removed since these are confirmed false detections. Final clean dataset: **248,371 rows.**

### Key EDA Findings

**Temporal patterns:**
- Violations are heavily concentrated in **late night and early morning** (peak at 5 AM, near-zero from 9 AM–6 PM)
- This reflects overnight illegal parking enforcement, not daytime commute congestion
- Weekend violations marginally higher than weekdays

**Spatial concentration:**
- Top 10 junctions account for **73% of all violations** — extreme hotspot concentration
- 168 unique named junctions, 54 police stations
- Complete GPS coverage (0 missing latitude/longitude)

**Vehicle composition:**
- Scooter (94,856), Car (88,870), Motorcycle (40,811) dominate
- Heavy vehicles (lorries, buses, tankers) are fewer but cause disproportionate road blockage

**Violation types:**
- `WRONG PARKING` (165K instances) and `NO PARKING` (139K) account for 86% of all violations
- 86.5% of records have exactly one violation type; 13.5% have multiple

**Validation status distribution:**
approved      115,400

rejected       49,754  ← removed

unvalidated   125,254  ← retained

created1        7,044

processing        678

duplicate         320  ← removed

---

## ⚙️ Feature Engineering

### Temporal Features

```python
hour, minute, time_mins          # basic time decomposition
is_peak   (hours 9–14)           # midday peak identified from EDA
is_low    (hours 17–21)          # evening low identified from EDA
is_midnight (hours 22–3)         # overnight enforcement window
hour_sin, hour_cos               # cyclical encoding
day_of_week_num                  # 0=Monday to 6=Sunday
is_weekend                       # binary flag
month                            # seasonal patterns
time_bucket = time_mins // 30    # 30-minute granularity slots (48 per day)
```

**Why cyclical encoding for hour:** Hour 23 and hour 0 are 15 minutes apart in reality but 23 units apart numerically. Sin/cos encoding wraps the 24-hour clock into a circle so the model correctly understands temporal proximity.

### Geohash / Spatial Features

```python
latitude, longitude              # junction-level mean coordinates
junction_name                    # passed natively to CatBoost as categorical
junction_total_violations        # historical total per junction
junction_unique_days             # activity frequency
junction_avg_per_day             # average daily load
```

**Junction name handling:** Rather than label-encoding `junction_name` (which proved to destroy predictive signal — see ML Journey below), the column was passed directly to CatBoost using its native `cat_features` parameter, which uses Ordered Target Statistics internally.

### Target Variable Engineering

Individual violation records were aggregated to **junction × date × hour** level:

```python
agg = df_clean.groupby(['junction_name', 'date', 'hour']).size()
             .reset_index(name='violation_count')
```

This produced 31,429 aggregated rows. The target (`violation_count`) was log-transformed:

```python
y = np.log1p(violation_count)
```

**Why log transform:** `violation_count` is heavily right-skewed (median 2, mean 7.9, max 268). Log transform compresses the upper tail and stabilizes variance, which is standard practice for count regression targets.

### Congestion Impact Score

A weighted severity score was engineered to quantify how much each hotspot actually blocks road capacity — since not all violations cause equal congestion:

```python
vehicle_weight = {
    'LORRY/GOODS VEHICLE': 3.0,
    'BUS (BMTC/KSRTC)': 3.0,
    'CAR': 1.5,
    'SCOOTER': 1.0,
    ...
}

violation_weight = {
    'DOUBLE PARKING': 2.5,
    'PARKING IN A MAIN ROAD': 2.0,
    'WRONG PARKING': 1.5,
    'NO PARKING': 1.3,
    ...
}

blocking_severity = vehicle_weight × violation_weight
congestion_impact_score = blocking_severity.rank(pct=True) × 100
```

Scores are **percentile-ranked 0–100** across all monitored locations, making them robust to outliers. A lorry double-parked on a main road scores ~3.0 × 2.5 = 7.5, versus a scooter wrongly parked scoring ~1.0 × 1.5 = 1.5 — a 5x difference in road-blocking severity.

---

## 🤖 ML Pipeline & Model Selection

### Aggregation Strategy

The pipeline does **not** model individual vehicle violations. Individual records are too noisy (random patrol timing, officer availability, logging delays). Instead, violations are aggregated to **junction × date × hour** level, giving a stable count that reflects the underlying parking behavior pattern at each location over time.

### Why CatBoostRegressor

**The model choice was validated through elimination, not assumption.**

Several alternatives were evaluated:

| Model | Approach | CV R² | Reason for rejection |
|---|---|---|---|
| LightGBM | Label-encoded junction | 0.207 | Junction encoding destroys signal |
| CatBoost | Label-encoded junction | 0.207 | Same encoding problem |
| CatBoost | Native cat_features | **0.555** | ✅ Selected |
| DummyRegressor | Mean prediction | -0.192 | Baseline only |
| Risk classification (lag-based) | Daily aggregation | Accuracy 0.23 | Worse than regression |

**Why CatBoost specifically over LightGBM or XGBoost:**

CatBoost uses **Ordered Target Statistics** for categorical features — it processes each training example using only the rows that came before it in a random permutation, preventing target leakage during categorical encoding. For a high-cardinality feature like `junction_name` (169 unique values), this is significantly more powerful than:

- Label encoding (assigns arbitrary integers with no meaning)
- One-hot encoding (169 new columns, sparse, computationally expensive)
- Manual target encoding (risks leakage without careful KFold implementation)

CatBoost handles this natively, correctly, and without additional code. This is why the jump from label-encoded LightGBM (0.207) to CatBoost with `cat_features=['junction_name']` (0.555) was so dramatic — it wasn't a model architecture difference, it was entirely about how the junction identity was represented.

**Why not neural networks:** 31,429 aggregated rows is insufficient for neural networks to outperform gradient boosting. Deep learning excels on large unstructured data (images, text). For tabular count data at this scale, gradient boosting consistently outperforms. Neural networks also require significantly more hyperparameter tuning, which was impractical within a 4-day prototype timeline.

### Final Model Parameters

```python
CatBoostRegressor(
    iterations=500,
    depth=6,
    learning_rate=0.05,
    loss_function='RMSE',
    cat_features=['junction_name'],
    random_seed=42
)
```

### Validation Strategy

Three validation strategies were used to ensure the model is not overfitting or leaking future data:

**1. 5-Fold Cross Validation:**
CV R² = 0.555 ± 0.011

**2. Temporal Holdout (most important):**
Data sorted chronologically. Train on first 80% of dates, validate on last 20%. This simulates real-world deployment where the model always predicts forward in time.
Temporal Holdout R² = 0.544
The negligible gap between CV (0.555) and temporal holdout (0.544) confirms no temporal leakage — the model genuinely generalizes to future periods.

**3. Dummy Baseline:**
DummyRegressor(strategy='mean') R² = -0.192
The model dramatically outperforms a trivial baseline, confirming it is learning real structure, not noise.

### Risk Classification Layer

Model predictions are converted to 5-tier risk levels using quantile-based thresholds:

```python
pd.qcut(avg_predicted_violations,
        q=[0, 0.5, 0.8, 0.95, 0.99, 1.0],
        labels=['Low','Medium','High','Critical','Extreme'])
```

**Distribution across 169 monitored locations:**
Low:      85 locations (50%)

Medium:   50 locations (30%)

High:     25 locations (15%)

Critical:  7 locations (4%)

Extreme:   2 locations (1%)

---

## 🗺️ Project Journey — Attempts, Failures & Final Success

### Phase 1: Initial Formulation (Day 1)

The first instinct was to directly aggregate violations by `junction_name + date + hour` and train a regression model to predict `violation_count`. Features included temporal variables (hour, day of week, month), cyclical encodings, and junction historical statistics.

`junction_name` was label-encoded (standard practice for categorical variables) and passed to both LightGBM and CatBoost.

**Result:** Both models produced CV R² ≈ 0.207 — significantly below expectations for structured tabular data.

Initial hypothesis: *"Maybe predicting exact violation counts is inherently noisy — perhaps the problem formulation itself is wrong."*

---

### Phase 2: Wrong Turn — Reformulation Attempt (Day 1)

Acting on the hypothesis that exact count prediction was infeasible, the problem was reformulated:

- Aggregated to `junction + date` level (daily rather than hourly)
- Added lag features (lag_1, lag_3, lag_7, rolling_mean_7)
- Converted target to 3-tier risk classification (Low/Medium/High)

**Result:**
CatBoostClassifier Accuracy: 0.234 ± 0.163

Macro F1: 0.185 ± 0.069

Classification performed dramatically worse than regression. This was a critical observation — the problem formulation was not the issue. Abandoning the original regression formulation was unjustified by evidence.

**Lesson learned:** Don't change the problem formulation based on one set of poor results. Diagnose the root cause first.

---

### Phase 3: Root Cause Diagnosis — Encoding Bug (Day 2)

Returning to the original regression formulation, a different hypothesis was tested:

*"Perhaps the issue is not the target variable. Perhaps the issue is how junction information is being represented."*

CatBoost has a specific design feature: it can handle categorical variables natively using Ordered Target Statistics, which is fundamentally different from label encoding. When `junction_name` was passed as a `cat_features` column instead of a pre-encoded integer:

**Result:**
CatBoost with cat_features=['junction_name'] CV R² = 0.580

A jump from 0.207 to 0.580 from a single change. The root cause was identified: label encoding destroyed the junction signal entirely. The model had been trying to learn patterns from meaningless integers (junction 47, junction 83) when it should have been learning from junction identity directly.

---

### Phase 4: Leakage Investigation (Day 2)

Feature importance analysis revealed `junction_hour_te` (target encoding of junction × hour) dominated at 73% importance. This raised concern:

*"Is the model genuinely learning, or is it almost entirely dependent on a feature that encodes the target?"*

Target encoding was removed entirely and the model was retrained with only structural features:
CatBoost NO-TE CV R² = 0.555

Gap from 0.580 to 0.555 — only 2.5 percentage points. The majority of predictive power survived without target encoding. This confirmed:
- Junction identity contains real signal
- Temporal features contain real signal  
- The model is not solely dependent on target encoding

**Decision:** Use the 0.555 model (no target encoding) for the final prototype. Lower score, but fully defensible and explainable — a lorry that always parks illegally at junction X at 2 AM can be predicted from junction identity and time alone, without needing historical demand statistics.

---

### Phase 5: Temporal Validation (Day 2)

Random KFold CV on time-series data risks temporal leakage — a fold might train on April data and predict November. A proper temporal holdout was implemented:
Train: first 80% of dates chronologically

Validate: last 20% of dates

Temporal Holdout R² = 0.544

Near-identical to CV R² (0.555), confirming no temporal leakage. Model genuinely generalizes to future dates — critical for a real-world deployment claim.

---

### Phase 6: Congestion Impact Layer (Day 3)

The official problem statement asked to "quantify impact on traffic flow" — but the dataset contains only violation records, not traffic speed or density data. This gap needed addressing without manufacturing false signals.

**Solution:** Engineer a proxy metric from data that actually exists — vehicle type and violation type both directly determine how much road capacity a parked vehicle consumes. A lorry double-parked on a main road causes objectively more congestion than a scooter in a side lane.

Weights were assigned based on vehicle size (lorry = 3.0, car = 1.5, scooter = 1.0) and violation severity (double parking = 2.5, wrong parking = 1.5). Scores are percentile-ranked 0-100 for interpretability.

This turns a limitation ("we don't have traffic speed data") into a strength ("we quantify road-blocking severity directly from violation characteristics").

---

### Phase 7: Streamlit Dashboard (Day 3–4)

Three-page dashboard built:

1. **Overview Dashboard** — metrics, top 10 hotspots, risk distribution chart, interactive Folium map
2. **Congestion Impact Analysis** — weighted severity scores, impact distribution, second map colored by impact rather than risk
3. **Enforcement Planner** — dynamic risk filter, fixed resource allocation by risk tier, downloadable CSV deployment plan, deployment map with pin markers

**Key UI decisions:**
- No ML jargon visible to end users — "R²" appears only once in the metrics bar
- Color coding matches universal traffic light intuition (green→red→darkred)
- Download button generates a shift-ready CSV officers can act on immediately
- `CartoDB positron` tiles chosen for clean, professional map appearance

---

## 🖥️ Dashboard Features

| Page | Features |
|---|---|
| Overview Dashboard | 4 KPI metrics, Top 10 hotspot table, Risk distribution chart, Interactive Folium map with 169 markers |
| Congestion Impact Analysis | Weighted impact scores (0-100), Top 20 contributors table, Impact distribution chart, Second map colored by impact score |
| Enforcement Planner | Risk level filter, Fixed officer/tow truck allocation, Downloadable deployment CSV, Deployment map with pin markers |

---

## 🚀 How to Run

```bash
# Clone the repository
git clone https://github.com/Chinmay-Kumar-Rath/Theme-poor-Visibility-on-Parking-Induced-Congestion.git
cd Theme-poor-Visibility-on-Parking-Induced-Congestion
# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run app.py
```

---

## 📦 Tech Stack

| Component | Technology |
|---|---|
| ML Model | CatBoostRegressor |
| Data Processing | pandas, numpy |
| Validation | scikit-learn |
| Dashboard | Streamlit |
| Maps | Folium, streamlit-folium |
| Charts | Plotly Express |
| Environment | Python 3.11, Anaconda |

---
## Dataset

The original dataset was provided as part of the Flipkart Gridlock 2.0 Hackathon and is not included in this repository.

To reproduce the training pipeline, place the competition dataset in the appropriate data directory before running the notebook.
---

## 📁 Project Structure
gridlock-parking-intelligence/

│

├── app.py                          # Streamlit dashboard

├── requirements.txt

├── README.md

│

├── data/

│   ├── dashboard_predictions.csv   # Junction-hour predictions

│   └── location_summary.csv        # 169 hotspot summaries with risk scores

│

├── model/

│   └── parking_congestion_model.cbm  # Trained CatBoost model

│

├── notebook/

│   └── Theme poor Visibility on Parking Induced Congestion_pipeline.ipynb  # Complete ML pipeline

│

└── PPT Theme poor Visibility on Parking Induced Congestion

└── SreenShots/

---

## 📈 Results Summary

| Metric | Value |
|---|---|
| Records Processed | 248,371 |
| Hotspots Identified | 169 |
| Model CV R² | 0.555 |
| Temporal Holdout R² | 0.544 |
| Dummy Baseline R² | -0.192 |
| Extreme Risk Zones | 2 |
| Critical Risk Zones | 7 |

---

## 🔮 Future Roadmap

- **Real-time CCTV Integration** — Live violation detection using computer vision on ASTraM camera feeds
- **Dynamic Enforcement Routing** — Optimal patrol route generation based on predicted hotspot windows
- **Traffic Speed Correlation** — Integration with MapMyIndia traffic API to directly measure congestion impact
- **City-wide Scalability** — Extend beyond 169 named junctions to all GPS-tagged violation points
- **Mobile Interface** — Officer-facing mobile app for shift-ready deployment plans

---

## 📚 References

1. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD '16*.
2. Prokhorenkova, L., et al. (2018). CatBoost: Unbiased Boosting with Categorical Features. *NeurIPS 2018*.
3. Dorogush, A. V., et al. (2018). CatBoost: gradient boosting with categorical features support. *arXiv:1810.11363*.
4. Bengaluru Traffic Police ASTraM Unit — dataset provided via HackerEarth Gridlock Hackathon 2.0.
5. Folium Documentation — https://python-visualization.github.io/folium/
6. Streamlit Documentation — https://docs.streamlit.io/
7. scikit-learn Documentation — https://scikit-learn.org/stable/
8. Bengaluru Traffic Police — https://btp.karnataka.gov.in/

---

## 👤 Author

**Chinmay Kumar Rath**
B.Tech Computer Science Engineering (2027 Batch)
ITER, SOA University, Bhubaneswar


GitHub: [Chinmay-Kumar-Rath](https://github.com/Chinmay-Kumar-Rath)

*Submitted as part of Flipkart Gridlock Hackathon 2.0 — Round 2 Prototype Phase*
*June 2026*