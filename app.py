import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Bengaluru Parking Intelligence",
    layout="wide"
)

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────

@st.cache_data
def load_data():
    dashboard_df = pd.read_csv("data/dashboard_predictions.csv")
    location_summary = pd.read_csv("data/location_summary.csv")
    return dashboard_df, location_summary

dashboard_df, location_summary = load_data()

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

RISK_ORDER = ['Low', 'Medium', 'High', 'Critical', 'Extreme']

RISK_COLORS = {
    'Low': 'green',
    'Medium': 'blue',
    'High': 'orange',
    'Critical': 'red',
    'Extreme': 'darkred'
}

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

st.sidebar.title("🚦 Navigation")
st.sidebar.caption("Bengaluru Parking Congestion Intelligence Platform")

page = st.sidebar.radio(
    "Go to",
    ["Overview Dashboard", "Congestion Impact Analysis", "Enforcement Planner"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Model:** CatBoostRegressor")
st.sidebar.markdown("**Validation R²:** 0.555")
st.sidebar.markdown("**Locations Monitored:** 169")
st.sidebar.markdown("**Data:** Bengaluru Traffic Police (ASTraM)")

# ─────────────────────────────────────────────
# PAGE 1 — OVERVIEW DASHBOARD
# ─────────────────────────────────────────────

if page == "Overview Dashboard":

    st.title("🚦 Bengaluru Parking Congestion Intelligence Platform")
    st.caption(
        "AI-powered detection of illegal parking hotspots and targeted enforcement prioritization — "
        "built on real Bengaluru Traffic Police data."
    )

    st.markdown(
        """
        > **Problem:** On-street illegal parking near commercial areas, metro stations, and intersections 
        chokes carriageways. Enforcement is currently patrol-based and reactive — no heatmap of 
        where violations cluster or when they peak. This platform changes that.
        """
    )

    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Locations Monitored", len(location_summary))
    col2.metric("Extreme Risk Zones", 
                len(location_summary[location_summary['risk_level'] == 'Extreme']))
    col3.metric("Critical Risk Zones", 
                len(location_summary[location_summary['risk_level'] == 'Critical']))
    col4.metric("Model Validation R²", "0.555")

    st.markdown("---")

    # Top 10 hotspots
    st.subheader("🔥 Top 10 Highest Risk Locations")
    top10 = location_summary.sort_values(
        'avg_predicted_violations', ascending=False
    ).head(10)

    st.dataframe(
        top10[[
            'display_location', 'avg_predicted_violations',
            'congestion_impact_score', 'risk_level',
            'officers', 'tow_trucks'
        ]].rename(columns={
            'display_location': 'Location',
            'avg_predicted_violations': 'Avg Predicted Violations',
            'congestion_impact_score': 'Congestion Impact Score',
            'risk_level': 'Risk Level',
            'officers': 'Officers Required',
            'tow_trucks': 'Tow Trucks Required'
        }),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    col_left, col_right = st.columns([1, 2])

    with col_left:
        import plotly.express as px

        risk_order = ['Low', 'Medium', 'High', 'Critical', 'Extreme']
        risk_counts = (
            location_summary['risk_level']
            .value_counts()
            .reindex(risk_order)
            .fillna(0)
            .reset_index()
        )
        risk_counts.columns = ['Risk Level', 'Count']
        
        fig = px.bar(
            risk_counts,
            x='Risk Level',
            y='Count',
            color='Risk Level',
            color_discrete_map={
                'Low': 'green',
                'Medium': 'blue',
                'High': 'orange',
                'Critical': 'red',
                'Extreme': 'darkred'
            },
            title='Risk Level Distribution Across 169 Monitored Locations'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("🗺️ Bengaluru Parking Hotspot Map")

        m = folium.Map(
            location=[12.9716, 77.5946],
            zoom_start=12,
            tiles='CartoDB positron'
        )

        max_v = location_summary['avg_predicted_violations'].max()

        for _, row in location_summary.iterrows():
            radius = 5 + (row['avg_predicted_violations'] / max_v) * 15
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=radius,
                color=RISK_COLORS.get(row['risk_level'], 'gray'),
                fill=True,
                fill_opacity=0.75,
                popup=folium.Popup(
                    f"<b>{row['display_location']}</b><br>"
                    f"Risk Level: <b>{row['risk_level']}</b><br>"
                    f"Avg Violations: {row['avg_predicted_violations']:.1f}<br>"
                    f"Congestion Impact: {row['congestion_impact_score']:.0f}/100",
                    max_width=250
                )
            ).add_to(m)

        st_folium(m, width=700, height=450)

    st.markdown("---")
    st.subheader("⚙️ How This System Works")
    st.markdown("""
    1. **Data Collection** — 248,371 real parking violation records from Bengaluru Traffic Police (ASTraM)
    2. **AI Forecasting** — CatBoostRegressor predicts future violation intensity per junction per hour
    3. **Risk Classification** — Locations ranked into 5 tiers: Low → Extreme
    4. **Congestion Impact Scoring** — Weighted by vehicle size and violation severity (e.g. lorries blocking main roads score highest)
    5. **Enforcement Planning** — Officers and tow trucks allocated proportionally to predicted risk
    """)

# ─────────────────────────────────────────────
# PAGE 2 — CONGESTION IMPACT ANALYSIS
# ─────────────────────────────────────────────

if page == "Congestion Impact Analysis":

    st.title("🚗 Congestion Impact Analysis")
    st.caption(
        "How much does each hotspot actually impact traffic flow? "
        "Scores are weighted by vehicle type (lorries > cars > bikes) "
        "and violation severity (double parking > wrong parking)."
    )

    st.info(
        "**Methodology:** The dataset contains parking violations, not direct traffic speed data. "
        "Congestion Impact Score is computed using violation frequency weighted by "
        "vehicle size and violation type — since a lorry double-parked on a main road "
        "causes significantly more congestion than a scooter in a side lane. "
        "Scores are percentile-ranked 0–100 across all monitored locations."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Extreme Impact Zones (Score 90+)",
                len(location_summary[location_summary['congestion_impact_score'] >= 90]))
    col2.metric("High Impact Zones (Score 70+)",
                len(location_summary[location_summary['congestion_impact_score'] >= 70]))
    col3.metric("Low Impact Zones (Score < 40)",
                len(location_summary[location_summary['congestion_impact_score'] < 40]))

    st.markdown("---")

    st.subheader("Top 20 Congestion Contributors")
    impact_top = location_summary.sort_values(
        'congestion_impact_score', ascending=False
    ).head(20)

    st.dataframe(
        impact_top[[
            'display_location', 'congestion_impact_score',
            'avg_predicted_violations', 'risk_level'
        ]].rename(columns={
            'display_location': 'Location',
            'congestion_impact_score': 'Congestion Impact Score (0-100)',
            'avg_predicted_violations': 'Avg Predicted Violations',
            'risk_level': 'Risk Level'
        }),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    import plotly.express as px

    risk_order = ['Low', 'Medium', 'High', 'Critical', 'Extreme']
    risk_counts = (
        location_summary['risk_level']
        .value_counts()
        .reindex(risk_order)
        .fillna(0)
        .reset_index()
    )
    risk_counts.columns = ['Risk Level', 'Count']
    
    fig = px.bar(
        risk_counts,
        x='Risk Level',
        y='Count',
        color='Risk Level',
        color_discrete_map={
            'Low': 'green',
            'Medium': 'blue',
            'High': 'orange',
            'Critical': 'red',
            'Extreme': 'darkred'
        },
        title='Risk Level Distribution Across 169 Monitored Locations'
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Congestion Impact Map")

    m2 = folium.Map(
        location=[12.9716, 77.5946],
        zoom_start=12,
        tiles='CartoDB positron'
    )

    for _, row in location_summary.iterrows():
        score = row['congestion_impact_score']
        if score >= 80:
            color = 'darkred'
        elif score >= 60:
            color = 'red'
        elif score >= 40:
            color = 'orange'
        elif score >= 20:
            color = 'blue'
        else:
            color = 'green'

        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=5 + (score / 100) * 15,
            color=color,
            fill=True,
            fill_opacity=0.75,
            popup=folium.Popup(
                f"<b>{row['display_location']}</b><br>"
                f"Congestion Impact: <b>{score:.0f}/100</b><br>"
                f"Risk Level: {row['risk_level']}",
                max_width=250
            )
        ).add_to(m2)

    st_folium(m2, width=1200, height=500)

# ─────────────────────────────────────────────
# PAGE 3 — ENFORCEMENT PLANNER
# ─────────────────────────────────────────────

if page == "Enforcement Planner":

    st.title("👮 Enforcement Resource Planner")
    st.caption(
        "Allocate officers and tow trucks to priority zones based on predicted congestion risk."
    )

    # Resource sliders
    st.subheader("Available Resources Today")
    col1, col2 = st.columns(2)
    with col1:
        total_officers = st.slider("Total Officers Available", 10, 200, 50)
    with col2:
        total_trucks = st.slider("Total Tow Trucks Available", 1, 30, 5)

    # Risk filter
    selected_risks = st.multiselect(
        "Deploy to which risk levels?",
        options=RISK_ORDER,
        default=['Critical', 'Extreme']
    )

    deployment = location_summary[
        location_summary['risk_level'].isin(selected_risks)
    ].sort_values('avg_predicted_violations', ascending=False).copy()

    if len(deployment) == 0:
        st.warning("No locations match the selected risk levels.")
    else:
        # Proportional allocation
        # Fixed allocation based on risk level — more operationally realistic
        officer_map   = {'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 5, 'Extreme': 8}
        tow_truck_map = {'Low': 0, 'Medium': 1, 'High': 1, 'Critical': 2, 'Extreme': 3}
        
        deployment['officers_assigned']   = deployment['risk_level'].map(officer_map)
        deployment['tow_trucks_assigned'] = deployment['risk_level'].map(tow_truck_map)

        st.markdown("---")
        st.subheader(f"Deployment Plan — {len(deployment)} Priority Zones")

        col1, col2, col3 = st.columns(3)
        col1.metric("Zones Selected", len(deployment))
        col2.metric("Officers to Deploy", total_officers)
        col3.metric("Tow Trucks to Deploy", total_trucks)

        st.dataframe(
            deployment[[
                'display_location', 'risk_level',
                'avg_predicted_violations', 'congestion_impact_score',
                'officers_assigned', 'tow_trucks_assigned'
            ]].rename(columns={
                'display_location': 'Location',
                'risk_level': 'Risk Level',
                'avg_predicted_violations': 'Predicted Violations',
                'congestion_impact_score': 'Impact Score',
                'officers_assigned': 'Officers',
                'tow_trucks_assigned': 'Tow Trucks'
            }),
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "📥 Download Deployment Plan (CSV)",
            deployment.to_csv(index=False),
            "enforcement_deployment_plan.csv",
            "text/csv"
        )

        st.markdown("---")
        st.subheader("📍 Deployment Map")

        m3 = folium.Map(
            location=[12.9716, 77.5946],
            zoom_start=12,
            tiles='CartoDB positron'
        )

        for _, row in deployment.iterrows():
            folium.Marker(
                location=[row['latitude'], row['longitude']],
                popup=folium.Popup(
                    f"<b>{row['display_location']}</b><br>"
                    f"Risk: {row['risk_level']}<br>"
                    f"Officers: {row['officers_assigned']}<br>"
                    f"Tow Trucks: {row['tow_trucks_assigned']}",
                    max_width=250
                ),
                icon=folium.Icon(
                    color=RISK_COLORS.get(row['risk_level'], 'gray'),
                    icon='info-sign'
                )
            ).add_to(m3)

        st_folium(m3, width=1200, height=500)