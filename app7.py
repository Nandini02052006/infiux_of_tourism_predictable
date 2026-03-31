import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Global Tourism Dashboard",
    page_icon="🌍",
    layout="wide"
)

# -----------------------------
# Cream Theme Styling
# -----------------------------
st.markdown("""
<style>

/* Background */
.stApp {
    background: linear-gradient(to right, #d9f1ff, #e6f7ff);
}

/* All text */
html, body, [class*="css"]  {
    color: black !important;
}

/* Metrics FIX */
div[data-testid="stMetric"] * {
    color: black !important;
}

/* Metric label */
div[data-testid="stMetricLabel"] {
    color: black !important;
}

/* Metric value */
div[data-testid="stMetricValue"] {
    color: black !important;
}

/* Success message */
div[data-testid="stAlert"] {
    color: black !important;
}
.stApp {
    background: linear-gradient(to right, #d9f1ff, #e6f7ff);
    color: black;
}

/* Headings */
h1, h2, h3, h4, h5, h6 {
    color: black;
}

/* Metrics (VERY IMPORTANT) */
div[data-testid="stMetric"] {
    color: black !important;
}
div[data-testid="stMetricLabel"] {
    color: black !important;
}
div[data-testid="stMetricValue"] {
    color: black !important;
}

/* Success message */
div[data-testid="stAlert"] {
    color: black !important;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Title
# -----------------------------
st.markdown("<h1 style='text-align:center;'>🌍 Global Tourism Intelligence Dashboard</h1>", unsafe_allow_html=True)

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    with open("auto_arima_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# -----------------------------
# Sidebar Settings
# -----------------------------
st.sidebar.header("⚙️ Settings")

steps = st.sidebar.slider("Forecast Periods (Quarters)", 1, 50, 6)

# -----------------------------
# ALL COUNTRIES LIST
# -----------------------------
countries = [
"Canada","United States Of America","Argentina","Brazil","Mexico",
"Austria","Belgium","Denmark","Finland","France","Germany","Greece","Ireland","Italy",
"Netherlands","Norway","Portugal","Spain","Sweden","Switzerland","United Kingdom",
"Czech Rep.","Hungary","Kazakhstan","Poland","Russian Federation","Ukraine",
"Egypt","Kenya","Mauritius","Nigeria","South Africa","Sudan","Tanzania",
"Bahrain","Iraq","Israel","Oman","Saudi Arabia","Turkey","United Arab Emirates","Yemen",
"Afghanistan","Bangladesh","Bhutan","Iran","Maldives","Nepal","Pakistan","Sri Lanka",
"Indonesia","Malaysia","Myanmar","Philippines","Singapore","Thailand","Vietnam",
"China","Japan","Korea (Republic Of)","Taiwan","Australia","New Zealand"
]

place = st.sidebar.selectbox("🌍 Select Destination", countries)

# -----------------------------
# Dynamic Info Generator
# -----------------------------
def get_info(place):
    return {
        "hotels": [f"{place} Grand Hotel", f"{place} Palace", f"{place} Residency"],
        "transport": ["Flights ✈️", "Trains 🚄", "Cabs 🚕"],
        "places": [f"Famous Spot in {place}", f"City Center {place}", f"Top Attraction {place}"]
    }

info = get_info(place)

# -----------------------------
# Display Info
# -----------------------------
st.markdown(f"## 📍 {place}")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🏨 Hotels")
    for h in info["hotels"]:
        st.write("✔️", h)

with col2:
    st.markdown("### 🚗 Transport")
    for t in info["transport"]:
        st.write("✔️", t)

with col3:
    st.markdown("### 📸 Attractions")
    for p in info["places"]:
        st.write("✔️", p)

# -----------------------------
# Forecast
# -----------------------------
if st.sidebar.button("🚀 Generate Forecast"):

    forecast = model.predict(n_periods=int(steps))

    future_dates = pd.date_range(
        start=pd.Timestamp.today(),
        periods=steps,
        freq='Q'
    )

    forecast_df = pd.DataFrame({
        "Date": future_dates.date,
        "Time": future_dates.time,
        "Tourists": forecast.astype(int)
    })

    # -----------------------------
    # Metrics
    # -----------------------------
    st.markdown("<h2 style='color:black;'>📊 Insights</h2>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(label="Average Tourists", value=int(forecast.mean()))

    with col2:
        st.metric(label="Peak Value", value=int(forecast.max()))

    with col3:
        st.metric(label="Minimum Value", value=int(forecast.min()))
    # -----------------------------
    # Table
    # -----------------------------
    st.markdown("## 📋 Forecast Table")

    st.dataframe(
        forecast_df.style.set_properties(**{
            'background-color': '#fff5e6',
            'color': '#5a4634'
        }),
        use_container_width=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # -----------------------------
    # Graph
    # -----------------------------
    st.markdown("## 📈 Forecast Trend")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(future_dates, forecast, marker='o')
    ax.set_xlabel("Date")
    ax.set_ylabel("Tourists")
    ax.grid(True)

    st.pyplot(fig)

    # -----------------------------
    # Download
    # -----------------------------
    csv = forecast_df.to_csv(index=False).encode('utf-8')

    st.download_button(
        "📥 Download Data",
        csv,
        "tourism_forecast.csv",
        "text/csv"
    )

    st.markdown(
    "<div style='color:black; font-weight:bold;'>✅ Forecast generated successfully!</div>",
    unsafe_allow_html=True
)
# -----------------------------
# Footer
# -----------------------------
st.markdown(
    "<hr><p style='text-align:center;'>🌍 Explore • Analyze • Predict ✨</p>",
    unsafe_allow_html=True
)
