import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import base64

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Tourism Forecasting",
    page_icon="🌍",
    layout="wide"
)

# -----------------------------
# Background Image Function
# -----------------------------
def set_bg(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    bg_img = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
    }}
    </style>
    """
    st.markdown(bg_img, unsafe_allow_html=True)

# 👉 Put your image in same folder and name it "bg.jpg"
try:
    set_bg("bg.jpg")
except:
    pass

# -----------------------------
# Title Section
# -----------------------------
st.markdown(
    "<h1 style='text-align: center; color: white;'>🌍 Tourism Inflow Forecasting</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<h4 style='text-align: center; color: white;'>Predict future tourist trends using Auto ARIMA</h4>",
    unsafe_allow_html=True
)

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    with open("auto_arima_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# Debug check (optional)
# st.write(type(model))

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("⚙️ Settings")

steps = st.sidebar.slider(
    "Select Forecast Periods (Quarters)",
    min_value=1,
    max_value=20,
    value=4
)

# -----------------------------
# Predict Button
# -----------------------------
if st.sidebar.button("🚀 Predict"):

    try:
        forecast = model.predict(n_periods=int(steps))

        future_dates = pd.date_range(
            start=pd.Timestamp.today(),
            periods=steps,
            freq='Q'
        )

        forecast_df = pd.DataFrame({
            "Date": future_dates,
            "Forecast": forecast
        })

        # -----------------------------
        # Layout Columns
        # -----------------------------
        col1, col2 = st.columns(2)

        # -----------------------------
        # Table
        # -----------------------------
        with col1:
            st.markdown("### 📋 Forecast Data")
            st.dataframe(forecast_df)

        # -----------------------------
        # Graph
        # -----------------------------
        with col2:
            st.markdown("### 📈 Forecast Graph")
            fig, ax = plt.subplots()
            ax.plot(forecast_df["Date"], forecast_df["Forecast"], marker='o')
            ax.set_xlabel("Date")
            ax.set_ylabel("Tourist Inflow")
            ax.grid()

            st.pyplot(fig)

        # -----------------------------
        # Download Button
        # -----------------------------
        csv = forecast_df.to_csv(index=False).encode('utf-8')

        st.download_button(
            label="📥 Download Forecast as CSV",
            data=csv,
            file_name="forecast.csv",
            mime='text/csv'
        )

        # -----------------------------
        # Success Message
        # -----------------------------
        st.success("✅ Forecast generated successfully!")

    except Exception as e:
        st.error(f"❌ Error: {e}")

# -----------------------------
# Footer
# -----------------------------
st.markdown(
    "<br><hr><p style='text-align:center;color:white;'> ❤️ </p>",
    unsafe_allow_html=True
)
