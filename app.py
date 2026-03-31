import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Title
# -----------------------------
st.title("📊 Tourism Forecasting App (Auto ARIMA)")

# -----------------------------
# Load Model (cached)
# -----------------------------
@st.cache_resource
def load_model():
    with open("auto_arima_Model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

st.success("✅ Model loaded successfully!")

# -----------------------------
# User Input
# -----------------------------
steps = st.number_input(
    "Enter number of future quarters:",
    min_value=1,
    max_value=20,
    value=4
)

# -----------------------------
# Predict
# -----------------------------
if st.button("Predict"):

    try:
        forecast = model.predict(n_periods=int(steps))

        # Create future dates
        future_dates = pd.date_range(
            start=pd.Timestamp.today(),
            periods=steps,
            freq='Q'
        )

        # DataFrame output
        forecast_df = pd.DataFrame({
            "Date": future_dates,
            "Forecast": forecast
        })

        # -----------------------------
        # Show Table
        # -----------------------------
        st.subheader("📋 Forecasted Values")
        st.dataframe(forecast_df)

        # -----------------------------
        # Plot Graph
        # -----------------------------
        fig, ax = plt.subplots()
        ax.plot(forecast_df["Date"], forecast_df["Forecast"], marker='o')
        ax.set_title("📈 Future Forecast")
        ax.set_xlabel("Date")
        ax.set_ylabel("Tourist Inflow")

        st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Error: {e}")
