
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Stock Analysis Pro",
    page_icon="📈",
    layout="wide"
)

# --- FUNCTIONS ---

# Cache akan membuat versi baru saat ticker, start_date, atau end_date berubah
@st.cache_data
def load_data(ticker, start_date, end_date):
    """Downloads stock data from Yahoo Finance."""
    data = yf.download(ticker, start=start_date, end=end_date)
    if data.empty:
        return pd.DataFrame()
    return data[['Close']]

# Cache akan membuat model baru saat 'data_series' yang menjadi input berubah
@st.cache_resource
def train_arima_model(data_series):
    """Trains an ARIMA(5,1,1) model."""
    model = ARIMA(data_series, order=(5, 1, 1))
    model_fit = model.fit()
    return model_fit

def calculate_var(data):
    """Calculates the 95% Daily Value at Risk."""
    daily_returns = data['Close'].pct_change().dropna()
    if daily_returns.empty:
        return 0.0
    var_95 = daily_returns.quantile(0.05)
    return var_95.item()

# --- APP LAYOUT ---

st.title('📈 Stock Analysis & Forecasting Dashboard')
st.caption(f"Last updated: {pd.to_datetime('today').strftime('%Y-%m-%d')}")

# --- SIDEBAR FOR USER INPUT ---
with st.sidebar:
    st.header("⚙️ User Inputs")
    ticker_symbol = st.text_input("Enter Stock Ticker", "AAPL", key="ticker_input").upper()
    start_date = st.date_input("Start Date", pd.to_datetime('2020-01-01'), key="start_date")
    end_date = st.date_input("End Date", pd.to_datetime('today'), key="end_date")
    run_button = st.button("Run Analysis", type="primary", key="run_analysis_button")

# --- MAIN CONTENT ---
if run_button:
    with st.spinner(f"Loading data for {ticker_symbol}..."):
        close_df = load_data(ticker_symbol, start_date, end_date)

    if close_df.empty:
        st.error("Could not load data. Please check the ticker symbol and date range.")
    else:
        st.success(f"Data for {ticker_symbol} loaded successfully!")

        st.header("Historical Price Data")
        st.line_chart(close_df['Close'])

        col1, col2 = st.columns(2)

        with col1:
            st.header("ARIMA Forecast")
            with st.spinner("Training ARIMA model and forecasting..."):
                # Panggil fungsi dengan data yang benar
                model_fit = train_arima_model(close_df['Close'])

                forecast_result = model_fit.get_forecast(steps=30)
                predicted_mean = forecast_result.predicted_mean

                last_date = close_df.index[-1]
                future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30)
                forecast_df = pd.DataFrame({'Forecast': predicted_mean.values}, index=future_dates)

                st.line_chart(forecast_df)
                st.write("Forecasted values for the next 30 days:")
                st.dataframe(forecast_df.style.format("{:.2f}"))

        with col2:
            st.header("Risk Analysis (VaR)")
            with st.spinner("Calculating Value at Risk..."):
                var_95_value = calculate_var(close_df)

                st.metric(
                    label="Daily Value at Risk (95%)",
                    value=f"{var_95_value:.2%}",
                    help="This means we are 95% confident that the maximum loss in one day will not exceed this value."
                )

        st.info(
            "**Note on Hybrid Model (ARIMA+XGBoost):** "
            "For simplicity, this app uses the ARIMA model for forecasting. "
            "Integrating the XGBoost model for forecasting residuals requires a more complex, iterative prediction loop which is better suited for offline analysis."
        )
else:
    st.info("Please enter a stock ticker and click 'Run Analysis' to begin.")
