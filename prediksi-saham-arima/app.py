import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
import io

st.set_page_config(page_title="ARIMA Stock Forecaster", page_icon="📈", layout="wide")

@st.cache_data(ttl=3600)
def load_data(ticker):
    try:
        data = yf.download(tickers=[ticker], start='2020-01-01', end=pd.to_datetime('today').strftime('%Y-%m-%d'))
        if data.empty:
            return None
        
        # Fix deprecated fillna method
        data = data.asfreq('B').ffill()
        
        # Ensure timezone-naive datetime index for Streamlit compatibility
        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)
            
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

with st.sidebar:
    st.title("📈 ARIMA Forecaster")
    st.markdown("---")
    ticker_input = st.text_input('Masukkan Ticker Saham', 'BBCA.JK')
    st.markdown("---")
    st.subheader('Pengaturan Model')
    p_param = st.slider('Order AR (p)', 0, 5, 1)
    d_param = st.slider('Order Differencing (d)', 0, 2, 1)
    q_param = st.slider('Order MA (q)', 0, 5, 1)
    st.markdown("---")
    st.subheader('Pengaturan Forecast')
    forecast_days = st.slider('Pilih Jumlah Hari Forecast', 7, 180, 30)
    run_button = st.button('🚀 Jalankan Analisis', type="primary", use_container_width=True)

st.header(f'Analisis & Peramalan Saham: {ticker_input.upper()}', divider='rainbow')

if not ticker_input:
    st.warning("Silakan masukkan ticker saham di sidebar.")
else:
    data_df = load_data(ticker_input.upper())
    if data_df is None or data_df.empty:
        st.error(f'Ticker "{ticker_input}" tidak ditemukan.')
    else:
        data_close = data_df['Close']
        
        # Debug information
        st.write(f"Data shape: {data_close.shape}")
        st.write(f"Data range: {data_close.index.min()} to {data_close.index.max()}")
        st.write(f"Has NaN values: {data_close.isna().sum()}")
        
        # Alternative approach for displaying historical price
        st.subheader('Data Harga Saham Historis')
        
        # Method 1: Use st.line_chart with clean data
        try:
            clean_data = data_close.dropna()
            if len(clean_data) > 0:
                st.line_chart(clean_data)
            else:
                st.error("No valid data points to display")
        except Exception as e:
            st.error(f"Error with st.line_chart: {e}")
            # Fallback to matplotlib
            fig_hist, ax_hist = plt.subplots(figsize=(12, 6))
            ax_hist.plot(data_close.index, data_close.values)
            ax_hist.set_title('Historical Stock Price')
            ax_hist.set_xlabel('Date')
            ax_hist.set_ylabel('Price')
            ax_hist.grid(True)
            st.pyplot(fig_hist)

        with st.expander("Lihat Analisis Diagnostik Awal"):
            clean_data = data_close.dropna()
            if len(clean_data) > 0:
                adf_result = adfuller(clean_data)
                st.write(f'**Hasil Uji ADF:** P-value = `{adf_result[1]:.4f}`')
                col1, col2 = st.columns(2)
                with col1:
                    fig_acf, ax_acf = plt.subplots(figsize=(6,3))
                    plot_acf(clean_data, ax=ax_acf, lags=min(40, len(clean_data)//4))
                    ax_acf.set_title('ACF Plot')
                    st.pyplot(fig_acf)
                with col2:
                    fig_pacf, ax_pacf = plt.subplots(figsize=(6,3))
                    plot_pacf(clean_data, ax=ax_pacf, lags=min(40, len(clean_data)//4))
                    ax_pacf.set_title('PACF Plot')
                    st.pyplot(fig_pacf)
            else:
                st.error("Insufficient data for analysis")

        if run_button:
            with st.spinner('Melatih model...'):
                try:
                    df_train = data_close.dropna()
                    if len(df_train) < 10:
                        st.error("Insufficient data for training ARIMA model")
                        st.stop()
                        
                    model = ARIMA(df_train, order=(p_param, d_param, q_param))
                    results = model.fit()
                    
                    st.markdown("---")
                    st.subheader('Hasil Model & Forecast')
                    
                    # Display model summary
                    with st.expander("Model Summary"):
                        st.text(str(results.summary()))
                    
                    forecast_result = results.get_forecast(steps=forecast_days)
                    forecast_df = forecast_result.summary_frame(alpha=0.05)
                    
                    # Create forecast dates
                    last_date = df_train.index[-1]
                    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                                 periods=forecast_days, freq='B')
                    forecast_df.index = forecast_dates
                    
                    # Plot forecast
                    fig_fc, ax_fc = plt.subplots(figsize=(12, 6))
                    
                    # Plot last 60 days of historical data
                    recent_data = df_train.tail(60)
                    ax_fc.plot(recent_data.index, recent_data, label='Data Historis', color='blue')
                    ax_fc.plot(forecast_df.index, forecast_df['mean'], color='red', linestyle='--', label='Forecast')
                    ax_fc.fill_between(forecast_df.index, 
                                     forecast_df['mean_ci_lower'], 
                                     forecast_df['mean_ci_upper'], 
                                     color='pink', alpha=0.5, label='Confidence Interval (95%)')
                    ax_fc.set_title(f'Forecast Harga Saham {ticker_input.upper()}')
                    ax_fc.set_xlabel('Tanggal')
                    ax_fc.set_ylabel('Harga')
                    ax_fc.legend()
                    ax_fc.grid(True)
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig_fc)
                    
                    # Display forecast table
                    st.subheader('Tabel Forecast')
                    forecast_display = forecast_df[['mean', 'mean_ci_lower', 'mean_ci_upper']].round(2)
                    forecast_display.columns = ['Prediksi', 'Batas Bawah', 'Batas Atas']
                    st.dataframe(forecast_display)

                except Exception as e:
                    st.error(f"Gagal melatih model: {e}")
                    st.write("Error details:", str(e))
