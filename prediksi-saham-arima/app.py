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
    data = yf.download(tickers=[ticker], start='2020-01-01', end=pd.to_datetime('today').strftime('%Y-%m-%d'))
    if data.empty:
        return None
    data = data.asfreq('B').fillna(method='ffill')
    return data

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
        
        # =======================================================================
        # PLOT HISTORIS MENGGUNAKAN MATPLOTLIB
        st.subheader('Data Harga Saham Historis')
        fig_hist, ax_hist = plt.subplots(figsize=(12, 6))
        ax_hist.plot(data_close.index, data_close, label='Harga Penutupan Historis')
        ax_hist.set_xlabel('Tanggal')
        ax_hist.set_ylabel('Harga')
        ax_hist.legend()
        ax_hist.grid(True)
        st.pyplot(fig_hist)
        # =======================================================================

        with st.expander("Lihat Analisis Diagnostik Awal"):
            adf_result = adfuller(data_close.dropna())
            st.write(f'**Hasil Uji ADF:** P-value = `{adf_result[1]:.4f}`')
            col1, col2 = st.columns(2)
            with col1:
                fig_acf, ax_acf = plt.subplots(figsize=(6,3))
                plot_acf(data_close.dropna(), ax=ax_acf, lags=40)
                st.pyplot(fig_acf)
            with col2:
                fig_pacf, ax_pacf = plt.subplots(figsize=(6,3))
                plot_pacf(data_close.dropna(), ax=ax_pacf, lags=40)
                st.pyplot(fig_pacf)

        if run_button:
            with st.spinner('Melatih model...'):
                try:
                    df_train = data_close.dropna()
                    model = ARIMA(df_train, order=(p_param, d_param, q_param))
                    results = model.fit()
                    
                    st.markdown("---")
                    st.subheader('Hasil Model & Forecast')
                    
                    forecast_result = results.get_forecast(steps=forecast_days)
                    forecast_df = forecast_result.summary_frame(alpha=0.05)
                    
                    # =======================================================================
                    # PLOT FORECAST MENGGUNAKAN MATPLOTLIB
                    fig_fc, ax_fc = plt.subplots(figsize=(12, 6))
                    ax_fc.plot(df_train.index, df_train, label='Data Historis')
                    ax_fc.plot(forecast_df.index, forecast_df['mean'], color='red', linestyle='--', label='Forecast')
                    ax_fc.fill_between(forecast_df.index, 
                                     forecast_df['mean_ci_lower'], 
                                     forecast_df['mean_ci_upper'], 
                                     color='pink', alpha=0.5, label='Batas Atas & Bawah (95%)')
                    ax_fc.set_title(f'Forecast Harga Saham')
                    ax_fc.set_xlabel('Tanggal')
                    ax_fc.set_ylabel('Harga')
                    ax_fc.legend()
                    ax_fc.grid(True)
                    st.pyplot(fig_fc)
                    # =======================================================================

                except Exception as e:
                    st.error(f"Gagal melatih model: {e}")
