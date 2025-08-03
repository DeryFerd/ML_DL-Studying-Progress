import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
import io

st.set_page_config(page_title="ARIMA Stock Forecaster", page_icon="📈", layout="wide")

@st.cache_data(ttl=3600)
def load_data(ticker):
    data = yf.download(tickers=[ticker], start='2020-01-01', end=pd.to_datetime('today').strftime('%Y-%m-%d'))
    if data.empty:
        return None
    data = data.asfreq('B').fillna(method='ffill')
    data.ffill(inplace=True) 
    return data

with st.sidebar:
    st.title("📈 ARIMA Forecaster")
    st.markdown("---")
    ticker_input = st.text_input('Enter Stock Ticker', 'BBCA.JK')
    st.markdown("---")
    st.subheader('Model Settings')
    p_param = st.slider('AR Order (p)', 0, 5, 1)
    d_param = st.slider('Differencing Order (d)', 0, 2, 1)
    q_param = st.slider('MA Order (q)', 0, 5, 1)
    st.markdown("---")
    st.subheader('Forecast Settings')
    forecast_days = st.slider('Forecast Horizon (Days)', 7, 180, 30)
    run_button = st.button('🚀 Run Analysis', type="primary", use_container_width=True)

st.header(f'Stock Analysis & Forecast: {ticker_input.upper()}', divider='rainbow')

if not ticker_input:
    st.warning("Please enter a stock ticker in the sidebar.")
else:
    data_df = load_data(ticker_input.upper())
    if data_df is None or data_df.empty:
        st.error(f'Ticker "{ticker_input}" not found.')
    else:
        data_close = data_df['Close']
        
        st.subheader('Historical Stock Price Data')
        
        is_indonesian_stock = ticker_input.upper().endswith('.JK')

        if is_indonesian_stock:
            st.caption("Plotting with Matplotlib for IDX stock compatibility.")
            fig_hist, ax_hist = plt.subplots(figsize=(12, 6))
            ax_hist.plot(data_close.index, data_close, label='Historical Price')
            ax_hist.set_xlabel('Date')
            ax_hist.set_ylabel('Price')
            ax_hist.legend()
            ax_hist.grid(True)
            st.pyplot(fig_hist)
        else:
            st.caption("Plotting with st.line_chart for interactive view.")
            st.line_chart(data_close)

        with st.expander("View Initial Diagnostic Analysis"):
            adf_result = adfuller(data_close.dropna())
            st.write(f'**ADF Test Result:** P-value = `{adf_result[1]:.4f}`')
            if adf_result[1] > 0.05:
                st.warning('⚠️ Data is likely non-stationary. Consider using `d > 0`.')
            else:
                st.success('✅ Data is likely stationary. You might be able to use `d = 0`.')

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
            with st.spinner('Training ARIMA model...'):
                try:
                    df_train = data_close.dropna()
                    model = ARIMA(df_train, order=(p_param, d_param, q_param))
                    results = model.fit()
                    
                    st.markdown("---")
                    st.subheader('Model Results & Forecast')

                    with st.expander("View Model Validation Details", expanded=True):
                        summary_html = results.summary().as_html()
                        st.markdown(summary_html, unsafe_allow_html=True)
                        st.markdown("---")
                        param_p_values = results.pvalues.drop('const', errors='ignore')
                        if (param_p_values > 0.05).any():
                            st.error('❌ Parameter Significance Warning: One or more AR/MA parameters are not statistically significant (p-value > 0.05).')
                        else:
                            st.success('✅ Parameter Significance OK.')
                        lb_test = acorr_ljungbox(results.resid, lags=[10], return_df=True)
                        lb_p_value = lb_test['lb_pvalue'].iloc[0]
                        st.write(f'**Ljung-Box Test on Residuals:** P-value = `{lb_p_value:.4f}`')
                        if lb_p_value < 0.05:
                            st.error('❌ Residuals Are Not White Noise: Autocorrelation patterns still exist in the residuals.')
                        else:
                            st.success('✅ Residuals Are White Noise.')
                    
                    forecast_result = results.get_forecast(steps=forecast_days)
                    forecast_df = forecast_result.summary_frame(alpha=0.05)
                    
                    st.subheader('Forecast Plot')
                    
                    if is_indonesian_stock:
                        st.caption("Plotting with Matplotlib for IDX stock compatibility.")
                        fig_fc, ax_fc = plt.subplots(figsize=(12, 6))
                        ax_fc.plot(df_train.index, df_train, label='Historical Data')
                        ax_fc.plot(forecast_df.index, forecast_df['mean'], color='red', linestyle='--', label='Forecast')
                        ax_fc.fill_between(forecast_df.index, 
                                         forecast_df['mean_ci_lower'], 
                                         forecast_df['mean_ci_upper'], 
                                         color='pink', alpha=0.5, label='95% Confidence Interval')
                        ax_fc.set_title(f'Stock Price Forecast')
                        ax_fc.set_xlabel('Date')
                        ax_fc.set_ylabel('Price')
                        ax_fc.legend()
                        ax_fc.grid(True)
                        st.pyplot(fig_fc)
                    else:
                        st.caption("Plotting with st.line_chart for interactive view.")
                        
                        # Gabungkan data dengan cara yang lebih aman
                        plot_df = pd.DataFrame({
                            'Historical': df_train,
                            'Forecast': forecast_df['mean'],
                            'Upper Bound': forecast_df['mean_ci_upper'],
                            'Lower Bound': forecast_df['mean_ci_lower']
                        })
                        
                        st.line_chart(plot_df)

                except Exception as e:
                    st.error(f"Failed to train model: {e}")
