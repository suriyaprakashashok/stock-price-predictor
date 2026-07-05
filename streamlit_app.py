import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
)

from statsmodels.tsa.arima.model import ARIMA
from datetime import timedelta

# -------------------- PAGE CONFIG --------------------

st.set_page_config(
    page_title="📈 Stock Price Predictor",
    page_icon="📈",
    layout="wide",
)

st.title("📈 AI Stock Price Predictor")

st.markdown("""
Predict future stock prices using **Linear Regression** and **ARIMA**.

**Disclaimer:** Educational purpose only. Not financial advice.
""")

# -------------------- SIDEBAR --------------------

st.sidebar.header("Configuration")

popular_tickers = {
    "Apple (AAPL)": "AAPL",
    "Microsoft (MSFT)": "MSFT",
    "Google (GOOGL)": "GOOGL",
    "Amazon (AMZN)": "AMZN",
    "NVIDIA (NVDA)": "NVDA",
    "Tesla (TSLA)": "TSLA",
    "Meta (META)": "META",
    "Netflix (NFLX)": "NFLX",
    "Reliance (RELIANCE.NS)": "RELIANCE.NS",
    "TCS (TCS.NS)": "TCS.NS",
    "Infosys (INFY.NS)": "INFY.NS",
    "HDFC Bank (HDFCBANK.NS)": "HDFCBANK.NS",
    "ICICI Bank (ICICIBANK.NS)": "ICICIBANK.NS",
    "SBI (SBIN.NS)": "SBIN.NS",
    "Bitcoin (BTC-USD)": "BTC-USD",
    "Ethereum (ETH-USD)": "ETH-USD",
    "NIFTY 50 (^NSEI)": "^NSEI",
    "SENSEX (^BSESN)": "^BSESN",
    "S&P 500 (^GSPC)": "^GSPC",
    "NASDAQ (^IXIC)": "^IXIC"
}

selected_stock = st.sidebar.selectbox(
    "Choose a Stock",
    list(popular_tickers.keys())
)

custom_ticker = st.sidebar.text_input(
    "Or Enter Any Yahoo Finance Ticker",
    value=""
).strip().upper()

ticker = custom_ticker if custom_ticker else popular_tickers[selected_stock]

start_date = st.sidebar.date_input(
    "Start Date",
    value=pd.to_datetime("2020-01-01")
)

end_date = st.sidebar.date_input(
    "End Date",
    value=pd.Timestamp.today()
)

days_to_predict = st.sidebar.slider(
    "Prediction Days",
    min_value=1,
    max_value=30,
    value=7
)

model_choice = st.sidebar.selectbox(
    "Prediction Model",
    [
        "Linear Regression",
        "ARIMA"
    ]
)
# -------------------- LOAD DATA --------------------

@st.cache_data
def load_stock(symbol, start, end):
    df = yf.download(
        symbol,
        start=start,
        end=end,
        progress=False
    )

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df

data = load_stock(
    ticker,
    start_date,
    end_date
)

if data.empty:
    st.error("No stock data found.")
    st.stop()

# -------------------- COMPANY INFO --------------------

stock = yf.Ticker(ticker)

try:
    info = stock.info
except:
    info = {}

company = info.get("longName", ticker)
sector = info.get("sector", "N/A")
industry = info.get("industry", "N/A")
marketcap = info.get("marketCap", 0)
pe = info.get("trailingPE", "N/A")
website = info.get("website", "")

st.subheader(company)

c1, c2, c3, c4 = st.columns(4)

current_price = float(data["Close"].iloc[-1])
previous_price = float(data["Close"].iloc[-2])

change = current_price - previous_price
change_pct = change / previous_price * 100

c1.metric(
    "Current Price",
    f"${current_price:.2f}",
    f"{change_pct:.2f}%"
)

c2.metric(
    "Sector",
    sector
)

c3.metric(
    "Industry",
    industry
)

if marketcap != 0:
    c4.metric(
        "Market Cap",
        f"${marketcap/1e9:.2f} B"
    )
else:
    c4.metric(
        "Market Cap",
        "N/A"
    )

# -------------------- MOVING AVERAGES --------------------

data["MA20"] = data["Close"].rolling(20).mean()
data["MA50"] = data["Close"].rolling(50).mean()

# -------------------- LINEAR REGRESSION --------------------

def linear_forecast(df, days):

    close = df["Close"].values

    X = np.arange(len(close)).reshape(-1,1)
    y = close

    split = int(len(close)*0.8)

    X_train = X[:split]
    X_test = X[split:]

    y_train = y[:split]
    y_test = y[split:]

    model = LinearRegression()

    model.fit(X_train,y_train)

    test_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test,test_pred))
    mae = mean_absolute_error(y_test,test_pred)
    mape = mean_absolute_percentage_error(y_test,test_pred)
    r2 = r2_score(y_test,test_pred)

    futureX = np.arange(len(close),len(close)+days).reshape(-1,1)

    future_pred = model.predict(futureX)

    future_dates = pd.bdate_range(
        df.index[-1]+timedelta(days=1),
        periods=days
    )

    forecast = pd.DataFrame({
        "Date":future_dates,
        "Predicted Close":future_pred
    })

    metrics = {
        "RMSE":rmse,
        "MAE":mae,
        "MAPE":mape,
        "R2":r2
    }

    return forecast,metrics
# -------------------- ARIMA MODEL --------------------

def arima_forecast(df, days):

    close = df["Close"].dropna().astype(float)

    train_size = int(len(close) * 0.8)

    train = close[:train_size]
    test = close[train_size:]

    try:
        model = ARIMA(train, order=(5, 1, 0))
        fitted = model.fit()

        test_pred = fitted.forecast(steps=len(test))

        rmse = np.sqrt(mean_squared_error(test, test_pred))
        mae = mean_absolute_error(test, test_pred)
        mape = mean_absolute_percentage_error(test, test_pred)

        try:
            r2 = r2_score(test, test_pred)
        except:
            r2 = 0

        final_model = ARIMA(close, order=(5, 1, 0)).fit()

        future_pred = final_model.forecast(steps=days)

        future_dates = pd.bdate_range(
            df.index[-1] + timedelta(days=1),
            periods=days
        )

        forecast = pd.DataFrame({
            "Date": future_dates,
            "Predicted Close": future_pred.values
        })

        metrics = {
            "RMSE": rmse,
            "MAE": mae,
            "MAPE": mape,
            "R2": r2
        }

        return forecast, metrics

    except Exception as e:
        st.error(f"ARIMA Error: {e}")
        st.stop()


# -------------------- RUN PREDICTION --------------------

if model_choice == "Linear Regression":
    forecast_df, metrics = linear_forecast(data, days_to_predict)
else:
    forecast_df, metrics = arima_forecast(data, days_to_predict)


# -------------------- CANDLESTICK CHART --------------------

st.subheader("Historical Stock Price")

fig = go.Figure()

fig.add_trace(
    go.Candlestick(
        x=data.index,
        open=data["Open"],
        high=data["High"],
        low=data["Low"],
        close=data["Close"],
        name="Candlestick"
    )
)

fig.add_trace(
    go.Scatter(
        x=data.index,
        y=data["MA20"],
        mode="lines",
        name="MA20"
    )
)

fig.add_trace(
    go.Scatter(
        x=data.index,
        y=data["MA50"],
        mode="lines",
        name="MA50"
    )
)

fig.update_layout(
    height=650,
    xaxis_title="Date",
    yaxis_title="Price (USD)",
    xaxis_rangeslider_visible=False,
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)
# -------------------- FORECAST CHART --------------------

st.subheader("Future Price Prediction")

forecast_fig = go.Figure()

# Historical Close Price
forecast_fig.add_trace(
    go.Scatter(
        x=data.index,
        y=data["Close"],
        mode="lines",
        name="Historical Close",
        line=dict(color="blue", width=2)
    )
)

# Predicted Price
forecast_fig.add_trace(
    go.Scatter(
        x=forecast_df["Date"],
        y=forecast_df["Predicted Close"],
        mode="lines+markers",
        name="Forecast",
        line=dict(color="red", width=3)
    )
)

forecast_fig.update_layout(
    template="plotly_white",
    height=600,
    xaxis_title="Date",
    yaxis_title="Price (USD)",
    hovermode="x unified"
)

st.plotly_chart(forecast_fig, use_container_width=True)

# -------------------- FORECAST TABLE --------------------

st.subheader("Predicted Prices")

display_forecast = forecast_df.copy()
display_forecast["Date"] = display_forecast["Date"].dt.strftime("%Y-%m-%d")
display_forecast["Predicted Close"] = display_forecast["Predicted Close"].round(2)

st.dataframe(
    display_forecast,
    use_container_width=True,
    hide_index=True
)

# -------------------- MODEL PERFORMANCE --------------------

st.subheader("Model Performance")

m1, m2, m3, m4 = st.columns(4)

m1.metric("RMSE", f"{metrics['RMSE']:.4f}")
m2.metric("MAE", f"{metrics['MAE']:.4f}")
m3.metric("MAPE", f"{metrics['MAPE'] * 100:.2f}%")
m4.metric("R² Score", f"{metrics['R2']:.4f}")

# -------------------- DOWNLOAD CSV --------------------

csv = forecast_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="📥 Download Forecast CSV",
    data=csv,
    file_name=f"{ticker}_forecast.csv",
    mime="text/csv"
)

# -------------------- SUMMARY --------------------

st.subheader("Prediction Summary")

last_price = float(data["Close"].iloc[-1])
future_price = float(forecast_df["Predicted Close"].iloc[-1])

difference = future_price - last_price
percent_change = (difference / last_price) * 100

if difference > 0:
    st.success(
        f"The selected model predicts an increase of "
        f"{percent_change:.2f}% over the next {days_to_predict} trading day(s)."
    )
elif difference < 0:
    st.error(
        f"The selected model predicts a decrease of "
        f"{abs(percent_change):.2f}% over the next {days_to_predict} trading day(s)."
    )
else:
    st.info("The model predicts little or no price change.")
    # -------------------- COMPANY INFORMATION --------------------

st.subheader("Company Information")

info_col1, info_col2 = st.columns(2)

with info_col1:
    st.write(f"**Company:** {company}")
    st.write(f"**Ticker:** {ticker}")
    st.write(f"**Sector:** {sector}")
    st.write(f"**Industry:** {industry}")

with info_col2:
    st.write(f"**Market Cap:** {'${:,.2f} B'.format(marketcap/1e9) if marketcap else 'N/A'}")
    st.write(f"**P/E Ratio:** {pe}")

    if website:
        st.markdown(f"**Website:** [{website}]({website})")
    else:
        st.write("**Website:** N/A")


# -------------------- STOCK STATISTICS --------------------

st.subheader("Stock Statistics")

stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)

high_52 = float(data["High"].max())
low_52 = float(data["Low"].min())
avg_close = float(data["Close"].mean())
avg_volume = int(data["Volume"].mean())

stats_col1.metric("Highest Price", f"${high_52:.2f}")
stats_col2.metric("Lowest Price", f"${low_52:.2f}")
stats_col3.metric("Average Close", f"${avg_close:.2f}")
stats_col4.metric("Average Volume", f"{avg_volume:,}")


# -------------------- DATASET PREVIEW --------------------

st.subheader("Historical Dataset")

preview = data.copy()

preview = preview.reset_index()

preview["Date"] = preview["Date"].dt.strftime("%Y-%m-%d")

st.dataframe(
    preview,
    use_container_width=True,
    hide_index=True
)


# -------------------- RAW DATA DOWNLOAD --------------------

raw_csv = preview.to_csv(index=False).encode("utf-8")

st.download_button(
    "📥 Download Historical Data",
    raw_csv,
    file_name=f"{ticker}_historical_data.csv",
    mime="text/csv"
)


# -------------------- ABOUT --------------------

with st.expander("About This Project"):

    st.markdown("""
### 📈 AI Stock Price Predictor

This application demonstrates two commonly used forecasting approaches:

- **Linear Regression** – Fits a linear trend using historical closing prices.
- **ARIMA (AutoRegressive Integrated Moving Average)** – A statistical time-series forecasting model.

### Features
- Live stock data from Yahoo Finance
- Candlestick chart
- 20-day & 50-day moving averages
- Future price prediction
- Performance metrics (RMSE, MAE, MAPE, R²)
- CSV download for historical and forecast data

### Technologies Used
- Streamlit
- yFinance
- Plotly
- NumPy
- Pandas
- Scikit-learn
- Statsmodels
""")


# -------------------- FOOTER --------------------

st.markdown("---")

st.markdown(
    """
<div style="text-align:center;">
    <h4>📈 AI Stock Price Predictor</h4>
    <p>Developed using Streamlit, Plotly, yFinance, Scikit-learn and Statsmodels.</p>
    <p><b>Disclaimer:</b> Predictions are generated for educational purposes only and should not be considered financial advice.</p>
</div>
""",
    unsafe_allow_html=True,
)