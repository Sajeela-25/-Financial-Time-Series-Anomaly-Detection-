#!/usr/bin/env python
# coding: utf-8

# In[1]:


import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from prophet import Prophet

# Step 1: Download historical stock data
ticker = 'AAPL'  # Change as needed
data = yf.download(ticker, start='2018-01-01', end='2024-12-31', auto_adjust=True)
data = data[['Close']].dropna()

# Step 2: Compute Financial Indicators
data['SMA_20'] = data['Close'].rolling(window=20).mean()
data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
data['STD_20'] = data['Close'].rolling(window=20).std()
data['Upper_BB'] = data['SMA_20'] + 2 * data['STD_20']
data['Lower_BB'] = data['SMA_20'] - 2 * data['STD_20']

# RSI function
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

data['RSI_14'] = compute_rsi(data['Close'])

# Step 3: Anomaly Detection using Isolation Forest
features = data[['Close', 'SMA_20', 'EMA_20', 'RSI_14']].dropna()
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

model = IsolationForest(contamination=0.01, random_state=42)
data['anomaly'] = -1
data.loc[features.index, 'anomaly'] = model.fit_predict(scaled_features)

# Step 4: Time-Series Forecasting using Prophet
df_prophet = data[['Close']].dropna().copy()
df_prophet['ds'] = df_prophet.index  # set index as datetime
df_prophet['y'] = df_prophet['Close']
df_prophet = df_prophet[['ds', 'y']]

prophet_model = Prophet(daily_seasonality=True)
prophet_model.fit(df_prophet)

future = prophet_model.make_future_dataframe(periods=60)
forecast = prophet_model.predict(future)

# Step 5: Visualization
plt.figure(figsize=(14, 7))
plt.plot(data.index, data['Close'], label='Close Price')
plt.scatter(data[data['anomaly'] == -1].index, data[data['anomaly'] == -1]['Close'],
            color='red', label='Anomaly', marker='x')
plt.title(f"{ticker} - Anomaly Detection")
plt.legend()
plt.grid()
plt.show()

# Forecast plot
prophet_model.plot(forecast)
plt.title(f"{ticker} - Price Forecast with Prophet")
plt.show()


# In[ ]:




