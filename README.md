# 📈 AI Stock Price Predictor

## Overview

The **AI Stock Price Predictor** is an interactive web application built with **Streamlit** that forecasts stock prices using historical market data from Yahoo Finance. It enables users to analyze stock performance through interactive visualizations and generate future price predictions using machine learning and time-series forecasting techniques.

Users can select from a variety of global stocks, indices, and cryptocurrencies—or enter any valid Yahoo Finance ticker symbol—to view historical trends and predicted future prices.

## Prediction Models

### 📊 Linear Regression

A machine learning model that identifies the overall trend in historical closing prices and predicts future values based on that trend.

### 📈 ARIMA (AutoRegressive Integrated Moving Average)

A statistical time-series forecasting model that captures historical patterns and trends in stock prices to generate future predictions.

## Features

* 📡 Live stock market data from Yahoo Finance
* 📉 Interactive candlestick chart with Plotly
* 📊 20-day and 50-day Moving Average analysis
* 🔮 Future stock price prediction
* 📅 Custom date range selection
* 📈 Multiple forecasting models (Linear Regression & ARIMA)
* 📋 Model evaluation metrics (RMSE, MAE, MAPE, and R² Score)
* 📄 Historical and forecast data tables
* 📥 Download historical and predicted data as CSV files
* 🌍 Support for international stocks, market indices, and cryptocurrencies
* 🎨 Clean and responsive Streamlit user interface

## Technologies Used

* Python
* Streamlit
* yFinance
* Pandas
* NumPy
* Plotly
* Scikit-learn
* Statsmodels

## Disclaimer

This application is developed for educational and research purposes only. The predictions are generated using mathematical and statistical models and should not be considered financial or investment advice. Always conduct your own research before making investment decisions.

##Hosted Website Link

https://stock-price-predictor-06.streamlit.app/
