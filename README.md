# Taiwan Stock Quantitative Trading System (ML Optimized)

A Streamlit-based web application for backtesting Taiwan stock trading strategies. This system integrates traditional moving average (SMA) crossovers with Machine Learning (Random Forest) to optimize trading signals and reduce friction costs through hysteresis.

## Features
* **Automated Data Pipeline**: Fetches daily market data (OHLCV) directly via the FinMind API.
* **Advanced Feature Engineering**: Utilizes `pandas_ta` to extract momentum, trend, and volatility indicators (MACD, RSI, Bollinger Bands, ATR, etc.).
* **Machine Learning Optimization**: Employs a Random Forest Classifier to predict future returns.
* **Hysteresis Filter (Confidence Thresholds)**: Implements upper and lower probability thresholds to filter out market noise and minimize transaction costs.
* **Vectorized Backtesting Engine**: Simulates realistic trading environments by accounting for market friction (slippage and commission).
* **Interactive Dashboard**: Provides a dynamic UI using Streamlit and Plotly for real-time parameter tuning and equity curve visualization.

## Usage
After executing `streamlit run smart_trade.py`, simply open `http://localhost:8501`. You can dynamically adjust the stock symbol, date range, and ML/SMA parameters via the sidebar.