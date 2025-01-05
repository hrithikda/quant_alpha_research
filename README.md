# **Quantitative Alpha Research Platform**

## **Overview**  
The **Quantitative Alpha Research Platform** is an interactive tool for conducting financial research and backtesting trading strategies using machine learning and technical indicators. This project integrates data collection, feature engineering, model training, and a web-based dashboard to visualize stock performance and trading signals.

---

## **Key Features**
- **Data Collection:** Pulls historical stock prices using Yahoo Finance (`yfinance`).
- **Feature Engineering:** Computes key technical indicators:
  - **RSI (Relative Strength Index)**  
  - **MACD (Moving Average Convergence Divergence)**  
  - **Bollinger Bands**  
  - **Lagged Returns**
- **Machine Learning Model:** Trains a **Random Forest Classifier** to predict buy/sell signals.
- **Backtesting:** Simulates a trading strategy using **Backtrader** and evaluates performance.
- **Streamlit Dashboard:** Displays interactive charts, key metrics (ROC AUC, Confusion Matrix), and prediction samples.

---

## **Technologies Used**
- **Programming Language:** Python 3.x  
- **Libraries:**
  - `yfinance`: Historical stock data fetching.
  - `pandas`, `numpy`: Data manipulation and feature engineering.
  - `scikit-learn`: Machine learning model training and evaluation.
  - `backtrader`: Trading strategy simulation.
  - `streamlit`: Interactive web dashboard.
  - `matplotlib`: Data visualization for backtest results.

---

## **Project Structure**
```plaintext
quant_alpha_research/
├── main.py           # Main Python script
├── README.md         # Project description and instructions
├── requirements.txt  # List of dependencies
└── .gitignore        # Files to exclude from GitHub
