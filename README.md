Quantitative Alpha Research Platform
=====================================

Overview
--------

The **Quantitative Alpha Research Platform** is an interactive tool for conducting financial research and backtesting trading strategies using machine learning and technical indicators. This project integrates data collection, feature engineering, model training, and a web-based dashboard to visualize stock performance and trading signals.

Key Features
------------

- **Data Collection:** Pulls historical stock prices using Yahoo Finance (`yfinance`).
- **Feature Engineering:** Computes key technical indicators:
  - *RSI (Relative Strength Index)*  
  - *MACD (Moving Average Convergence Divergence)*  
  - *Bollinger Bands*  
  - *Lagged Returns*
- **Machine Learning Model:** Trains a *Random Forest Classifier* to predict buy/sell signals.
- **Backtesting:** Simulates a trading strategy using *Backtrader* and evaluates performance.
- **Streamlit Dashboard:** Displays interactive charts, key metrics (ROC AUC, Confusion Matrix), and prediction samples.

Technologies Used
-----------------

- **Programming Language:** Python 3.x  
- **Libraries:**

  .. code-block:: bash

     yfinance      # Historical stock data fetching
     pandas, numpy # Data manipulation and feature engineering
     scikit-learn  # Machine learning model training and evaluation
     backtrader    # Trading strategy simulation
     streamlit     # Interactive web dashboard
     matplotlib    # Data visualization for backtest results

Project Structure
-----------------

.. code-block:: text

   quant_alpha_research/
   ├── main.py           # Main Python script
   ├── README.rst        # Project description and instructions
   ├── requirements.txt  # List of dependencies
   └── .gitignore        # Files to exclude from GitHub

Installation Guide
------------------

1. **Clone the Repository:**

   .. code-block:: bash

      git clone https://github.com/hrithikda/QuantAlphaResearchPlatform.git
      cd QuantAlphaResearchPlatform

2. **Install Dependencies:**

   Ensure you have Python installed, then run:

   .. code-block:: bash

      pip install -r requirements.txt

3. **Run the Application:**

   .. code-block:: bash

      streamlit run main.py

   The dashboard will open in your default web browser.

Usage Instructions
------------------

1. Enter the desired stock ticker (e.g., ``AAPL`` for Apple) and set the date range.
2. The platform will:
   - Fetch historical stock data.
   - Compute technical indicators (RSI, MACD, Bollinger Bands, etc.).
   - Train a Random Forest model to generate buy/sell signals.
   - Run backtests and display performance metrics.
3. View results in the dashboard, including:
   - Stock price and indicator plots.
   - Model accuracy (ROC AUC score).
   - Confusion matrix and prediction samples.

Future Enhancements
-------------------

- Add live trading support using APIs (e.g., Alpaca, Interactive Brokers).
- Include Explainable AI (XAI) features to improve transparency.
- Add hyperparameter tuning improvements and support for additional models (e.g., XGBoost).

Contributions
-------------

Contributions are welcome! Feel free to fork this repository, make enhancements, and submit a pull request.

License
-------

This project is licensed under the MIT License.

Acknowledgments
---------------

- **Yahoo Finance API** for financial data.
- **Backtrader** for providing a robust backtesting framework.
- **Streamlit** for making interactive dashboards simple to create.
