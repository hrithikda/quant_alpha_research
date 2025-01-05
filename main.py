# Importing necessary libraries
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import backtrader as bt
import streamlit as st

# Step 1: Data Collection
def fetch_stock_data(ticker, start_date, end_date):
    print(f"Fetching data for {ticker} from {start_date} to {end_date}...")
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    stock_data['Returns'] = stock_data['Close'].pct_change()
    stock_data.dropna(inplace=True)
    return stock_data

# Load data for AAPL stock
data = fetch_stock_data('AAPL', '2015-01-01', '2023-01-01')

# Step 2: Feature Engineering - Adding technical indicators
def calculate_indicators(df):
    # Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14).mean()
    avg_loss = pd.Series(loss).rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD (Moving Average Convergence Divergence)
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']

    # Bollinger Bands
    df['Rolling_Mean'] = df['Close'].rolling(window=20).mean()
    df['Bollinger_Upper'] = df['Rolling_Mean'] + 2 * df['Close'].rolling(window=20).std()
    df['Bollinger_Lower'] = df['Rolling_Mean'] - 2 * df['Close'].rolling(window=20).std()

    # Lagged Returns
    df['Lagged_Returns'] = df['Returns'].shift(1)

    # Clean up NaNs from calculations
    df.dropna(inplace=True)
    return df

# Add features to the data
data = calculate_indicators(data)

# Step 3: Train-Test Split and Model Training
features = ['RSI', 'MACD', 'Lagged_Returns', 'Rolling_Mean', 'Bollinger_Upper', 'Bollinger_Lower']
X = data[features]
y = (data['Returns'] > 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model with hyperparameter tuning
print("Training the Random Forest classifier...")
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='roc_auc')
grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_

# Step 4: Model Evaluation
predictions = best_rf.predict(X_test)
roc_auc = roc_auc_score(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)
print("\nModel Performance:")
print(f"ROC AUC Score: {roc_auc:.2f}")
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", classification_report(y_test, predictions))

# Step 5: Backtesting Strategy in Backtrader
class MLBacktestStrategy(bt.Strategy):
    def __init__(self):
        self.dataclose = self.datas[0].close
        self.model = best_rf

    def next(self):
        current_data = pd.Series({
            'RSI': self.data.close[0],  # Assume most recent features
            'MACD': self.data.close[0] - self.data.close[-12],
            'Lagged_Returns': self.data.close[-1] / self.data.close[-2],
            'Rolling_Mean': self.data.close[-10:].mean(),
            'Bollinger_Upper': self.data.close[-1] + 2 * self.data.close[-1:].std(),
            'Bollinger_Lower': self.data.close[-1] - 2 * self.data.close[-1:].std()
        })

        signal = self.model.predict([current_data])[0]
        if signal == 1 and self.position.size == 0:
            self.buy()
        elif signal == 0 and self.position.size > 0:
            self.sell()

# Backtrader Engine
print("\nRunning backtest...")
cerebro = bt.Cerebro()
cerebro.addstrategy(MLBacktestStrategy)
cerebro.adddata(bt.feeds.PandasData(dataname=data))
cerebro.broker.setcash(100000)
cerebro.run()
cerebro.plot()

# Step 6: Interactive Streamlit Dashboard
def run_dashboard():
    st.title("Quant Alpha Research Platform")
    st.subheader("Stock Price and Technical Indicators")
    st.line_chart(data[['Close', 'Rolling_Mean', 'Bollinger_Upper', 'Bollinger_Lower']])
    st.write(f"**ROC AUC Score:** {roc_auc:.2f}")
    st.write("**Confusion Matrix:**")
    st.dataframe(pd.DataFrame(conf_matrix, columns=["Predicted 0", "Predicted 1"], index=["Actual 0", "Actual 1"]))
    st.write("Sample Predictions:")
    st.dataframe(pd.DataFrame({"Actual": y_test.values, "Predicted": predictions}).head())

if __name__ == "__main__":
    run_dashboard()
