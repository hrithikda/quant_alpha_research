# Import necessary libraries
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

# Step 2: Feature Engineering - Adding technical indicators
def calculate_indicators(df):
    if len(df) < 20:
        raise ValueError("Not enough data points to calculate indicators. Please provide more historical data.")
    
    # Relative Strength Index (RSI)
    delta = df['Close'].diff().fillna(0)  # Replace NaNs with 0 to avoid rolling issues
    gain = np.where(delta > 0, delta, 0).flatten()  # Flatten to ensure 1D array
    loss = np.where(delta < 0, -delta, 0).flatten()  # Flatten to ensure 1D array

    # Convert to Pandas Series with index matching the DataFrame
    gain = pd.Series(gain, index=df.index)
    loss = pd.Series(loss, index=df.index)

    avg_gain = gain.rolling(window=14, min_periods=1).mean()  # 14-period rolling average
    avg_loss = loss.rolling(window=14, min_periods=1).mean()

    # Compute RSI
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD (Moving Average Convergence Divergence)
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']

    # Bollinger Bands
    df['Rolling_Mean'] = df['Close'].rolling(window=20, min_periods=1).mean()
    df['Bollinger_Upper'] = df['Rolling_Mean'] + 2 * df['Close'].rolling(window=20, min_periods=1).std()
    df['Bollinger_Lower'] = df['Rolling_Mean'] - 2 * df['Close'].rolling(window=20, min_periods=1).std()

    # Lagged Returns
    df['Lagged_Returns'] = df['Returns'].shift(1)

    # Drop any remaining NaN values after calculations
    df.dropna(inplace=True)
    return df

# Fetch historical data for stock and add technical indicators
data = fetch_stock_data('AAPL', '2015-01-01', '2023-01-01')
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
        # Prepare the current data as a feature set for prediction
        current_data = pd.Series({
            'RSI': self.dataclose[-1],  # Assume most recent price is equivalent for RSI in simplified backtest
            'MACD': self.dataclose[-1] - self.dataclose[-12],
            'Lagged_Returns': self.dataclose[-1] / self.dataclose[-2],
            'Rolling_Mean': self.dataclose[-10:].mean(),
            'Bollinger_Upper': self.dataclose[-1] + 2 * self.dataclose[-10:].std(),
            'Bollinger_Lower': self.dataclose[-1] - 2 * self.dataclose[-10:].std()
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
