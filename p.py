import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Function for stock price prediction using Linear Regression
def stock_price_prediction(data):
    features = data[['Close']]
    data['Target'] = data['Close'].shift(-7)
    data.dropna(inplace=True)
    train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)
    model = LinearRegression()
    model.fit(train_data[['Close']], train_data['Target'])
    predictions = model.predict(data[['Close']])  # Predict on the entire dataset
    rmse = np.sqrt(mean_squared_error(data['Target'], predictions))
    print(f"Root Mean Squared Error: {rmse}")
    plt.plot(data.index, data['Target'], label='Actual Prices')
    plt.plot(data.index, predictions, label='Predicted Prices')
    plt.legend()
    plt.show()

    # Calculate train accuracy
    train_predictions = model.predict(train_data[['Close']])
    train_accuracy = accuracy_score(train_data['Target'], train_predictions > train_data['Close'])
    print(f"Train Accuracy: {train_accuracy}")

# Function for classification - Buy/Sell signals
def classification_buy_sell_signals(data):
    data['Target'] = (data['Close'].shift(-7) > data['Close']).astype(int)
    data.dropna(inplace=True)
    train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)
    model = RandomForestClassifier()
    model.fit(train_data[['Close']], train_data['Target'])
    predictions = model.predict(test_data[['Close']])

    # Plot confusion matrix as a heatmap
    cm = confusion_matrix(test_data['Target'], predictions)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Sell', 'Buy'], yticklabels=['Sell', 'Buy'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    # Plot bar chart for precision, recall, and F1-score
    report = classification_report(test_data['Target'], predictions, output_dict=True)
    metrics = report['1']  # Considering '1' as the positive class, adjust if needed
    plt.bar(metrics.keys(), metrics.values())
    plt.xlabel('Metrics')
    plt.ylabel('Values')
    plt.title('Classification Report Metrics')
    plt.show()

    # Calculate train accuracy
    train_predictions = model.predict(train_data[['Close']])
    train_accuracy = accuracy_score(train_data['Target'], train_predictions > train_data['Close'])
    print(f"Train Accuracy: {train_accuracy}")

# Function for technical indicator analysis
def technical_indicator_analysis(data):
    data['MA_50'] = data['Close'].rolling(window=50).mean()
    delta = data['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    average_gain = gain.rolling(window=14).mean()
    average_loss = loss.rolling(window=14).mean()
    rs = average_gain / average_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    plt.plot(data.index, data['Close'], label='Close Prices')
    plt.plot(data.index, data['MA_50'], label='50-day Moving Average')
    plt.plot(data.index, data['RSI'], label='RSI')
    plt.legend()
    plt.show()

    # Dummy train accuracy calculation (since no model training)
    print("Train Accuracy: N/A (No model trained)")

# Prompt the user to choose the analysis or visualization
print("Choose an option:")
print("1. Stock Price Prediction")
print("2. Classification - Buy/Sell Signals")
print("3. Technical Indicator Analysis")

choice = int(input("Enter the number of your choice: "))

# Fetch stock data
csv_filename = 'AAPL_stock_data.csv' # Replace with your actual CSV file
stock_data = pd.read_csv(csv_filename)
stock_data['Date'] = pd.to_datetime(stock_data['Date'])
stock_data.set_index('Date', inplace=True)

# Perform the selected analysis or visualization
if choice == 1:
    stock_price_prediction(stock_data)
elif choice == 2:
    classification_buy_sell_signals(stock_data)
elif choice == 3:
    technical_indicator_analysis(stock_data)
else:
    print("Invalid choice. Please choose a number between 1 and 3.")
