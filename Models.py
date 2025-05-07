import matplotlib.pyplot as plt

def simple_moving_average(df, window=5):
    """
    Predicts the next day’s price using Simple Moving Average.
    :param df: DataFrame with 'Close' price
    :param window: The size of the moving window (e.g., 5 days)
    :return: SMA predictions
    """
    df['SMA'] = df['Close'].rolling(window=window).mean()
    # The last value in the moving average is the prediction for the next day
    return df['SMA'].shift(-1)  # Predicts the next day's closing price

def plot_simple_moving_average(df):
    # Plot the test set predictions vs actual values (this is for model evaluation)
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Close'], label='Actual Price (Test)', color='blue')
    plt.plot(df.index, df['SMA_Prediction'], label='SMA Prediction (Test)', color='orange')
    plt.title('Tesla Stock Price vs SMA Prediction (Test Set)')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

