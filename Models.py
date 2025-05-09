import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression



#Simple Moving Average (SMA) Model

def simple_moving_average_model(df, window=5):
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
    

# Linear Regression Model

# Function to train and test the Linear Regression model
def linear_regression_model(df_train, df_test):
    """
    Trains the Linear Regression model on the training data and tests it on the test data.
    :param df_train: Training data
    :param df_test: Test data
    :return: y_test: Actual values for the test set, y_pred: Predicted values for the test set
    """

    # Initialize the Linear Regression model
    model = LinearRegression()
    
    # Train the model
    model.fit(df_train[['Low','High','Open','Close','Volume' ,'Monthly_Return', 'MA5', 'MA10', 'MA20']], df_train[['Target']])

    # Make predictions on the test set
    y_pred = model.predict(df_test[['Low','High','Open','Close','Volume' ,'Monthly_Return', 'MA5', 'MA10', 'MA20']])

    # Return actual and predicted values
    return df_test[['Target']], y_pred


# Random Forest Model

from sklearn.ensemble import RandomForestRegressor

def random_forest_model(df_train, df_test):
    """
    Trains the Random Forest Regressor model on the training data and tests it on the test data.
    :param df_train: Training data
    :param df_test: Test data
    :return: y_test: Actual values for the test set, y_pred: Predicted values for the test set
    """
    
    # Initialize the Random Forest Regressor model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Train the model using features and target from training data
    model.fit(df_train[['Low', 'High', 'Open', 'Close', 'Volume', 'Monthly_Return', 'MA5', 'MA10', 'MA20']], df_train['Target'])
    
    # Make predictions on the test set
    y_pred = model.predict(df_test[['Low', 'High', 'Open', 'Close', 'Volume', 'Monthly_Return', 'MA5', 'MA10', 'MA20']])
    
    # Return actual and predicted values
    return df_test['Target'], y_pred

from sklearn.svm import SVR

def svr_model(df_train, df_test):
    """
    Trains the Support Vector Regressor (SVR) model on the training data and tests it on the test data.
    :param df_train: Training data
    :param df_test: Test data
    :return: y_test: Actual values for the test set, y_pred: Predicted values for the test set
    """
    
    # Initialize the Support Vector Regressor model
    model = SVR(kernel='rbf', C=100, epsilon=0.1)
    
    # Train the model using features and target from training data
    model.fit(df_train[['Low', 'High', 'Open', 'Close', 'Volume', 'Monthly_Return', 'MA5', 'MA10', 'MA20']], df_train['Target'])
    
    # Make predictions on the test set
    y_pred = model.predict(df_test[['Low', 'High', 'Open', 'Close', 'Volume', 'Monthly_Return', 'MA5', 'MA10', 'MA20']])
    
    # Return actual and predicted values
    return df_test['Target'], y_pred

from sklearn.tree import DecisionTreeRegressor

def decision_tree_model(df_train, df_test):
    """
    Trains the Decision Tree Regressor model on the training data and tests it on the test data.
    :param df_train: Training data
    :param df_test: Test data
    :return: y_test: Actual values for the test set, y_pred: Predicted values for the test set
    """
    
    # Initialize the Decision Tree Regressor model
    model = DecisionTreeRegressor(random_state=42)
    
    # Train the model using features and target from training data
    model.fit(df_train[['Low', 'High', 'Open', 'Close', 'Volume', 'Monthly_Return', 'MA5', 'MA10', 'MA20']], df_train['Target'])
    
    # Make predictions on the test set
    y_pred = model.predict(df_test[['Low', 'High', 'Open', 'Close', 'Volume', 'Monthly_Return', 'MA5', 'MA10', 'MA20']])
    
    # Return actual and predicted values    
    return df_test['Target'], y_pred


import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def create_sliding_window(data, window_size=5):
    """
    Converts time series data into sliding windows for LSTM input.
    :param data: The input time series data (e.g., stock prices).
    :param window_size: The number of time steps used to predict the next step.
    :return: X (input features), y (target/labels)
    """
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])  # Input sequence (window)
        y.append(data[i + window_size])    # Target value (next value in time series)
    return np.array(X), np.array(y)

def preprocess_for_lstm(df, window_size=5):
    """
    Preprocess the data for LSTM by creating sliding windows.
    :param df: DataFrame containing the time series data
    :param window_size: The size of the sliding window for creating sequences
    :return: X (features), y (targets)
    """
    # Select multiple features, not just 'Close'
    features = ['Close', 'Open', 'High', 'Low', 'Volume', 'MA5', 'MA10', 'MA20']  # Add any other features you want
    data = df[features].values  # Extract multiple features as input

    # Create sliding windows
    X, y = create_sliding_window(data, window_size)

    # Reshape X for LSTM: (samples, time_steps, features)
    X = X.reshape((X.shape[0], X.shape[1], len(features)))  # Adjust for the number of features

    return X, y


def build_lstm_model(X_train):
    """
    Builds and compiles an LSTM model for time series prediction.
    :param X_train: Training data
    :return: model: Compiled LSTM model
    """
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])))  # Corrected input shape
    model.add(Dense(1))  # Output layer
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def lstm_model(df_train, df_test, window_size=5):
    """
    Trains the LSTM model on the training data and makes predictions on the test data.
    :param df_train: Training data
    :param df_test: Test data
    :param window_size: Size of the sliding window
    :return: y_test, y_pred: Actual values for the test set and predicted values
    """
    # Preprocess data for LSTM
    X_train, y_train = preprocess_for_lstm(df_train, window_size)
    X_test, y_test = preprocess_for_lstm(df_test, window_size)
    
    print(len(X_test))


    # Build the LSTM model
    model = build_lstm_model(X_train)

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32)

    # Make predictions
    y_pred = model.predict(X_test)
    
    # Flatten the arrays to make them 1D
    y_test = y_test.flatten()  # Ensure y_test is 1D
    y_pred = y_pred.flatten()  # Ensure y_pred is 1D
    
    assert len(y_test) == len(y_pred), f"Mismatch in number of samples: {len(y_test)} vs {len(y_pred)}"

    # Return actual and predicted values
    return y_test, y_pred





    
