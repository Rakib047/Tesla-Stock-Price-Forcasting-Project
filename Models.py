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
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt 







    
