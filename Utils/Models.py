import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import joblib
import pandas as pd


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
def linear_regression_model(df_train, df_test,model_filename='Models/linear_regression_model.pkl'):
    """
    Trains the Linear Regression model on the training data and tests it on the test data.
    :param df_train: Training data
    :param df_test: Test data
    :return: y_test: Actual values for the test set, y_pred: Predicted values for the test set
    """

    # Initialize the Linear Regression model
    model = LinearRegression()
    
    # Train the model
    model.fit(df_train[['Open', 'High', 'Low', 'Close', 'Volume']], df_train[['Target']])
    
        # Save the model to a file
    joblib.dump(model, model_filename)

    # Make predictions on the test set
    y_pred = model.predict(df_test[['Open', 'High', 'Low', 'Close', 'Volume']])

    # Return actual and predicted values
    return df_test['Target'], y_pred


# Random Forest Model

from sklearn.ensemble import RandomForestRegressor

def random_forest_model(df_train, df_test,model_filename='Models/random_forest_model.pkl'):
    """
    Trains the Random Forest Regressor model on the training data and tests it on the test data.
    :param df_train: Training data
    :param df_test: Test data
    :return: y_test: Actual values for the test set, y_pred: Predicted values for the test set
    """
    
    # Initialize the Random Forest Regressor model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Train the model using features and target from training data
    model.fit(df_train[['Open', 'High', 'Low', 'Close', 'Volume']], df_train['Target'])
    
    joblib.dump(model, model_filename)
    
    # Make predictions on the test set
    y_pred = model.predict(df_test[['Open', 'High', 'Low', 'Close', 'Volume']])
    
    # Return actual and predicted values
    return df_test['Target'], y_pred

from sklearn.svm import SVR

def svr_model(df_train, df_test,model_filename='Models/svr_model.pkl'):
    """
    Trains the Support Vector Regressor (SVR) model on the training data and tests it on the test data.
    :param df_train: Training data
    :param df_test: Test data
    :return: y_test: Actual values for the test set, y_pred: Predicted values for the test set
    """
    
    # Initialize the Support Vector Regressor model
    model = SVR(kernel='rbf', C=100, epsilon=0.1)
    
    # Train the model using features and target from training data
    model.fit(df_train[['Open', 'High', 'Low', 'Close', 'Volume']], df_train['Target'])
    
    joblib.dump(model, model_filename)
    
    # Make predictions on the test set
    y_pred = model.predict(df_test[['Open', 'High', 'Low', 'Close', 'Volume']])
    
    # Return actual and predicted values
    return df_test['Target'], y_pred

from sklearn.tree import DecisionTreeRegressor

def decision_tree_model(df_train, df_test,model_filename='Models/decision_tree_model.pkl'):
    """
    Trains the Decision Tree Regressor model on the training data and tests it on the test data.
    :param df_train: Training data
    :param df_test: Test data
    :return: y_test: Actual values for the test set, y_pred: Predicted values for the test set
    """
    
    # Initialize the Decision Tree Regressor model
    model = DecisionTreeRegressor(random_state=42)
    
    # Train the model using features and target from training data
    model.fit(df_train[['Open', 'High', 'Low', 'Close', 'Volume']], df_train['Target'])
    
    joblib.dump(model, model_filename)
    
    # Make predictions on the test set
    y_pred = model.predict(df_test[['Open', 'High', 'Low', 'Close', 'Volume']])
    
    # Return actual and predicted values    
    return df_test['Target'], y_pred


#XGBoost Model
import xgboost as xgb
def xgboost_model(df_train, df_test,model_filename='Models/xgboost_model.pkl'):
    """
    Trains the XGBoost model on the training data and tests it on the test data.
    :param df_train: Training data
    :param df_test: Test data
    :return: y_test: Actual values for the test set, y_pred: Predicted values for the test set
    """
    
    # Initialize the XGBoost model
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    
    # Train the model using features and target from training data
    model.fit(df_train[['Open', 'High', 'Low', 'Close', 'Volume']], df_train['Target'])
    
    joblib.dump(model, model_filename)
    
    # Make predictions on the test set
    y_pred = model.predict(df_test[['Open', 'High', 'Low', 'Close', 'Volume']])
    
    # Return actual and predicted values
    return df_test['Target'], y_pred

# Voting Ensemble

from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import xgboost as xgb

def voting_model(df_train, df_test,model_filename='Models/voting_model.pkl'):
    """
    Trains a Voting Regressor model on the training data and tests it on the test data.
    :param df_train: Training data
    :param df_test: Test data
    :return: y_test: Actual values for the test set, y_pred: Predicted values for the test set
    """
    
    # Define the base models for voting
    model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
    model_lr = LinearRegression()
    model_svr = SVR(kernel='rbf', C=100, epsilon=0.1)
    model_xgb = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)

    # Initialize the Voting Regressor
    model = VotingRegressor(estimators=[('rf', model_rf), ('lr', model_lr), ('svr', model_svr), ('xgb', model_xgb)])

    # Train the model
    model.fit(df_train[['Open', 'High', 'Low', 'Close', 'Volume']], df_train['Target'])
    
    joblib.dump(model, model_filename)

    # Make predictions on the test set
    y_pred = model.predict(df_test[['Open', 'High', 'Low', 'Close', 'Volume']])

    # Return actual and predicted values
    return df_test['Target'], y_pred



import keras_tuner as kt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

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
    features = ['Open', 'High', 'Low', 'Close', 'Volume']  # Add any other features you want
    data = df[features].values

    X, y = create_sliding_window(data, window_size)
    y = df['Target'].values[window_size:]

    X = X.reshape((X.shape[0], X.shape[1], len(features)))  # Adjust for the number of features

    return X, y

# Function to build LSTM model with hyperparameters
def build_lstm_model(hp, window_size=5):
    model = Sequential()
    
    # Tuning the number of units for LSTM layer
    model.add(LSTM(units=hp.Int('lstm_units_1', min_value=50, max_value=200, step=50), 
                   return_sequences=True, input_shape=(window_size, 5)))
    model.add(Dropout(hp.Float('dropout_1', min_value=0.1, max_value=0.5, step=0.1)))
    
    model.add(LSTM(units=hp.Int('lstm_units_2', min_value=50, max_value=200, step=50), return_sequences=False))
    model.add(Dropout(hp.Float('dropout_2', min_value=0.1, max_value=0.5, step=0.1)))
    
    # Dense layer after LSTM layers
    model.add(Dense(units=hp.Int('dense_units', min_value=64, max_value=128, step=32), activation='relu'))
    
    model.add(Dense(1))  # Output layer

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='LOG')),
                  loss='mean_squared_error', 
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])
    
    # After training your model  # or use .keras or SavedModel folder format

    
    return model

# Hyperparameter tuning function
def tune_lstm_hyperparameters(X_train, y_train, window_size=5):
    tuner = kt.RandomSearch(
        build_lstm_model,
        objective='val_loss',
        max_trials=5,  # The number of configurations to try
        executions_per_trial=1,  # Number of models to train per configuration
        directory='hyperparameter_tuning',
        project_name='lstm_hyperparameter_tuning'
    )

    tuner.search(X_train, y_train, epochs=30, batch_size=64, validation_split=0.2)

    best_hp = tuner.oracle.get_best_trials(num_trials=1)[0].hyperparameters
    print(f"Best hyperparameters: {best_hp.values}")

    best_model = tuner.hypermodel.build(best_hp)
    
    return best_model

def plot_training_history(history):
    # Plot training and validation loss
    plt.figure(figsize=(16,10))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss During Training')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Main function to train the LSTM model
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

    # Hyperparameter tuning
    best_model = tune_lstm_hyperparameters(X_train, y_train, window_size)

    # Train the model
    history_lstm = best_model.fit(X_train, y_train, epochs=30, batch_size=64, validation_split=0.2, verbose=1)

    best_model.save('Models/lstm_model.h5')
    # Plot the training history
    plot_training_history(history_lstm)

    # Make predictions
    y_pred = best_model.predict(X_test)
    
    # Return actual and predicted values
    return y_test, y_pred




#GRU Model

def preprocess_for_gru(df, window_size=5):
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    data = df[features].values

    X, y = create_sliding_window(data, window_size)
    y = df['Target'].values[window_size:]


    X = X.reshape((X.shape[0], X.shape[1], len(features)))
    
    return X, y



import keras_tuner as kt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

# Define the GRU model with hyperparameters
def build_gru_model(hp, window_size=5):
    model = Sequential()
    
    # Tuning the number of units for GRU layers
    model.add(GRU(units=hp.Int('gru_units_1', min_value=50, max_value=200, step=50), 
                  return_sequences=True, input_shape=(window_size, 5)))
    model.add(Dropout(hp.Float('dropout_1', min_value=0.1, max_value=0.5, step=0.1)))
    
    model.add(GRU(units=hp.Int('gru_units_2', min_value=50, max_value=200, step=50), return_sequences=True))
    model.add(Dropout(hp.Float('dropout_2', min_value=0.1, max_value=0.5, step=0.1)))
    
    model.add(GRU(units=hp.Int('gru_units_3', min_value=50, max_value=200, step=50), return_sequences=False))
    model.add(Dense(units=hp.Int('dense_units', min_value=64, max_value=128, step=32), activation='relu'))
    
    model.add(Dropout(hp.Float('dropout_3', min_value=0.1, max_value=0.5, step=0.1)))
    model.add(Dense(1))  # Output layer

    # Compile the model with Adam optimizer and an exponentially decaying learning rate
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='LOG'),
        decay_steps=1000,
        decay_rate=0.9
    )
    
    model.compile(optimizer=Adam(learning_rate=lr_schedule),
                  loss='mean_squared_error',
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])
      # Save the model after training
    return model


# Preprocess the data for GRU model (same as before)
def preprocess_for_gru(df, window_size=5):
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    data = df[features].values

    X, y = create_sliding_window(data, window_size)
    y = df['Target'].values[window_size:]

    X = X.reshape((X.shape[0], X.shape[1], len(features)))
    
    return X, y


# Hyperparameter tuning function for GRU model
def tune_gru_hyperparameters(X_train, y_train, window_size=5):
    tuner = kt.RandomSearch(
        build_gru_model,
        objective='val_loss',
        max_trials=5,  # The number of configurations to try
        executions_per_trial=1,  # Number of models to train per configuration
        directory='hyperparameter_tuning_gru',
        project_name='gru_hyperparameter_tuning'
    )

    tuner.search(X_train, y_train, epochs=30, batch_size=64, validation_split=0.2)

    best_hp = tuner.oracle.get_best_trials(num_trials=1)[0].hyperparameters
    print(f"Best hyperparameters: {best_hp.values}")

    best_model = tuner.hypermodel.build(best_hp)
    
    return best_model


# Final GRU model function that trains the model with the best hyperparameters found from tuning
def gru_model(df_train, df_test, window_size=5):
    X_train, y_train = preprocess_for_gru(df_train, window_size)
    X_test, y_test = preprocess_for_gru(df_test, window_size)

    # Tune the hyperparameters using the function
    best_model = tune_gru_hyperparameters(X_train, y_train, window_size)

    early_stop = EarlyStopping(patience=50, restore_best_weights=True)

    # Train the best model found from the tuning
    history_gru=best_model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=64,
        batch_size=8,
        callbacks=[early_stop],
        verbose=1
    )
    
    best_model.save('Models/gru_model.h5')
    
    plot_training_history(history_gru)

    # Make predictions on the test set
    y_pred = best_model.predict(X_test)
    
    # Return actual and predicted values
    return y_test, y_pred

from sklearn.metrics import mean_squared_error

def walk_forward_validation(df, model_func, feature_cols, target_col='Target', start=100, step=30):
    """
    Perform walk-forward validation.

    Parameters:
        df: DataFrame containing all features and target
        model_func: function that returns a fitted model (e.g., LinearRegression)
        feature_cols: list of feature column names
        target_col: name of the target column
        start: number of initial samples to train on
        step: how many steps ahead to predict (default 1)

    Returns:
        y_true: list of actual values
        y_pred: list of predicted values
    """
    y_true, y_pred = [], []
    
    train_sizes = []
    train_errors = []

    for i in range(start, len(df) - step):
        train_data = df.iloc[:i]
        test_data = df.iloc[i:i+step]

        X_train = train_data[feature_cols]
        y_train = train_data[target_col]

        X_test = test_data[feature_cols]
        y_test = test_data[target_col]

        # Train model
        model = model_func()
        model.fit(X_train, y_train)

        # Predict
        pred = model.predict(X_test)
        
                # Predict on training data to get training error
        train_pred = model.predict(X_train)
        train_mse = mean_squared_error(y_train, train_pred)

        y_true.extend(y_test.values)
        y_pred.extend(pred.flatten())
        
        train_sizes.append(len(X_train))
        train_errors.append(train_mse)
    
    # Plot training curve
    plt.figure(figsize=(8,5))
    plt.plot(train_sizes, train_errors, label='Training MSE')
    plt.xlabel('Training Set Size')
    plt.ylabel('Mean Squared Error')
    plt.title('Training Error Curve Over Increasing Training Size')
    plt.legend()
    plt.grid(True)
    plt.show()

    return y_true, y_pred

from prophet import Prophet
import plotly.graph_objects as go

def run_and_predict_prophet(df_full, future_periods,df_actual_2024=None):
    """
    Train Prophet on the full dataset and predict future values.
    
    :param df_full: DataFrame with DateTime index and 'Close' column
    :param future_periods: int, number of future periods (days) to forecast
    :return: forecast DataFrame with predictions
    """
    df_prophet = df_full[['Close']].copy()
    df_prophet.reset_index(inplace=True)
    df_prophet.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)
    
    # Train on full data
    m = Prophet()
    m.fit(df_prophet)
    
    # Save the model
    joblib.dump(m, 'Models/prophet_model.pkl')
    
    # Create dataframe to hold future dates including historical + future
    future = m.make_future_dataframe(periods=future_periods)
    
    # Predict future values
    forecast = m.predict(future)
    
    # Plot interactive plotly plot (optional)
    from prophet.plot import plot_plotly
    fig = plot_plotly(m, forecast)
    
    
    fig.add_trace(go.Scatter(
    x=df_actual_2024['ds'],
    y=df_actual_2024['y'],
    mode='lines',
    name='Actual 2024',
    marker=dict(color='red', size=6)
    ))

    
    fig.show()
    
    return forecast


# Python







    
