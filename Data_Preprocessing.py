import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import zscore, skew, kurtosis

# # Function to load and preprocess the dataset
def load_data(file_path):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df 

# Function to create features (returns, moving averages, volatility, target)
def create_features(df):
    # Copy the original DataFrame to preserve original features
    df_copy = df.copy()

    # Calculate new features
    df_copy['Monthly_Return'] = df_copy['Close'].pct_change() * 100
    df_copy['MA5'] = df_copy['Close'].rolling(window=5).mean()
    df_copy['MA10'] = df_copy['Close'].rolling(window=10).mean()
    df_copy['MA20'] = df_copy['Close'].rolling(window=20).mean()
    df_copy['Volatility_5'] = df_copy['Close'].rolling(window=5).std()
    df_copy['Volatility_10'] = df_copy['Close'].rolling(window=10).std()
    df_copy['Volatility_20'] = df_copy['Close'].rolling(window=20).std()

    # Create target variable (next day's closing price)
    df_copy['Target'] = df_copy['Close'].shift(-1)

    # Drop NaN values after feature creation
    df_copy.dropna(inplace=True)

    return df_copy


# Function to split the dataset into train and test sets
def split_data(df, train_size=0.8):
    """
    Splits the data into training and test sets based on the specified ratio, 
    without shuffling the data.
    
    :param df: The DataFrame with time series data
    :param train_size: The proportion of data to be used for training (default 0.8)
    :return: df_train: Training data, df_test: Test data
    """
    # Ensure the data is sorted by the Date index (important for time series)
    df = df.sort_index()

    # Calculate the index where to split the data
    split_index = int(len(df) * train_size)

    # Split the data
    df_train = df.iloc[:split_index].copy()  # First 80% for training
    df_test = df.iloc[split_index:].copy()   # Remaining 20% for testing

    return df_train, df_test


# Function to scale features using MinMaxScaler
def scale_features(df_train, df_test, feature_cols):
    scaler = MinMaxScaler()
    df_train_scaled = df_train.copy()
    df_test_scaled = df_test.copy()
    df_train_scaled[feature_cols] = scaler.fit_transform(df_train[feature_cols])
    df_test_scaled[feature_cols] = scaler.transform(df_test[feature_cols])
    return df_train_scaled, df_test_scaled, scaler