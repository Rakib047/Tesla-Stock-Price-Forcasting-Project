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
    df['Monthly_Return'] = df['Close'].pct_change() * 100
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['Volatility_5'] = df['Close'].rolling(window=5).std()
    df['Volatility_10'] = df['Close'].rolling(window=10).std()
    df['Volatility_20'] = df['Close'].rolling(window=20).std()
    df['Target'] = df['Close'].shift(-1)
    df.dropna(inplace=True)
    return df
# Function to split the dataset into train and test sets
def split_data(df, split_date='2021-01-01'):
    df_train = df[df.index < split_date].copy()
    df_test = df[df.index >= split_date].copy()
    return df_train, df_test
# Function to scale features using MinMaxScaler
def scale_features(df_train, df_test, feature_cols):
    scaler = MinMaxScaler()
    df_train_scaled = df_train.copy()
    df_test_scaled = df_test.copy()
    df_train_scaled[feature_cols] = scaler.fit_transform(df_train[feature_cols])
    df_test_scaled[feature_cols] = scaler.transform(df_test[feature_cols])
    return df_train_scaled, df_test_scaled, scaler