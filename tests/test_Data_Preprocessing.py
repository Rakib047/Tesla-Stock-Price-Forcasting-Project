import pytest
import pandas as pd
import pandas as pd
import numpy as np

import sys
import os
sys.path.append(os.path.abspath('/Users/rakibabdullah/Desktop/Tesla-S'))

from Utils import Data_Preprocessing as dp

from sklearn.preprocessing import MinMaxScaler

# Sample CSV data to simulate loading
@pytest.fixture
def sample_csv(tmp_path):


    dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
    close_prices = np.linspace(100, 150, 50) + np.random.normal(0, 1, 50)  # smooth trend + noise
    
    df = pd.DataFrame({'Date': dates, 'Close': close_prices})
    file = tmp_path / "tesla_sample.csv"
    df.to_csv(file, index=False)
    return str(file)


def test_load_data(sample_csv):
    df = dp.load_data(sample_csv)
    assert isinstance(df.index, pd.DatetimeIndex)
    assert 'Close' in df.columns
    # Check first date is as expected
    assert df.index[0] == pd.Timestamp('2023-01-01')

def test_create_features(sample_csv):
    df = dp.load_data(sample_csv)
    df_features = dp.create_features(df)
    # Expected columns created
    expected_cols = ['Close', 'Monthly_Return', 'MA5', 'MA10', 'MA20',
                     'Volatility_5', 'Volatility_10', 'Volatility_20', 'Target']
    for col in expected_cols:
        assert col in df_features.columns
    # No NaNs remain
    assert not df_features.isnull().values.any()

def test_split_data(sample_csv):
    df = dp.load_data(sample_csv)

    df_train, df_test = dp.split_data(df, train_size=0.8)
    total_len = len(df)
    # Lengths sum to total
    assert len(df_train) + len(df_test) == total_len
    # Train is approx 80%
    assert abs(len(df_train) / total_len - 0.8) < 0.05
    # Test is approx 20%
    assert abs(len(df_test) / total_len - 0.2) < 0.05
    # Indexes sorted
    assert df_train.index[-1] < df_test.index[0]




def test_scale_features(sample_csv):
    # Load and preprocess data
    df = dp.load_data(sample_csv)
    df_features = dp.create_features(df)
    df_train, df_test = dp.split_data(df_features, train_size=0.8)
    
    feature_cols = ['Monthly_Return', 'MA5', 'MA10', 'MA20', 'Volatility_5', 'Volatility_10', 'Volatility_20']
    target_cols = ['Target']

    # Call the function to test
    df_train_scaled, df_test_scaled, scaler_X, scaler_y = dp.scale_features(df_train, df_test, feature_cols, target_cols)
    
    tolerance = 1e-8

    # Check scaled feature values in train are between 0 and 1 (within tolerance)
    for col in feature_cols:
        assert col in df_train_scaled.columns
        assert df_train_scaled[col].min() >= 0 - tolerance
        assert df_train_scaled[col].max() <= 1 + tolerance

    # Check scaled target values in train are between 0 and 1 (within tolerance)
    for col in target_cols:
        assert col in df_train_scaled.columns
        assert df_train_scaled[col].min() >= 0 - tolerance
        assert df_train_scaled[col].max() <= 1 + tolerance

    # For test set, values may go slightly outside [0,1], so just check columns exist and types
    for col in feature_cols + target_cols:
        assert col in df_test_scaled.columns
        assert df_test_scaled[col].dtype.kind in 'fi'  # float or int
        assert len(df_test_scaled[col]) > 0

    # Check scalers are instances of MinMaxScaler
    assert isinstance(scaler_X, MinMaxScaler)
    assert isinstance(scaler_y, MinMaxScaler)

