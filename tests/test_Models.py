import sys
import os
sys.path.append(os.path.abspath('/Users/rakibabdullah/Desktop/Tesla-S'))
import pytest
import numpy as np
from Utils.Models import *

def test_simple_moving_average_model():
    data = {'Close': [10, 20, 30, 40, 50, 60, 70]}
    df = pd.DataFrame(data)

    window = 3
    sma_pred = simple_moving_average_model(df, window=window)

    expected_sma = df['Close'].rolling(window=window).mean().shift(-1)

    # Compare ignoring the Series name, allowing tolerance for floats
    pd.testing.assert_series_equal(
        sma_pred.reset_index(drop=True),
        expected_sma.reset_index(drop=True),
        check_names=False,
        check_dtype=False,
        atol=1e-8,
        rtol=1e-5
    )
    
    # Assert output length matches input length
    assert len(sma_pred) == len(df)




@pytest.fixture
def sample_train_test_data(tmp_path):
    def _factory(model_name='model.pkl'):
        # Create sample training data
        df_train = pd.DataFrame({
            'Open': [100, 101, 102, 103, 104,105,106,107,108,109],
            'High': [101, 102, 103, 104, 105,106,107,108,109,110],
            'Low': [99, 100, 101, 102, 103,104,105,106,107,108],
            'Close': [100.5, 101.5, 102.5, 103.5, 104.5,105.5,106.5, 107.5, 108.5, 109.5],
            'Volume': [1000, 1100, 1200, 1300, 1400,1500,1600,1700,1800,1900],
            'Target': [101, 102, 103, 104, 105,106,107,108,109,110]
        })

        # Create sample test data
        df_test = pd.DataFrame({
            'Open': [105, 106, 107, 108, 109, 110],
            'High': [106, 107, 108, 109, 110, 111],
            'Low': [104, 105, 106, 107, 108, 109],
            'Close': [105.5, 106.5, 107.5, 108.5, 109.5, 110.5],
            'Volume': [1500, 1600, 1700, 1800, 1900, 2000],
            'Target': [106, 107, 108, 109, 110, 111]
        })

        model_path = tmp_path / model_name
        return df_train, df_test, str(model_path)

    return _factory

def test_classical_model(sample_train_test_data):
    models = {
        "decision_tree_model.pkl": decision_tree_model,
        "linear_regression_model.pkl": linear_regression_model,
        "random_forest_model.pkl": random_forest_model,
        "xgboost_model.pkl": xgboost_model,
        "voting_model.pkl": voting_model,
        "svr_model.pkl": svr_model,
    }
    
    for model_name,model_func in models.items():
        df_train, df_test, model_path = sample_train_test_data(model_name)

        y_test, y_pred = model_func(df_train, df_test, model_filename=model_path)

        # Check types
        assert isinstance(y_test, pd.Series), "y_test should be a pandas Series"
        assert isinstance(y_pred, np.ndarray), "y_pred should be a numpy array"

        # Check lengths match
        assert len(y_test) == len(df_test)
        assert len(y_pred) == len(df_test)

        # Check model file was created
        assert os.path.exists(model_path)

        # Flatten y_pred to 1D array
        y_pred_flat = y_pred.flatten()

        #check predicted values are floats
        assert all(isinstance(x, (float, np.floating)) for x in y_pred_flat)

        #simple sanity check that predictions are not all zeros
        assert not np.allclose(y_pred, 0)

        # Edge case 1: Test set same as train set
        y_test_same, y_pred_same = model_func(df_train, df_train, model_filename=model_path)
        
        # When predicting training data, predictions should be close to target
        assert np.allclose(y_pred_same.flatten(), y_test_same.values, atol=1.0)

        # Edge case 3: Single sample test set
        single_test = df_test.iloc[[0]]
        y_test_single, y_pred_single = model_func(df_train, single_test, model_filename=model_path)
        assert len(y_test_single) == 1
        assert len(y_pred_single) == 1


from unittest.mock import MagicMock, patch
import tensorflow as tf

def test_create_sliding_window():
    data = np.arange(12)
    X, y = create_sliding_window(data, window_size=5)
    assert X.shape == (7, 5)
    assert y.shape == (7,)
    print(X[0])
    np.testing.assert_array_equal(X[0], np.array([0,1,2,3,4]))
    assert y[0] == 5

def test_preprocess_for_lstm():
    df = pd.DataFrame({
        'Open': np.arange(10),
        'High': np.arange(10) + 1,
        'Low': np.arange(10) - 1,
        'Close': np.arange(10) + 2,
        'Volume': np.arange(10) * 10,
        'Target': np.arange(10) * 5
    })
    X, y = preprocess_for_lstm(df, window_size=5)
    assert X.shape == (5, 5, 5)
    assert y.shape == (5,)


def test_build_lstm_model():
    from kerastuner.engine.hyperparameters import HyperParameters
    hp = HyperParameters()
    model = build_lstm_model(hp, window_size=5)
    assert isinstance(model, tf.keras.Model)
    assert model.output_shape == (None, 1)


from unittest.mock import MagicMock, patch
import numpy as np
import tensorflow as tf



#LSTM Model Test

import pytest
import numpy as np
import pandas as pd
import tensorflow as tf
from unittest.mock import MagicMock, patch

# Adjust this import to your actual module path
# from Utils.Models import lstm_model



@patch("Utils.Models.plot_training_history")
@patch("Utils.Models.tune_lstm_hyperparameters")
def test_lstm_model(mock_tuner, mock_plot, sample_train_test_data):
    # Create dummy model
    dummy_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(5, 5)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1)
    ])
    dummy_model.compile(optimizer='adam', loss='mse')

    df_train, df_test, _ = sample_train_test_data()

    expected_samples = len(df_test) - 5  # window_size = 5

    # Mock fit and predict behavior with correct shape
    dummy_model.fit = MagicMock(return_value=MagicMock(history={'loss': [0.1], 'val_loss': [0.2]}))
    dummy_model.predict = MagicMock(return_value=np.arange(1, expected_samples + 1).reshape(-1, 1))

    # Mock tuner and plot
    mock_tuner.return_value = dummy_model
    mock_plot.return_value = None

    # Run the lstm_model function
    y_test, y_pred = lstm_model(df_train, df_test, window_size=5)

    # Assertions
    assert isinstance(y_test, np.ndarray)
    assert isinstance(y_pred, np.ndarray)
    assert y_test.shape[0] == y_pred.shape[0]
    np.testing.assert_array_equal(y_pred.flatten(), np.arange(1, expected_samples + 1))













