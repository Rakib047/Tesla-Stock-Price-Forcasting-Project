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
            'Open': [100, 101, 102, 103, 104],
            'High': [101, 102, 103, 104, 105],
            'Low': [99, 100, 101, 102, 103],
            'Close': [100.5, 101.5, 102.5, 103.5, 104.5],
            'Volume': [1000, 1100, 1200, 1300, 1400],
            'Target': [101, 102, 103, 104, 105]
        })

        # Create sample test data
        df_test = pd.DataFrame({
            'Open': [105, 106],
            'High': [106, 107],
            'Low': [104, 105],
            'Close': [105.5, 106.5],
            'Volume': [1500, 1600],
            'Target': [106, 107]
        })

        model_path = tmp_path / model_name
        return df_train, df_test, str(model_path)

    return _factory

def test_linear_regression_model(sample_train_test_data):
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

        # Optional: check predicted values are floats
        assert all(isinstance(x, (float, np.floating)) for x in y_pred_flat)

        # Optional: simple sanity check that predictions are not all zeros
        assert not np.allclose(y_pred, 0)

