import sys
import os
sys.path.append(os.path.abspath('/Users/rakibabdullah/Desktop/Tesla-S'))
import pytest
import numpy as np
from Utils.Evaluation import *

def test_evaluate_model_metrics_and_output(capsys):
    # Sample data
    y_true = np.array([3.0, -0.5, 2.0, 7.0])
    y_pred = np.array([2.5, 0.0, 2.1, 7.8])

    mae, mse, rmse, r2, mape = evaluate_model(y_true, y_pred)

    # Assert types
    assert isinstance(mae, float)
    assert isinstance(mse, float)
    assert isinstance(rmse, float)
    assert isinstance(r2, float)
    assert isinstance(mape, float)

    # Assert reasonable value ranges (example)
    assert mae >= 0
    assert mse >= 0
    assert rmse >= 0
    assert -1 <= r2 <= 1
    assert mape >= 0

    # Capture printed output and check for expected strings
    captured = capsys.readouterr()
    assert "Mean Absolute Error" in captured.out
    assert "Mean Squared Error" in captured.out
    assert "Root Mean Squared Error" in captured.out
    assert "R-squared" in captured.out
    assert "Mean Absolute Percentage Error" in captured.out

@pytest.mark.parametrize("y_true, y_pred", [
    (np.array([0, 0, 0]), np.array([0, 0, 0])),  # all zero true values (check MAPE safe)
    (np.array([1, 2, 3]), np.array([1, 2, 3])),
])
def test_evaluate_model_handles_edge_cases(y_true, y_pred):
    mae, mse, rmse, r2, mape = evaluate_model(y_true, y_pred)
    # For identical predictions, errors should be zero
    if np.allclose(y_true, y_pred):
        assert mae == 0
        assert mse == 0
        assert rmse == 0
        assert r2 == 1
        assert mape == 0 or np.isnan(mape)  # MAPE may be nan if y_true zeros




@pytest.fixture
def sample_volatility_data():
    # Create sample df_test with Close prices and dates
    dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
    close_prices = np.linspace(100, 150, 50) + np.random.normal(0, 1, 50)
    df_test = pd.DataFrame({'Date': dates, 'Close': close_prices})
    df_test.set_index('Date', inplace=True)

    # Create df_test_scaled with Target values (simulate scaled test data)
    df_test_scaled = df_test.copy()
    df_test_scaled['Target'] = close_prices + np.random.normal(0, 0.5, 50)  # some noise around Close

    # Simulated predictions close to target but with some noise
    y_pred = df_test_scaled['Target'] + np.random.normal(0, 0.2, 50)

    return df_test, df_test_scaled, y_pred

def test_volatility_error_plot_calculations(sample_volatility_data):
    df_test, df_test_scaled, y_pred = sample_volatility_data

    # Call the function (it returns MAE and RMSE)
    mae_vol, rmse_vol = volatility_error_plot(df_test.copy(), df_test_scaled.copy(), y_pred)

    # Check returned values are floats and positive
    assert isinstance(mae_vol, float)
    assert mae_vol >= 0
    assert isinstance(rmse_vol, float)
    assert rmse_vol >= 0


