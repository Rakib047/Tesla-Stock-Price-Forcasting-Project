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