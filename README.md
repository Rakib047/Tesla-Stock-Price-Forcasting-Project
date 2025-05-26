# Tesla Stock Price Forecasting Project

## Overview

This project implements a comprehensive pipeline for forecasting Tesla (TSLA) stock prices using multiple machine learning and deep learning techniques. The project covers data preprocessing, exploratory data analysis (EDA), model training (classical and neural network models), hyperparameter tuning, model evaluation, and visualization of results.

The aim is to predict stock price trends accurately and analyze model performance during both normal and volatile market conditions.

---

## Features

* **Data Loading & Preprocessing**

  * Handles date parsing, feature engineering (returns, moving averages, volatility)
  * Data splitting and scaling with saved scaler models
  * Data augmentation techniques for improved training robustness

* **Exploratory Data Analysis (EDA)**

  * Distribution plots, seasonal trends, volatility, correlation heatmaps
  * Outlier detection using z-score and IQR methods
  * Visualization of volume vs price movement and cumulative returns

* **Model Implementations**

  * Classical ML models: Linear Regression, Random Forest, Decision Tree, Support Vector Regression (SVR), XGBoost, Voting Ensemble
  * Deep learning models: LSTM and GRU with hyperparameter tuning (using Keras Tuner)
  * Time series forecasting using Facebook Prophet

* **Model Evaluation**

  * Metrics: MAE, MSE, RMSE, R², MAPE
  * Performance analysis during high-volatility periods
  * Walk-forward validation for robust time series evaluation

* **Visualization**

  * Interactive plots using Plotly and static plots with Matplotlib
  * Comparison of actual vs predicted prices for train/test datasets
  * Performance comparison charts across models

* **Testing**

  * Unit tests for data preprocessing, modeling, and evaluation modules using pytest

* **Model Inference**

  * Utilities to load saved models (.pkl and .h5) and verify their structures

---

## Repository Structure

```
├── Data/
│   ├── Tesla_Stock_Updated_V2.csv    # Historical stock data CSV
│   ├── df_train.csv                   # Training split CSV
│   ├── df_test.csv                    # Testing split CSV
│   ├── tesla_2024.csv                 # Actual 2024 data for Prophet validation
│   └── tesla_2025.csv                 # Raw stock data including future dates
├── Models/                           # Saved model files (.pkl, .h5)
├── Scaler/                          # Saved scalers for features and target
├── Utils/
│   ├── Data_Preprocessing.py         # Data loading, feature engineering, scaling
│   ├── EDA.py                        # Exploratory data analysis functions
│   ├── Models.py                     # Model definitions, training and prediction
│   ├── Evaluation.py                 # Model evaluation metrics and error analysis
│   ├── plot_prediction.py            # Visualization of predictions
│   └── tests/                      # Unit tests for each module
├── hyperparameter_tuning/            # Keras Tuner setups for LSTM and GRU
├── model_training.ipynb              # End-to-end training and evaluation notebook
├── model_inferencing.py              # Utilities to load saved models
├── requirements.txt                  # Required Python packages
├── prediction_results.png            # Sample output visualization
└── README.md                        # This file
```

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Rakib047/Tesla-Stock-Price-Forcasting-Project.git
cd Tesla-Stock-Price-Forcasting-Project
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

Ensure you have Python 3.8+ and packages including TensorFlow, scikit-learn, xgboost, keras-tuner, prophet, plotly, pandas, matplotlib, seaborn, and pytest.

---

## Usage

### Data Preparation and EDA

* Load raw Tesla stock data and create features such as returns, moving averages, and volatility.
* Split the dataset into train and test sets chronologically.
* Scale the features and target using MinMaxScaler.
* Perform EDA with various plots to understand data distribution, seasonality, and outliers.

### Model Training

* Run the full training pipeline in `model_training.ipynb`.
* Train classical models (Linear Regression, Random Forest, SVR, XGBoost, Voting Ensemble).
* Train deep learning models (LSTM and GRU) with hyperparameter tuning.
* Evaluate each model using multiple metrics and visualize results.
* Perform walk-forward validation for time series robustness.

### Forecasting with Prophet

* Use Facebook Prophet for long-term forecasting and compare predictions against actual 2024 data.

### Model Inference

* Load saved models from the `Models/` directory using `model_inferencing.py`.
* Use the trained models to make predictions on new data.

---

## Testing

Unit tests ensure reliability for data processing, modeling, and evaluation. Run tests with:

```bash
pytest
```

Tests cover:

* Data loading, feature engineering, scaling
* Model training functions for classical and deep learning models
* Evaluation metrics and edge cases

---

## Results

* Metrics and visualization charts summarize model performances.
* Evaluation during volatile periods helps understand model robustness.
* Interactive plots facilitate detailed comparison of actual vs predicted prices.

---

## Acknowledgments

* Data sourced from public Tesla stock price datasets.
* Utilizes open-source Python libraries including TensorFlow, scikit-learn, XGBoost, Prophet, and Plotly.

