# Tesla Stock Price Forecasting Project

## Overview

This project implements a comprehensive pipeline for forecasting Tesla (TSLA) stock prices using multiple machine learning and deep learning techniques. The project covers data preprocessing, exploratory data analysis (EDA), model training (classical and neural network models), hyperparameter tuning, model evaluation, and visualization of results.

The aim is to predict stock price trends accurately and analyze model performance during both normal and volatile market conditions.

## Problem Statement / Motivation
Stock price prediction, particularly for volatile stocks like Tesla (TSLA), presents a significant challenge due to market dynamics, investor sentiment, and numerous macroeconomic factors. This project aims to tackle this challenge by implementing and comparing various machine learning and deep learning models. The goal is to identify robust forecasting techniques that can provide insights into future price movements, aiding in more informed decision-making.

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

  * Classical ML models: Linear Regression, Random Forest, Decision Tree, Support Vector Regression (SVR), XGBoost, Voting Ensemble (saved as `.pkl` files using `joblib`)
  * Deep learning models: LSTM and GRU with hyperparameter tuning (using Keras Tuner) (saved as `.h5` files)
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
├── hyperparameter_tuning/            # Keras Tuner setups for LSTM and GRU
├── tests/                            # Unit tests for each module
├── model_training.ipynb              # End-to-end training and evaluation notebook
├── model_inferencing.ipynb           # Utilities to load saved models
├── requirements.txt                  # Required Python packages
├── prediction_results.png            # Sample output visualization
└── README.md                        # This file
```

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Rakib047/Tesla-Stock-Price-Forecasting-Project.git
cd Tesla-Stock-Price-Forecasting-Project
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

*   Run the `model_training.ipynb` notebook. This Jupyter Notebook provides a step-by-step guide through the entire pipeline:
    *   Loading and preprocessing the Tesla stock data.
    *   Performing Exploratory Data Analysis (EDA).
    *   Training various models: Linear Regression, Random Forest, Decision Tree, SVR, XGBoost, Voting Ensemble, LSTM, and GRU.
    *   Conducting hyperparameter tuning for LSTM and GRU models using Keras Tuner.
    *   Evaluating model performance using metrics like MAE, MSE, RMSE, R², and MAPE.
    *   Visualizing actual vs. predicted stock prices and comparing model performances.

### Forecasting with Prophet

*   The `model_training.ipynb` also includes a section for time series forecasting using Facebook Prophet, including comparison against actual 2024 data.

### Model Inference

*   Refer to the `model_inferencing.ipynb` notebook. This notebook demonstrates how to:
    *   Load pre-trained classical machine learning models (saved as `.pkl` files).
    *   Load pre-trained deep learning models (saved as `.h5` files).
    *   Make predictions on new, unseen data using these loaded models.
    *   Verify the structure and configuration of the loaded models.

### Utility Scripts (`Utils/`)
The Python scripts within the `Utils/` directory (`Data_Preprocessing.py`, `EDA.py`, `Models.py`, `Evaluation.py`, `plot_prediction.py`) contain the core functions for data handling, analysis, model definitions, training logic, evaluation metrics, and plotting. These scripts are primarily imported as modules into the Jupyter notebooks (`model_training.ipynb`, `model_inferencing.ipynb`) and are not typically run as standalone scripts by the end-user.

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

The `model_training.ipynb` notebook showcases the detailed performance of each model, including metrics (MAE, MSE, RMSE, R², MAPE) and various visualization charts that summarize model performances. These visualizations include comparisons of actual vs. predicted stock prices and performance during volatile market periods. The `prediction_results.png` file in the root directory provides an example of the kind of graphical output generated by the notebooks.

Key aspects highlighted in the results include:
*   Comprehensive metrics for both training and testing datasets.
*   Evaluation of model robustness, particularly during high-volatility periods.
*   Interactive plots (where applicable, e.g., using Plotly within the notebooks) that allow for detailed inspection of prediction accuracy over time.

---

## Acknowledgments

* Data sourced from public Tesla stock price datasets.
* Utilizes open-source Python libraries including TensorFlow, scikit-learn, XGBoost, Prophet, and Plotly.

## Contributing
Contributions are welcome! If you have suggestions for improvements or encounter any issues, please feel free to open an issue or submit a pull request.

When contributing to this repository, please first discuss the change you wish to make via an issue with the owners of this repository before making a change.

Please follow these steps to contribute:
1.  Fork the repository.
2.  Create a new branch for your feature or bug fix (`git checkout -b feature/AmazingFeature` or `git checkout -b bugfix/IssueDescription`).
3.  Make your changes and commit them (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

## License
This project is licensed under the MIT License.

A copy of the MIT License is typically included in a `LICENSE` file in the root of the repository. While not currently present in this project, the MIT License allows for broad use and modification, provided that the original copyright and license notice are included in any substantial portions of the software.

## Running with Docker

This project includes a `Dockerfile` to make it easy to build and run in a containerized environment, ensuring all dependencies and settings are consistent.

### Prerequisites

*   [Docker](https://docs.docker.com/get-docker/) installed on your system.

### Building the Docker Image

1.  Clone this repository to your local machine.
2.  Navigate to the root directory of the project (where the `Dockerfile` is located).
3.  Run the following command to build the Docker image. Replace `tesla-stock-forecasting` with your preferred image name if desired:

    ```bash
    docker build -t tesla-stock-forecasting .
    ```

### Running the Jupyter Notebook Server

After the image is built successfully, you can run a container to start the Jupyter Notebook server:

```bash
docker run -p 8888:8888 tesla-stock-forecasting
```

*   The `-p 8888:8888` flag maps port 8888 on your host machine to port 8888 in the container, where Jupyter Notebook is running.
*   When the container starts, Jupyter Notebook will output a URL in the terminal, usually something like `http://127.0.0.1:8888/?token=xxxxxxxxxxxxxxxxxxxxxxxxxxxx`.
*   Copy this URL and paste it into your web browser to access the Jupyter environment. You will see the project files and can run the notebooks (`model_training.ipynb`, `model_inferencing.ipynb`).

### Running Other Scripts or Accessing the Shell

If you want to run a specific script or access the shell within the container, you can override the default command. For example, to get an interactive bash shell:

```bash
docker run -it tesla-stock-forecasting bash
```

Once inside the container, you can navigate the file system, run Python scripts (e.g., `python Utils/Data_Preprocessing.py`), or execute other commands.

