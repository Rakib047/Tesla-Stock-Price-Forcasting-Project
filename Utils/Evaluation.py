# model_evaluation.py
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def evaluate_model(y_true, y_pred):
    """
    Evaluates the performance of any model using common regression metrics.
    
    Parameters:
        y_true (array-like): Actual values (ground truth).
        y_pred (array-like): Predicted values from the model.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Mean Absolute Error (MAE)
    mae = mean_absolute_error(y_true, y_pred)

    # Mean Squared Error (MSE)
    mse = mean_squared_error(y_true, y_pred)

    # Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)

    # R-squared (R²)
    r2 = r2_score(y_true, y_pred)

    # Mean Absolute Percentage Error (MAPE) with safe division
    non_zero_indices = y_true != 0
    if np.any(non_zero_indices):
        mape = np.mean(np.abs((y_true[non_zero_indices] - y_pred[non_zero_indices]) / y_true[non_zero_indices])) * 100
    else:
        mape = np.nan  # Or set to 0 or some other default

    # Print evaluation metrics
    print(f"Model Evaluation Metrics:")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R-squared (R²): {r2:.4f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.4f}%")
    print("\n")
    
    return mae, mse, rmse, r2, mape


# def evaluate_model(y_true, y_pred):
#     """
#     Evaluates the performance of any model using common regression metrics.
    
#     Parameters:
#         y_true (array-like): Actual values (ground truth).
#         y_pred (array-like): Predicted values from the model.
#     """
#     # Mean Absolute Error (MAE)
#     mae = mean_absolute_error(y_true, y_pred)
    
#     # Mean Squared Error (MSE)
#     mse = mean_squared_error(y_true, y_pred)
    
#     # Root Mean Squared Error (RMSE)
#     rmse = np.sqrt(mse)
    
#     # R-squared (R²)
#     r2 = r2_score(y_true, y_pred)
    
#     # Mean Absolute Percentage Error (MAPE)
#     mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
#     # Print evaluation metrics
#     print(f"Model Evaluation Metrics:")
#     print(f"Mean Absolute Error (MAE): {mae:.4f}")
#     print(f"Mean Squared Error (MSE): {mse:.4f}")
#     print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
#     print(f"R-squared (R²): {r2:.4f}")
#     print(f"Mean Absolute Percentage Error (MAPE): {mape:.4f}%")
#     print("\n")


def volatility_error_plot(df_test,df_test_scaled,y_pred):
    # Calculate daily returns (percentage change)
    df_test['Returns'] = df_test['Close'].pct_change() * 100

    # Calculate rolling standard deviation (volatility) with a window of 30 days
    df_test['Rolling_Volatility'] = df_test['Returns'].rolling(window=30).std()

    # Define a threshold for high volatility (e.g., top 10% of volatility)
    volatility_threshold = df_test['Rolling_Volatility'].quantile(0.9)

    # Mark periods of high volatility
    df_test['High_Volatility'] = df_test['Rolling_Volatility'] > volatility_threshold

    # # Assume y_pred is your predicted values and y_test is the actual test values
    # # Calculate model errors
    # Set the index of y_pred to match the index of df_test for alignment
    df_test_scaled['Predicted'] = y_pred

    # Now, calculate the error between actual and predicted values for the test set
    df_test_scaled['Error'] = df_test_scaled['Target'] - df_test_scaled['Predicted']

    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import numpy as np

    # Filter errors during high volatility periods
    high_volatility_data = df_test_scaled[df_test['High_Volatility']]

    # Calculate error metrics during high volatility periods
    mae_volatility = mean_absolute_error(high_volatility_data['Target'], high_volatility_data['Predicted'])
    rmse_volatility = np.sqrt(mean_squared_error(high_volatility_data['Target'], high_volatility_data['Predicted']))

    # # Print the error metrics
    print(f'Mean Absolute Error (MAE) during High Volatility: {mae_volatility}')
    print(f'Root Mean Squared Error (RMSE) during High Volatility: {rmse_volatility}')



    # Plot the volatility
    import plotly.graph_objects as go

    # Plot volatility
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_test.index,
        y=df_test['Rolling_Volatility'],
        mode='lines',
        name='Rolling Volatility',
        line=dict(color='blue')
    ))

    fig.add_trace(go.Scatter(
        x=[df_test.index.min(), df_test.index.max()],
        y=[volatility_threshold, volatility_threshold],
        mode='lines',
        name='High Volatility Threshold',
        line=dict(color='red', dash='dash')
    ))

    fig.update_layout(
        title='Volatility (Rolling 30-day Std Dev of Returns)',
        xaxis_title='Date',
        yaxis_title='Volatility',
        legend_title='',
        width=900, height=450,
        template='plotly_white',
        showlegend=True
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')

    fig.show()


    # Plot model errors during high volatility
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=high_volatility_data.index,
        y=high_volatility_data['Error'],
        mode='lines',
        name='Model Error During High Volatility',
        line=dict(color='red')
    ))

    fig.update_layout(
        title='Model Error During High Volatility Periods',
        xaxis_title='Date',
        yaxis_title='Error',
        legend_title='',
        width=900, height=450,
        template='plotly_white',
        showlegend=True
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')

    fig.show()

    
    return mae_volatility, rmse_volatility



def plot_model_comparsion(results):
    # Assuming you have already populated `results_normal_training` with results for all models

    # Convert the results into a DataFrame
    df_results = pd.DataFrame(results)
    
    # Your list of full model names and their abbreviations
    model_refs = {
        "Linear Regression": "LR",
        "Support Vector Regression": "SVR",
        "Random Forest": "RF",
        "Decision Tree": "DT",
        "XGBoost": "XGB",
        "Stacked Linear Regression": "SLR",
        "Voting Regressor": "VR",
        "Prophet": "PR",
        "LSTM": "LSTM",
        "GRU": "GRU"
    }



    # Plotting: Side by side bar charts for MAE, MSE, RMSE, R², and MAPE
    fig, axes = plt.subplots(3, 2, figsize=(18, 12))
    
    plt.subplots_adjust(right=0.75)
    
    
    
    legend_text = '\n'.join([f"{k:<27}: {v}" for k, v in model_refs.items()])
    fig.text(0.68, 0.2, legend_text,
            fontsize=12, va='center', ha='left', fontfamily='monospace',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))



    # Plot MAE (Mean Absolute Error)
    mae_bars = df_results.set_index('Model')['MAE'].plot(kind='bar', ax=axes[0, 0], color='blue')
    axes[0, 0].set_title('Mean Absolute Error (MAE)')
    axes[0, 0].set_ylabel('Error')
    axes[0, 0].tick_params(axis='x', rotation=45)
    # Add percentage labels above each bar
    mae_bars.bar_label(mae_bars.containers[0], labels=[f'{v:.2f}' for v in df_results['MAE']])

    # Plot MSE (Mean Squared Error)
    mse_bars = df_results.set_index('Model')['MSE'].plot(kind='bar', ax=axes[0, 1], color='green')
    axes[0, 1].set_title('Mean Squared Error (MSE)')
    axes[0, 1].set_ylabel('Error')
    axes[0, 1].tick_params(axis='x', rotation=45)
    # Add percentage labels above each bar
    mse_bars.bar_label(mse_bars.containers[0], labels=[f'{v:.6f}' for v in df_results['MSE']])

    # Plot RMSE (Root Mean Squared Error)
    rmse_bars = df_results.set_index('Model')['RMSE'].plot(kind='bar', ax=axes[1, 0], color='red')
    axes[1, 0].set_title('Root Mean Squared Error (RMSE)')
    axes[1, 0].set_ylabel('Error')
    axes[1, 0].tick_params(axis='x', rotation=45)
    # Add percentage labels above each bar
    rmse_bars.bar_label(rmse_bars.containers[0], labels=[f'{v:.2f}' for v in df_results['RMSE']])

    # Plot R² (R-squared)
    r2_bars = df_results.set_index('Model')['R2'].plot(kind='bar', ax=axes[1, 1], color='cyan')
    axes[1, 1].set_title('R² Score')
    axes[1, 1].set_ylabel('R²')
    axes[1, 1].tick_params(axis='x', rotation=45)
    # Add percentage labels above each bar
    r2_bars.bar_label(r2_bars.containers[0], labels=[f'{v:.2f}' for v in df_results['R2']])

    # Plot MAPE (Mean Absolute Percentage Error)
    mape_bars = df_results.set_index('Model')['MAPE'].plot(kind='bar', ax=axes[2, 0], color='orange')
    axes[2, 0].set_title('Mean Absolute Percentage Error (MAPE)')
    axes[2, 0].set_ylabel('Error (%)')
    axes[2, 0].tick_params(axis='x', rotation=45)
    # Add percentage labels above each bar
    mape_bars.bar_label(mape_bars.containers[0], labels=[f'{v:.2f}%' for v in df_results['MAPE']])

    # Hide the last subplot (empty)
    axes[2, 1].axis('off')

    # Tight layout to ensure no clipping of labels
    plt.tight_layout()
    plt.show()