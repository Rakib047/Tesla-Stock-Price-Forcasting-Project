import matplotlib.pyplot as plt
# Plot the train, test, and XGBoost prediction data
def plot_test_train_prediction(model_name,df_train, df_test, y_pred):
    plt.figure(figsize=(16, 10))
    
    # Plot the train data
    plt.plot(df_train.index, df_train['Close'], label='Train', color='teal')
    
    # Plot the test data
    plt.plot(df_test.index, df_test['Close'], label='Test', color='magenta')
    
    
    # Plot the model predictions
    plt.plot(df_test.index, y_pred, label=f'{model_name} Predictions', color='blue', linestyle='dashed')
    
    
    # Title and labels
    plt.title(f'{model_name} Predictions vs Actual Test Data')
    plt.xlabel('Date')
    plt.ylabel('Close Price USD ($)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    
import matplotlib.pyplot as plt

def plot_test_train_prediction_log(model_name, df_train, df_test, y_pred, log_scale=True):
    """
    Plots the training, testing, and predicted data.
    Optionally uses a logarithmic scale on the Y-axis.
    
    Parameters:
    - model_name: str, name of the model for the legend/title
    - df_train: DataFrame containing the training data (must have 'Close' column)
    - df_test: DataFrame containing the test data (must have 'Close' column)
    - y_pred: Predicted values (must match length of df_test)
    - log_scale: bool, whether to use logarithmic scale on y-axis
    """
    
    plt.figure(figsize=(16, 10))
    
    # Plot training data
    plt.plot(df_train.index, df_train['Close'], label='Train', color='teal')
    
    # Plot test data
    plt.plot(df_test.index, df_test['Close'], label='Test', color='magenta')
    
    # Plot model predictions
    plt.plot(df_test.index, y_pred, label=f'{model_name} Predictions', color='blue', linestyle='dashed')
    
    # Set log scale if requested
    if log_scale:
        plt.yscale('log')
    
    # Titles and labels
    plt.title(f'{model_name} Predictions vs Actual Test Data {"(Log Scale)" if log_scale else ""}')
    plt.xlabel('Date')
    plt.ylabel('Close Price USD ($)')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()
    



