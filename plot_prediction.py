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