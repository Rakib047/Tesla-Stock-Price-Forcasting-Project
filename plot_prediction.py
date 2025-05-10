# Plot the train, test, and XGBoost prediction data
def plot_xgboost_predictions(df_train, df_test, y_pred):
    plt.figure(figsize=(16, 10))
    
    # Plot the train data
    plt.plot(df_train.index, df_train['Close'], label='Train', color='teal')
    
    # Plot the test data
    plt.plot(df_test.index, df_test['Close'], label='Test', color='magenta')
    
    
    # Plot the XGBoost predictions
    plt.plot(df_test.index, y_pred, label='XGBoost Predictions', color='blue', linestyle='dashed')
    
    # Title and labels
    plt.title('XGBoost Predictions vs Actual Test Data')
    plt.xlabel('Date')
    plt.ylabel('Close Price USD ($)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()