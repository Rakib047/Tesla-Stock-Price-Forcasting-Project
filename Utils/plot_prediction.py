import plotly.graph_objects as go

def plot_test_train_prediction(model_name, df_train, df_test, y_pred):
    fig = go.Figure()

    # Train data
    fig.add_trace(go.Scatter(
        x=df_train.index, y=df_train['Close'],
        mode='lines',
        name='Train',
        line=dict(color='teal')
    ))

    # Test data
    fig.add_trace(go.Scatter(
        x=df_test.index, y=df_test['Close'],
        mode='lines',
        name='Test',
        line=dict(color='magenta')
    ))

    # Prediction data
    fig.add_trace(go.Scatter(
        x=df_test.index, y=y_pred,
        mode='lines',
        name=f'{model_name} Predictions',
        line=dict(color='blue', dash='dash')
    ))

    fig.update_layout(
        title=f'{model_name} Predictions vs Actual Test Data',
        xaxis_title='Date',
        yaxis_title='Close Price USD ($)',
        legend_title='',
        width=900, height=600,
        template='plotly_white',
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')

    fig.show()


def plot_test_train_prediction_log(model_name, df_train, df_test, y_pred, log_scale=True):
    fig = go.Figure()

    # Train data
    fig.add_trace(go.Scatter(
        x=df_train.index, y=df_train['Close'],
        mode='lines',
        name='Train',
        line=dict(color='teal')
    ))

    # Test data
    fig.add_trace(go.Scatter(
        x=df_test.index, y=df_test['Close'],
        mode='lines',
        name='Test',
        line=dict(color='magenta')
    ))

    # Prediction data
    fig.add_trace(go.Scatter(
        x=df_test.index, y=y_pred,
        mode='lines',
        name=f'{model_name} Predictions',
        line=dict(color='blue', dash='dash')
    ))

    fig.update_layout(
        title=f'{model_name} Predictions vs Actual Test Data {"(Log Scale)" if log_scale else ""}',
        xaxis_title='Date',
        yaxis_title='Close Price USD ($)',
        legend_title='',
        width=900, height=600,
        template='plotly_white',
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray',
                     type='log' if log_scale else 'linear')

    fig.show()
