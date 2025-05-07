# Function to plot distribution of monthly returns
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import zscore, skew, kurtosis

def plot_monthly_return_distribution(df):
    plt.figure(figsize=(12, 6))
    sns.histplot(df['Monthly_Return'], bins=50, kde=True, color='blue', alpha=0.7)
    plt.title('Distribution of Tesla Monthly Returns')
    plt.xlabel('Monthly Return (%)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
# Function to plot stock price trend
def plot_stock_price_trend(df):
    plt.figure(figsize=(14, 6))
    plt.plot(df.index, df['Close'], label='Close Price', linewidth=1)
    plt.title('Tesla Stock Price Trend', fontsize=16)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
# Function to plot moving averages
def plot_moving_averages(df):
    plt.figure(figsize=(14, 6))
    plt.plot(df.index, df['MA5'], label='MA5 (5-day)', linewidth=1.5)
    plt.plot(df.index, df['MA10'], label='MA10 (10-day)', linewidth=1.5)
    plt.plot(df.index, df['MA20'], label='MA20 (20-day)', linewidth=1.5)
    plt.title('Tesla Stock Price Moving Average Trend', fontsize=16)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
# Function to plot average weekly return
def plot_weekly_avg_return(df):
    df['Weekday'] = df.index.weekday
    weekly_avg_return = df.groupby('Weekday')['Monthly_Return'].mean()
    plt.figure(figsize=(10, 5))
    weekly_avg_return.plot(kind='bar', color='lightcoral')
    plt.title('Average Weekly Return')
    plt.xlabel('Weekday (0 = Monday, ..., 6 = Sunday)')
    plt.ylabel('Avg Return (%)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
# Function to plot quarterly seasonality
def plot_quarterly_seasonality(df):
    df['Quarter'] = df.index.quarter
    quarterly_avg_return = df.groupby('Quarter')['Monthly_Return'].mean()
    plt.figure(figsize=(10, 5))
    quarterly_avg_return.plot(kind='bar', color='lightblue')
    plt.title('Average Quarterly Return')
    plt.xlabel('Quarter (1 = Q1, 2 = Q2, 3 = Q3, 4 = Q4)')
    plt.ylabel('Avg Return (%)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
# Function to plot volatility over time
def plot_volatility(df):
    plt.figure(figsize=(14, 6))
    plt.plot(df.index, df['Volatility_5'], label='Volatility (5 days)', color='red')
    plt.plot(df.index, df['Volatility_10'], label='Volatility (10 days)', color='orange')
    plt.plot(df.index, df['Volatility_20'], label='Volatility (20 days)', color='green')
    plt.title('Tesla Stock Volatility Over Time')
    plt.xlabel('Date')
    plt.ylabel('Volatility (Rolling Std Dev)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
# Function to calculate and plot correlation heatmap
def plot_correlation_heatmap(df):
    corr_matrix = df[['Close', 'Monthly_Return', 'MA5', 'MA10', 'MA20', 'Volatility_5', 'Volatility_10', 'Volatility_20']].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Heatmap of Tesla Stock Features')
    plt.tight_layout()
    plt.show()
# Function to plot volume vs price movement
def plot_volume_vs_price(df):
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(df.index, df['Close'], label='Close Price', color='blue')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Close Price', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax2 = ax1.twinx()
    ax2.bar(df.index, df['Volume'], label='Volume', color='red', alpha=0.3)
    ax2.set_ylabel('Volume', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    plt.title('Tesla Price Movement and Volume')
    plt.tight_layout()
    plt.show()
# Function to calculate and plot Z-score outliers
def plot_zscore_outliers(df):
    df['Z-Score_Return'] = zscore(df['Monthly_Return'].dropna())
    outliers_zscore_return = df[abs(df['Z-Score_Return']) > 3]
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Monthly_Return'], label='Monthly Returns', color='blue', alpha=0.6)
    plt.scatter(outliers_zscore_return.index, outliers_zscore_return['Monthly_Return'], color='red', label='Outliers (Z-Score > 3 or < -3)', alpha=0.7)
    plt.title('Tesla Monthly Returns with Outliers (Z-Score > 3)')
    plt.xlabel('Date')
    plt.ylabel('Monthly Return (%)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
# Function to calculate and plot IQR-based outliers
def plot_iqr_outliers(df):
    Q1 = df['Monthly_Return'].quantile(0.25)
    Q3 = df['Monthly_Return'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers_iqr_return = df[(df['Monthly_Return'] < lower_bound) | (df['Monthly_Return'] > upper_bound)]
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Monthly_Return'], label='Monthly Returns', color='blue', alpha=0.6)
    plt.scatter(outliers_iqr_return.index, outliers_iqr_return['Monthly_Return'], color='orange', label='Outliers (IQR Method)', alpha=0.7)
    plt.title('Tesla Monthly Returns with Outliers (IQR Method)')
    plt.xlabel('Date')
    plt.ylabel('Monthly Return (%)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
# Function to calculate Sharpe Ratio
def calculate_sharpe_ratio(df, risk_free_rate=0.02):
    risk_free_rate /= 12  # Convert to monthly risk-free rate
    mean_return = df['Monthly_Return'].mean()
    std_dev = df['Monthly_Return'].std()
    sharpe_ratio = (mean_return - risk_free_rate) / std_dev
    print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
# Function to calculate and plot cumulative returns
def plot_cumulative_returns(df):
    df['Daily_Return'] = df['Close'].pct_change() * 100
    df = df.dropna(subset=['Daily_Return'])
    df['Cumulative_Return'] = (1 + df['Daily_Return'] / 100).cumprod() - 1
    df_yearly = df.resample('Y').last()
    df_yearly['Yearly_Growth'] = df_yearly['Cumulative_Return'] * 100
    plt.figure(figsize=(12, 6))
    plt.bar(df_yearly.index.year, df_yearly['Yearly_Growth'], color='teal', alpha=0.7)
    plt.title('Tesla Yearly Growth (Cumulative Return)')
    plt.xlabel('Year')
    plt.ylabel('Yearly Growth (%)')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()