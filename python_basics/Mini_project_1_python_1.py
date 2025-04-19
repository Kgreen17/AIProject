import requests          # For fetching real-time financial data
import pandas as pd      # For organizing and analyzing the data
import matplotlib.pyplot as plt  # For graphing
import yfinance as yf  # Install with `pip install yfinance

from contourpy.util import data


def reward_points():
    data = yf.download("AAPL", start="2023-01-01", end="2024-01-01")

    print(data.head())

def calculate_moving_average(prices, window=5):
    from calculations import calculate_moving_average
    return prices.rolling(window=window).mean()

if __name__ == '__main__':
    data['MA_20'] = calculate_moving_average(data['Close'], 20)

    plt.figure(figsize=(12, 6))
    plt.plot(data['Close'], label='Close Price')
    plt.plot(data['MA_20'], label='20-Day MA', linestyle='--')
    plt.title("AAPL Price with Moving Average")
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.grid(True)
    plt.show()




