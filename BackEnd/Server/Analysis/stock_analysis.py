import yfinance as yf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from alpha_vantage.timeseries import TimeSeries
from config import ALPHA_VANTAGE_KEY

def get_stock_symbol(company_name):
    try:
        stock = yf.Ticker(company_name)
        return stock.ticker if stock.ticker else None
    except Exception:
        return None

def fetch_stock_data(symbol):
    try:
        ts = TimeSeries(key=ALPHA_VANTAGE_KEY, output_format="pandas")
        stock_data, _ = ts.get_daily(symbol=symbol, outputsize="compact")
        return stock_data if isinstance(stock_data, pd.DataFrame) else None
    except Exception:
        return None

def visualize_stock_trends(stock_data, symbol):
    if isinstance(stock_data, pd.DataFrame):
        stock_data["date"] = stock_data.index
        stock_data["close"] = stock_data["4. close"]
        plt.figure(figsize=(12, 6))
        sns.lineplot(x="date", y="close", data=stock_data, color="blue")
        plt.title(f"Stock Closing Price Trend for {symbol}", fontsize=16)
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Closing Price", fontsize=12)
        plt.xticks(rotation=45)
        plt.grid()
        plt.show()
        print("\nStock Data Summary:")
        print(stock_data[["close"]].describe())