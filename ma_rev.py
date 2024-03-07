import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from statsmodels.tsa.stattools import adfuller

# Create a list of US stocks 
stock_symbols = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'BRK-A', 'NVDA',
    'JPM', 'JNJ', 'V', 'PG', 'UNH', 'MA', 'DIS', 'HD', 'BAC', 'VZ',
    'INTC', 'KO', 'PFE', 'WMT', 'MRK', 'PEP', 'T', 'BA', 'XOM', 'ABBV',
    'NKE', 'MCD', 'CSCO', 'DOW', 'ADBE', 'IBM', 'CVX', 'CRM', 'ABT', 'MDT',
    'PYPL', 'NEE', 'COST', 'AMGN', 'CMCSA', 'NFLX', 'ORCL', 'PM', 'HON', 'ACN',
    'TMO', 'AVGO'
]

# FOREX STOCKS
# stock_symbols = [
#     ''
# ]

# Fetch the data from Yahoo Finance
df = {}
for symbol in stock_symbols:
    # print(symbol)
    data = yf.download(symbol, start='2015-01-01', end='2023-01-01')
    # if symbol == 'IBM':
    #     print(data)
    # df[symbol] = data['Close']
    df[symbol] = data

# print('df**********')
# print(df)
# print('df**********')
stationary_stocks = []
p_values = []  

for symbol, data in df.items():
    # print('datadata', data.iloc[1])
    result = adfuller(data['Close'])
    # result = adfuller(data)
    # A p-value less than 0.05 indicates that the data is stationary
    p_value = result[1]
    if p_value <= 0.05:
        stationary_stocks.append(symbol)
        p_values.append(p_value)

print("Stocks suitable for mean reversion strategy:")
for stock, p_value in zip(stationary_stocks, p_values):  # Use zip to iterate over both lists simultaneously
    print(f"Stock: {stock}, p-value: {p_value:.4f}")

class MeanReversion(Strategy):
    n1 = 150  # Period for the moving average
    
    def init(self):
        # Compute moving average
        self.offset = 0.01  # Buy/sell when price is 1% below/above the moving average
        prices = self.data['Close']
        # prices = self.data.iloc[1]
        self.ma = self.I(self.compute_rolling_mean, prices, self.n1)

    def compute_rolling_mean(self, prices, window):
        return [(sum(prices[max(0, i - window):i]) / min(i, window)) if i > 0 else np.nan for i in range(len(prices))]

    def next(self):
        size = 0.1
        # If price drops to more than offset% below n1-day moving average, buy
        if self.data['Close'] < self.ma[-1] * (1 - self.offset):
            if self.position.size < 0:  # Check for existing short position
                self.buy()  # Close short position
            self.buy(size=size)

        # If price rises to more than offset% above n1-day moving average, sell
        elif self.data['Close'] > self.ma[-1] * (1 + self.offset):
            if self.position.size > 0:  # Check for existing long position
                self.sell()  # Close long position
            self.sell(size=size)

stock_to_backtest = stationary_stocks[0]
df = df[stock_to_backtest]
df = pd.DataFrame.from_dict(df)
# print('df----------')
# print(df)
# print('df----------')
bt = Backtest(df, MeanReversion, cash=100000, commission=.002)
stats = bt.run()
# bt.plot()
print(stats)

# def plot_stationary_stocks(df, stationary_stocks):
#     for stock in stationary_stocks:
#         data = df[stock].Close
        
#         # Calculate rolling statistics
#         rolling_mean = data.rolling(window=30).mean()  # 30-day rolling mean
#         rolling_std = data.rolling(window=30).std()   # 30-day rolling standard deviation
        
#         # Plot the statistics
#         plt.figure(figsize=(12, 6))
#         plt.plot(data, label=f'{stock} Prices', color='blue')
#         plt.plot(rolling_mean, label='Rolling Mean', color='red')
#         plt.plot(rolling_std, label='Rolling Std. Dev.', color='black')
#         plt.title(f'Stationarity Check for {stock}')
#         plt.xlabel('Date')
#         plt.ylabel('Prices')
#         plt.legend()
#         plt.grid(True)
#         plt.show()


# Calling the function
# plot_stationary_stocks(df, stationary_stocks)