"""

@author: Rachit Jain

"""

import yfinance as yf
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Download historical data for S&P 500
ticker = "^GSPC"
SnP = yf.download(ticker, start="1990-04-20", end="2017-12-30")
SnP = SnP.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1)
SnP['Return'] = SnP['Adj Close'].pct_change()

# Compute the Rolling 200 Day Moving Average
SnP['200DMA'] = SnP['Adj Close'].rolling(200).mean()
SnP.dropna(axis=0, how='any', inplace=True)

def Ann_Ret(DF, column_name):
    "function to calculate the Annualized Return of a trading strategy"
    df = DF.to_frame()
    df["cum_return"] = (1 + df[column_name]).cumprod()
    n = len(df)/252
    cagr_final = (df["cum_return"][-1])**(1/n) - 1
    return cagr_final

def Cum_Ret(DF, column_name):
    "function to calculate the Cumulative Annual Return of a trading strategy"
    df = DF.to_frame()
    df["cum_return"] = (1 + df[column_name]).cumprod()
    cum_ret = (df["cum_return"][-1]) - 1
    return cum_ret

def volatility(DF, column_name):
    "function to calculate annualized volatility of a trading strategy"
    df = DF.to_frame()
    vol = df[column_name].std() * np.sqrt(252)
    return vol

def max_dd(DF, column_name):
    "function to calculate max drawdown"
    df = DF.to_frame()
    df["cum_ret"] = (1+df[column_name]).cumprod()
    df["max_rolling_cum"] = df["cum_ret"].cummax()
    df["drawdown"] = df["max_rolling_cum"] - df["cum_ret"]
    df["drawdown_pct"] = df["drawdown"]/df["max_rolling_cum"]
    max_drawdown = df["drawdown_pct"].max()
    return max_drawdown

def test_vol_after_200DMA(DF, n):
    """ This function is constructed to test the returns and volatility of S&P Returns after 200DMA breach
    by certain percentage"""
    df = DF.copy()
    for i in range(0, n+1):
        column_name = str('Exceeds 200DMA by {}%'.format(i))
        df[column_name] = np.NaN
        df[column_name][0] = (1 if DF['Adj Close'][0] >= (DF['200DMA'][0] * (1 + (i / 100))) else 0 if DF['Adj Close'][0] <= (DF['200DMA'][0] * (1 + (i / 100))) else 0)
        for k in range(0, len(df[column_name])-1):
            if DF['Adj Close'][k] >= (DF['200DMA'][k] * (1 + (i / 100))):
                df[column_name][k + 1] = 1
            elif DF['Adj Close'][k] <= (DF['200DMA'][k] * (1 - (i / 100))):
                df[column_name][k + 1] = 0
            else:
                df[column_name][k + 1] = df[column_name][k]
    return df

# Positions for all algorithms
Pos = test_vol_after_200DMA(SnP,10)

# Daily Returns for all algorithms
SnP_Returns = SnP.copy().drop(columns=['Return'])
for j in range(0, 11):
    column_name = str('Position Returns for {}% strategy'.format(j))
    column_name_pos = str('Exceeds 200DMA by {}%'.format(j))
    SnP_Returns[column_name] = np.where(Pos[column_name_pos] == 1, Pos['Return'], 0)

# Computing Annualized Returns, std.dev, Max DD, Ann Ret/MaxDD, Cumulative Returns
strategy_perf = pd.DataFrame(index=['Ann_Ret', 'Std.dev', 'MaxDD', 'Ann_Ret/MaxDD', 'Cum_Ret'], columns=Pos.columns[3:14])
for s in range(0, 11):
    column_name = str('Position Returns for {}% strategy'.format(s))
    column_name_pos = str('Exceeds 200DMA by {}%'.format(s))
    strategy_perf.loc['Ann_Ret', column_name_pos] = Ann_Ret(SnP_Returns[column_name], column_name)
    strategy_perf.loc['Std.dev', column_name_pos] = volatility(SnP_Returns[column_name], column_name)
    strategy_perf.loc['MaxDD', column_name_pos] = max_dd(SnP_Returns[column_name], column_name)
    strategy_perf.loc['Ann_Ret/MaxDD', column_name_pos] = strategy_perf.loc['Ann_Ret', column_name_pos] / strategy_perf.loc['MaxDD', column_name_pos]
    strategy_perf.loc['Cum_Ret', column_name_pos] = Cum_Ret(SnP_Returns[column_name], column_name)

strategy_perf = strategy_perf.astype('float64').round(4).T


# Visualizing the Investment Strategies performance
max_sr_idx = strategy_perf[['Ann_Ret/MaxDD']].idxmax()
ret_arr = np.array(strategy_perf['Ann_Ret'])
vol_arr = np.array(strategy_perf['Std.dev'])
retbydd_arr = np.array(strategy_perf['Ann_Ret/MaxDD'])
plt.figure(figsize=(12, 6))
plt.scatter(vol_arr, ret_arr, c=retbydd_arr, cmap='viridis', marker='o')
plt.colorbar(label='Ann_Ret / MaxDD')
plt.xlabel('Volatility')
plt.ylabel('Return')
plt.scatter(strategy_perf.loc[max_sr_idx[0], 'Std.dev'], strategy_perf.loc[max_sr_idx[0], 'Ann_Ret'], c='red', s=50, marker='*', label=max_sr_idx[0]+' is the best strategy')  # red dot
plt.legend()
plt.show()

# At last, make the performance table look more readable
strategy_perf['Ann_Ret'] = pd.Series(["{0:.2f}%".format(val * 100) for val in strategy_perf['Ann_Ret']], index=strategy_perf.index)
strategy_perf['Std.dev'] = pd.Series(["{0:.2f}%".format(val * 100) for val in strategy_perf['Std.dev']], index=strategy_perf.index)
strategy_perf['MaxDD'] = pd.Series(["{0:.2f}%".format(val * 100) for val in strategy_perf['MaxDD']], index=strategy_perf.index)
strategy_perf['Cum_Ret'] = pd.Series(["{0:.2f}%".format(val * 100) for val in strategy_perf['Cum_Ret']], index=strategy_perf.index)

# If required, download a copy of the performance report
# SnP_Returns.to_csv('/Users/rachnish/PycharmProjects/TWSA_Sess2/Project2_SnPReturns.csv', index=True)