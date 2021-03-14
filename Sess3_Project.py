"""

@author: Rachit Jain

"""

import yfinance as yf
import datetime as dt
import pandas as pd
import numpy as np
from pandas.plotting import table
import matplotlib.pyplot as plt
from scipy.stats import levene

# Download historical data for S&P 500
ticker = "^GSPC"
SnP = yf.download(ticker, start="1991-02-01", end="2018-06-01")
SnP = SnP.drop(['Open','High','Low','Close','Volume'], axis=1)
SnP['Return'] = SnP['Adj Close'].pct_change()

def Ret_everyndays(DF,n):
    """ This function takes in the SnP data, calculates the returns every n days, returns a list"""
    df = DF.copy().drop('Return', axis = 1).iloc[::n, :]
    ret = df['Adj Close'].pct_change().to_list()
    return ret

def MV_Breach(mvg_avg_days,DF):
    """this function takes in the MA days, df of close prices & outputs events when MA was breached
    . In order for the moving average to be breached, the previous day’s closing price
    has to be ABOVE the moving average and today's close must be BELOW the moving average"""
    df = DF.copy().drop("Return", axis=1)
    df["Moving Average Price"] = df["Adj Close"].rolling(mvg_avg_days).mean()
    last_close_price = df["Adj Close"].iloc[mvg_avg_days-2]
    df = df.iloc[mvg_avg_days-2:, ]
    df_BreakingDMA = df[(df["Adj Close"].shift(1) > df["Moving Average Price"].shift(1)) & (df["Adj Close"] < df["Moving Average Price"])]
    df_BreakingDMA = df_BreakingDMA.reset_index().rename(columns={'Date': f'Date {mvg_avg_days}d MA is breached','Adj Close': 'Closing Price on Day 0'})
    df_BreakingDMA = df_BreakingDMA[[f'Date {mvg_avg_days}d MA is breached', 'Moving Average Price', 'Closing Price on Day 0']]
    return df_BreakingDMA

def strategyretdata(Price,breachdata,n,N):
    """ Extract the close prices 1d,2d,..,nd from the breach date. Then calculate the returns for each of such intervals
    taking the price as on breach date as the base price"""
    price = Price.copy()
    price = price.reset_index()
    dict = {}
    for i in breachdata[f'Date {N}d MA is breached']:
        x = price[price['Date'] == i]['Adj Close'].index.values
        S = pd.Series(SnP['Adj Close'][x[0]:x[0] + n])
        first_element = S[0]
        dict[i] = list(S.apply(lambda y: ((y / first_element) - 1)))
    return dict

# Create a DataFrame with SnP returns from every 1d,2d,...,40d
SnP_Returns = pd.DataFrame()
df = pd.DataFrame()
for k in range(1, 41):
    Column_Name = str(f'Date - EVERY {k} DAYS')
    df[Column_Name] = np.array(Ret_everyndays(SnP, k))
    SnP_Returns = pd.concat([SnP_Returns, df], axis=1)
    df = pd.DataFrame()
    continue
SnP_Returns.drop(index=0, inplace=True)

# Create DataFrame with the Close Prices on the Breach Date
Breach_data_50DMA = MV_Breach(50,SnP)   # 50 DMA
Breach_data_100DMA = MV_Breach(100,SnP)  # 100 DMA
Breach_data_200DMA = MV_Breach(200,SnP)  # 200 DMA

# Create a DataFrame with the Strategy Returns every 1d,2d,....,40d from the Breach Date
Breach_ret_50DMA = pd.DataFrame(dict((k, v) for k, v in strategyretdata(SnP, Breach_data_50DMA, 41, 50).items() if len(v)==41)).transpose().drop(columns=0)
Breach_ret_100DMA = pd.DataFrame(dict((k, v) for k, v in strategyretdata(SnP, Breach_data_100DMA, 41, 100).items() if len(v)==41)).transpose().drop(columns=0)
Breach_ret_200DMA = pd.DataFrame(dict((k, v) for k, v in strategyretdata(SnP, Breach_data_200DMA, 41, 200).items() if len(v)==41)).transpose().drop(columns=0)

# Performing Levene's Test on 50d,100d,200d against S&P_Ret for very 1d,2d,3d,...,40d holding period
P_Values = pd.DataFrame(index=range(1,41), columns=["Levene's test p-value MA 200 ","Levene's test p-value MA 100","Levene's test p-value MA 50"])
for i in range(0, 40):
    stat1, p1 = levene(list(Breach_ret_50DMA.iloc[:, i].dropna()), list(SnP_Returns.iloc[:, i].dropna()))
    stat2, p2 = levene(list(Breach_ret_100DMA.iloc[:, i].dropna()), list(SnP_Returns.iloc[:, i].dropna()))
    stat3, p3 = levene(list(Breach_ret_200DMA.iloc[:, i].dropna()), list(SnP_Returns.iloc[:, i].dropna()))
    P_Values.iloc[i, 2] = p1
    P_Values.iloc[i, 1] = p2
    P_Values.iloc[i, 0] = p3

# Analyzing the p-values for 50d,100d and 200d MA
mu1 = round(np.mean(P_Values.iloc[:, 0]), 2)
sigma1 = round(np.std(P_Values.iloc[:, 0]), 2)
plt.subplot(311)
plt.hist(P_Values.iloc[:,0], 20, density=True)
plt.title("Histogram of 'p-value - MA 200': '$\mu={}$, $\sigma={}$'".format(mu1, sigma1))
plt.xticks([])  # Disables xticks
plt.axvline(x=0.05, color='r', label='p-value of 0.05', linestyle='--', linewidth=1)
plt.legend()

mu2 = round(np.mean(P_Values.iloc[:, 1]), 2)
sigma2 = round(np.std(P_Values.iloc[:, 1]), 2)
plt.subplot(312)
plt.hist(P_Values.iloc[:, 1])
plt.title("Histogram of 'p-value - MA 100': '$\mu={}$, $\sigma={}$'".format(mu2, sigma2))
plt.xticks([])
plt.axvline(x=0.05, color='r', label='p-value of 0.05', linestyle='--', linewidth=1)
plt.legend()

mu3 = round(np.mean(P_Values.iloc[:, 2]), 2)
sigma3 = round(np.std(P_Values.iloc[:, 2]), 2)
plt.subplot(313)
plt.hist(P_Values.iloc[:, 2])
plt.title("Histogram of 'p-value - MA 50': '$\mu={}$, $\sigma={}$'".format(mu3, sigma3))
plt.axvline(x=0.05, color='r', label='p-value of 0.05', linestyle='--', linewidth=1)
plt.legend()

plt.show()

# Time Series Plot
plt.plot(P_Values)
plt.axhline(y=0.05, color='r', label='p-value of 0.05', linestyle='--', linewidth=1)
plt.title("Time Series of P-Values for Trades at 1-40 Days from Breach of MA")
plt.legend(('MA 200','MA 100', 'MA 50','p-value of 0.05'))

plt.show()


