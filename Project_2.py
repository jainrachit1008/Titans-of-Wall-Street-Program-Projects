import datetime as dt
import pandas as pd
import numpy as np
from pandas.plotting import table
import matplotlib.pyplot as plt

def ann_return(DF):
    "function to calculate the Annualized return from monthly prices of a fund/sript"
    df = DF.copy()
    df["mon_ret"] = df["NAV"].pct_change()
    df["cum_return"] = (1 + df["mon_ret"]).cumprod()
    n = len(df)/12
    ann_ret = (df["cum_return"][-1])**(1/n) - 1
    return ann_ret

def ann_volatility(DF):
    "function to calculate annualized volatility of a trading strategy"
    df = DF.copy()
    df["mon_ret"] = df["NAV"].pct_change()
    vol = df["mon_ret"].std() * np.sqrt(12)
    return vol

def sharpe(DF, rf):
    "function to calculate sharpe ratio ; rf is the risk free rate"
    df = DF.copy()
    sr = (ann_return(df) - rf) / ann_volatility(df)
    return sr

# Import the Lipper Hedge Fund Data
Lip = pd.read_csv(r'/Users/rachnish/Dropbox/TWSA Session #1 - Wed Nov 20/Kapil_Data.csv', index_col='Date')

# format the date columns to datetime format
Lip['Performance Start Date'] = pd.to_datetime(Lip['Performance Start Date'], errors='raise', dayfirst=True)
Lip['Performance End Date'] = pd.to_datetime(Lip['Performance End Date'], errors='raise', dayfirst=True)
Lip.index = pd.to_datetime(Lip.index, errors='raise', dayfirst=True)

# Filter Funds with a continuous track record from 1995/1/1 to 1995/12/1
Yearly_data = Lip.copy()
Yearly_data = Yearly_data[(Yearly_data['Performance Start Date'] <= '1995-01-01') & (Yearly_data['Performance End Date'] >= '1995-12-31')]
Yearly_data = Yearly_data[(Yearly_data.index >= '1995-01-01') & (Yearly_data.index <= '1995-12-31')]

# Calculate Sharpe Ratio for each Fund in the selected database
HF = list(Yearly_data['Fund Name'].unique())
HF_stats = pd.DataFrame(columns=['SharpeRatioPast', 'PercentileRankingPast'], index=HF)
for i in HF:
    HF_stats['SharpeRatioPast'].loc[i] = sharpe(Yearly_data.loc[Yearly_data['Fund Name'] == i], 0.00)

# Calculate percentile ranking for each Fund in the selected database
ranks = HF_stats.SharpeRatioPast.rank(ascending=False)
HF_stats['PercentileRankingPast'] = (ranks - ranks.min())/(ranks.max() - ranks.min())*100



