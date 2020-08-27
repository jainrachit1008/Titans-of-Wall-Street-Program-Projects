import datetime as dt
import pandas as pd
import numpy as np
from pandas.plotting import table
import matplotlib.pyplot as plt

# Import the Lipper Hedge Fund Data
Lip = pd.read_csv(r'/Users/rachnish/Dropbox/TWSA Session #1 - Wed Nov 20/Kapil_Data.csv', index_col='Date')

# format the date columns to datetime format
Lip['Performance Start Date'] = pd.to_datetime(Lip['Performance Start Date'], errors='raise', dayfirst=True)
Lip['Performance End Date'] = pd.to_datetime(Lip['Performance End Date'], errors='raise', dayfirst=True)
Lip.index = pd.to_datetime(Lip.index, errors='raise', dayfirst=True)

# Filter Funds with a continuous track record from 2008/1/1 to 2017/12/31
Lip = Lip[(Lip['Performance Start Date'] <= '2008-01-01') & (Lip['Performance End Date'] >= '2017-12-31')]
Lip = Lip[(Lip.index >= '2008-01-01') & (Lip.index <= '2017-12-31')]

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

def max_dd(DF):
    "function to calculate max drawdown"
    df = DF.copy()
    df["mon_ret"] = df["NAV"].pct_change()
    df["cum_ret"] = (1+df["mon_ret"]).cumprod()
    df["max_rolling_cum"] = df["cum_ret"].cummax()
    df["drawdown"] = df["max_rolling_cum"] - df["cum_ret"]
    df["drawdown_pct"] = df["drawdown"]/df["max_rolling_cum"]
    max_drawdown = df["drawdown_pct"].max()
    return max_drawdown

def dur_dd(DF):
    "function to calculate the max. duration of drawdown"
    df = DF.copy()
    df["mon_ret"] = df["NAV"].pct_change()
    df["cum_ret"] = (1+df["mon_ret"]).cumprod()
    df["max_rolling_cum"] = df["cum_ret"].cummax()
    a = 0
    dur_dd = []
    max_rolling_cum = df["max_rolling_cum"].tolist()
    for i in range(len(max_rolling_cum)):
        if i == 0:
            dur_dd.append(0)
        elif max_rolling_cum[i] == max_rolling_cum[i-1]:
            dur_dd.append(a+1)
            a += 1
        elif max_rolling_cum[i] != max_rolling_cum[i-1]:
            a = 0
            dur_dd.append(a)
    df['dur_dd'] = np.array(dur_dd)
    max_dur = df['dur_dd'].max()/12
    return max_dur

def calmer(DF):
    "function to calculate calmar ratio"
    df = DF.copy()
    clmr = ann_return(df)/max_dd(df)
    return clmr

# KPIs for each Hedge Funds falling in our criteria
HF = list(Lip['Fund Name'].unique())
HF_stats = pd.DataFrame(
    columns=['Annualized Return', 'Annualized Volatility', 'Maximum Drawdown', 'Drawdown Duration', 'Calmar Ratio',
             'Sharpe Ratio'], index=HF)
for i in HF:
    HF_stats['Annualized Return'].loc[i] = ann_return(Lip.loc[Lip['Fund Name'] == i])
    HF_stats['Annualized Volatility'].loc[i] = ann_volatility(Lip.loc[Lip['Fund Name'] == i])
    HF_stats['Maximum Drawdown'].loc[i] = max_dd(Lip.loc[Lip['Fund Name'] == i])
    HF_stats['Drawdown Duration'].loc[i] = dur_dd(Lip.loc[Lip['Fund Name'] == i])
    HF_stats['Calmar Ratio'].loc[i] = calmer(Lip.loc[Lip['Fund Name'] == i])
    HF_stats['Sharpe Ratio'].loc[i] = sharpe(Lip.loc[Lip['Fund Name'] == i], 0.00)

HF_stats.to_excel(r'/Users/rachnish/PycharmProjects/TWSA_Sess1_Pr1/HF_Stats.xlsx', sheet_name='HF Stats', index=True)

# Histograms for all KPIs

# To plot percentiles
"this func is sourced from StackOverflow"
def percentile(n):
    def percentile_(x):
        return x.quantile(n)
    percentile_.__name__ = 'percentile_{:2.0f}'.format(n*100)
    return percentile_

## PLotting Returns
fig, ax = plt.subplots(1, 1)
plt.style.use('ggplot')
table(ax, np.round(HF_stats['Annualized Return'].agg([np.min,np.max,np.mean,np.median,percentile(0.1),percentile(0.9)]), 2), loc='center', colWidths=[0.15])
ax.set(title='Annualized Returns of HFs from 2008-2017', xlabel="Returns", ylabel = "# of Funds")
plt.hist(HF_stats['Annualized Return'], align='right',edgecolor='black')

## PLotting Volatility
fig, ax = plt.subplots(1, 1)
plt.style.use('ggplot')
table(ax, np.round(HF_stats['Annualized Volatility'].agg([np.min,np.max,np.mean,np.median,percentile(0.1),percentile(0.9)]), 2), loc='center', colWidths=[0.15])
ax.set(title='Annualized Volatility of HFs from 2008-2017', xlabel="Volatility", ylabel = "# of Funds")
plt.hist(HF_stats['Annualized Volatility'], align='right',edgecolor='black')

## PLotting Max. Drawdown
fig, ax = plt.subplots(1, 1)
plt.style.use('ggplot')
table(ax, np.round(HF_stats['Maximum Drawdown'].agg([np.min,np.max,np.mean,np.median,percentile(0.1),percentile(0.9)]), 2), loc='center', colWidths=[0.15])
ax.set(title='Maximum Drawdown of HFs from 2008-2017', xlabel="Maximum Drawdown", ylabel = "# of Funds")
plt.hist(HF_stats['Maximum Drawdown'], align='right',edgecolor='black')

## Plotting maximum Duration of Drawdown
fig, ax = plt.subplots(1, 1)
plt.style.use('ggplot')
table(ax, np.round(HF_stats['Drawdown Duration'].agg([np.min,np.max,np.mean,np.median,percentile(0.1),percentile(0.9)]), 2), loc='center', colWidths=[0.15])
ax.set(title='Maximum Drawdown Duration of HFs from 2008-2017', xlabel="Maximum Drawdown Duration", ylabel = "# of Funds")
plt.hist(HF_stats['Drawdown Duration'], align='right',edgecolor='black')

## Plotting Calmar Ratio
fig, ax = plt.subplots(1, 1)
plt.style.use('ggplot')
table(ax, np.round(HF_stats['Calmar Ratio'].agg([np.min,np.max,np.mean,np.median,percentile(0.1),percentile(0.9)]), 2), loc='center', colWidths=[0.15])
ax.set(title='Calmar Ratio of HFs from 2008-2017', xlabel="Calmar Ratio", ylabel = "# of Funds")
plt.hist(HF_stats['Calmar Ratio'], align='right',edgecolor='black')

## Plotting Sharpe Ratio
fig, ax = plt.subplots()
plt.style.use('ggplot')
table(ax, np.round(HF_stats['Sharpe Ratio'].agg([np.min,np.max,np.mean,np.median,percentile(0.1),percentile(0.9)]), 2), loc='center', colWidths=[0.15])
ax.set(title='Sharpe Ratio of HFs from 2008-2017', xlabel="Sharpe Ratio", ylabel = "# of Funds")
plt.hist(HF_stats['Sharpe Ratio'], align='right',edgecolor='black')
