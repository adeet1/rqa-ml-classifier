from project import df, X_test, Y_test_pred, Y_test, split_ind, open_prices
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

# Constants (can be tuned) ======================================
# Number of shares to buy/sell at each trade
shares = 10

# Initial portfolio value (i.e. starting cash amount)
init_cash = 100000

# Commission amount charged per trade
commission = 4.95
# ===============================================================

# Buy or sell indicator array (1 = buy, 0 = sell)
# Note: the "_bh" suffix indicates the corresponding arrays for a buy-and-hold AAPL strategy
buy_or_sell = Y_test_pred.values
buy_or_sell_bh = np.zeros_like(Y_test_pred)
buy_or_sell_bh[0] = 1

# Convert prices array from type object to type float
prices = open_prices[split_ind:].astype(float).values

# The change in shares
# positive = shares bought on day i (number of shares increases)
# negative = shares sold on day i (number of shares decreases)
shares_change = np.zeros_like(buy_or_sell).astype(int)
np.place(shares_change, buy_or_sell == 1, shares)
np.place(shares_change, buy_or_sell == 0, -shares)
shares_change_bh = np.zeros_like(buy_or_sell_bh).astype(int)
shares_change_bh[0] = shares

# The change in cash
# positive = gross proceeds from selling shares
# negative = purchase amount for buying shares
cash_change = np.zeros_like(buy_or_sell).astype(float)

# Populate cash_change array
for i in range(len(buy_or_sell)):
    # Buy
    if buy_or_sell[i] == 1:
        amount = shares * prices[i] + commission
        cash_change[i] = -amount
    
    # Sell
    if buy_or_sell[i] == 0:
        amount = shares * prices[i] - commission
        cash_change[i] = amount

cash_change_bh = np.zeros_like(buy_or_sell).astype(float)
for i in range(len(buy_or_sell)):
    # Buy
    if buy_or_sell_bh[i] == 1:
        amount = shares * prices[i] + commission
        cash_change_bh[i] = -amount

# Cash over time
# (array index i is the amount of cash we have on day i)
# An increase in cash --> added gross proceeds from selling shares
# A decrease in cash --> subtracted purchase amount from buying shares
cash_over_time = init_cash + np.cumsum(cash_change)
cash_over_time_bh = init_cash + np.cumsum(cash_change_bh)

# Shares over time
# (array index i is the number of shares we own on day i)
# An increase in shares --> shares bought
# A decrease in shares --> shares sold
shares_over_time = np.cumsum(shares_change).astype(int)
shares_over_time_bh = np.cumsum(shares_change_bh).astype(int)

# Securities value over time
# (array index i is the securities value on day i)
sec_over_time = np.multiply(shares_over_time, prices)
sec_over_time_bh = np.multiply(shares_over_time_bh, prices)

# Total portfolio value over time
# (array index i is the portfolio value on day i)
portfolio_over_time = np.add(cash_over_time, sec_over_time)
portfolio_over_time_bh = np.add(cash_over_time_bh, sec_over_time_bh)

# =====================
# Plots
# =====================

prices_qqq = pd.read_csv("qqq.csv")["Close"][split_ind+14:].values
prices_qqq_percent_change = np.divide(np.diff(prices_qqq), prices_qqq[:-1])
value_qqq = init_cash * np.cumprod(1 + prices_qqq_percent_change)

# Plot graph of portfolio value
plt.figure()
plt.plot(portfolio_over_time / 1000, color = "blue")
plt.plot(portfolio_over_time_bh / 1000, color = "orange")
plt.plot(value_qqq / 1000, color = "purple")
plt.xlabel("Day #")
plt.ylabel("Portfolio Value ($) in Thousands")
plt.title("Portfolio Value over Time (in Thousands)")
plt.legend(["ML Algorithm", "Buy and Hold AAPL", "QQQ"])
plt.show()

# Plot histogram of daily portfolio returns
portfolio_change = np.diff(portfolio_over_time)
percent_returns = np.divide(portfolio_change, portfolio_over_time[:-1]) * 100

plt.hist(percent_returns, bins = 10, color = "blue")
plt.xlabel("Daily Return (%)")
plt.ylabel("Frequency")
plt.title("Histogram of Daily Portfolio Returns: Realized and Unrealized\n(ML Algorithm)")
plt.show()

portfolio_change_bh = np.diff(portfolio_over_time_bh)
percent_returns_bh = np.divide(portfolio_change_bh, portfolio_over_time_bh[:-1]) * 100

plt.hist(percent_returns_bh, bins = 10, color = "orange")
plt.xlabel("Daily Return (%)")
plt.ylabel("Frequency")
plt.title("Histogram of Daily Portfolio Returns: Realized and Unrealized\n(Buy and Hold AAPL)")
plt.show()

stats.probplot(percent_returns, plot = plt)
plt.title("Normal Probability Plot of Daily Returns\n(ML Algorithm)")
plt.show()

stats.probplot(percent_returns_bh, plot = plt)
plt.title("Normal Probability Plot of Daily Returns\n(Buy and Hold AAPL)")
plt.show()

print("From both normal probability plots, it seems like the daily portfolio returns are approximately normally distributed.")
print("")

# =====================
# Metrics
# =====================

# Mean and standard deviation of daily returns
mean = np.mean(percent_returns)
std = np.std(percent_returns, ddof = 1)
mean_bh = np.mean(percent_returns_bh)
std_bh = np.std(percent_returns_bh, ddof = 1)

# CAGR
end_val = portfolio_over_time[-1]
end_val_bh = portfolio_over_time_bh[-1]
years = prices.size / 365
cagr = ((end_val / init_cash)**(1 / years) - 1) * 100
cagr_bh = ((end_val_bh / init_cash)**(1 / years) - 1) * 100

# Sharpe ratio
# Treasury rates obtained from https://www.treasury.gov/resource-center/data-chart-center/interest-rates/pages/TextView.aspx?data=yieldYear&year=2018
treasury = pd.read_csv("treasury_rates.csv")
treasury_rates = treasury.loc[:, ["1 yr"]]
treasury_rates.index = pd.to_datetime(treasury.loc[:, "Date"])

daily_returns = pd.DataFrame()
daily_returns["Return"] = percent_returns
daily_returns.index = pd.Series(df.loc[split_ind+1:, "Date"])

portfolio_vs_treas = daily_returns.join(how = "left", other = treasury_rates, on = daily_returns.index, lsuffix = "_1", rsuffix = "_2")
portfolio_vs_treas.columns = ["Daily Return", "1 Year Treasury"]

excess_returns = portfolio_vs_treas["Daily Return"] - portfolio_vs_treas["1 Year Treasury"]
treas_mean = np.mean(portfolio_vs_treas["1 Year Treasury"])
excess_returns_std = np.std(excess_returns, ddof = 1)

sharpe_ratio = (cagr - treas_mean) / excess_returns_std
sharpe_ratio = np.round(sharpe_ratio, 2)

print("---------------------------------------")
print("ML Algorithm")
print("---------------------------------------")
print("Mean daily return:", str(np.round(mean, 5)) + "%")
print("Standard deviation of daily returns (volatility):", str(np.round(std, 5)) + "%")
print("Annual return:", str(np.round(cagr, 2)) + "%")
print("Sharpe ratio:", sharpe_ratio)
print("")

print("---------------------------------------")
print("Buy and Hold AAPL")
print("---------------------------------------")
print("Mean daily return:", str(np.round(mean_bh, 5)) + "%")
print("Standard deviation of daily returns (volatility):", str(np.round(std_bh, 5)) + "%")
print("Annual return:", str(np.round(cagr_bh, 2)) + "%")

# Max drawdown for ML algorithm
# https://quant.stackexchange.com/questions/18094/how-can-i-calculate-the-maximum-drawdown-mdd-in-python
peak = pd.Series(portfolio_over_time).cummax()
daily_drawdown = prices / peak - 1.0
max_daily_drawdown = pd.Series(daily_drawdown).cummin()

# Plot max drawdown
plt.plot(max_daily_drawdown, color = "blue")
plt.xlabel("Day #")
plt.ylabel("Max Daily Drawdown")
plt.title("Max Daily Drawdown over Time\n(ML Algorithm)")
plt.show()

# Max drawdown for buy and hold AAPL
peak_bh = pd.Series(portfolio_over_time_bh).cummax()
daily_drawdown_bh = prices / peak_bh - 1.0
max_daily_drawdown_bh = pd.Series(daily_drawdown_bh).cummin()

# Plot max drawdown
plt.plot(max_daily_drawdown_bh, color = "orange")
plt.xlabel("Day #")
plt.ylabel("Max Daily Drawdown")
plt.title("Max Daily Drawdown over Time\n(Buy and Hold AAPL)")
plt.show()