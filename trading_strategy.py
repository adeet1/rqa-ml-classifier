from project import df, X_test, Y_test_pred, Y_test, split_ind
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
buy_or_sell = Y_test_pred.values

# Convert prices array from type object to type float
prices = df.loc[split_ind:, "Close"].astype(float).values

# The change in shares
# positive = shares bought on day i (number of shares increases)
# negative = shares sold on day i (number of shares decreases)
shares_change = np.zeros_like(buy_or_sell).astype(int)
np.place(shares_change, buy_or_sell == 1, shares)
np.place(shares_change, buy_or_sell == 0, -shares)

# The change in cash
# positive = gross proceeds from selling shares
# negative = purchase amount for buying shares
cash_change = np.zeros_like(buy_or_sell).astype(float)

# Populate cash_change and sec_change arrays
for i in range(len(buy_or_sell)):
    # Buy
    if buy_or_sell[i] == 1:
        amount = shares * prices[i] + commission
        cash_change[i] = -amount
    
    # Sell
    if buy_or_sell[i] == 0:
        amount = shares * prices[i] - commission
        cash_change[i] = amount

# Cash over time
# (array index i is the amount of cash we have on day i)
# An increase in cash --> added gross proceeds from selling shares
# A decrease in cash --> subtracted purchase amount from buying shares
cash_over_time = init_cash + np.cumsum(cash_change)

# Shares over time
# (array index i is the number of shares we own on day i)
# An increase in shares --> shares bought
# A decrease in shares --> shares sold
shares_over_time = np.cumsum(shares_change).astype(int)

# Securities value over time
# (array index i is the securities value on day i)
sec_over_time = np.multiply(shares_over_time, prices)

# Total portfolio value over time
# (array index i is the portfolio value on day i)
portfolio_over_time = np.add(cash_over_time, sec_over_time)

# =====================
# Plots
# =====================

# Plot graph of portfolio value
plt.figure()
plt.plot(portfolio_over_time / 1000)
plt.xlabel("Day #")
plt.ylabel("Portfolio Value ($) in Thousands")
plt.title("Portfolio Value over Time (in Thousands)")
plt.show()

# Plot histogram of daily portfolio returns
portfolio_change = np.diff(portfolio_over_time)
percent_returns = np.divide(portfolio_change, portfolio_over_time[:-1]) * 100

plt.hist(percent_returns, bins = 10)
plt.xlabel("Daily Return (%)")
plt.ylabel("Frequency")
plt.title("Histogram of Daily Portfolio Returns (Realized and Unrealized)")
plt.show()

stats.probplot(percent_returns, plot = plt)
plt.show()

print("From the normal probability plot, it seems like the daily portfolio returns are approximately normally distributed.")
print("")

# =====================
# Metrics
# =====================

# Mean and standard deviation of daily returns
mean = np.mean(percent_returns)
std = np.std(percent_returns, ddof = 1)
print("Mean daily return:", str(np.round(mean, 5)) + "%")
print("Standard deviation of daily returns (volatility):", str(np.round(std, 5)) + "%")

# CAGR
end_val = portfolio_over_time[-1]
years = prices.size / 365
cagr = ((end_val / init_cash)**(1 / years) - 1) * 100
print("Annual return:", str(np.round(cagr, 2)) + "%")

# Max drawdown
# https://quant.stackexchange.com/questions/18094/how-can-i-calculate-the-maximum-drawdown-mdd-in-python
peak = pd.Series(portfolio_over_time).cummax()
daily_drawdown = prices / peak - 1.0
max_daily_drawdown = pd.Series(daily_drawdown).cummin()

# Plot max drawdown
plt.plot(max_daily_drawdown)
plt.xlabel("Day #")
plt.ylabel("Max Daily Drawdown")
plt.title("Max Daily Drawdown over Time")
plt.show()