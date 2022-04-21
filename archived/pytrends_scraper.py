import pandas as pd
import os
import pytrends
from pytrends.request import TrendReq

#####################
# Get pytrends data #
#####################

kw_list = ["Apple", "Apple Inc.", "IPhone", "IPad", "MacBook", "MacOS"]

begin_date = [2015, 7, 1]
end_date = [2018, 8, 31]

pytrends = TrendReq(hl='en-US', tz=360, retries=10)

#%%

pytrends_hourly = \
pytrends.get_historical_interest(kw_list, \
                                 year_start = 2015, month_start = 7, day_start = 1, hour_start = 0, \
                                 year_end = 2018, month_end = 8, day_end = 31, hour_end = 23, \
                                 cat = 0, geo = '', gprop = '', sleep = 60)

#%%

# Delete dummy variables
del begin_date, end_date, kw_list

# Delete the isPartial column
pytrends_hourly = pytrends_hourly.iloc[:, 0:-1]

##############################################
# Convert pytrends data from hourly to daily #
##############################################

# Define a remove_duplicates function for the purposes of creating a list of distinct
# dates from the pytrends dataframe
def remove_duplicates(thelist):
    newlist = []
    for item in thelist:
        if item not in newlist:
            newlist.append(item)
    
    return newlist

# Create a list of all dates in the dataframe
dates = []
for i in range(0, len(pytrends_hourly)):
    current_date = str(pytrends_hourly.index[i]).split(" ")[0]
    dates.append(current_date)

# Add a column to the dataframe with the date as a string
pytrends_hourly["Date"] = dates

# Remove duplicates to get individual dates, e.g. 2018-01-01, 2018-01-02, etc.
dates = remove_duplicates(dates)

# Split the giant dataframe into smaller dataframes containing values for one particular day
pytrends_split_daily = []
for i in range(0, len(dates)):
    x = pytrends_hourly[pytrends_hourly["Date"] == dates[i]]
    pytrends_split_daily.append(x)

# Total up each of those smaller dataframes, to get the values for each entire day
counts1 = []
counts2 = []
counts3 = []
counts4 = []
for dataframe in pytrends_split_daily:
    total1 = dataframe[["iphone"]].sum().iloc[0]
    total2 = dataframe[["ipad"]].sum().iloc[0]
    total3 = dataframe[["macbook"]].sum().iloc[0]
    total4 = dataframe[["macos"]].sum().iloc[0]
    counts1.append(total1)
    counts2.append(total2)
    counts3.append(total3)
    counts4.append(total4)

# Delete dummy variables
del current_date, dataframe, i, pytrends_split_daily, total1, total2, total3, total4, x

# Create a dataframe with the daily data
pytrends_daily = pd.DataFrame({"Date": dates, "iPhone": counts1, "iPad": counts2, "MacBook": counts3, "MacOS": counts4})

# Delete dummy variables
del dates, counts1, counts2, counts3, counts4