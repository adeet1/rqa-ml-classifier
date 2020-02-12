# Stock Price Direction ML Classifier

## Overview
This is a machine learning classification project in which the direction of Apple Inc.'s (AAPL) next-day stock price movement is predicted using information such such as Wikipedia page traffic, Google News traffic, and other technical indicators. The ML model outputs a binary outcome:
* 1 (the stock price increased)
* 0 (the stock price decreased/didn't change)

This project is an attempt to replicate the findings (in Chapter 2) of a whitepaper written by Bin Weng. The paper has been included in this repository, and can be accessed [here](whitepaper.pdf).

## Authors
* Adeet Patel
* Andrew Holpe ([Richmond Quantitative Advisors](https://www.richmondquant.com/))

## Dataset
The [aapl.xlsx](aapl.xlsx) file contains data for Apple ranging from July 1, 2015 to August 31, 2018.

### Columns
* **Open**: The open stock price for the given day.
* **Close**: The close stock price for the given day.
* **High**: The highest stock price reached during the given day.
* **Low**: The lowest stock price reached during the given day.
* **Change in Close**: The previous day's close price subtracted from that of the given day.
* **Gain**: Equal to the absolute value of Change in Close if that value is positive, otherwise blank.
* **Loss**: Equal to the absolute value of Change in Close if that value is zero or negative, otherwise blank.
* **Average Gain**: The trailing 14-day average of the price gain.
* **Average Loss**: The trailing 14-day average of the price loss.
* **RS**: A momentum indicator, specifically the average gain divided by the average loss.
* **Wiki Traffic**: The amount of Wikipedia page traffic, calculated as the total of the page traffic for the following keywords:
    - AAPL
    - Apple Inc.
    - iPhone
    - iPad
    - MacBook
    - MacOS
* **Wiki Traffic- 1 Day Lag**: Equal to Wiki Traffic, but offset by 1 day. This is due to the fact that the Wiki traffic for a given day is not known until the next day. This feature is used in the model as it is more representative of the available data at a given point in time.
* **PE Ratio**: This feature is ignored.
* **Wiki 5day disparity**: The disparity for Wikipedia page traffic over a 5-day time period.
* **Wiki Move**: Boolean value; equal to 1 if the given day's Wiki traffic is larger than that of the previous day, and 0 otherwise.
* **Wiki MA3 Move**: The trailing 3-day average of the Wiki traffic.
* **Wiki MA5 Move**: The trailing 5-day average of the Wiki traffic.
* **Wiki EMA5 Move**: The exponential moving average of the Wiki traffic, calculated from Wiki MA5 Move.
* **Wiki 5day Disparity Move**: Boolean value; equal to 1 if the given day's Wiki 5-day disparity is larger than that of the previous day, and 0 otherwise.
* **Goog Total**: The amount of Google News traffic, calculated as the total of the traffic for the following keywords:
    - iPhone
    - iPad
    - MacBook
    - Apple Inc
    - iPod
    - Apple
* **Change in Goog**: The previous day's Google News traffic subtracted from that of the given day.
* **Goog Gain**: Equal to the absolute value of Change in Goog if that value is positive, otherwise blank.
* **Goog Loss**: Equal to the absolute value of Change in Goog if that value is zero or negative, otherwise blank.
* **Goog Avg. Gain**: The trailing 14-day average of the Google News gain.
* **Goog Avg. Loss**: The trailing 14-day average of the Google News loss.
* **Goog RS**: A momentum indicator, specifically the average Google gain divided by the average Google loss.
* **Goog ROC**: The rate of change for Google traffic.
* **Goog MA3**: The trailing 3-day average of the Google traffic.
* **Goog MA5**: The trailing 5-day average of the Google traffic.
* **Goog EMA5**: The exponential moving average of the Google traffic, calculated from the Goog MA5.
* **Goog EMA5 Move**: Boolean value; equal to 1 if the given day's Goog EMA5 is larger than that of the previous day, and 0 otherwise.
* **Goog 3day Disparity**: The disparity for Google News traffic over a 3-day time period.
* **Goog 3day Disparity Move**: Boolean value; equal to 1 if the given day's Goog 3day Disparity is larger than that of the previous day, and 0 otherwise.
* **Goog ROC Move**: Boolean value; equal to 1 if the given day's Goog ROC is larger than that of the previous day, and 0 otherwise.
* **Goog RSI (14 days)**: The relative strength index for Google traffic, measured across a 14-day time period.
* **Goog RSI Move**: Boolean value; equal to 1 if the given day's Goog RSI is larger than that of the previous day, and 0 otherwise.
* **Wiki 3day Disparity**: The disparity for Wikipedia page traffic over a 3-day time period.
* **Stochastic Oscillator (14 days)**: The stochastic oscillator for the low stock price, measured across a 14-day time period.
* **Price RSI (14 days)**: The relative strength index of the stock price, measured across a 14-day time period.
* **Price RSI Move**: Boolean value; equal to 1 if the given day's Price RSI is larger than that of the previous day, and 0 otherwise.
* **Google MA6**: The trailing 6-day average of the Google traffic.
* **Google_Move**: Boolean value; equal to 1 if the given day's total Google traffic is larger than that of the previous day, and 0 otherwise.
* **Target**: Boolean value; equal to 1 if the given day's close stock price is larger than that of the previous day, and 0 otherwise. This value is predicted by the model.

## Issues & Next Steps
Due to the low signal-to-noise ratio of this project's dataset (and financial market data in general), many of the attempted models (random forest, artificial neural networks, recurrent neural networks) are overfitting.

As an example, the metrics for the random forest model are shown below:
```
           Training Set  Test Set
Accuracy           73.6      45.9
Precision          69.8       0.0
Recall             87.2       0.0
F1 Score           77.5       0.0
```

The file [random-forest.py](random-forest.py) contains an in-progress implementation of a feature selection method. This implementation creates clustermaps via ```seaborn```, which provide a visual representation of the correlations between features. An attempt has been made to filter features based on their correlation.

Therefore, the next major step for this project will be feature selection, as it will dictate the reliability of the models.