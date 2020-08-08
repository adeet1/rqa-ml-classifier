# Import data
import pandas as pd
df = pd.read_excel("aapl.xlsx").iloc[2:, :].reset_index(drop = True)

# Remove the first 14 rows and the last row (we don't have future data in the present)
df = df.iloc[14:-1, :].reset_index(drop = True)
df["Wiki Move"] = df["Wiki Move"].astype(int)
df["Goog ROC"] = df["Goog ROC"].astype(float)

# Remove the columns that are used to calculate features
"""
These columns need to be removed for the following reasons:

Open, Close, High, Low................. reflected in features
Gain, Loss............................. reflected in Change in Close
Average Gain, Average Loss............. reflected in RS
RS..................................... reflected in 14-day Price RSI
Wiki Traffic, Wiki Traffic- 1 Day Lag.. reflected in Wiki features
PE Ratio............................... ignore this feature entirely
Wiki 5day disparity.................... reflected in Wiki 5day Disparity Move
Wiki MA3 Move.......................... reflected in Wiki 3day Disparity
Wiki MA5 Move.......................... reflected in Wiki EMA5 Move
Goog Total............................. reflected in Goog features
Goog Gain, Goog Loss................... reflected in Change in Goog
Goog Avg. Gain, Goog Avg. Loss......... reflected in RS
Goog RS................................ reflected in Change in Goog
Goog ROC............................... reflected in Goog ROC Move
Goog MA3............................... reflected in Goog 3day Disparity
Goog EMA5.............................. reflected in Goog EMA5 Move
Goog 3day Disparity.................... reflected in Goog 3day Disparity Move
Goog RSI (14 days)..................... reflected in Goog RSI Move
Price RSI (14 days).................... reflected in Price RSI Move
"""

df.drop(columns = ["Open", "High", "Low", "Gain", "Loss", \
                   "Average Gain", "Average Loss", "RS", "Wiki Traffic", \
                   "Wiki Traffic- 1 Day Lag", "PE Ratio", "Wiki 5day disparity", \
                   "Wiki MA3 Move", "Wiki MA5 Move", "Goog Total", "Goog Gain", \
                   "Goog Loss", "Goog Avg. Gain", "Goog Avg. Loss", "Goog RS", \
                   "Goog ROC", "Goog MA3", "Goog EMA5", "Goog 3day Disparity", \
                   "Goog RSI (14 days)", "Price RSI (14 days)"], inplace = True)

# Fix column data format
df["Stochastic Oscillator (14 days)"] = df["Stochastic Oscillator (14 days)"].astype(float)

# Train/test split
X = df.iloc[:, 2:-1]
Y = df.iloc[:, -1]

def dataset_split(X, Y, test_size):
    ind = int((1 - test_size) * len(X))
    X_train = X.iloc[:ind, :]
    X_test = X.iloc[ind:, :]
    Y_train = Y.iloc[:ind]
    Y_test = Y.iloc[ind:]
    return X_train, X_test, Y_train, Y_test, ind

X_train, X_test, Y_train, Y_test, split_ind = dataset_split(X, Y, test_size = 0.2)

# Analyze the balance of the classification problem
# i.e. the percentage of 1s and 0s in the target
counts = Y_train.groupby(by = Y_train).count()
counts = pd.Series(counts)
print(counts)

# Feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler().fit(X_train)
X_train = pd.DataFrame(sc.transform(X_train), columns = X.columns)
X_test = pd.DataFrame(sc.transform(X_test), columns = X.columns)

# Each feature's correlation with every other feature
corr_matrix = X_train.corr()

# Each feature's correlation with the target variable
feature_target_corr = X_train.corrwith(Y_train, method = "pearson")
feature_target_corr = feature_target_corr.sort_values(ascending = False)

# Each feature's correlation with every other feature
corr_matrix = X_train.corr()

# Select features based on correlation
all_features = X.columns.to_list()
selected_features = all_features[:]
for i in range(0, len(corr_matrix)):
    for j in range(i, len(corr_matrix)):
        # Process every pair of features
        corr = corr_matrix.iloc[i, j] 
        if abs(corr) > 0.5 and corr != 1:
            feature1 = all_features[i]
            feature2 = all_features[j]
            
            # Out of the two features in the current pair, remove the feature that
            # is less correlated with the target variable
            corr1 = feature_target_corr[feature1]
            corr2 = feature_target_corr[feature2]
            try:
                if abs(corr1) < abs(corr2):
                    selected_features.remove(feature1)
                    print("Removed from list of features:", feature1)
                else:
                    selected_features.remove(feature2)
                    print("Removed from list of features:", feature2)
            except ValueError:
                # Catch the error just in case we're trying to remove a feature
                # that's already been removed
                pass

X_train = X_train[selected_features]
X_test = X_test[selected_features]

corr_matrix_selected_features = X_train.corr()

# Plot number of components vs. explained variance for PCA algorithm
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
pca = PCA().fit(X_train)
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') # for each component
plt.show()

# Run PCA algorithm (the optimal number of components is wherever the elbow in the graph occurs)
pca = PCA(n_components = 5).fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

# Model =======================================================================================
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth = 4, random_state = 0)
model.fit(X_train_pca, Y_train)

# Make predictions
Y_test_pred = pd.Series(model.predict(X_test_pca)).astype(int)
Y_train_pred = pd.Series(model.predict(X_train_pca)).astype(int)

from sklearn.metrics import classification_report
print(classification_report(Y_test_pred, Y_test, target_names = ["be flat", "be long"]))

# Evaluate model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# Print metrics
metrics = \
[ \
[accuracy_score(Y_train, Y_train_pred), accuracy_score(Y_test, Y_test_pred)],
[precision_score(Y_train, Y_train_pred), precision_score(Y_test, Y_test_pred)], [recall_score(Y_train, Y_train_pred), recall_score(Y_test, Y_test_pred)],
[f1_score(Y_train, Y_train_pred), f1_score(Y_test, Y_test_pred)]
]
metrics = np.array(metrics) * 100

metrics = pd.DataFrame(metrics, columns = ["Training Set", "Test Set"])
metrics.insert(0, "Metric", ["Accuracy", "Precision", "Recall", "F1 Score"])
metrics.set_index("Metric", inplace = True)
print(metrics)
